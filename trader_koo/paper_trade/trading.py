"""Trade lifecycle helpers for paper trades."""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig
from trader_koo.paper_trade.config import config_snapshot
from trader_koo.paper_trade.decision import (
    compute_position_plan,
    compute_stop_and_target,
    evaluate_setup_for_paper_trade,
)
from trader_koo.paper_trade.schema import ensure_paper_trade_schema, register_bot_version
from trader_koo.paper_trade.summary import update_portfolio_snapshot

LOG = logging.getLogger(__name__)


def _build_review(
    *,
    exit_reason: str,
    r_multiple: float | None,
    expected_r_multiple: float | None,
) -> tuple[str, str]:
    expected_text = (
        f" vs plan {expected_r_multiple:.2f}R"
        if isinstance(expected_r_multiple, (int, float))
        else ""
    )
    if exit_reason == "target_hit":
        achieved = f"{r_multiple:.2f}R" if isinstance(r_multiple, (int, float)) else "target"
        return (
            "target_hit",
            f"Plan worked: target reached at {achieved}{expected_text}. Review whether a trailing exit could preserve trend continuation.",
        )
    if exit_reason == "stopped_out":
        return (
            "stopped_out",
            f"Invalidation was hit{expected_text}. Review whether the entry was early, the setup family is weakening, or confirmation should be stricter.",
        )
    if exit_reason == "expired":
        return (
            "timed_out",
            "Time stop triggered before the move resolved. Review whether entries need stronger momentum confirmation or shorter holding windows.",
        )
    return (
        "closed",
        f"Trade closed with {f'{r_multiple:.2f}R' if isinstance(r_multiple, (int, float)) else 'an unscored outcome'}{expected_text}. Compare discretion with the original plan.",
    )


def compute_pnl(
    direction: str,
    entry_price: float,
    current_price: float,
) -> float:
    """Return P&L percentage."""
    if direction == "long":
        return ((current_price / entry_price) - 1.0) * 100.0
    return (1.0 - (current_price / entry_price)) * 100.0


def compute_r_multiple(
    direction: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float | None,
    *,
    config: PaperTradeConfig,
) -> float | None:
    """Return R-multiple (profit / initial risk)."""
    if stop_loss is None:
        risk = entry_price * (config.default_stop_pct / 100.0)
    else:
        risk = abs(entry_price - stop_loss)
    if risk <= 0:
        return None

    if direction == "long":
        pnl_per_share = exit_price - entry_price
    else:
        pnl_per_share = entry_price - exit_price
    return round(pnl_per_share / risk, 2)


def _close_trade(
    conn: sqlite3.Connection,
    trade_id: int,
    exit_price: float,
    exit_date: str,
    exit_reason: str,
    direction: str,
    entry_price: float,
    stop_loss: float | None,
    *,
    config: PaperTradeConfig,
) -> None:
    pnl = round(compute_pnl(direction, entry_price, exit_price), 2)
    r_mult = compute_r_multiple(
        direction,
        entry_price,
        exit_price,
        stop_loss,
        config=config,
    )
    status = exit_reason if exit_reason in ("stopped_out", "target_hit", "expired") else "closed"
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    meta_row = conn.execute(
        "SELECT expected_r_multiple FROM paper_trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    expected_r_multiple = float(meta_row[0]) if meta_row and meta_row[0] is not None else None
    review_status, review_summary = _build_review(
        exit_reason=exit_reason,
        r_multiple=r_mult,
        expected_r_multiple=expected_r_multiple,
    )

    conn.execute(
        """
        UPDATE paper_trades SET
            status = ?,
            exit_price = ?,
            exit_date = ?,
            exit_reason = ?,
            pnl_pct = ?,
            r_multiple = ?,
            current_price = ?,
            unrealized_pnl_pct = NULL,
            review_status = ?,
            review_summary = ?,
            updated_ts = ?
        WHERE id = ?
        """,
        (
            status,
            exit_price,
            exit_date,
            exit_reason,
            pnl,
            r_mult,
            exit_price,
            review_status,
            review_summary,
            now,
            trade_id,
        ),
    )


def create_paper_trades_from_report(
    conn: sqlite3.Connection,
    *,
    setup_rows: list[dict[str, Any]],
    report_date: str,
    generated_ts: str,
    config: PaperTradeConfig,
) -> int:
    """Create paper trades from qualifying daily report setups."""
    if not report_date or not setup_rows:
        return 0

    ensure_paper_trade_schema(conn)
    register_bot_version(
        conn,
        bot_version=config.bot_version,
        decision_version=config.decision_version,
        config_json=json.dumps(config_snapshot(config)),
        notes="Current champion paper-trade policy snapshot.",
    )

    open_count = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
    ).fetchone()[0]

    if open_count >= config.max_open:
        LOG.info(
            "Paper trades: %d open trades already at max (%d), skipping creation",
            open_count, config.max_open,
        )
        return 0

    remaining_slots = config.max_open - open_count
    inserted = 0

    for row in setup_rows:
        if inserted >= remaining_slots:
            break
        if not isinstance(row, dict):
            continue
        evaluation = evaluate_setup_for_paper_trade(row, config=config)
        if not evaluation["approved"]:
            continue

        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue

        direction = str(evaluation["direction"])
        entry_price = float(row["close"])
        levels = compute_stop_and_target(row, direction, config=config)
        plan = compute_position_plan(row, evaluation, levels, config=config)

        expected_r_multiple = plan.get("expected_r_multiple")
        if (
            isinstance(expected_r_multiple, (int, float))
            and expected_r_multiple < config.min_reward_r_multiple
        ):
            LOG.info(
                "Paper trade skipped: %s %s only offers %.2fR (< %.2fR minimum)",
                direction.upper(),
                ticker,
                float(expected_r_multiple),
                config.min_reward_r_multiple,
            )
            continue

        # Capture market context at entry (VIX, regime, HMM state)
        from trader_koo.paper_trade.context import capture_market_context

        market_ctx = capture_market_context(conn)
        market_ctx["bot_version"] = config.bot_version

        before_changes = conn.total_changes
        conn.execute(
            """
            INSERT INTO paper_trades (
                report_date, generated_ts, ticker, direction,
                entry_price, entry_date, target_price, stop_loss, atr_at_entry,
                status, current_price, unrealized_pnl_pct,
                high_water_mark, low_water_mark,
                setup_family, setup_tier, score, signal_bias, actionability,
                observation, action_text, risk_note,
                yolo_pattern, yolo_recency, debate_agreement_score,
                decision_version, decision_state, analyst_stage, debate_stage,
                risk_stage, portfolio_decision, decision_summary,
                decision_reasons, risk_flags,
                position_size_pct, risk_budget_pct, stop_distance_pct,
                expected_reward_pct, expected_r_multiple,
                entry_plan, exit_plan, sizing_summary,
                review_status, review_summary,
                bot_version, vix_at_entry, vix_percentile_at_entry,
                regime_state_at_entry, hmm_regime_at_entry, hmm_confidence_at_entry
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                'open', ?, 0.0,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(report_date, ticker, direction) DO NOTHING
            """,
            (
                report_date,
                generated_ts,
                ticker,
                direction,
                entry_price,
                report_date,
                levels["target_price"],
                levels["stop_loss"],
                levels["atr_at_entry"],
                entry_price,
                entry_price,
                entry_price,
                row.get("setup_family"),
                row.get("setup_tier"),
                row.get("score"),
                row.get("signal_bias"),
                row.get("actionability"),
                row.get("observation"),
                row.get("action"),
                row.get("risk_note"),
                row.get("yolo_pattern"),
                row.get("yolo_recency"),
                row.get("debate_agreement_score"),
                evaluation["decision_version"],
                evaluation["decision_state"],
                evaluation["analyst_stage"],
                evaluation["debate_stage"],
                evaluation["risk_stage"],
                evaluation["portfolio_decision"],
                evaluation["decision_summary"],
                json.dumps(evaluation["decision_reasons"]),
                json.dumps(evaluation["risk_flags"]),
                plan["position_size_pct"],
                plan["risk_budget_pct"],
                plan["stop_distance_pct"],
                plan["expected_reward_pct"],
                plan["expected_r_multiple"],
                plan["entry_plan"],
                plan["exit_plan"],
                plan["sizing_summary"],
                plan["review_status"],
                plan["review_summary"],
                market_ctx["bot_version"],
                market_ctx["vix_at_entry"],
                market_ctx["vix_percentile_at_entry"],
                market_ctx["regime_state_at_entry"],
                market_ctx["hmm_regime_at_entry"],
                market_ctx["hmm_confidence_at_entry"],
            ),
        )
        if conn.total_changes > before_changes:
            inserted += 1
            LOG.info(
                "Paper trade created: %s %s @ %.2f (stop=%.2f target=%.2f)",
                direction.upper(), ticker, entry_price,
                levels["stop_loss"], levels["target_price"],
            )

    return inserted


def mark_to_market(
    conn: sqlite3.Connection,
    *,
    config: PaperTradeConfig,
) -> dict[str, Any]:
    """Update all open paper trades with latest prices."""
    ensure_paper_trade_schema(conn)

    open_rows = conn.execute(
        """
        SELECT id, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, high_water_mark, low_water_mark
        FROM paper_trades WHERE status = 'open'
        """
    ).fetchall()

    if not open_rows:
        update_portfolio_snapshot(conn)
        return {"open_trades": 0, "updated": 0, "closed": 0}

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    updated = 0
    closed = 0

    for row in open_rows:
        trade_id, ticker, direction, entry_price, entry_date = row[:5]
        target_price, stop_loss, high_water_mark, low_water_mark = row[5:]

        price_row = conn.execute(
            "SELECT CAST(close AS REAL), date FROM price_daily "
            "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()

        if not price_row or price_row[0] is None:
            continue

        current_price = float(price_row[0])
        price_date = price_row[1]
        unrealized = round(compute_pnl(direction, entry_price, current_price), 2)
        new_hwm = max(high_water_mark or current_price, current_price)
        new_lwm = min(low_water_mark or current_price, current_price)

        hit_stop = False
        if stop_loss is not None:
            if direction == "long" and current_price <= stop_loss:
                hit_stop = True
            elif direction == "short" and current_price >= stop_loss:
                hit_stop = True

        hit_target = False
        if target_price is not None:
            if direction == "long" and current_price >= target_price:
                hit_target = True
            elif direction == "short" and current_price <= target_price:
                hit_target = True

        expired = False
        try:
            entry_dt = dt.datetime.strptime(entry_date, "%Y-%m-%d")
            today_dt = dt.datetime.strptime(today, "%Y-%m-%d")
            if (today_dt - entry_dt).days >= config.expiry_days:
                expired = True
        except (ValueError, TypeError):
            pass

        if hit_stop:
            _close_trade(
                conn,
                trade_id,
                current_price,
                today,
                "stopped_out",
                direction,
                entry_price,
                stop_loss,
                config=config,
            )
            closed += 1
        elif hit_target:
            _close_trade(
                conn,
                trade_id,
                current_price,
                today,
                "target_hit",
                direction,
                entry_price,
                stop_loss,
                config=config,
            )
            closed += 1
        elif expired:
            _close_trade(
                conn,
                trade_id,
                current_price,
                today,
                "expired",
                direction,
                entry_price,
                stop_loss,
                config=config,
            )
            closed += 1
        else:
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE paper_trades SET
                    current_price = ?, unrealized_pnl_pct = ?,
                    last_mtm_date = ?, high_water_mark = ?, low_water_mark = ?,
                    updated_ts = ?
                WHERE id = ?
                """,
                (current_price, unrealized, price_date, new_hwm, new_lwm, now, trade_id),
            )
        updated += 1

    update_portfolio_snapshot(conn)
    return {"open_trades": len(open_rows) - closed, "updated": updated, "closed": closed}


def manually_close_trade(
    conn: sqlite3.Connection,
    *,
    trade_id: int,
    exit_price: float | None = None,
    exit_reason: str = "manual_close",
    config: PaperTradeConfig,
) -> dict[str, Any]:
    """Manually close an open paper trade."""
    row = conn.execute(
        "SELECT ticker, direction, entry_price, stop_loss, status FROM paper_trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Paper trade {trade_id} not found")
    ticker, direction, entry_price, stop_loss, status = row
    if status != "open":
        raise ValueError(f"Paper trade {trade_id} is already {status}")

    if exit_price is None:
        price_row = conn.execute(
            "SELECT CAST(close AS REAL) FROM price_daily WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if not price_row or price_row[0] is None:
            raise ValueError(f"No price data for {ticker} to close trade")
        exit_price = float(price_row[0])

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    _close_trade(
        conn,
        trade_id,
        exit_price,
        today,
        exit_reason,
        direction,
        entry_price,
        stop_loss,
        config=config,
    )
    conn.commit()

    pnl = round(compute_pnl(direction, entry_price, exit_price), 2)
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "direction": direction,
        "exit_price": exit_price,
        "pnl_pct": pnl,
        "status": "closed",
    }

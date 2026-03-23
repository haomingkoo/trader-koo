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
    raw_pnl = compute_pnl(direction, entry_price, exit_price)

    # Deduct trading costs from P&L
    # 1. Commission: entry + exit as % of entry price
    position_row = conn.execute(
        "SELECT position_size_pct FROM paper_trades WHERE id = ?", (trade_id,),
    ).fetchone()
    pos_pct = float(position_row[0] or 8.0) if position_row and position_row[0] is not None else 8.0
    starting_capital = 1_000_000.0
    notional = starting_capital * (pos_pct / 100)
    commission_cost_pct = (config.commission_per_trade * 2 / notional) * 100 if notional > 0 else 0

    # 2. Short borrow cost (annualized, pro-rated to TRADING days held)
    borrow_cost_pct = 0.0
    if direction == "short":
        entry_date_row = conn.execute(
            "SELECT entry_date FROM paper_trades WHERE id = ?", (trade_id,),
        ).fetchone()
        if entry_date_row and entry_date_row[0]:
            try:
                # Count actual trading days (rows in price_daily) between entry and exit
                trading_days_row = conn.execute(
                    "SELECT COUNT(*) FROM price_daily "
                    "WHERE ticker = 'SPY' AND date > ? AND date <= ?",
                    (entry_date_row[0], exit_date),
                ).fetchone()
                trading_days = int(trading_days_row[0]) if trading_days_row and trading_days_row[0] else 0
                if trading_days == 0:
                    # Fallback: calendar days if no SPY data
                    trading_days = max(1, (
                        dt.datetime.strptime(exit_date, "%Y-%m-%d")
                        - dt.datetime.strptime(entry_date_row[0], "%Y-%m-%d")
                    ).days)
                borrow_cost_pct = config.short_borrow_annual_pct * trading_days / 252
            except (ValueError, TypeError):
                pass

    pnl = round(raw_pnl - commission_cost_pct - borrow_cost_pct, 2)
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

    # Portfolio drawdown circuit breaker — halt new entries if drawdown exceeds limit
    try:
        snapshot_row = conn.execute(
            "SELECT equity_index FROM paper_portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        ).fetchone()
        if snapshot_row and snapshot_row[0] is not None:
            equity_index = float(snapshot_row[0])
            drawdown_pct = (100.0 - equity_index)  # equity starts at 100
            if drawdown_pct >= config.max_drawdown_pct:
                LOG.warning(
                    "CIRCUIT BREAKER: portfolio drawdown %.1f%% exceeds %.1f%% limit, blocking new entries",
                    drawdown_pct, config.max_drawdown_pct,
                )
                return 0
    except Exception as exc:
        LOG.debug("Drawdown check skipped: %s", exc)

    # Daily loss circuit breaker — halt if today's realized losses exceed limit
    try:
        daily_loss_row = conn.execute(
            "SELECT SUM(pnl_pct) FROM paper_trades "
            "WHERE exit_date = ? AND status != 'open' AND pnl_pct IS NOT NULL",
            (report_date,),
        ).fetchone()
        daily_loss = float(daily_loss_row[0]) if daily_loss_row and daily_loss_row[0] else 0.0
        if daily_loss < 0 and abs(daily_loss) >= config.max_daily_loss_pct:
            LOG.warning(
                "CIRCUIT BREAKER: daily loss %.1f%% exceeds %.1f%% limit, blocking new entries",
                abs(daily_loss), config.max_daily_loss_pct,
            )
            return 0
    except Exception as exc:
        LOG.debug("Daily loss check skipped: %s", exc)

    remaining_slots = config.max_open - open_count
    inserted = 0

    # Pre-fetch VIX level once for position sizing (used by all trades this batch)
    _vix_level: float | None = None
    try:
        _vix_row = conn.execute(
            "SELECT CAST(close AS REAL) FROM price_daily "
            "WHERE ticker = '^VIX' AND close IS NOT NULL ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if _vix_row and _vix_row[0] is not None:
            _vix_level = float(_vix_row[0])
    except Exception:
        pass

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

        # Entry price = NEXT DAY OPEN (signal generates after close,
        # earliest possible entry is next trading day's open).
        # Fall back to today's close if next-day open not yet available.
        next_open_row = conn.execute(
            "SELECT CAST(open AS REAL), date FROM price_daily "
            "WHERE ticker = ? AND date > ? ORDER BY date ASC LIMIT 1",
            (ticker, report_date),
        ).fetchone()
        if next_open_row and next_open_row[0] is not None:
            raw_entry = float(next_open_row[0])
            entry_date_actual = next_open_row[1]
        else:
            # Next day data not available yet (live trading edge);
            # use today's close as best estimate
            raw_entry = float(row["close"])
            entry_date_actual = report_date

        # Apply entry slippage on top of open price
        slip_mult = config.entry_slippage_bps / 10_000
        if direction == "long":
            entry_price = round(raw_entry * (1 + slip_mult), 4)
        else:
            entry_price = round(raw_entry * (1 - slip_mult), 4)
        levels = compute_stop_and_target(row, direction, config=config)
        plan = compute_position_plan(row, evaluation, levels, config=config, vix_level=_vix_level)

        # ADV liquidity check: reject if position > max_adv_pct of daily volume
        try:
            vol_row = conn.execute(
                "SELECT AVG(vol) FROM ("
                "  SELECT CAST(volume AS REAL) AS vol FROM price_daily"
                "  WHERE ticker = ? AND volume IS NOT NULL"
                "  ORDER BY date DESC LIMIT 20"
                ")",
                (ticker,),
            ).fetchone()
            if vol_row and vol_row[0] and vol_row[0] > 0:
                avg_daily_volume = float(vol_row[0])
                position_pct = float(plan.get("position_size_pct") or 8.0)
                position_dollars = 1_000_000.0 * (position_pct / 100)
                position_shares = position_dollars / entry_price if entry_price > 0 else 0
                adv_pct = (position_shares / avg_daily_volume) * 100 if avg_daily_volume > 0 else 0
                if adv_pct > config.max_adv_pct:
                    LOG.info(
                        "Paper trade skipped: %s %s position is %.1f%% of ADV (> %.1f%% max)",
                        direction.upper(), ticker, adv_pct, config.max_adv_pct,
                    )
                    continue
        except Exception as exc:
            LOG.debug("ADV check skipped: %s", exc)

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

        # ML score — OBSERVATION MODE: score trades but never reject.
        # The model's AUC (0.5235) is too low to gate trades reliably.
        # Scores are recorded on the trade for post-hoc analysis so we
        # can evaluate when the model improves enough to re-enable filtering.
        ml_prediction: dict[str, Any] = {}
        if config.ml_enabled:
            try:
                from trader_koo.ml.scorer import score_single_ticker

                ml_score = score_single_ticker(
                    conn, ticker=ticker, as_of_date=report_date,
                )
                if (
                    ml_score.get("model_available")
                    and ml_score.get("predicted_win_prob") is not None
                ):
                    ml_prediction = ml_score
                    LOG.info(
                        "ML observation: %s %s win_prob=%.2f (threshold %.2f, NOT filtering)",
                        direction.upper(),
                        ticker,
                        ml_score["predicted_win_prob"],
                        config.ml_min_win_prob,
                    )
                else:
                    LOG.debug(
                        "ML scoring unavailable for %s: %s",
                        ticker, ml_score.get("note", "no model"),
                    )
            except Exception as exc:
                LOG.warning("ML scoring failed (allowing trade): %s", exc)

        # Capture market context at entry (VIX, regime, HMM state)
        from trader_koo.paper_trade.context import capture_market_context

        market_ctx = capture_market_context(conn)
        market_ctx["bot_version"] = config.bot_version

        # Critic review — devil's advocate that kills low-conviction trades
        try:
            from trader_koo.paper_trade.critic import critic_review

            critic = critic_review(
                conn,
                row=row,
                evaluation=evaluation,
                plan=plan,
                market_ctx=market_ctx,
                max_open=config.max_open,
            )
            if not critic["approved"]:
                LOG.info(
                    "Critic REJECTED: %s %s — %s",
                    direction.upper(),
                    ticker,
                    critic["rejections"][0] if critic["rejections"] else "failed critic review",
                )
                continue
            LOG.info(
                "Critic APPROVED: %s %s (%s) — %d/%d checks passed",
                direction.upper(),
                ticker,
                critic["conviction_grade"],
                critic["checks_passed"],
                critic["checks_total"],
            )
        except Exception as exc:
            LOG.warning("Critic check failed (allowing trade): %s", exc)

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
                regime_state_at_entry, hmm_regime_at_entry, hmm_confidence_at_entry,
                directional_regime_at_entry, directional_regime_confidence,
                ml_predicted_win_prob, ml_confidence, ml_signal
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
                ?, ?, ?,
                ?, ?,
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
                entry_date_actual,
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
                market_ctx.get("directional_regime_at_entry"),
                market_ctx.get("directional_regime_confidence"),
                ml_prediction.get("predicted_win_prob"),
                ml_prediction.get("confidence"),
                ml_prediction.get("signal"),
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
            "SELECT CAST(close AS REAL), date, "
            "CAST(high AS REAL), CAST(low AS REAL), CAST(open AS REAL) "
            "FROM price_daily "
            "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()

        if not price_row or price_row[0] is None:
            continue

        current_price = float(price_row[0])
        price_date = price_row[1]
        day_high = float(price_row[2]) if price_row[2] is not None else current_price
        day_low = float(price_row[3]) if price_row[3] is not None else current_price
        day_open = float(price_row[4]) if price_row[4] is not None else current_price
        unrealized = round(compute_pnl(direction, entry_price, current_price), 2)
        # Track HWM/LWM using intraday extremes for realistic trailing stops
        new_hwm = max(high_water_mark or day_high, day_high)
        new_lwm = min(low_water_mark or day_low, day_low)

        # --- Stop / target detection using OHLC ---
        # Priority 1: Check if OPEN itself breaches stop or target (no ambiguity)
        # Priority 2: Check intraday high/low
        # Priority 3: If both stop AND target hit intraday, assume stop first (conservative)
        hit_stop = False
        hit_target = False

        if direction == "long":
            open_hits_stop = stop_loss is not None and day_open <= stop_loss
            open_hits_target = target_price is not None and day_open >= target_price
            intraday_hits_stop = stop_loss is not None and day_low <= stop_loss
            intraday_hits_target = target_price is not None and day_high >= target_price
        else:  # short
            open_hits_stop = stop_loss is not None and day_open >= stop_loss
            open_hits_target = target_price is not None and day_open <= target_price
            intraday_hits_stop = stop_loss is not None and day_high >= stop_loss
            intraday_hits_target = target_price is not None and day_low <= target_price

        # Apply exit slippage multiplier
        exit_slip = config.exit_slippage_bps / 10_000

        if open_hits_stop:
            # Open gapped past stop - fill at OPEN (realistic gap loss)
            hit_stop = True
            current_price = day_open
        elif open_hits_target:
            # Open gapped past target - fill at target (limit order fills at limit)
            hit_target = True
            current_price = target_price
        elif intraday_hits_stop and intraday_hits_target:
            # Both hit intraday - conservative: assume stop hit first
            hit_stop = True
            if direction == "long":
                current_price = round(stop_loss * (1 - exit_slip), 4)
            else:
                current_price = round(stop_loss * (1 + exit_slip), 4)
        elif intraday_hits_stop:
            hit_stop = True
            # Stop is a market order - apply slippage against you
            if direction == "long":
                current_price = round(stop_loss * (1 - exit_slip), 4)
            else:
                current_price = round(stop_loss * (1 + exit_slip), 4)
        elif intraday_hits_target:
            # Target is a limit order - fills at exact level (no slippage)
            hit_target = True
            current_price = target_price

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
            # Trailing stop logic: protect profits on winning trades
            new_stop = stop_loss
            if stop_loss is not None and entry_price > 0:
                risk = abs(entry_price - stop_loss)
                if risk > 0:
                    if direction == "long":
                        current_r = (new_hwm - entry_price) / risk
                        if current_r >= 1.5:
                            # Trail stop at HWM minus 0.5R
                            trail_stop = new_hwm - (0.5 * risk)
                            new_stop = max(stop_loss, trail_stop)
                        elif current_r >= 1.0:
                            # Move stop to breakeven
                            new_stop = max(stop_loss, entry_price)
                    else:  # short
                        current_r = (entry_price - new_lwm) / risk
                        if current_r >= 1.5:
                            trail_stop = new_lwm + (0.5 * risk)
                            new_stop = min(stop_loss, trail_stop)
                        elif current_r >= 1.0:
                            new_stop = min(stop_loss, entry_price)

            now = dt.datetime.now(dt.timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE paper_trades SET
                    current_price = ?, unrealized_pnl_pct = ?,
                    last_mtm_date = ?, high_water_mark = ?, low_water_mark = ?,
                    stop_loss = ?, updated_ts = ?
                WHERE id = ?
                """,
                (current_price, unrealized, price_date, new_hwm, new_lwm, new_stop, now, trade_id),
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

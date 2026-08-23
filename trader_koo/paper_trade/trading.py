"""Trade lifecycle helpers for paper trades."""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig
from trader_koo.paper_trade.config import config_snapshot
from trader_koo.paper_trade.chronology import (
    next_scheduled_session_after,
    publication_precedes_session_open,
)
from trader_koo.paper_trade.campaign import (
    canonical_hash,
    canonical_json,
    decide_candidate,
    DivergentDecisionSetError,
    persist_decision_set,
)
from trader_koo.paper_trade.decision import (
    compute_position_plan,
    compute_stop_and_target,
    evaluate_setup_for_paper_trade,
)
from trader_koo.paper_trade.schema import ensure_paper_trade_schema, register_bot_version
from trader_koo.paper_trade.shadow import (
    record_breadth_shadow,
    resolve_breadth_shadow_outcomes,
)
from trader_koo.paper_trade.summary import update_portfolio_snapshot
from trader_koo.paper_trade.portfolio_accounting import reconcile_portfolio
from trader_koo.db.price_contract import research_price_contract
from trader_koo.research.next_open_baseline import (
    adverse_fill_price,
    resolve_barrier_exit,
)

LOG = logging.getLogger(__name__)


def _pending_order_hash(
    *, order_id: str, report_run_id: str, report_date: str, generated_ts: str,
    campaign_id: str, policy_version: str, candidate_rank: int, ticker: str,
    direction: str, candidate_json: str, critic_json: str,
    market_context_json: str, avg_daily_volume: float | None,
) -> str:
    return canonical_hash({
        "order_id": order_id, "report_run_id": report_run_id,
        "report_date": report_date, "generated_ts": generated_ts,
        "campaign_id": campaign_id, "policy_version": policy_version,
        "candidate_rank": candidate_rank, "ticker": ticker,
        "direction": direction, "candidate_json": candidate_json,
        "critic_json": critic_json, "market_context_json": market_context_json,
        "avg_daily_volume": avg_daily_volume,
    })


def _record_trade_event(
    conn: sqlite3.Connection,
    *,
    trade_id: int,
    event_type: str,
    event_date: str,
    payload: dict[str, Any],
) -> None:
    payload_json = canonical_json(payload)
    conn.execute(
        """INSERT OR IGNORE INTO paper_trade_events
               (trade_id,event_type,event_date,payload_json,payload_hash)
           VALUES (?,?,?,?,?)""",
        (trade_id, event_type, event_date, payload_json, canonical_hash(payload)),
    )


def _entry_accounting(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    entry_price: float,
    position_size_pct: float,
    config: PaperTradeConfig,
) -> dict[str, float] | None:
    account = reconcile_portfolio(
        conn,
        campaign_id=campaign_id,
        starting_capital=config.starting_capital,
    )
    target_notional = float(account["equity"]) * position_size_pct / 100
    quantity = int(target_notional / entry_price) if entry_price > 0 else 0
    commission = float(config.commission_per_trade)
    while quantity > 0 and quantity * entry_price + commission > float(account["cash"]):
        quantity -= 1
    if quantity < 1:
        return None
    return {
        "quantity": float(quantity),
        "entry_notional": quantity * entry_price,
        "entry_commission": commission,
        "equity_before": float(account["equity"]),
        "cash_before": float(account["cash"]),
    }


def _run_owned_transaction(conn: sqlite3.Connection, operation: Any) -> Any:
    """Run a mutation in one snapshot without committing caller-owned work."""
    owned = not conn.in_transaction
    if owned:
        conn.execute("BEGIN")
    try:
        result = operation()
        if owned:
            conn.commit()
        return result
    except Exception:
        if owned:
            conn.rollback()
        raise


def _require_published_canonical_report(
    conn: sqlite3.Connection,
    report_run_id: str,
    *,
    require_current: bool = True,
) -> dict[str, Any]:
    """Resolve lineage only from #230's authoritative registry."""
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='report_runs'"
    ).fetchone()
    if not table:
        raise ValueError("paper admission requires the authoritative report_runs registry")
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(report_runs)")}
    required = {"run_id", "artifact_path", "publication_verified"}
    if not required.issubset(columns):
        raise ValueError("report_runs registry does not expose verified publication lineage")
    row = conn.execute(
        "SELECT artifact_path,published_ts FROM report_runs WHERE run_id=?",
        (report_run_id,),
    ).fetchone()
    if not row or not str(row[0] or "").strip():
        raise ValueError("paper admission requires a verified report artifact")
    from pathlib import Path
    from trader_koo.report.runs import resolve_published_report

    resolved = resolve_published_report(
        conn,
        report_dir=Path(str(row[0])).parent,
        run_id=report_run_id,
        require_current=require_current,
    )
    if resolved is None:
        raise ValueError("paper admission requires the current verified publication")
    return {
        "report_complete": True,
        "is_canonical": True,
        "published_ts": str(row[1] or ""),
    }


def _advance_paper_book(
    conn: sqlite3.Connection,
    *,
    config: PaperTradeConfig,
    through_date: str,
) -> None:
    """Apply each session's opens before its barriers and closing marks."""
    earliest = conn.execute(
        "SELECT MIN(report_date) FROM paper_pending_orders WHERE status='pending'"
    ).fetchone()[0]
    if not earliest:
        _mark_to_market(conn, config=config, through_date=through_date)
        return
    sessions = [
        str(row[0])
        for row in conn.execute(
            """SELECT date FROM price_daily
               WHERE ticker='SPY' AND date>? AND date<=? AND open IS NOT NULL
               ORDER BY date""",
            (str(earliest), through_date),
        )
    ]
    if not sessions:
        _mark_to_market(conn, config=config, through_date=through_date)
        return
    for session_date in sessions:
        fill_pending_paper_orders(
            conn, config=config, through_date=session_date,
        )
        _mark_to_market(conn, config=config, through_date=session_date)


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
    if exit_reason == "trailing_stop":
        achieved = f"{r_multiple:.2f}R" if isinstance(r_multiple, (int, float)) else "a protected gain"
        return (
            "trailing_stop",
            f"Protective trailing stop was hit at {achieved}{expected_text}. Review whether the trail locked gains too early or the target was too ambitious.",
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


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _append_unique(items: list[str], value: Any) -> None:
    text = _clean_text(value)
    if text and text not in items:
        items.append(text)


def _critic_reason_text(raw: Any) -> str:
    text = _clean_text(raw)
    if not text:
        return ""
    if "]: " in text:
        return text.split("]: ", 1)[1]
    return text


def _build_entry_rationale(
    *,
    ticker: str,
    direction: str,
    row: dict[str, Any],
    evaluation: dict[str, Any],
    plan: dict[str, Any],
    market_ctx: dict[str, Any],
    critic: dict[str, Any],
    ml_prediction: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact trade-entry journal for auditability and UI display."""
    family = _clean_text(row.get("setup_family")) or "setup"
    tier = _clean_text(row.get("setup_tier")).upper() or "?"
    score = row.get("score")
    score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "n/a"
    agreement = row.get("debate_agreement_score")
    agreement_text = f"{float(agreement):.0f}%" if isinstance(agreement, (int, float)) else "n/a"
    critic_grade = _clean_text(critic.get("conviction_grade")).upper() or "APPROVED"
    checks_passed = critic.get("checks_passed")
    checks_total = critic.get("checks_total")
    checks_text = (
        f"{checks_passed}/{checks_total}"
        if isinstance(checks_passed, int) and isinstance(checks_total, int)
        else "critic"
    )

    entry_reason = (
        f"{ticker} {direction}: {family} {tier}-tier setup, score {score_text}, "
        f"debate agreement {agreement_text}; critic {critic_grade} passed {checks_text} checks."
    )

    evidence: list[str] = []
    _append_unique(evidence, row.get("observation"))
    _append_unique(evidence, row.get("action"))
    _append_unique(
        evidence,
        f"Planned reward/risk {plan.get('expected_r_multiple')}R; "
        f"size {plan.get('position_size_pct')}% notional; "
        f"risk budget {plan.get('risk_budget_pct')}%.",
    )
    yolo_pattern = _clean_text(row.get("yolo_pattern"))
    if yolo_pattern:
        recency = _clean_text(row.get("yolo_recency")) or "unknown"
        _append_unique(evidence, f"Pattern context: {recency} {yolo_pattern}.")
    vix = market_ctx.get("vix_at_entry")
    regime = _clean_text(market_ctx.get("regime_state_at_entry"))
    if isinstance(vix, (int, float)) or regime:
        _append_unique(
            evidence,
            f"Market context: VIX {float(vix):.1f}" if isinstance(vix, (int, float)) else f"Market context: {regime}",
        )
    if ml_prediction.get("predicted_win_prob") is not None:
        label = _clean_text(ml_prediction.get("prediction_label")) or "ML probability"
        _append_unique(
            evidence,
            f"{label}: {float(ml_prediction['predicted_win_prob']) * 100:.0f}% observation-only.",
        )

    for raw in critic.get("critic_reasons") or []:
        if not str(raw).startswith("PASS"):
            continue
        text = _critic_reason_text(raw)
        if any(
            key in str(raw)
            for key in ("conviction_grade", "debate_strength", "risk_reward", "regime_alignment", "family_edge")
        ):
            _append_unique(evidence, text)
        if len(evidence) >= 8:
            break

    risks: list[str] = []
    _append_unique(risks, row.get("risk_note"))
    for flag in evaluation.get("risk_flags") or []:
        _append_unique(risks, flag)
    if evaluation.get("decision_state") == "approved_with_flags":
        _append_unique(risks, "Approved with caution flags.")
    for raw in critic.get("critic_reasons") or []:
        if not str(raw).startswith("PASS"):
            _append_unique(risks, _critic_reason_text(raw))
    if not ml_prediction:
        _append_unique(risks, "ML did not filter this entry; model remains observation-only or unavailable.")

    return {
        "entry_reason": entry_reason,
        "entry_evidence": evidence[:8],
        "entry_risks": risks[:6],
    }


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


def _stop_exit_reason(direction: str, entry_price: float, exit_price: float) -> str:
    """Return a clearer reason for stops that close after protecting gains."""
    pnl = compute_pnl(direction, entry_price, exit_price)
    return "trailing_stop" if pnl > 0 else "stopped_out"


def _compute_spy_return_pct(
    conn: sqlite3.Connection,
    *,
    entry_date: str | None,
    exit_date: str,
) -> float | None:
    if not entry_date:
        return None
    start_row = conn.execute(
        "SELECT CAST(close AS REAL) FROM price_daily "
        "WHERE ticker = 'SPY' AND date >= ? ORDER BY date ASC LIMIT 1",
        (entry_date,),
    ).fetchone()
    end_row = conn.execute(
        "SELECT CAST(close AS REAL) FROM price_daily "
        "WHERE ticker = 'SPY' AND date <= ? ORDER BY date DESC LIMIT 1",
        (exit_date,),
    ).fetchone()
    if not start_row or not end_row or start_row[0] is None or end_row[0] is None:
        return None
    start = float(start_row[0])
    end = float(end_row[0])
    if start <= 0:
        return None
    return round((end / start - 1.0) * 100.0, 2)


def _lesson_from_outcome(
    *,
    ticker: str,
    direction: str,
    setup_family: str | None,
    exit_reason: str,
    pnl_pct: float,
    r_multiple: float | None,
    alpha_vs_spy_pct: float | None,
) -> str:
    family = setup_family or "unclassified setup"
    r_text = f", {r_multiple:+.2f}R" if isinstance(r_multiple, (int, float)) else ""
    alpha_text = (
        f", alpha vs SPY {alpha_vs_spy_pct:+.2f}pp"
        if isinstance(alpha_vs_spy_pct, (int, float))
        else ""
    )
    if pnl_pct > 0 and (alpha_vs_spy_pct is None or alpha_vs_spy_pct >= 0):
        verdict = "worked and beat the benchmark"
    elif pnl_pct > 0:
        verdict = "made money but lagged SPY"
    elif exit_reason == "stopped_out":
        verdict = "failed at invalidation"
    else:
        verdict = "did not produce enough edge"
    return (
        f"{ticker} {direction} {family} {verdict}: "
        f"{pnl_pct:+.2f}%{r_text}{alpha_text}. "
        "Compare future entries in this family/regime against this outcome."
    )


def _record_trade_reflection(
    conn: sqlite3.Connection,
    *,
    trade_id: int,
    exit_date: str,
    exit_reason: str,
    pnl_pct: float,
    r_multiple: float | None,
) -> None:
    row = conn.execute(
        """
        SELECT ticker, direction, setup_family, entry_date
        FROM paper_trades
        WHERE id = ?
        """,
        (trade_id,),
    ).fetchone()
    if not row:
        return
    ticker, direction, setup_family, entry_date = row
    spy_return = _compute_spy_return_pct(conn, entry_date=entry_date, exit_date=exit_date)
    alpha = round(pnl_pct - spy_return, 2) if isinstance(spy_return, (int, float)) else None
    lesson = _lesson_from_outcome(
        ticker=str(ticker),
        direction=str(direction),
        setup_family=str(setup_family) if setup_family else None,
        exit_reason=exit_reason,
        pnl_pct=pnl_pct,
        r_multiple=r_multiple,
        alpha_vs_spy_pct=alpha,
    )
    conn.execute(
        """
        INSERT INTO paper_trade_reflections (
            trade_id, ticker, direction, setup_family, entry_date, exit_date,
            exit_reason, pnl_pct, r_multiple, spy_return_pct, alpha_vs_spy_pct,
            lesson_summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_id) DO UPDATE SET
            exit_date = excluded.exit_date,
            exit_reason = excluded.exit_reason,
            pnl_pct = excluded.pnl_pct,
            r_multiple = excluded.r_multiple,
            spy_return_pct = excluded.spy_return_pct,
            alpha_vs_spy_pct = excluded.alpha_vs_spy_pct,
            lesson_summary = excluded.lesson_summary
        """,
        (
            trade_id,
            ticker,
            direction,
            setup_family,
            entry_date,
            exit_date,
            exit_reason,
            pnl_pct,
            r_multiple,
            spy_return,
            alpha,
            lesson,
        ),
    )


def compute_trailing_stop(
    *,
    direction: str,
    entry_price: float,
    original_risk: float,
    current_hwm: float,
    current_lwm: float,
    current_stop: float | None,
    config: PaperTradeConfig,
) -> float | None:
    """Compute the new trailing stop using graduated 4-level logic.

    Levels (for longs — shorts mirror with min/LWM):
      R >= trail_tight_r  (2.0): HWM - tight_cushion_r * risk  (lock gains)
      R >= trail_mid_r    (1.5): HWM - mid_cushion_r * risk    (wide cushion)
      R >= trail_breakeven_r (1.25): entry price                (breakeven)
      R <  trail_breakeven_r: no change                         (original stop)

    Returns the new stop value, guaranteed to never loosen (only tighten).
    """
    if original_risk <= 0 or entry_price <= 0:
        return current_stop

    if direction == "long":
        current_r = (current_hwm - entry_price) / original_risk
        if current_r >= config.trail_tight_r:
            trail = current_hwm - config.trail_tight_cushion_r * original_risk
            return max(current_stop or 0, trail)
        if current_r >= config.trail_mid_r:
            trail = current_hwm - config.trail_mid_cushion_r * original_risk
            return max(current_stop or 0, trail)
        if current_r >= config.trail_breakeven_r:
            return max(current_stop or 0, entry_price)
        return current_stop
    else:  # short
        current_r = (entry_price - current_lwm) / original_risk
        if current_r >= config.trail_tight_r:
            trail = current_lwm + config.trail_tight_cushion_r * original_risk
            return min(current_stop or entry_price, trail)
        if current_r >= config.trail_mid_r:
            trail = current_lwm + config.trail_mid_cushion_r * original_risk
            return min(current_stop or entry_price, trail)
        if current_r >= config.trail_breakeven_r:
            return min(current_stop or entry_price, entry_price)
        return current_stop


def _resolve_original_risk(
    *,
    entry_price: float,
    current_stop: float | None,
    stop_distance_pct: float | None,
    atr_at_entry: float | None,
    config: PaperTradeConfig,
) -> float:
    """Reconstruct the original stop distance used for R-multiple trailing.

    Prefer the persisted entry stop distance when available. This keeps
    trailing-stop math anchored to the trade's original risk budget even
    after stop_loss has been tightened by MTM updates.
    """
    if entry_price <= 0:
        return 0.0

    if isinstance(stop_distance_pct, (int, float)) and float(stop_distance_pct) > 0:
        return entry_price * (float(stop_distance_pct) / 100.0)

    if isinstance(atr_at_entry, (int, float)) and float(atr_at_entry) > 0:
        return entry_price * (float(atr_at_entry) / 100.0) * config.stop_atr_mult

    if isinstance(current_stop, (int, float)) and float(current_stop) > 0:
        return abs(entry_price - float(current_stop))

    return entry_price * (config.default_stop_pct / 100.0)


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
        """SELECT position_size_pct,quantity,entry_notional,entry_commission,
                  accounting_status
           FROM paper_trades WHERE id=?""",
        (trade_id,),
    ).fetchone()
    pos_pct = float(position_row[0] or 8.0) if position_row and position_row[0] is not None else 8.0
    reconciled = bool(position_row and str(position_row[4]) == "reconciled")
    notional = (
        float(position_row[2]) if reconciled and position_row[2] is not None
        else config.starting_capital * (pos_pct / 100)
    )
    entry_commission = (
        float(position_row[3]) if reconciled and position_row[3] is not None
        else float(config.commission_per_trade)
    )
    exit_commission = float(config.commission_per_trade)
    commission_cost_pct = (
        (entry_commission + exit_commission) / notional * 100 if notional > 0 else 0
    )

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

    total_cost_pct = commission_cost_pct + borrow_cost_pct
    pnl = round(raw_pnl - total_cost_pct, 2)
    quantity = float(position_row[1]) if reconciled and position_row[1] is not None else None
    borrow_cost = notional * borrow_cost_pct / 100 if reconciled else None
    realized_pnl_usd = (
        (
            (exit_price - entry_price) * quantity
            if direction == "long"
            else (entry_price - exit_price) * quantity
        ) - entry_commission - exit_commission - float(borrow_cost or 0.0)
        if quantity is not None else None
    )
    if realized_pnl_usd is not None and notional > 0:
        pnl = round(realized_pnl_usd / notional * 100, 2)
    # R-multiple net of costs: adjust exit price by total cost drag
    if direction == "long":
        cost_adjusted_exit = exit_price * (1 - total_cost_pct / 100)
    else:
        cost_adjusted_exit = exit_price * (1 + total_cost_pct / 100)
    r_mult = compute_r_multiple(
        direction,
        entry_price,
        cost_adjusted_exit,
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
            last_mtm_date = ?,
            review_status = ?,
            review_summary = ?,
            exit_commission = ?,
            borrow_cost = ?,
            realized_pnl_usd = ?,
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
            exit_date,
            review_status,
            review_summary,
            exit_commission if reconciled else None,
            borrow_cost,
            realized_pnl_usd,
            now,
            trade_id,
        ),
    )
    _record_trade_reflection(
        conn,
        trade_id=trade_id,
        exit_date=exit_date,
        exit_reason=exit_reason,
        pnl_pct=pnl,
        r_multiple=r_mult,
    )
    _record_trade_event(
        conn,
        trade_id=trade_id,
        event_type="close",
        event_date=exit_date,
        payload={
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "status": status,
            "pnl_pct": pnl,
            "r_multiple": r_mult,
            "commission_cost_pct": commission_cost_pct,
            "borrow_cost_pct": borrow_cost_pct,
        },
    )


def _create_paper_trades_from_report(
    conn: sqlite3.Connection,
    *,
    setup_rows: list[dict[str, Any]],
    report_date: str,
    generated_ts: str,
    config: PaperTradeConfig,
    report_run_id: str | None = None,
    schema_ready: bool = False,
    expected_price_revision: str | None = None,
) -> int:
    """Atomically persist one report's trades and sealed decision ledger."""
    if not report_date:
        return 0

    if not schema_ready:
        ensure_paper_trade_schema(conn)
    if not str(report_run_id or "").strip():
        raise ValueError("paper-trade creation requires canonical report-run lineage")
    lineage = _require_published_canonical_report(conn, str(report_run_id))
    if expected_price_revision is not None:
        current_price_contract = research_price_contract(conn)
        if (
            not current_price_contract.get("eligible")
            or current_price_contract.get("revision") != expected_price_revision
        ):
            return 0
    _advance_paper_book(conn, config=config, through_date=report_date)
    resolve_breadth_shadow_outcomes(
        conn, through_date=report_date, base_config=config
    )
    register_bot_version(
        conn,
        bot_version=config.bot_version,
        decision_version=config.decision_version,
        config_json=json.dumps(config_snapshot(config)),
        notes="Current champion paper-trade policy snapshot.",
        schema_ready=True,
    )
    conn.execute("SAVEPOINT paper_report_admission")
    try:
        inserted = _create_paper_trades_from_report_in_transaction(
            conn,
            setup_rows=setup_rows,
            report_date=report_date,
            generated_ts=generated_ts,
            config=config,
            report_run_id=report_run_id,
            expected_price_revision=expected_price_revision,
        )
    except Exception:
        conn.execute("ROLLBACK TO paper_report_admission")
        conn.execute("RELEASE paper_report_admission")
        raise
    conn.execute("RELEASE paper_report_admission")
    return inserted


def _create_paper_trades_from_report_in_transaction(
    conn: sqlite3.Connection,
    *,
    setup_rows: list[dict[str, Any]],
    report_date: str,
    generated_ts: str,
    config: PaperTradeConfig,
    report_run_id: str | None = None,
    expected_price_revision: str | None = None,
) -> int:
    """Create paper trades from qualifying daily report setups."""
    if not report_date:
        return 0
    if not report_run_id:
        raise ValueError("paper campaign admission requires report_run_id")
    lineage = _require_published_canonical_report(conn, report_run_id)
    report_complete = bool(lineage["report_complete"])
    is_canonical = bool(lineage["is_canonical"])

    campaign_row = conn.execute(
        "SELECT status, policy_version, starting_capital, policy_hash FROM paper_campaigns WHERE campaign_id=?",
        (config.campaign_id,),
    ).fetchone()
    if not campaign_row:
        raise ValueError(f"paper campaign {config.campaign_id} is not registered")
    runtime_policy_hash = canonical_hash(config_snapshot(config))
    if str(campaign_row[1]) != config.decision_version or float(campaign_row[2]) != config.starting_capital:
        raise ValueError("runtime policy does not match immutable campaign registration")
    if str(campaign_row[3] or "") and str(campaign_row[3]) != runtime_policy_hash:
        raise ValueError("runtime policy hash does not match sealed campaign registration")
    if not str(campaign_row[3] or ""):
        conn.execute(
            "UPDATE paper_campaigns SET policy_hash=? WHERE campaign_id=?",
            (runtime_policy_hash, config.campaign_id),
        )
    campaign_active = str(campaign_row[0]) == "active"
    request_hash = canonical_hash({
        "report_run_id": report_run_id,
        "report_date": report_date,
        "generated_ts": generated_ts,
        "campaign_id": config.campaign_id,
        "report_complete": report_complete,
        "is_canonical": is_canonical,
        "policy": config_snapshot(config),
        "candidates": setup_rows,
    })
    existing_set = conn.execute(
        "SELECT request_hash FROM paper_decision_sets WHERE report_run_id=? AND campaign_id=?",
        (report_run_id, config.campaign_id),
    ).fetchone()
    if existing_set:
        if str(existing_set[0]) == request_hash:
            return 0
        raise DivergentDecisionSetError(
            f"divergent retry for report_run_id={report_run_id} campaign_id={config.campaign_id}"
        )
    record_breadth_shadow(
        conn,
        report_run_id=report_run_id,
        report_date=report_date,
        generated_ts=generated_ts,
        setup_rows=setup_rows,
        base_config=config,
    )
    open_count = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE campaign_id=? AND status='open'",
        (config.campaign_id,),
    ).fetchone()[0]

    global_block: tuple[str, str, str] | None = None
    if not report_complete or not is_canonical:
        global_block = (
            "report_integrity",
            "report_not_complete_canonical",
            "Only complete canonical reports may admit paper trades.",
        )
    elif not campaign_active:
        global_block = (
            "campaign_lifecycle",
            "campaign_not_active",
            "Campaign is not active; candidate is recorded in shadow mode.",
        )
    if global_block is None and open_count >= config.max_open:
        LOG.info(
            "Paper trades: %d open trades already at max (%d), skipping creation",
            open_count, config.max_open,
        )
        global_block = (
            "portfolio_capacity",
            "max_open_positions",
            f"Open positions {open_count} reached policy maximum {config.max_open}.",
        )

    # All admission risk gates consume the one reconciled account snapshot.
    try:
        snapshot_row = conn.execute(
            """SELECT drawdown_pct,equity,session_pnl_usd,accounting_breaks_json
               FROM paper_portfolio_snapshots WHERE campaign_id=?
               ORDER BY snapshot_date DESC LIMIT 1""",
            (config.campaign_id,),
        ).fetchone()
        if snapshot_row:
            accounting_breaks = json.loads(str(snapshot_row[3] or "[]"))
            if global_block is None and accounting_breaks:
                global_block = (
                    "portfolio_accounting",
                    "portfolio_reconciliation_failed",
                    "New entries are blocked until account reconciliation succeeds.",
                )
            drawdown_pct = float(snapshot_row[0] or 0.0)
            if global_block is None and drawdown_pct >= config.max_drawdown_pct:
                LOG.warning(
                    "CIRCUIT BREAKER: portfolio drawdown %.1f%% exceeds %.1f%% limit, blocking new entries",
                    drawdown_pct, config.max_drawdown_pct,
                )
                global_block = (
                    "portfolio_risk",
                    "max_drawdown_circuit_breaker",
                    f"Portfolio drawdown {drawdown_pct:.1f}% reached {config.max_drawdown_pct:.1f}% limit.",
                )
            equity = float(snapshot_row[1] or config.starting_capital)
            session_pnl = float(snapshot_row[2] or 0.0)
            session_start = equity - session_pnl
            daily_loss_pct = (
                -session_pnl / session_start * 100
                if session_pnl < 0 and session_start > 0 else 0.0
            )
        else:
            daily_loss_pct = 0.0
        if global_block is None and daily_loss_pct >= config.max_daily_loss_pct:
            LOG.warning(
                "CIRCUIT BREAKER: daily loss %.1f%% exceeds %.1f%% limit, blocking new entries",
                daily_loss_pct, config.max_daily_loss_pct,
            )
            global_block = (
                "portfolio_risk",
                "max_daily_loss_circuit_breaker",
                f"Session account loss {daily_loss_pct:.1f}% reached {config.max_daily_loss_pct:.1f}% limit.",
            )
    except Exception as exc:
        LOG.warning("Reconciled portfolio risk check failed: %s", exc)
        if global_block is None:
            global_block = (
                "portfolio_accounting",
                "portfolio_snapshot_unavailable",
                "New entries are blocked because the account snapshot is unavailable.",
            )

    remaining_slots = max(0, config.max_open - open_count)
    inserted = 0
    decisions: list[dict[str, Any]] = []
    _decision_runtime_context: dict[str, Any] = {}

    def record_decision(
        *,
        row: dict[str, Any],
        rank: int,
        evaluation: dict[str, Any],
        final_gate: str,
        reason_code: str,
        reasons: list[str],
        disposition: str = "rejected",
        levels: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        critic: dict[str, Any] | None = None,
    ) -> None:
        if critic is not None:
            _decision_runtime_context["critic_outcome"] = critic
        policy_decision = decide_candidate(
            row=row, rank=rank, config=config, context=_decision_runtime_context,
        )
        expected = (final_gate, reason_code, disposition)
        actual = (
            policy_decision["final_gate"], policy_decision["reason_code"],
            policy_decision["disposition"],
        )
        if actual != expected:
            raise RuntimeError(
                f"live policy branch drift at rank {rank}: expected {expected}, got {actual}"
            )
        decisions.append(policy_decision)

    # Pre-fetch VIX level once for position sizing (used by all trades this batch)
    _vix_level: float | None = None
    try:
        _vix_row = conn.execute(
            "SELECT CAST(close AS REAL) FROM price_daily "
            "WHERE ticker = '^VIX' AND close IS NOT NULL AND date <= ? "
            "ORDER BY date DESC LIMIT 1",
            (report_date,),
        ).fetchone()
        if _vix_row and _vix_row[0] is not None:
            _vix_level = float(_vix_row[0])
    except Exception:
        pass

    for rank, row in enumerate(setup_rows, start=1):
        _decision_runtime_context = {
            "vix_level": _vix_level,
            "campaign_active": campaign_active,
            "portfolio_block": (
                {"gate": global_block[0], "reason_code": global_block[1], "detail": global_block[2]}
                if global_block else None
            ),
            "portfolio_context": {"open_count": open_count, "remaining_slots": remaining_slots, "inserted_this_report": inserted},
            "source_context": {"report_date": report_date, "generated_ts": generated_ts, "report_run_id": report_run_id},
        }
        if not isinstance(row, dict):
            decisions.append(decide_candidate(
                row=row, rank=rank, config=config, context=_decision_runtime_context,
            ))
            continue
        evaluation = evaluate_setup_for_paper_trade(row, config=config)
        if not evaluation["approved"]:
            failures = list(evaluation.get("gate_failures") or [])
            first_failure = failures[0] if failures else {
                "gate": "eligibility", "reason_code": "eligibility_rejected"
            }
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                final_gate=str(first_failure["gate"]),
                reason_code=str(first_failure["reason_code"]),
                reasons=list(evaluation.get("decision_reasons") or []),
            )
            continue

        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                final_gate="candidate_identity",
                reason_code="missing_ticker",
                reasons=["Candidate ticker is missing."],
            )
            continue

        if global_block is not None:
            gate, code, detail = global_block
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                final_gate=gate,
                reason_code=code,
                reasons=[detail],
            )
            continue

        if inserted >= remaining_slots:
            _decision_runtime_context["portfolio_block"] = {
                "gate": "portfolio_capacity",
                "reason_code": "report_slots_exhausted",
                "detail": "Higher-ranked candidates filled all remaining policy slots.",
            }
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                final_gate="portfolio_capacity",
                reason_code="report_slots_exhausted",
                reasons=["Higher-ranked candidates filled all remaining policy slots."],
            )
            continue
        member = conn.execute(
            """SELECT 1 FROM report_run_decisions
               WHERE run_id=? AND ticker=? AND decision='accepted'""",
            (report_run_id, ticker),
        ).fetchone()
        if member is None:
            raise ValueError(f"{ticker} is not an accepted decision in report run {report_run_id}")
        direction = str(evaluation["direction"])

        # Entry price is strictly the ticker open on the immediate next SPY
        # session. A missing ticker bar cannot silently roll the fill forward.
        try:
            intended_session = next_scheduled_session_after(report_date)
            spy_ready = bool(
                intended_session
                and conn.execute(
                    "SELECT 1 FROM price_daily "
                    "WHERE ticker='SPY' AND date=? AND open IS NOT NULL",
                    (intended_session,),
                ).fetchone()
            )
            next_open_row = (
                conn.execute(
                    "SELECT CAST(open AS REAL),date FROM price_daily "
                    "WHERE ticker=? AND date=? AND open IS NOT NULL",
                    (ticker, intended_session),
                ).fetchone()
                if spy_ready
                else None
            )
            publication_ready = bool(
                intended_session
                and publication_precedes_session_open(
                    str(lineage.get("published_ts") or ""), intended_session
                )
            )
            if next_open_row and next_open_row[0] is not None:
                raw_entry = float(next_open_row[0])
                entry_date_actual = next_open_row[1]
                execution_ready = True
            else:
                raw_entry = float(row["close"])
                entry_date_actual = None
                execution_ready = False

            slip_mult = config.entry_slippage_bps / 10_000
            if direction == "long":
                entry_price = round(raw_entry * (1 + slip_mult), 4)
            else:
                entry_price = round(raw_entry * (1 - slip_mult), 4)
            _decision_runtime_context["entry_price"] = entry_price
            _decision_runtime_context["execution_ready"] = execution_ready
            _decision_runtime_context["execution_pending_reason"] = (
                "scheduled_spy_open_missing"
                if not spy_ready else "scheduled_ticker_open_missing"
                if not execution_ready else None
            )
            _decision_runtime_context["source_context"] = {
                "report_run_id": report_run_id,
                "intended_session": intended_session,
                "price_date": str(next_open_row[1]) if next_open_row else None,
            }
            levels = compute_stop_and_target(
                row, direction, config=config, entry_price=entry_price
            )
            plan = compute_position_plan(
                row,
                evaluation,
                levels,
                config=config,
                vix_level=_vix_level,
                entry_price=entry_price,
            )
        except (KeyError, TypeError, ValueError) as exc:
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                final_gate="trade_plan",
                reason_code="invalid_stop_target_or_fill",
                reasons=[f"Trade plan could not be computed: {type(exc).__name__}."],
            )
            continue

        if not publication_ready:
            reason_code = (
                "report_published_after_intended_open"
                if str(lineage.get("published_ts") or "")
                else "report_publication_timestamp_unavailable"
            )
            detail = (
                "Verified report publication did not precede the intended session open."
                if str(lineage.get("published_ts") or "")
                else "Verified report publication chronology is unavailable."
            )
            _decision_runtime_context["portfolio_block"] = {
                "gate": "execution.next_open",
                "reason_code": reason_code,
                "detail": detail,
            }
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                levels=levels,
                plan=plan,
                final_gate="execution.next_open",
                reason_code=reason_code,
                reasons=[detail],
            )
            continue

        if execution_ready and not research_price_contract(conn, [ticker]).get("eligible"):
            LOG.warning("Paper trade skipped: %s price series is unresolved", ticker)
            _decision_runtime_context["portfolio_block"] = {
                "gate": "price_basis",
                "reason_code": "price_series_revision_unavailable",
                "detail": "The executable next-open price series is not revision verified.",
            }
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                levels=levels,
                plan=plan,
                final_gate="price_basis",
                reason_code="price_series_revision_unavailable",
                reasons=["The executable next-open price series is not revision verified."],
            )
            continue

        # ADV liquidity check: reject if position > max_adv_pct of daily volume
        try:
            vol_row = conn.execute(
                "SELECT AVG(vol) FROM ("
                "  SELECT CAST(volume AS REAL) AS vol FROM price_daily"
                "  WHERE ticker = ? AND volume IS NOT NULL AND date <= ?"
                "  ORDER BY date DESC LIMIT 20"
                ")",
                (ticker, report_date),
            ).fetchone()
            if vol_row and vol_row[0] and vol_row[0] > 0:
                avg_daily_volume = float(vol_row[0])
                _decision_runtime_context["avg_daily_volume"] = avg_daily_volume
                position_pct = float(plan.get("position_size_pct") or 8.0)
                position_dollars = config.starting_capital * (position_pct / 100)
                position_shares = position_dollars / entry_price if entry_price > 0 else 0
                adv_pct = (position_shares / avg_daily_volume) * 100 if avg_daily_volume > 0 else 0
                if adv_pct > config.max_adv_pct:
                    LOG.info(
                        "Paper trade skipped: %s %s position is %.1f%% of ADV (> %.1f%% max)",
                        direction.upper(), ticker, adv_pct, config.max_adv_pct,
                    )
                    record_decision(
                        row=row,
                        rank=rank,
                        evaluation=evaluation,
                        levels=levels,
                        plan=plan,
                        final_gate="liquidity",
                        reason_code="position_exceeds_adv_limit",
                        reasons=[
                            f"Planned position is {adv_pct:.1f}% of ADV; policy maximum is {config.max_adv_pct:.1f}%."
                        ],
                    )
                    continue
        except Exception as exc:
            LOG.debug("ADV check skipped: %s", exc)

        expected_r_multiple = plan.get("expected_r_multiple")
        if not isinstance(expected_r_multiple, (int, float)) or (
            expected_r_multiple < config.min_reward_r_multiple
        ):
            LOG.info(
                "Paper trade skipped: %s %s only offers %.2fR (< %.2fR minimum)",
                direction.upper(),
                ticker,
                float(expected_r_multiple or 0),
                config.min_reward_r_multiple,
            )
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                levels=levels,
                plan=plan,
                final_gate="reward_risk",
                reason_code="minimum_reward_r_not_met",
                reasons=[
                    f"Expected {float(expected_r_multiple or 0):.2f}R is below policy minimum {config.min_reward_r_multiple:.2f}R."
                ],
            )
            continue

        # ML score — OBSERVATION MODE: score trades but never reject.
        # The current best local model is a barrier model, so the probability
        # means "long target-hit likelihood" rather than a generic short signal.
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
                    prediction_label = ml_score.get("prediction_label") or "model_probability"
                    LOG.info(
                        "ML observation: %s %s %s=%.2f (threshold %.2f, NOT filtering)",
                        direction.upper(),
                        ticker,
                        prediction_label,
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

        market_ctx = capture_market_context(conn, as_of_date=report_date)
        market_ctx["bot_version"] = config.bot_version
        _decision_runtime_context["market_context"] = market_ctx

        # Critic review — devil's advocate that kills low-conviction trades
        critic: dict[str, Any] = {}
        try:
            from trader_koo.paper_trade.critic import critic_review

            critic = critic_review(
                conn,
                row=row,
                evaluation=evaluation,
                plan=plan,
                market_ctx=market_ctx,
                max_open=config.max_open,
                campaign_id=config.campaign_id,
            )
            failed_check = None
            for raw_reason in critic.get("critic_reasons") or []:
                text = str(raw_reason)
                if text.startswith("FAIL [") and "]:" in text:
                    failed_check = text.split("FAIL [", 1)[1].split("]:", 1)[0]
                    break
            critic["failed_check"] = failed_check
            _decision_runtime_context["critic_outcome"] = critic
            if not critic["approved"]:
                failed_check = failed_check or "unknown"
                LOG.info(
                    "Critic REJECTED: %s %s — %s",
                    direction.upper(),
                    ticker,
                    critic["rejections"][0] if critic["rejections"] else "failed critic review",
                )
                record_decision(
                    row=row,
                    rank=rank,
                    evaluation=evaluation,
                    levels=levels,
                    plan=plan,
                    critic=critic,
                    final_gate=f"critic.{failed_check}",
                    reason_code=f"critic_{failed_check}_rejected",
                    reasons=list(critic.get("rejections") or ["Critic rejected candidate."]),
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
            if config.critic_fail_open:
                LOG.warning("Critic check failed (allowing trade by explicit config): %s", exc)
                critic = {
                    "approved": True,
                    "fail_open": True,
                    "error": type(exc).__name__,
                }
                _decision_runtime_context["critic_outcome"] = critic
            else:
                LOG.warning("Critic check failed (rejecting trade): %s", exc)
                record_decision(
                    row=row,
                    rank=rank,
                    evaluation=evaluation,
                    levels=levels,
                    plan=plan,
                    critic={"approved": False, "error": type(exc).__name__},
                    final_gate="critic",
                    reason_code="critic_infrastructure_error",
                    reasons=["Critic infrastructure failed; policy rejects by default."],
                )
                continue

        if not execution_ready:
            order_id = canonical_hash({
                "report_run_id": report_run_id,
                "campaign_id": config.campaign_id,
                "candidate_rank": rank,
                "ticker": ticker,
                "direction": direction,
            })
            order_payload = {
                "order_id": order_id, "report_run_id": report_run_id,
                "report_date": report_date, "ticker": ticker,
                "direction": direction, "candidate_rank": rank,
            }
            candidate_json = canonical_json(row)
            critic_json = canonical_json(critic)
            market_context_json = canonical_json(market_ctx)
            avg_daily_volume = _decision_runtime_context.get("avg_daily_volume")
            order_hash = _pending_order_hash(
                order_id=order_id, report_run_id=report_run_id,
                report_date=report_date, generated_ts=generated_ts,
                campaign_id=config.campaign_id,
                policy_version=config.decision_version, candidate_rank=rank,
                ticker=ticker, direction=direction,
                candidate_json=candidate_json, critic_json=critic_json,
                market_context_json=market_context_json,
                avg_daily_volume=avg_daily_volume,
            )
            conn.execute(
                """INSERT INTO paper_pending_orders
                   (order_id,report_run_id,report_date,generated_ts,campaign_id,
                    policy_version,candidate_rank,ticker,direction,candidate_json,
                    critic_json,market_context_json,avg_daily_volume,order_hash,status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending')
                   ON CONFLICT(order_id) DO NOTHING""",
                (order_id,report_run_id,report_date,generated_ts,config.campaign_id,
                 config.decision_version,rank,ticker,direction,candidate_json,
                 critic_json,market_context_json,avg_daily_volume,order_hash),
            )
            conn.execute(
                """INSERT OR IGNORE INTO paper_order_events
                   (order_id,event_type,event_date,payload_json,payload_hash)
                   VALUES (?,'created',?,?,?)""",
                (order_id,report_date,
                 canonical_json({**order_payload, "order_hash": order_hash}),
                 canonical_hash({**order_payload, "order_hash": order_hash})),
            )
            record_decision(
                row=row, rank=rank, evaluation=evaluation, levels=levels, plan=plan,
                critic=critic, final_gate="execution.next_open",
                reason_code=str(
                    _decision_runtime_context.get("execution_pending_reason")
                    or "scheduled_ticker_open_missing"
                ),
                reasons=["The exact scheduled-session observation is not available yet."],
                disposition="pending",
            )
            continue

        accounting = _entry_accounting(
            conn,
            campaign_id=config.campaign_id,
            entry_price=entry_price,
            position_size_pct=float(plan["position_size_pct"]),
            config=config,
        )
        if accounting is None:
            record_decision(
                row=row,
                rank=rank,
                evaluation=evaluation,
                levels=levels,
                plan=plan,
                critic=critic,
                final_gate="portfolio.cash",
                reason_code="insufficient_reconciled_cash",
                reasons=["Reconciled cash cannot fund one share plus commission."],
            )
            continue

        rationale = _build_entry_rationale(
            ticker=ticker,
            direction=direction,
            row=row,
            evaluation=evaluation,
            plan=plan,
            market_ctx=market_ctx,
            critic=critic,
            ml_prediction=ml_prediction,
        )

        before_changes = conn.total_changes
        insert_cursor = conn.execute(
            """
            INSERT INTO paper_trades (
                report_date, generated_ts, report_run_id, ticker, direction,
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
                entry_reason, entry_evidence, entry_risks,
                bot_version, vix_at_entry, vix_percentile_at_entry,
                regime_state_at_entry, hmm_regime_at_entry, hmm_confidence_at_entry,
                directional_regime_at_entry, directional_regime_confidence,
                ml_predicted_win_prob, ml_confidence, ml_signal,
                campaign_id, policy_version
            ) VALUES (
                ?, ?, ?, ?, ?,
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
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?
            )
            ON CONFLICT(campaign_id, report_date, ticker, direction) DO NOTHING
            """,
            (
                report_date,
                generated_ts,
                report_run_id,
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
                rationale["entry_reason"],
                json.dumps(rationale["entry_evidence"]),
                json.dumps(rationale["entry_risks"]),
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
                config.campaign_id,
                config.decision_version,
            ),
        )
        if conn.total_changes > before_changes:
            inserted += 1
            trade_id = int(insert_cursor.lastrowid)
            conn.execute(
                """UPDATE paper_trades SET
                       quantity=?,entry_notional=?,entry_commission=?,accounting_status='reconciled'
                   WHERE id=?""",
                (
                    accounting["quantity"],
                    accounting["entry_notional"],
                    accounting["entry_commission"],
                    trade_id,
                ),
            )
            _record_trade_event(
                conn,
                trade_id=trade_id,
                event_type="fill",
                event_date=entry_date_actual,
                payload={
                    "report_run_id": report_run_id,
                    "ticker": ticker,
                    "direction": direction,
                    "raw_open": raw_entry,
                    "fill_price": entry_price,
                    "fill_source": "immediate_next_session_open",
                    "policy_version": config.decision_version,
                    **accounting,
                },
            )
            LOG.info(
                "Paper trade created: %s %s @ %.2f (stop=%.2f target=%.2f) — %s",
                direction.upper(), ticker, entry_price,
                levels["stop_loss"], levels["target_price"], rationale["entry_reason"],
            )
            disposition = "admitted"
            reason_code = "admitted"
            reasons = ["Candidate passed the versioned paper campaign policy."]
        else:
            disposition = "duplicate"
            reason_code = "duplicate_candidate"
            reasons = ["A paper trade already exists for this report date, ticker, and direction."]
        _decision_runtime_context["duplicate"] = disposition == "duplicate"
        record_decision(
            row=row,
            rank=rank,
            evaluation=evaluation,
            levels=levels,
            plan=plan,
            critic=critic,
            final_gate="admission",
            reason_code=reason_code,
            reasons=reasons,
            disposition=disposition,
        )

    persist_decision_set(
        conn,
        report_run_id=report_run_id,
        report_date=report_date,
        generated_ts=generated_ts,
        campaign_id=config.campaign_id,
        policy_version=config.decision_version,
        request_hash=request_hash,
        policy_hash=canonical_hash(config_snapshot(config)),
        context_hash=canonical_hash({
            "report_date": report_date,
            "generated_ts": generated_ts,
            "report_complete": report_complete,
            "is_canonical": is_canonical,
            "campaign_active": campaign_active,
            "portfolio_block": global_block,
            "open_count": open_count,
            "candidate_context_hashes": [
                decision["context_hash"] for decision in decisions
            ],
        }),
        decisions=decisions,
        report_complete=report_complete,
        is_canonical=is_canonical,
    )
    update_portfolio_snapshot(conn, campaign_id=config.campaign_id)
    return inserted


def create_paper_trades_from_report(
    conn: sqlite3.Connection,
    *,
    setup_rows: list[dict[str, Any]],
    report_date: str,
    generated_ts: str,
    config: PaperTradeConfig,
    report_run_id: str | None = None,
    schema_ready: bool = False,
    expected_price_revision: str | None = None,
) -> int:
    return _run_owned_transaction(
        conn,
        lambda: _create_paper_trades_from_report(
            conn,
            setup_rows=setup_rows,
            report_date=report_date,
            generated_ts=generated_ts,
            config=config,
            report_run_id=report_run_id,
            schema_ready=schema_ready,
            expected_price_revision=expected_price_revision,
        ),
    )


def fill_pending_paper_orders(
    conn: sqlite3.Connection,
    *,
    config: PaperTradeConfig,
    through_date: str | None = None,
) -> dict[str, int]:
    """Resolve pending signals only from a real later-session open."""
    ensure_paper_trade_schema(conn)
    resolved = {"filled": 0, "rejected": 0, "still_pending": 0}
    rows = conn.execute(
        """SELECT order_id,report_run_id,report_date,generated_ts,campaign_id,
                  policy_version,candidate_rank,ticker,direction,candidate_json,
                  critic_json,market_context_json,avg_daily_volume,order_hash
           FROM paper_pending_orders WHERE status='pending'
           ORDER BY report_date,candidate_rank,order_id"""
    ).fetchall()
    for raw in rows:
        (order_id,report_run_id,report_date,generated_ts,campaign_id,
         policy_version,rank,ticker,direction,candidate_json,critic_json,
         market_json,avg_volume,order_hash) = raw
        expected_order_hash = _pending_order_hash(
            order_id=str(order_id), report_run_id=str(report_run_id),
            report_date=str(report_date), generated_ts=str(generated_ts),
            campaign_id=str(campaign_id), policy_version=str(policy_version),
            candidate_rank=int(rank), ticker=str(ticker), direction=str(direction),
            candidate_json=str(candidate_json), critic_json=str(critic_json),
            market_context_json=str(market_json),
            avg_daily_volume=float(avg_volume) if avg_volume is not None else None,
        )
        if str(order_hash) != expected_order_hash:
            raise ValueError(f"pending order {order_id} failed immutable hash verification")
        lineage = _require_published_canonical_report(
            conn, str(report_run_id), require_current=False
        )
        intended_session = next_scheduled_session_after(str(report_date))
        if (
            not intended_session
            or (through_date is not None and intended_session > through_date)
        ):
            resolved["still_pending"] += 1
            continue
        spy_ready = conn.execute(
            "SELECT 1 FROM price_daily "
            "WHERE ticker='SPY' AND date=? AND open IS NOT NULL",
            (intended_session,),
        ).fetchone()
        if not spy_ready:
            resolved["still_pending"] += 1
            continue
        open_row = conn.execute(
            "SELECT CAST(open AS REAL),date FROM price_daily "
            "WHERE ticker=? AND date=? AND open IS NOT NULL",
            (ticker, intended_session),
        ).fetchone()
        if not open_row:
            resolved["still_pending"] += 1
            continue
        campaign = conn.execute(
            "SELECT status,policy_version FROM paper_campaigns WHERE campaign_id=?",
            (campaign_id,),
        ).fetchone()
        open_count = int(conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE campaign_id=? AND status='open'",
            (campaign_id,),
        ).fetchone()[0])
        block = None
        if not publication_precedes_session_open(
            str(lineage.get("published_ts") or ""), intended_session
        ):
            block = {
                "gate": "execution.next_open",
                "reason_code": "report_published_after_intended_open",
                "detail": "Verified report publication did not precede the intended session open.",
            }
        elif not campaign or str(campaign[0]) != "active":
            block = {"gate": "campaign_lifecycle", "reason_code": "campaign_not_active", "detail": "Campaign is not active at execution."}
        elif str(campaign[1]) != str(policy_version):
            block = {"gate": "campaign_lifecycle", "reason_code": "policy_version_changed", "detail": "Pending order policy no longer matches campaign."}
        elif open_count >= config.max_open:
            block = {"gate": "portfolio_capacity", "reason_code": "max_open_positions", "detail": "No portfolio slot remained at execution."}
        elif not research_price_contract(conn, [str(ticker)]).get("eligible"):
            block = {"gate": "price_basis", "reason_code": "price_series_revision_unavailable", "detail": "The executable next-open price series is not revision verified."}
        raw_open = float(open_row[0])
        slip = config.entry_slippage_bps / 10_000
        entry_price = round(raw_open * (1 + slip if direction == "long" else 1 - slip), 4)
        row = json.loads(str(candidate_json))
        context = {
            "entry_price": entry_price, "avg_daily_volume": avg_volume,
            "portfolio_block": block, "critic_outcome": json.loads(str(critic_json)),
            "campaign_active": bool(campaign and str(campaign[0]) == "active"),
            "duplicate": False, "execution_ready": True,
            "market_context": json.loads(str(market_json)),
            "portfolio_context": {"open_count": open_count},
            "source_context": {"report_run_id": report_run_id,
                               "intended_session": intended_session,
                               "price_date": open_row[1]},
        }
        decision = decide_candidate(row=row, rank=int(rank), config=config, context=context)
        accounting = None
        if decision["disposition"] == "admitted":
            accounting = _entry_accounting(
                conn,
                campaign_id=str(campaign_id),
                entry_price=entry_price,
                position_size_pct=float(decision["plan"]["position_size_pct"]),
                config=config,
            )
            if accounting is None:
                context["portfolio_block"] = {
                    "gate": "portfolio.cash",
                    "reason_code": "insufficient_reconciled_cash",
                    "detail": "Reconciled cash cannot fund one share plus commission.",
                }
                decision = decide_candidate(
                    row=row, rank=int(rank), config=config, context=context,
                )
        event_payload = {
            "decision": decision, "raw_open": raw_open,
            "entry_price": entry_price, "entry_date": open_row[1],
        }
        if decision["disposition"] == "admitted":
            levels = decision["levels"]
            plan = decision["plan"]
            insert_cursor = conn.execute(
                """INSERT INTO paper_trades
                   (report_date,generated_ts,ticker,direction,entry_price,entry_date,
                    target_price,stop_loss,atr_at_entry,status,current_price,
                    high_water_mark,low_water_mark,setup_family,setup_tier,score,
                    signal_bias,actionability,position_size_pct,risk_budget_pct,
                    stop_distance_pct,expected_reward_pct,expected_r_multiple,
                    entry_plan,exit_plan,sizing_summary,campaign_id,report_run_id,
                    policy_version,decision_version,decision_state)
                   VALUES (?,?,?,?,?,?,?,?,?,'open',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(campaign_id,report_date,ticker,direction) DO NOTHING""",
                (report_date,generated_ts,ticker,direction,entry_price,open_row[1],
                 levels.get("target_price"),levels.get("stop_loss"),levels.get("atr_at_entry"),
                 entry_price,entry_price,entry_price,row.get("setup_family"),row.get("setup_tier"),
                 row.get("score"),row.get("signal_bias"),row.get("actionability"),
                 plan.get("position_size_pct"),plan.get("risk_budget_pct"),
                 plan.get("stop_distance_pct"),plan.get("expected_reward_pct"),
                 plan.get("expected_r_multiple"),plan.get("entry_plan"),plan.get("exit_plan"),
                 plan.get("sizing_summary"),campaign_id,report_run_id,policy_version,
                 policy_version,"admitted"),
            )
            if insert_cursor.rowcount == 1:
                trade_id = int(insert_cursor.lastrowid)
                conn.execute(
                    """UPDATE paper_trades SET
                           quantity=?,entry_notional=?,entry_commission=?,accounting_status='reconciled'
                       WHERE id=?""",
                    (
                        accounting["quantity"], accounting["entry_notional"],
                        accounting["entry_commission"], trade_id,
                    ),
                )
                _record_trade_event(
                    conn,
                    trade_id=trade_id,
                    event_type="fill",
                    event_date=str(open_row[1]),
                    payload={
                        "order_id": order_id,
                        "report_run_id": report_run_id,
                        "ticker": ticker,
                        "direction": direction,
                        "raw_open": raw_open,
                        "fill_price": entry_price,
                        "fill_source": "pending_immediate_next_session_open",
                        "policy_version": policy_version,
                        **accounting,
                    },
                )
            status = "filled"
            resolved["filled"] += 1
        else:
            status = "rejected"
            resolved["rejected"] += 1
        conn.execute(
            "UPDATE paper_pending_orders SET status=?,resolved_ts=CURRENT_TIMESTAMP WHERE order_id=? AND status='pending'",
            (status, order_id),
        )
        conn.execute(
            """INSERT INTO paper_order_events
               (order_id,event_type,event_date,payload_json,payload_hash)
               VALUES (?,?,?,?,?)""",
            (order_id,status,str(open_row[1]),canonical_json(event_payload),canonical_hash(event_payload)),
        )
    return resolved


def _trade_expired(
    conn: sqlite3.Connection,
    *,
    entry_date: str,
    price_date: str,
    config: PaperTradeConfig,
) -> bool:
    try:
        if config.expiry_use_trading_days:
            row = conn.execute(
                "SELECT COUNT(*) FROM price_daily "
                "WHERE ticker='SPY' AND date>? AND date<=?",
                (entry_date, price_date),
            ).fetchone()
            sessions = int(row[0]) if row and row[0] else 0
            if sessions:
                return sessions >= config.expiry_days
        entry = dt.date.fromisoformat(entry_date)
        current = dt.date.fromisoformat(price_date)
        return (current - entry).days >= config.expiry_days
    except (TypeError, ValueError):
        return False


def _apply_trade_bar(
    conn: sqlite3.Connection,
    *,
    trade: dict[str, Any],
    price_row: sqlite3.Row | tuple[Any, ...],
    config: PaperTradeConfig,
) -> bool:
    """Apply one chronological OHLC bar; return whether it closed the trade."""
    current_price = float(price_row[0])
    price_date = str(price_row[1])
    day_high = float(price_row[2]) if price_row[2] is not None else current_price
    day_low = float(price_row[3]) if price_row[3] is not None else current_price
    day_open = float(price_row[4]) if price_row[4] is not None else current_price
    new_hwm = max(trade["high_water_mark"] or day_high, day_high)
    new_lwm = min(trade["low_water_mark"] or day_low, day_low)
    _record_trade_event(
        conn,
        trade_id=trade["trade_id"],
        event_type="mark",
        event_date=price_date,
        payload={
            "ticker": trade["ticker"],
            "open": day_open,
            "high": day_high,
            "low": day_low,
            "close": current_price,
            "stop_before": trade["stop_loss"],
            "target": trade["target_price"],
        },
    )
    barrier = resolve_barrier_exit(
        direction=trade["direction"],
        open_price=day_open,
        high=day_high,
        low=day_low,
        close=current_price,
        stop_loss=trade["stop_loss"],
        target_price=trade["target_price"],
        expired=_trade_expired(
            conn,
            entry_date=trade["entry_date"],
            price_date=price_date,
            config=config,
        ),
    )
    if barrier is not None:
        exit_price = (
            adverse_fill_price(
                barrier.raw_price,
                trade["direction"],
                config.exit_slippage_bps,
                entry=False,
            )
            if barrier.apply_slippage else barrier.raw_price
        )
        reason = (
            _stop_exit_reason(trade["direction"], trade["entry_price"], exit_price)
            if barrier.reason == "stopped_out" else barrier.reason
        )
        _close_trade(
            conn,
            trade["trade_id"],
            exit_price,
            price_date,
            reason,
            trade["direction"],
            trade["entry_price"],
            trade["stop_loss"],
            config=config,
        )
        return True

    original_risk = _resolve_original_risk(
        entry_price=trade["entry_price"],
        current_stop=trade["stop_loss"],
        stop_distance_pct=trade["stop_distance_pct"],
        atr_at_entry=trade["atr_at_entry"],
        config=config,
    )
    new_stop = compute_trailing_stop(
        direction=trade["direction"],
        entry_price=trade["entry_price"],
        original_risk=original_risk,
        current_hwm=new_hwm,
        current_lwm=new_lwm,
        current_stop=trade["stop_loss"],
        config=config,
    )
    if new_stop != trade["stop_loss"]:
        _record_trade_event(
            conn,
            trade_id=trade["trade_id"],
            event_type="management",
            event_date=price_date,
            payload={
                "action": "tighten_stop",
                "stop_before": trade["stop_loss"],
                "stop_after": new_stop,
                "high_water_mark": new_hwm,
                "low_water_mark": new_lwm,
            },
        )
    conn.execute(
        """UPDATE paper_trades SET
               current_price=?, unrealized_pnl_pct=?, last_mtm_date=?,
               high_water_mark=?, low_water_mark=?, stop_loss=?, updated_ts=?
           WHERE id=?""",
        (
            current_price,
            round(compute_pnl(trade["direction"], trade["entry_price"], current_price), 2),
            price_date,
            new_hwm,
            new_lwm,
            new_stop,
            dt.datetime.now(dt.timezone.utc).isoformat(),
            trade["trade_id"],
        ),
    )
    trade.update(
        high_water_mark=new_hwm,
        low_water_mark=new_lwm,
        stop_loss=new_stop,
        last_mtm_date=price_date,
    )
    return False


def _mark_to_market(
    conn: sqlite3.Connection,
    *,
    config: PaperTradeConfig,
    through_date: str | None = None,
) -> dict[str, Any]:
    """Update all open paper trades with latest prices."""
    ensure_paper_trade_schema(conn)

    from trader_koo.report.runs import verified_report_run_ids

    linked_ids = {
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT report_run_id FROM paper_trades WHERE report_run_id IS NOT NULL"
        )
    }
    verified_ids = verified_report_run_ids(conn, linked_ids)
    if verified_ids:
        placeholders = ",".join("?" for _ in verified_ids)
        open_rows = conn.execute(
            f"""
        SELECT id, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, high_water_mark, low_water_mark,
               stop_distance_pct, atr_at_entry, last_mtm_date
        FROM paper_trades
        WHERE campaign_id=? AND status = 'open'
          AND report_run_id IN ({placeholders})
        """,
            (config.campaign_id, *tuple(sorted(verified_ids))),
        ).fetchall()
    else:
        open_rows = []

    if not open_rows:
        update_portfolio_snapshot(conn, campaign_id=config.campaign_id)
        return {"open_trades": 0, "updated": 0, "closed": 0}

    updated = 0
    closed = 0
    blocked: list[dict[str, Any]] = []

    for row in open_rows:
        trade = {
            "trade_id": int(row[0]),
            "ticker": str(row[1]),
            "direction": str(row[2]),
            "entry_price": float(row[3]),
            "entry_date": str(row[4]),
            "target_price": float(row[5]) if row[5] is not None else None,
            "stop_loss": float(row[6]) if row[6] is not None else None,
            "high_water_mark": float(row[7]) if row[7] is not None else None,
            "low_water_mark": float(row[8]) if row[8] is not None else None,
            "stop_distance_pct": float(row[9]) if row[9] is not None else None,
            "atr_at_entry": float(row[10]) if row[10] is not None else None,
            "last_mtm_date": str(row[11]) if row[11] is not None else None,
        }

        contract = research_price_contract(conn, [trade["ticker"]])
        if not contract.get("eligible"):
            blocked.append({
                "trade_id": trade["trade_id"],
                "ticker": trade["ticker"],
                "reason": str(contract.get("reason") or "price_series_unresolved"),
            })
            continue

        query = (
            "SELECT CAST(close AS REAL),date,CAST(high AS REAL),CAST(low AS REAL),"
            "CAST(open AS REAL) FROM price_daily "
            "WHERE ticker=? AND date>? AND close IS NOT NULL"
        )
        params: list[Any] = [
            trade["ticker"], trade["last_mtm_date"] or trade["entry_date"],
        ]
        if through_date is not None:
            query += " AND date<=?"
            params.append(through_date)
        query += " ORDER BY date ASC"
        price_rows = conn.execute(query, params).fetchall()
        if not price_rows:
            continue
        updated += 1
        for price_row in price_rows:
            if _apply_trade_bar(conn, trade=trade, price_row=price_row, config=config):
                closed += 1
                break

    update_portfolio_snapshot(conn, campaign_id=config.campaign_id)
    return {
        "open_trades": len(open_rows) - closed,
        "updated": updated,
        "closed": closed,
        "blocked": blocked,
    }


def mark_to_market(
    conn: sqlite3.Connection,
    *,
    config: PaperTradeConfig,
) -> dict[str, Any]:
    return _run_owned_transaction(conn, lambda: _mark_to_market(conn, config=config))


def _manually_close_trade(
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
    contract = research_price_contract(conn, [str(ticker)])
    if not contract.get("eligible"):
        raise ValueError(
            f"Price series for {ticker} is unresolved; trade remains open"
        )

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
    pnl = round(compute_pnl(direction, entry_price, exit_price), 2)
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "direction": direction,
        "exit_price": exit_price,
        "pnl_pct": pnl,
        "status": "closed",
    }


def manually_close_trade(
    conn: sqlite3.Connection,
    *,
    trade_id: int,
    exit_price: float | None = None,
    exit_reason: str = "manual_close",
    config: PaperTradeConfig,
) -> dict[str, Any]:
    return _run_owned_transaction(
        conn,
        lambda: _manually_close_trade(
            conn,
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            config=config,
        ),
    )

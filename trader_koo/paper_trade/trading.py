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
        "SELECT position_size_pct FROM paper_trades WHERE id = ?", (trade_id,),
    ).fetchone()
    pos_pct = float(position_row[0] or 8.0) if position_row and position_row[0] is not None else 8.0
    notional = config.starting_capital * (pos_pct / 100)
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

    total_cost_pct = commission_cost_pct + borrow_cost_pct
    pnl = round(raw_pnl - total_cost_pct, 2)
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
    _record_trade_reflection(
        conn,
        trade_id=trade_id,
        exit_date=exit_date,
        exit_reason=exit_reason,
        pnl_pct=pnl,
        r_multiple=r_mult,
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
            "WHERE ticker = '^VIX' AND close IS NOT NULL AND date <= ? "
            "ORDER BY date DESC LIMIT 1",
            (report_date,),
        ).fetchone()
        if _vix_row and _vix_row[0] is not None:
            _vix_level = float(_vix_row[0])
    except Exception:
        pass

    # Current portfolio equity for position sizing (adapts as equity changes)
    _current_equity = config.starting_capital
    try:
        _eq_row = conn.execute(
            "SELECT equity_index FROM paper_portfolio_snapshots "
            "ORDER BY snapshot_date DESC LIMIT 1"
        ).fetchone()
        if _eq_row and _eq_row[0] is not None:
            _current_equity = config.starting_capital * (float(_eq_row[0]) / 100.0)
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

        # Apply entry slippage on top of open price.
        # Entry price must be computed FIRST — stops and targets are anchored
        # to this actual fill price so stop distance reflects real trade risk.
        slip_mult = config.entry_slippage_bps / 10_000
        if direction == "long":
            entry_price = round(raw_entry * (1 + slip_mult), 4)
        else:
            entry_price = round(raw_entry * (1 - slip_mult), 4)
        levels = compute_stop_and_target(row, direction, config=config, entry_price=entry_price)
        plan = compute_position_plan(row, evaluation, levels, config=config, vix_level=_vix_level, entry_price=entry_price)

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
                position_dollars = config.starting_capital * (position_pct / 100)
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
            if config.critic_fail_open:
                LOG.warning("Critic check failed (allowing trade by explicit config): %s", exc)
            else:
                LOG.warning("Critic check failed (rejecting trade): %s", exc)
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
                entry_reason, entry_evidence, entry_risks,
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
            ),
        )
        if conn.total_changes > before_changes:
            inserted += 1
            LOG.info(
                "Paper trade created: %s %s @ %.2f (stop=%.2f target=%.2f) — %s",
                direction.upper(), ticker, entry_price,
                levels["stop_loss"], levels["target_price"], rationale["entry_reason"],
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
               target_price, stop_loss, high_water_mark, low_water_mark,
               stop_distance_pct, atr_at_entry
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
        target_price, stop_loss, high_water_mark, low_water_mark = row[5:9]
        stop_distance_pct, atr_at_entry = row[9:]

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
            if config.expiry_use_trading_days:
                # Count actual trading days using SPY price rows
                td_row = conn.execute(
                    "SELECT COUNT(*) FROM price_daily "
                    "WHERE ticker = 'SPY' AND date > ? AND date <= ?",
                    (entry_date, today),
                ).fetchone()
                days_held = int(td_row[0]) if td_row and td_row[0] else 0
                if days_held >= config.expiry_days:
                    expired = True
                elif days_held == 0:
                    # Fallback: no SPY data → use calendar days
                    entry_dt = dt.datetime.strptime(entry_date, "%Y-%m-%d")
                    today_dt = dt.datetime.strptime(today, "%Y-%m-%d")
                    if (today_dt - entry_dt).days >= config.expiry_days:
                        expired = True
            else:
                # Legacy calendar-day fallback
                entry_dt = dt.datetime.strptime(entry_date, "%Y-%m-%d")
                today_dt = dt.datetime.strptime(today, "%Y-%m-%d")
                if (today_dt - entry_dt).days >= config.expiry_days:
                    expired = True
        except (ValueError, TypeError):
            pass

        original_risk = _resolve_original_risk(
            entry_price=float(entry_price or 0.0),
            current_stop=float(stop_loss) if isinstance(stop_loss, (int, float)) else None,
            stop_distance_pct=float(stop_distance_pct) if isinstance(stop_distance_pct, (int, float)) else None,
            atr_at_entry=float(atr_at_entry) if isinstance(atr_at_entry, (int, float)) else None,
            config=config,
        )

        if hit_stop:
            exit_reason = _stop_exit_reason(direction, float(entry_price), current_price)
            _close_trade(
                conn,
                trade_id,
                current_price,
                today,
                exit_reason,
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
            # Graduated trailing stop (uses original_risk from ATR at entry)
            new_stop = compute_trailing_stop(
                direction=direction,
                entry_price=entry_price,
                original_risk=original_risk,
                current_hwm=new_hwm,
                current_lwm=new_lwm,
                current_stop=stop_loss,
                config=config,
            )

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

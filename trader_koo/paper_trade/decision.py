"""Decision and sizing logic for paper trades."""

from __future__ import annotations

import logging
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig

LOG = logging.getLogger(__name__)


def _validate_stop_and_target(
    direction: str,
    entry: float,
    stop_loss: float,
    target_price: float,
) -> None:
    """Validate stop/target geometry. Raises ValueError on invariant violations.

    Uses explicit raise (not assert) so it fires even under Python -O.
    Called at the end of compute_stop_and_target() to catch anchor bugs early.
    """
    if entry <= 0:
        raise ValueError(f"Entry price must be positive, got {entry}")

    if direction == "long":
        if stop_loss >= entry:
            raise ValueError(
                f"Long stop_loss {stop_loss:.4f} must be below entry {entry:.4f}"
            )
        if target_price <= entry:
            raise ValueError(
                f"Long target_price {target_price:.4f} must be above entry {entry:.4f}"
            )
    else:
        if stop_loss <= entry:
            raise ValueError(
                f"Short stop_loss {stop_loss:.4f} must be above entry {entry:.4f}"
            )
        if target_price >= entry:
            raise ValueError(
                f"Short target_price {target_price:.4f} must be below entry {entry:.4f}"
            )

    risk = abs(entry - stop_loss)
    reward = abs(target_price - entry)
    if risk <= 0:
        raise ValueError(f"Risk is zero (stop == entry = {entry:.4f})")
    rr = reward / risk
    if rr < 1.5:
        LOG.warning(
            "Low R:R %.2f (reward=%.4f risk=%.4f) on %s entry=%.4f — below 1.5 minimum",
            rr, reward, risk, direction, entry,
        )


def direction_from_row(row: dict[str, Any]) -> str:
    family = str(row.get("setup_family") or "").strip().lower()
    bias = str(row.get("signal_bias") or "").strip().lower()
    if family.startswith("bullish") or bias == "bullish":
        return "long"
    if family.startswith("bearish") or bias == "bearish":
        return "short"
    return "neutral"


def qualify_setup_for_paper_trade(
    row: dict[str, Any],
    *,
    config: PaperTradeConfig,
) -> bool:
    """Return True if a setup row qualifies as a paper trade entry."""
    return evaluate_setup_for_paper_trade(row, config=config)["approved"]


def evaluate_setup_for_paper_trade(
    row: dict[str, Any],
    *,
    config: PaperTradeConfig,
) -> dict[str, Any]:
    """Return staged decision metadata for paper-trade qualification."""
    tier = str(row.get("setup_tier") or "").strip().upper()
    reasons: list[str] = []
    risk_flags: list[str] = []
    analyst_stage = "pass"
    debate_stage = "pass"
    risk_stage = "pass"

    if tier not in config.qualifying_tiers:
        min_rank = config.tier_rank.get(config.min_tier, 2)
        if config.tier_rank.get(tier, 99) > min_rank:
            analyst_stage = "reject"
            reasons.append(
                f"Tier {tier or 'unknown'} is below paper-trade minimum {config.min_tier}."
            )

    score = row.get("score")
    if not isinstance(score, (int, float)) or float(score) < config.min_score:
        analyst_stage = "reject"
        reasons.append(
            f"Score {float(score):.1f}" if isinstance(score, (int, float)) else "Missing score"
        )

    actionability = str(row.get("actionability") or "").strip().lower()
    if actionability not in config.qualifying_actionability:
        debate_stage = "reject"
        reasons.append(
            f"Actionability '{actionability or 'unknown'}' is not eligible for paper trading."
        )
    elif actionability == "conditional":
        debate_stage = "caution"
        reasons.append("Setup is conditional rather than fully ready.")

    debate_score = row.get("debate_agreement_score")
    if isinstance(debate_score, (int, float)) and float(debate_score) < config.debate_caution_agreement:
        debate_stage = "caution" if debate_stage != "reject" else debate_stage
        reasons.append(f"Debate agreement is only {float(debate_score):.0f}%.")

    direction = direction_from_row(row)
    if direction not in config.qualifying_directions:
        analyst_stage = "reject"
        reasons.append("Signal direction is neutral or unsupported.")

    close = row.get("close")
    if not isinstance(close, (int, float)) or float(close) <= 0:
        analyst_stage = "reject"
        reasons.append("Entry price is missing or non-positive.")

    atr_pct = row.get("atr_pct_14")
    if isinstance(atr_pct, (int, float)) and float(atr_pct) >= config.high_vol_atr_pct:
        risk_stage = "caution"
        risk_flags.append(f"ATR {float(atr_pct):.1f}% suggests elevated volatility.")

    risk_note = str(row.get("risk_note") or "").strip().lower()
    if risk_note:
        for needle, label in (
            ("earnings", "Earnings event risk is still in play."),
            ("high volatility", "Risk note flags high volatility."),
            ("volatility", "Risk note mentions volatility."),
            ("gap", "Risk note mentions gap risk."),
            ("low liquidity", "Risk note mentions low liquidity."),
        ):
            if needle in risk_note and label not in risk_flags:
                risk_stage = "caution"
                risk_flags.append(label)

    yolo_recency = str(row.get("yolo_recency") or "").strip().lower()
    if "stale" in yolo_recency:
        risk_stage = "caution"
        risk_flags.append("YOLO context is stale.")

    approved = analyst_stage != "reject" and debate_stage != "reject"
    if not approved:
        portfolio_decision = "rejected"
        decision_state = "rejected"
        decision_summary = "Rejected by paper-trade gating."
    elif debate_stage == "caution" or risk_stage == "caution":
        portfolio_decision = "approved_with_flags"
        decision_state = "approved_with_flags"
        decision_summary = "Approved for paper trading with caution flags."
    else:
        portfolio_decision = "approved"
        decision_state = "approved"
        decision_summary = "Approved for paper trading."

    return {
        "approved": approved,
        "decision_version": config.decision_version,
        "decision_state": decision_state,
        "analyst_stage": analyst_stage,
        "debate_stage": debate_stage,
        "risk_stage": risk_stage,
        "portfolio_decision": portfolio_decision,
        "decision_summary": decision_summary,
        "decision_reasons": reasons,
        "risk_flags": risk_flags,
        "direction": direction,
    }


def compute_stop_and_target(
    row: dict[str, Any],
    direction: str,
    *,
    config: PaperTradeConfig,
    entry_price: float | None = None,
) -> dict[str, float | None]:
    """Compute stop_loss, target_price, and atr_at_entry from a setup row.

    Parameters
    ----------
    entry_price : float | None
        Actual fill price (next-day open + slippage). When provided, stops and
        targets are anchored to this value instead of the signal-day close so
        that stop distance reflects the real trade risk, not a stale price.
    """
    entry = entry_price if entry_price is not None and entry_price > 0 else float(row["close"])
    atr_pct = row.get("atr_pct_14")
    support = row.get("support_level")
    resistance = row.get("resistance_level")

    if isinstance(atr_pct, (int, float)) and float(atr_pct) > 0:
        atr_distance = (float(atr_pct) / 100.0) * entry * config.stop_atr_mult
    else:
        atr_distance = entry * (config.default_stop_pct / 100.0)

    if direction == "long":
        stop_loss = entry - atr_distance
        if isinstance(support, (int, float)) and float(support) > 0:
            support_stop = float(support) * 0.99
            if entry * 0.95 < support_stop < entry:
                stop_loss = max(stop_loss, support_stop)

        risk = entry - stop_loss
        if isinstance(resistance, (int, float)) and float(resistance) > entry:
            target_price = float(resistance)
        else:
            target_price = entry + (risk * 2.0)
    else:
        stop_loss = entry + atr_distance
        if isinstance(resistance, (int, float)) and float(resistance) > 0:
            resist_stop = float(resistance) * 1.01
            if entry < resist_stop < entry * 1.05:
                stop_loss = min(stop_loss, resist_stop)

        risk = stop_loss - entry
        if isinstance(support, (int, float)) and 0 < float(support) < entry:
            target_price = float(support)
        else:
            target_price = max(entry - (risk * 2.0), 0.01)  # floor at 0.01 — stock can't go negative

    # Enforce minimum stop distance: max(1.5x ATR, 2.5%) to avoid noise stops
    atr_pct_val = float(atr_pct) if isinstance(atr_pct, (int, float)) else 3.0
    min_stop_pct = max(atr_pct_val * 1.5 / 100.0, 0.025)
    min_stop_distance = entry * min_stop_pct
    if direction == "long" and (entry - stop_loss) < min_stop_distance:
        stop_loss = round(entry - min_stop_distance, 2)
    elif direction == "short" and (stop_loss - entry) < min_stop_distance:
        stop_loss = round(entry + min_stop_distance, 2)

    stop_loss = round(stop_loss, 2)
    target_price = round(target_price, 2)

    try:
        _validate_stop_and_target(direction, entry, stop_loss, target_price)
    except ValueError as exc:
        LOG.error("Stop/target validation failed: %s — levels may be unreliable", exc)
        raise

    return {
        "stop_loss": stop_loss,
        "target_price": target_price,
        "atr_at_entry": round(float(atr_pct), 2) if isinstance(atr_pct, (int, float)) else None,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_position_plan(
    row: dict[str, Any],
    evaluation: dict[str, Any],
    levels: dict[str, float | None],
    *,
    config: PaperTradeConfig,
    vix_level: float | None = None,
    entry_price: float | None = None,
) -> dict[str, float | str | None]:
    """Build a sizing and execution plan for a paper trade.

    Uses constant-dollar-risk sizing: every trade risks exactly
    ``config.risk_per_trade_pct`` of capital regardless of stop distance.
    The resulting notional is capped at the tier maximum to prevent
    overleveraging on unusually tight stops.

    Parameters
    ----------
    entry_price : float | None
        Actual fill price (next-day open + slippage). Used to compute
        stop_distance_pct from the real entry rather than the signal close.
    """
    entry = entry_price if entry_price is not None and entry_price > 0 else float(row["close"])
    direction = str(evaluation["direction"])
    tier = str(row.get("setup_tier") or "").strip().upper()
    score = float(row.get("score") or 0.0)
    atr_pct = row.get("atr_pct_14")
    atr_pct_num = float(atr_pct) if isinstance(atr_pct, (int, float)) else None
    risk_note = str(row.get("risk_note") or "").strip().lower()

    # Tier notional caps (hard ceiling — never exceed regardless of stop size)
    tier_notional_cap = {
        "A": config.tier_a_position_pct,
        "B": config.tier_b_position_pct,
        "C": config.tier_c_position_pct,
    }.get(tier, config.min_position_pct)
    sizing_notes = [f"tier {tier or 'fallback'} cap {tier_notional_cap:.1f}%"]

    # Compute stop distance first — needed for risk-based sizing
    stop_loss = levels.get("stop_loss")
    target_price = levels.get("target_price")
    stop_distance_pct: float | None = None
    expected_reward_pct: float | None = None
    expected_r_multiple: float | None = None

    if isinstance(stop_loss, (int, float)) and entry > 0:
        stop_distance_pct = abs(entry - float(stop_loss)) / entry * 100.0

    if isinstance(target_price, (int, float)) and entry > 0:
        expected_reward_pct = abs(float(target_price) - entry) / entry * 100.0

    if (
        isinstance(stop_distance_pct, (int, float))
        and stop_distance_pct > 0
        and isinstance(expected_reward_pct, (int, float))
    ):
        expected_r_multiple = expected_reward_pct / stop_distance_pct

    # Risk-based sizing: size so that hitting the stop costs exactly risk_per_trade_pct
    # position_size_pct × stop_distance_pct / 100 = risk_per_trade_pct
    # ⟹ position_size_pct = risk_per_trade_pct / stop_distance_pct × 100
    if isinstance(stop_distance_pct, (int, float)) and stop_distance_pct > 0:
        position_size_pct = (config.risk_per_trade_pct / stop_distance_pct) * 100.0
        sizing_notes.append(f"risk-based: {config.risk_per_trade_pct:.2f}% risk / {stop_distance_pct:.2f}% stop")
    else:
        # Fallback to tier base when stop distance is unavailable
        position_size_pct = tier_notional_cap
        sizing_notes.append("fallback to tier notional (no stop distance)")

    # Apply risk-environment haircuts (reduce size in adverse conditions)
    if evaluation.get("decision_state") == "approved_with_flags":
        position_size_pct *= config.caution_position_scale
        sizing_notes.append("caution haircut applied")

    if atr_pct_num is not None and atr_pct_num >= config.high_vol_atr_pct:
        position_size_pct *= config.high_vol_position_scale
        sizing_notes.append("high-volatility haircut applied")

    if "earnings" in risk_note:
        position_size_pct *= config.earnings_position_scale
        sizing_notes.append("event-risk haircut applied")

    if vix_level is not None:
        if vix_level < 15:
            vix_scale = 1.1
        elif vix_level < 20:
            vix_scale = 1.0
        elif vix_level < 25:
            vix_scale = 0.85
        elif vix_level < 30:
            vix_scale = 0.65
        else:
            vix_scale = 0.5
        position_size_pct *= vix_scale
        sizing_notes.append(f"VIX={vix_level:.1f} scale={vix_scale:.2f}")

    # Cap at tier notional ceiling (prevents overleveraging on very tight stops)
    position_size_pct = _clamp(
        position_size_pct,
        config.min_position_pct,
        min(tier_notional_cap, config.max_position_pct),
    )

    risk_budget_pct = (
        position_size_pct * stop_distance_pct / 100.0
        if isinstance(stop_distance_pct, (int, float))
        else None
    )

    if direction == "long":
        entry_plan = (
            f"Enter long around {entry:.2f} only while price remains above "
            f"{float(stop_loss):.2f} invalidation."
            if isinstance(stop_loss, (int, float))
            else f"Enter long around {entry:.2f} with disciplined risk."
        )
        exit_plan = (
            f"Initial stop {float(stop_loss):.2f}; objective {float(target_price):.2f}. "
            f"Reassess after +1R and avoid adding into exhaustion."
            if isinstance(stop_loss, (int, float)) and isinstance(target_price, (int, float))
            else "Use the initial stop and reassess at +1R."
        )
    else:
        entry_plan = (
            f"Enter short around {entry:.2f} only while price stays below "
            f"{float(stop_loss):.2f} invalidation."
            if isinstance(stop_loss, (int, float))
            else f"Enter short around {entry:.2f} with disciplined risk."
        )
        exit_plan = (
            f"Initial stop {float(stop_loss):.2f}; objective {float(target_price):.2f}. "
            f"Cover partial risk after +1R and avoid pressing into support."
            if isinstance(stop_loss, (int, float)) and isinstance(target_price, (int, float))
            else "Use the initial stop and reassess at +1R."
        )

    sizing_summary = (
        f"{position_size_pct:.1f}% notional, "
        f"{risk_budget_pct:.2f}% risk budget"
        if isinstance(risk_budget_pct, (int, float))
        else f"{position_size_pct:.1f}% notional"
    )

    return {
        "position_size_pct": round(position_size_pct, 2),
        "risk_budget_pct": round(risk_budget_pct, 2)
        if isinstance(risk_budget_pct, (int, float))
        else None,
        "stop_distance_pct": round(stop_distance_pct, 2)
        if isinstance(stop_distance_pct, (int, float))
        else None,
        "expected_reward_pct": round(expected_reward_pct, 2)
        if isinstance(expected_reward_pct, (int, float))
        else None,
        "expected_r_multiple": round(expected_r_multiple, 2)
        if isinstance(expected_r_multiple, (int, float))
        else None,
        "entry_plan": entry_plan,
        "exit_plan": exit_plan,
        "sizing_summary": "; ".join(sizing_notes + [sizing_summary]),
        "review_status": "monitoring",
        "review_summary": "Open trade. Compare progress against the initial stop/target plan.",
    }

"""Decision and sizing logic for paper trades."""

from __future__ import annotations

from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig


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
) -> dict[str, float | None]:
    """Compute stop_loss, target_price, and atr_at_entry from a setup row."""
    entry = float(row["close"])
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
            target_price = entry - (risk * 2.0)

    return {
        "stop_loss": round(stop_loss, 2),
        "target_price": round(target_price, 2),
        "atr_at_entry": round(float(atr_pct), 2) if isinstance(atr_pct, (int, float)) else None,
    }


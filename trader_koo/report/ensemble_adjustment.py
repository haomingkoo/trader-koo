"""Technical-ensemble adjustment of a setup row's score and tier (moved unchanged from generator)."""
from __future__ import annotations

from typing import Any

MAX_BOOST = 8.0          # score points added at most when the ensemble agrees
BOOST_PER_NET = 0.1      # per point of ensemble net score
MAX_PENALTY = 10.0       # score points removed at most when it disagrees
PENALTY_PER_NET = 0.12
HIGH_AGREEMENT_PCT = 80  # ensemble agreement at which the effect is scaled up
HIGH_AGREEMENT_MULT = 1.5
DOWNGRADE_PENALTY = 8.0  # a penalty this large downgrades an A or B tier
MAX_SCORE = 100.0


def apply_ensemble_adjustment(row: dict[str, Any]) -> None:
    """Adjust score and tier based on technical ensemble agreement.

    If the 5-strategy ensemble agrees with confluence bias, boost score.
    If it disagrees, penalize. High agreement (>70%) = bigger effect.
    """
    ensemble_bias = row.get("ensemble_bias")
    confluence_bias = row.get("signal_bias")
    agreement = row.get("ensemble_agreement_pct", 0)
    net_score = row.get("ensemble_net_score", 0)

    if not ensemble_bias or not confluence_bias:
        return

    score = float(row.get("score") or 0)
    tier = str(row.get("setup_tier") or "D")

    if ensemble_bias == confluence_bias:
        # Agreement: boost score proportionally to ensemble confidence
        boost = min(MAX_BOOST, abs(net_score) * BOOST_PER_NET)
        if agreement >= HIGH_AGREEMENT_PCT:
            boost *= HIGH_AGREEMENT_MULT
        score = min(MAX_SCORE, score + boost)
    elif ensemble_bias != "neutral" and confluence_bias != "neutral" and ensemble_bias != confluence_bias:
        # Disagreement: penalize and potentially downgrade tier
        penalty = min(MAX_PENALTY, abs(net_score) * PENALTY_PER_NET)
        if agreement >= HIGH_AGREEMENT_PCT:
            penalty *= HIGH_AGREEMENT_MULT
        score = max(0.0, score - penalty)
        # Strong disagreement with high agreement: downgrade tier
        if penalty >= DOWNGRADE_PENALTY and tier in {"A", "B"}:
            tier = "B" if tier == "A" else "C"

    row["score"] = round(score, 1)
    row["confluence_score"] = round(score, 1)
    row["setup_tier"] = tier

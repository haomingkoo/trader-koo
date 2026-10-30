"""The calibration re-grade keeps the debate's tier ceiling and earlier downgrades."""
from __future__ import annotations

from trader_koo.report.setup_scoring import (
    _apply_agreement_tier_adjustment,
    _apply_debate_guardrails,
    _apply_setup_eval_fields,
)

FAMILY = ("bullish_continuation", "long")


def _row(score: float, state: str, agreement: float = 60.0) -> dict:
    tier = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"
    return {
        "ticker": "TEST", "score": score, "setup_tier": tier, "signal_bias": "bullish",
        "setup_family": "bullish_continuation", "actionability": "higher-probability",
        "debate_v1": {"consensus": {"consensus_state": state, "consensus_bias": "bullish",
                                    "agreement_score": agreement, "disagreement_count": 0}},
    }


def _calibrate(row: dict, adjustment: float) -> dict:
    _apply_setup_eval_fields(
        [row], reliability_lookup={}, min_sample=5, baseline_hit_rate_pct=50.0,
        calibration_state={FAMILY: {"score_adjustment": adjustment}},
    )
    return row


def test_watch_cap_survives_a_score_bonus():
    row = _row(74.5, "watch")  # B by score, capped to C by the debate (production: IFF 2026-10-29)
    _apply_debate_guardrails([row])
    assert row["setup_tier"] == "C"
    assert _calibrate(row, 10.0)["setup_score_adjusted"] == 84.5
    assert row["setup_tier"] == "C"


def test_watch_cap_survives_a_score_cut():
    row = _row(69.0, "watch")  # production: XEL, cut by 4 points yet shown as B
    _apply_debate_guardrails([row])
    assert _calibrate(row, -4.0)["setup_tier"] == "C"


def test_conditional_row_is_not_lifted_to_a():
    row = _row(78.0, "conditional")
    _apply_debate_guardrails([row])
    assert _calibrate(row, 5.0)["setup_score_adjusted"] == 83.0
    assert row["setup_tier"] == "B"


def test_low_agreement_downgrade_survives_the_regrade():
    row = _row(82.0, "ready", agreement=40.0)
    _apply_agreement_tier_adjustment(row)
    _apply_debate_guardrails([row])
    assert row["setup_tier"] == "B"
    assert _calibrate(row, 2.0)["setup_tier"] == "B"


def test_an_uncapped_row_can_still_move_both_ways():
    up = _row(78.0, "ready")
    _apply_debate_guardrails([up])
    assert _calibrate(up, 5.0)["setup_tier"] == "A"
    down = _row(66.0, "ready")
    _apply_debate_guardrails([down])
    assert _calibrate(down, -4.0)["setup_tier"] == "C"


def test_inline_adjustment_compares_hit_rate_with_the_baseline_hit_rate():
    """Production family stats 2026-10-29: 33.47% hit over 505 calls, -0.77% average."""
    row = _row(70.0, "ready")
    stat = {"hit_rate_pct": 33.47, "calls": 505, "avg_signed_return_pct": -0.77}
    _apply_setup_eval_fields([row], reliability_lookup={FAMILY: stat}, min_sample=5,
                             baseline_hit_rate_pct=49.0)
    assert row["setup_score_adjustment"] < 0  # was +6.8 when 0.3 (a return %) stood in for a hit rate


def test_inline_adjustment_without_a_baseline_uses_returns_only():
    row = _row(70.0, "ready")
    stat = {"hit_rate_pct": 90.0, "calls": 505, "avg_signed_return_pct": -0.5}
    _apply_setup_eval_fields([row], reliability_lookup={FAMILY: stat}, min_sample=5,
                             baseline_hit_rate_pct=None)
    assert row["setup_score_adjustment"] == -0.8

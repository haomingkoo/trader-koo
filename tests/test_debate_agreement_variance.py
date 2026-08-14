"""Test that debate engine produces varied agreement scores, not always 100%."""

import pytest
from trader_koo.debate_engine import (
    _aggregate_roles,
    _bear_researcher,
    _momentum_role,
    build_setup_debate,
)
from trader_koo.report.setup_scoring import _apply_debate_guardrails


def test_debate_produces_varied_agreement_scores():
    """Agreement scores should vary based on signal quality, not always 100%."""

    # Strong bullish setup - all analysts bullish
    bullish_setup = {
        "trend_state": "uptrend",
        "breakout_state": "breakout_up",
        "ma_major_signal": "golden_cross",
        "pct_change": 2.5,
        "volume_ratio_20": 1.8,
        "candle_bias": "bullish",
        "discount_pct": 15.0,
        "peg": 1.2,
        "yolo_bias": "bullish",
        "yolo_recency": "fresh",
        "yolo_direction_conflict": False,
    }

    # Mixed signals - some bullish, some bearish
    mixed_setup = {
        "trend_state": "uptrend",
        "pct_change": -0.5,
        "volume_ratio_20": 0.6,
        "candle_bias": "bearish",
        "stretch_state": "extended_up",
        "pct_vs_ma20": 9.5,
        "discount_pct": -5.0,
        "peg": 4.2,
        "yolo_bias": "bearish",
        "yolo_direction_conflict": True,
    }

    bullish_result = build_setup_debate(bullish_setup)
    mixed_result = build_setup_debate(mixed_setup)

    bullish_agreement = bullish_result["consensus"]["agreement_score"]
    mixed_agreement = mixed_result["consensus"]["agreement_score"]

    # Agreement scores should be different
    assert bullish_agreement != mixed_agreement, \
        "Agreement scores should vary based on signal quality"

    # Neither should be 100% (that's the bug we're fixing)
    assert bullish_agreement < 100.0, \
        "Even strong setups shouldn't have 100% agreement"
    assert mixed_agreement < 100.0, \
        "Mixed signals shouldn't have 100% agreement"

    # Aligned directional evidence should score higher than a split panel.
    assert 40.0 <= bullish_agreement <= 90.0, \
        f"Bullish agreement {bullish_agreement}% should be in reasonable range"
    assert mixed_agreement < bullish_agreement


def test_bull_bear_researchers_debate():
    """Bull and Bear researchers should often disagree."""

    # Setup with conflicting signals
    conflicted_setup = {
        "trend_state": "uptrend",  # Bullish
        "pct_change": -2.0,  # Bearish
        "volume_ratio_20": 1.5,  # Bullish
        "stretch_state": "extended_up",  # Bearish risk
        "pct_vs_ma20": 10.0,  # Bearish risk
        "yolo_bias": "bullish",  # Bullish
        "yolo_direction_conflict": True,  # Bearish risk
    }

    result = build_setup_debate(conflicted_setup)
    consensus = result["consensus"]

    # Should have debate info
    assert "debate" in consensus, "Consensus should include debate details"

    debate = consensus["debate"]
    bull_stance = debate["bull_researcher"]["stance"]
    bear_stance = debate["bear_researcher"]["stance"]

    # Bull and Bear should have different stances
    assert bull_stance != bear_stance or bull_stance == "neutral" or bear_stance == "neutral", \
        "Bull and Bear researchers should disagree or one should be neutral"


def test_unanimous_analyst_agreement_penalty():
    """When all analysts agree, agreement score should be reduced (suspicious)."""

    # Extreme bullish setup where all analysts would agree
    unanimous_setup = {
        "trend_state": "uptrend",
        "breakout_state": "breakout_up",
        "ma_major_signal": "golden_cross",
        "ma_signal": "bullish_20_50_cross",
        "level_event": "resistance_breakout",
        "pct_change": 3.0,
        "volume_ratio_20": 2.0,
        "candle_bias": "bullish",
        "stretch_state": "normal",
        "pct_vs_ma20": 2.0,
        "discount_pct": 20.0,
        "peg": 0.7,
        "yolo_bias": "bullish",
        "yolo_recency": "fresh",
        "yolo_direction_conflict": False,
        "actionability": "ready",
    }

    result = build_setup_debate(unanimous_setup)
    agreement = result["consensus"]["agreement_score"]

    # Even with perfect setup, agreement should be < 90% due to penalty
    assert agreement < 90.0, \
        "Unanimous analyst agreement should trigger skepticism penalty"


def test_strong_aligned_setup_can_reach_ready():
    setup = {
        "trend_state": "uptrend",
        "breakout_state": "breakout_up",
        "ma_major_signal": "golden_cross",
        "ma_signal": "bullish_20_50_cross",
        "level_event": "resistance_breakout",
        "pct_change": 3.0,
        "volume_ratio_20": 2.0,
        "candle_bias": "bullish",
        "stretch_state": "normal",
        "pct_vs_ma20": 2.0,
        "discount_pct": 20.0,
        "peg": 0.7,
        "yolo_bias": "bullish",
        "yolo_recency": "fresh",
        "yolo_direction_conflict": False,
        "actionability": "ready",
    }

    consensus = build_setup_debate(setup)["consensus"]

    assert consensus["consensus_bias"] == "bullish"
    assert consensus["agreement_score"] == 75.0
    assert consensus["consensus_state"] == "ready"


def test_empty_input_has_no_directional_agreement():
    consensus = build_setup_debate({})["consensus"]

    assert consensus["consensus_bias"] == "neutral"
    assert consensus["agreement_score"] == 0.0
    assert consensus["consensus_state"] == "watch"


def test_opposing_confidence_gap_does_not_increase_agreement():
    def role(stance: str, confidence: float) -> dict:
        return {
            "role": stance,
            "stance": stance,
            "confidence": confidence,
            "evidence": [],
            "risk_flags": [],
            "weight": 0.5,
        }

    balanced = _aggregate_roles(
        [role("bullish", 0.5), role("bearish", 0.5)],
        {},
    )
    lopsided = _aggregate_roles(
        [role("bullish", 0.9), role("bearish", 0.5)],
        {},
    )

    assert balanced["agreement_score"] == 0.0
    assert lopsided["agreement_score"] == balanced["agreement_score"]


def test_risk_caution_does_not_create_directional_consensus():
    row = {
        "signal_bias": "neutral",
        "actionability": "wait",
        "trend_state": "mixed",
        "pct_change": 0.0,
        "candle_bias": "neutral",
        "yolo_bias": "neutral",
        "yolo_recency": "stale",
        "yolo_direction_conflict": True,
        "stretch_state": "extended_up",
    }
    row["debate_v1"] = build_setup_debate(row)

    assert row["debate_v1"]["consensus"]["consensus_bias"] == "neutral"
    _apply_debate_guardrails([row])
    assert row["signal_bias"] == "neutral"


def test_risk_flags_can_still_downgrade_ready_state():
    setup = {
        "trend_state": "uptrend",
        "breakout_state": "breakout_up",
        "ma_major_signal": "golden_cross",
        "level_event": "resistance_breakout",
        "pct_change": 3.0,
        "volume_ratio_20": 2.0,
        "candle_bias": "bullish",
        "stretch_state": "extended_up",
        "discount_pct": 20.0,
        "peg": 0.7,
        "yolo_bias": "bullish",
        "yolo_recency": "fresh",
        "actionability": "wait",
    }

    consensus = build_setup_debate(setup)["consensus"]

    assert consensus["consensus_bias"] == "bullish"
    assert consensus["consensus_state"] == "conditional"
    assert consensus["safety_adjustment"] == ["risk_manager_flags"]


@pytest.mark.parametrize("volume_ratio", [0.0, 0.01])
def test_low_volume_values_are_not_treated_as_missing(volume_ratio):
    momentum = _momentum_role({"volume_ratio_20": volume_ratio})
    bear = _bear_researcher([], {"volume_ratio_20": volume_ratio})

    assert "low participation" in momentum["risk_flags"]
    assert "weak volume confirmation" in bear["evidence"]


def test_missing_volume_keeps_neutral_default():
    momentum = _momentum_role({})
    bear = _bear_researcher([], {})

    assert "low participation" not in momentum["risk_flags"]
    assert "weak volume confirmation" not in bear["evidence"]


def test_debate_structure():
    """Verify debate output includes bull/bear researcher details."""

    setup = {
        "trend_state": "uptrend",
        "pct_change": 1.0,
        "yolo_bias": "bullish",
    }

    result = build_setup_debate(setup)
    consensus = result["consensus"]

    assert "debate" in consensus
    assert "bull_researcher" in consensus["debate"]
    assert "bear_researcher" in consensus["debate"]

    bull = consensus["debate"]["bull_researcher"]
    bear = consensus["debate"]["bear_researcher"]

    # Each researcher should have stance, confidence, evidence
    assert "stance" in bull
    assert "confidence" in bull
    assert "evidence" in bull
    assert isinstance(bull["evidence"], list)

    assert "stance" in bear
    assert "confidence" in bear
    assert "evidence" in bear
    assert isinstance(bear["evidence"], list)


def test_debate_includes_news_and_options_research_roles():
    setup = {
        "trend_state": "uptrend",
        "pct_change": 1.2,
        "signal_bias": "bullish",
        "news_sentiment_score": 72,
        "macro_news_score": 58,
        "news_context": {
            "ticker_headlines": [{"title": "AMD rallies on strong demand"}],
        },
        "options_context": {
            "signal": "underpriced_positioning",
            "iv_rank_pct": 22,
            "oi_rank_pct": 91,
            "underpriced_score": 74,
            "positioning_skew": "call_oi_skew",
        },
    }

    result = build_setup_debate(setup)
    roles = {role["role"]: role for role in result["roles"]}

    assert result["version"] == "v2"
    assert "news_catalyst" in roles
    assert "options_positioning" in roles
    assert roles["options_positioning"]["stance"] == "bullish"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

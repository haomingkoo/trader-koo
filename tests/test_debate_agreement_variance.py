"""Test that debate engine produces varied agreement scores, not always 100%."""

import pytest
from trader_koo.debate_engine import build_setup_debate


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

    # Both should be in reasonable range (not always 100%)
    assert 40.0 <= bullish_agreement <= 90.0, \
        f"Bullish agreement {bullish_agreement}% should be in reasonable range"
    assert 40.0 <= mixed_agreement <= 90.0, \
        f"Mixed agreement {mixed_agreement}% should be in reasonable range"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from __future__ import annotations

from trader_koo.report.suggestions import build_suggestions


def test_build_suggestions_returns_compact_top_ideas():
    rows = [
        {
            "ticker": "AMD",
            "score": 74.0,
            "setup_tier": "B",
            "setup_family": "bullish_continuation",
            "signal_bias": "bullish",
            "calibrated_hit_prob": 0.61,
            "probability_sample_size": 8,
            "debate_agreement_score": 72,
            "debate_consensus_bias": "bullish",
            "options_positioning_signal": "underpriced_positioning",
            "news_sentiment_score": 66,
            "support_level": 100.5,
            "action": "Hold above breakout shelf.",
        },
        {
            "ticker": "XYZ",
            "score": 52.0,
            "setup_tier": "C",
            "signal_bias": "neutral",
        },
    ]

    payload = build_suggestions(rows)

    assert payload["count"] == 1
    assert "not financial advice" in payload["note"]
    item = payload["items"][0]
    assert item["ticker"] == "AMD"
    assert item["action"] == "Paper Long"
    assert item["conviction"] in {"Medium", "Higher"}
    assert len(item["why"]) <= 3
    assert "Invalid below support" in item["invalidation"]


def test_build_suggestions_prefers_watch_for_low_conviction():
    payload = build_suggestions([
        {
            "ticker": "LOWQ",
            "score": 58.0,
            "setup_tier": "C",
            "signal_bias": "bearish",
            "calibrated_hit_prob": 0.52,
            "probability_sample_size": 0,
        }
    ])

    assert payload["items"][0]["action"] == "Watch"

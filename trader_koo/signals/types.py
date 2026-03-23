"""Standardized signal output format for all signal sources.

Every signal source (YOLO patterns, candle patterns, technical indicators,
etc.) should produce SignalOutput instances so they can be aggregated
uniformly by the scoring and ensemble systems.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalOutput:
    """One directional signal from any source."""

    signal_type: str        # e.g., "yolo_pattern", "candle", "trend", "mean_reversion"
    bias: str               # "bullish" | "bearish" | "neutral"
    confidence: float       # 0-100
    reasoning: str          # human-readable explanation
    weight: float = 1.0     # relative importance (default 1.0)


def aggregate_signals(signals: list[SignalOutput]) -> dict[str, float | str | int]:
    """Combine multiple signals into weighted bull/bear scores.

    Returns a dict with:
        bias: "bullish" | "bearish" | "neutral"
        bull_score: weighted bullish confidence
        bear_score: weighted bearish confidence
        net_score: bull_score - bear_score
        signal_count: total signals
        agreement_pct: % of signals agreeing with majority direction
    """
    if not signals:
        return {
            "bias": "neutral",
            "bull_score": 0.0,
            "bear_score": 0.0,
            "net_score": 0.0,
            "signal_count": 0,
            "agreement_pct": 0.0,
        }

    total_weight = sum(s.weight for s in signals)
    if total_weight <= 0:
        total_weight = 1.0

    bull_score = sum(
        s.confidence * s.weight for s in signals if s.bias == "bullish"
    ) / total_weight
    bear_score = sum(
        s.confidence * s.weight for s in signals if s.bias == "bearish"
    ) / total_weight
    net_score = bull_score - bear_score

    if net_score > 10:
        bias = "bullish"
    elif net_score < -10:
        bias = "bearish"
    else:
        bias = "neutral"

    # Agreement: how many signals agree with the majority direction
    bull_count = sum(1 for s in signals if s.bias == "bullish")
    bear_count = sum(1 for s in signals if s.bias == "bearish")
    majority_count = max(bull_count, bear_count)
    agreement_pct = round(majority_count / len(signals) * 100, 1) if signals else 0.0

    return {
        "bias": bias,
        "bull_score": round(bull_score, 2),
        "bear_score": round(bear_score, 2),
        "net_score": round(net_score, 2),
        "signal_count": len(signals),
        "agreement_pct": agreement_pct,
    }

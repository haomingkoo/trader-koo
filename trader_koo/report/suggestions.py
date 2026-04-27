"""Small, opinionated suggestion layer for the report.

This deliberately compresses the wider setup table into a few research-ready
ideas. The goal is product clarity: show what looks worth paper-tracking, why,
and what would invalidate it.
"""
from __future__ import annotations

from typing import Any

from trader_koo.report.utils import _clamp, _to_float


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _direction_label(row: dict[str, Any]) -> str:
    bias = _as_text(row.get("signal_bias")).lower()
    if bias == "bullish":
        return "Paper Long"
    if bias == "bearish":
        return "Paper Short"
    return "Watch"


def _persona(row: dict[str, Any]) -> str:
    family = _as_text(row.get("setup_family")).lower()
    if "continuation" in family:
        return "Trend continuation"
    if "reversal" in family:
        return "Mean-reversion reversal"
    if row.get("discount_pct") is not None or row.get("peg") is not None:
        return "Value plus technical trigger"
    if row.get("options_positioning_signal"):
        return "Options-positioning watch"
    return "Evidence-stack setup"


def _quality_score(row: dict[str, Any]) -> float:
    score = _to_float(row.get("score")) or 0.0
    prob = _to_float(row.get("calibrated_hit_prob"))
    agreement = _to_float(row.get("debate_agreement_score")) or 0.0
    sample = _to_float(row.get("probability_sample_size")) or 0.0
    quality = score
    if prob is not None:
        quality += (float(prob) - 0.5) * 40.0
    quality += _clamp((agreement - 50.0) / 8.0, -4.0, 4.0)
    quality += _clamp(sample / 10.0, 0.0, 3.0)
    if row.get("earnings_within_5d"):
        quality -= 4.0
    if _as_text(row.get("options_positioning_signal")) == "elevated_iv_event_risk":
        quality -= 4.0
    if _as_text(row.get("probability_label")) == "low":
        quality -= 6.0
    return round(_clamp(quality, 0.0, 100.0), 1)


def _conviction(row: dict[str, Any], quality: float) -> str:
    tier = _as_text(row.get("setup_tier")).upper()
    prob = _to_float(row.get("calibrated_hit_prob"))
    sample = int(_to_float(row.get("probability_sample_size")) or 0)
    if quality >= 72 and tier in {"A", "B"} and (prob is None or prob >= 0.56) and sample >= 5:
        return "Higher"
    if quality >= 62 and tier in {"A", "B", "C"}:
        return "Medium"
    return "Low"


def _why(row: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    prob = _to_float(row.get("calibrated_hit_prob"))
    sample = int(_to_float(row.get("probability_sample_size")) or 0)
    if prob is not None:
        if sample > 0:
            bullets.append(f"Calibrated setup probability {prob * 100:.0f}% from {sample} comparable calls.")
        else:
            bullets.append(f"Evidence score maps to {prob * 100:.0f}% prior probability; history is still thin.")
    if row.get("debate_agreement_score") is not None:
        bullets.append(f"Agent agreement {float(row.get('debate_agreement_score') or 0):.0f}% with {row.get('debate_consensus_bias') or 'neutral'} consensus.")
    option_signal = _as_text(row.get("options_positioning_signal"))
    if option_signal == "underpriced_positioning":
        bullets.append("Options context says OI is elevated while IV is subdued versus local snapshots.")
    elif option_signal == "subdued_iv":
        bullets.append("Options context says IV is subdued versus local snapshots.")
    elif option_signal == "elevated_iv_event_risk":
        bullets.append("Options context warns IV may already price the move.")
    news_score = _to_float(row.get("news_sentiment_score"))
    if news_score is not None:
        direction = _as_text(row.get("signal_bias")).lower()
        if (direction == "bullish" and news_score >= 55) or (direction == "bearish" and news_score <= 45):
            bullets.append(f"News pulse supports the setup direction ({news_score:.0f}/100).")
        else:
            bullets.append(f"News pulse is mixed or conflicts ({news_score:.0f}/100).")
    if row.get("yolo_pattern"):
        recency = _as_text(row.get("yolo_recency")) or "unknown"
        bullets.append(f"{recency.title()} chart pattern: {row.get('yolo_pattern')}.")
    action = _as_text(row.get("action"))
    if action and len(bullets) < 3:
        bullets.append(action)
    return bullets[:3]


def _risk(row: dict[str, Any]) -> str:
    notes = _as_text(row.get("risk_note"))
    if notes:
        return notes.split(". ")[0].strip()
    if row.get("earnings_within_5d"):
        return "Upcoming earnings can dominate the technical setup."
    if _as_text(row.get("options_positioning_signal")) == "crowded_open_interest":
        return "Open interest is crowded versus local option snapshots."
    return "Position size should stay small until the setup has more closed-out history."


def _invalidation(row: dict[str, Any]) -> str:
    bias = _as_text(row.get("signal_bias")).lower()
    support = _to_float(row.get("support_level"))
    resistance = _to_float(row.get("resistance_level"))
    if bias == "bullish" and support is not None:
        return f"Invalid below support near {support:.2f}."
    if bias == "bearish" and resistance is not None:
        return f"Invalid above resistance near {resistance:.2f}."
    tech = _as_text(row.get("technical_read"))
    if tech:
        return tech.split(". ")[0].strip()
    return "Invalid if price rejects the stated setup level or market regime turns risk-off."


def _data_gaps(row: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if not row.get("options_positioning_signal"):
        gaps.append("options history unavailable")
    if row.get("news_sentiment_score") is None:
        gaps.append("fresh news not point-in-time or unavailable")
    if int(_to_float(row.get("probability_sample_size")) or 0) < 5:
        gaps.append("thin closed-out sample")
    return gaps[:3]


def build_suggestions(setup_rows: list[dict[str, Any]], *, limit: int = 3) -> dict[str, Any]:
    """Return the top compact research suggestions."""
    candidates: list[dict[str, Any]] = []
    for row in setup_rows or []:
        if not isinstance(row, dict) or not row.get("ticker"):
            continue
        bias = _as_text(row.get("signal_bias")).lower()
        if bias not in {"bullish", "bearish"}:
            continue
        score = _to_float(row.get("score")) or 0.0
        prob = _to_float(row.get("calibrated_hit_prob"))
        if score < 58.0 and (prob is None or prob < 0.52):
            continue
        quality = _quality_score(row)
        action = _direction_label(row)
        conviction = _conviction(row, quality)
        suggestion = {
            "ticker": _as_text(row.get("ticker")).upper(),
            "action": action if conviction != "Low" else "Watch",
            "direction": "long" if bias == "bullish" else "short",
            "conviction": conviction,
            "quality_score": quality,
            "probability_pct": round(float(prob) * 100.0, 1) if prob is not None else None,
            "sample_size": int(_to_float(row.get("probability_sample_size")) or 0),
            "persona": _persona(row),
            "title": f"{_as_text(row.get('ticker')).upper()} {action.lower()} setup",
            "why": _why(row),
            "risk": _risk(row),
            "invalidation": _invalidation(row),
            "data_gaps": _data_gaps(row),
            "source_tier": row.get("setup_tier"),
            "setup_family": row.get("setup_family"),
        }
        candidates.append(suggestion)

    candidates.sort(
        key=lambda item: (
            float(item.get("quality_score") or 0.0),
            float(item.get("probability_pct") or 0.0),
            -len(item.get("data_gaps") or []),
        ),
        reverse=True,
    )
    return {
        "version": "v1",
        "count": min(len(candidates), int(limit)),
        "items": candidates[: max(1, int(limit))],
        "note": "Research suggestions only, not financial advice. Top ideas are compressed from setup quality, calibration, news, options, and debate evidence.",
    }

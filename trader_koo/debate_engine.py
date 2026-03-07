from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _fundamental_bias(discount: Any, peg: Any) -> tuple[str, float, list[str], list[str]]:
    long_score = 0.0
    short_score = 0.0
    evidence: list[str] = []
    risks: list[str] = []

    discount_v = _to_float(discount)
    if discount_v is not None:
        if discount_v >= 25:
            long_score += 0.7
            evidence.append("deep discount")
        elif discount_v >= 10:
            long_score += 0.4
            evidence.append("discounted")
        elif discount_v <= 0:
            short_score += 0.5
            risks.append("no valuation cushion")

    peg_v = _to_float(peg)
    if peg_v is not None and peg_v > 0:
        if peg_v <= 0.8:
            long_score += 0.6
            evidence.append("low PEG")
        elif peg_v <= 1.5:
            long_score += 0.35
            evidence.append("reasonable PEG")
        elif peg_v >= 5.0:
            short_score += 0.7
            risks.append("high PEG")
        elif peg_v >= 3.0:
            short_score += 0.45
            risks.append("rich PEG")

    if long_score > short_score + 0.2:
        return "bullish", _clamp(0.45 + long_score * 0.25, 0.25, 0.95), evidence, risks
    if short_score > long_score + 0.2:
        return "bearish", _clamp(0.45 + short_score * 0.25, 0.25, 0.95), evidence, risks
    return "neutral", 0.4, evidence, risks


def _trend_role(row: dict[str, Any]) -> dict[str, Any]:
    trend = str(row.get("trend_state") or "mixed").lower()
    breakout = str(row.get("breakout_state") or "none").lower()
    ma_major = str(row.get("ma_major_signal") or "").lower()
    ma_signal = str(row.get("ma_signal") or "").lower()
    level_event = str(row.get("level_event") or "none").lower()
    structure = str(row.get("structure_state") or "normal").lower()

    score = 0.0
    evidence: list[str] = []
    risks: list[str] = []

    if trend == "uptrend":
        score += 0.45
        evidence.append("uptrend")
    elif trend == "downtrend":
        score -= 0.45
        evidence.append("downtrend")

    if breakout in {"breakout_up", "failed_breakdown_down"}:
        score += 0.35
        evidence.append(breakout.replace("_", " "))
    elif breakout in {"breakout_down", "failed_breakout_up"}:
        score -= 0.35
        evidence.append(breakout.replace("_", " "))

    if level_event in {"resistance_breakout", "support_reclaim"}:
        score += 0.25
        evidence.append(level_event.replace("_", " "))
    elif level_event in {"support_breakdown", "resistance_reject"}:
        score -= 0.25
        evidence.append(level_event.replace("_", " "))

    if ma_major == "death_cross":
        score -= 0.2
        risks.append("death cross regime")
    elif ma_major == "golden_cross":
        score += 0.2
        evidence.append("golden cross regime")

    if ma_signal == "bearish_20_50_cross":
        score -= 0.12
        risks.append("recent bearish 20/50 cross")
    elif ma_signal == "bullish_20_50_cross":
        score += 0.12
        evidence.append("recent bullish 20/50 cross")

    if structure == "tight_consolidation_high":
        evidence.append("tight consolidation below resistance")
    elif structure == "tight_consolidation_low":
        evidence.append("tight consolidation above support")

    if score > 0.15:
        stance = "bullish"
    elif score < -0.15:
        stance = "bearish"
    else:
        stance = "neutral"

    confidence = _clamp(0.45 + abs(score) * 0.5, 0.35, 0.92)
    return {
        "role": "trend_structure",
        "stance": stance,
        "confidence": round(confidence, 2),
        "evidence": evidence[:4],
        "risk_flags": risks[:4],
        "weight": 0.28,
    }


def _momentum_role(row: dict[str, Any]) -> dict[str, Any]:
    pct_change = _to_float(row.get("pct_change")) or 0.0
    volume_ratio = _to_float(row.get("volume_ratio_20")) or 1.0
    candle_bias = str(row.get("candle_bias") or "neutral").lower()
    stretch = str(row.get("stretch_state") or "normal").lower()
    pct_vs_ma20 = _to_float(row.get("pct_vs_ma20"))

    score = 0.0
    evidence: list[str] = []
    risks: list[str] = []

    if pct_change >= 1.0:
        score += 0.22
        evidence.append(f"up day {pct_change:.2f}%")
    elif pct_change <= -1.0:
        score -= 0.22
        evidence.append(f"down day {pct_change:.2f}%")

    if volume_ratio >= 1.25:
        if pct_change > 0:
            score += 0.2
            evidence.append(f"buy participation {volume_ratio:.2f}x")
        elif pct_change < 0:
            score -= 0.2
            evidence.append(f"sell participation {volume_ratio:.2f}x")
    elif volume_ratio <= 0.75:
        risks.append("low participation")

    if candle_bias == "bullish":
        score += 0.15
        evidence.append("bullish candle")
    elif candle_bias == "bearish":
        score -= 0.15
        evidence.append("bearish candle")

    if stretch == "extended_up":
        score -= 0.12
        risks.append("extended above trend")
    elif stretch == "extended_down":
        score += 0.12
        evidence.append("washed-out move")

    if pct_vs_ma20 is not None:
        if pct_vs_ma20 >= 5.0:
            risks.append("far above MA20")
        elif pct_vs_ma20 <= -5.0:
            risks.append("far below MA20")

    if score > 0.12:
        stance = "bullish"
    elif score < -0.12:
        stance = "bearish"
    else:
        stance = "neutral"

    confidence = _clamp(0.42 + abs(score) * 0.65, 0.34, 0.9)
    return {
        "role": "momentum_participation",
        "stance": stance,
        "confidence": round(confidence, 2),
        "evidence": evidence[:4],
        "risk_flags": risks[:4],
        "weight": 0.2,
    }


def _valuation_role(row: dict[str, Any]) -> dict[str, Any]:
    stance, confidence, evidence, risks = _fundamental_bias(row.get("discount_pct"), row.get("peg"))
    return {
        "role": "valuation",
        "stance": stance,
        "confidence": round(confidence, 2),
        "evidence": evidence[:4],
        "risk_flags": risks[:4],
        "weight": 0.16,
    }


def _yolo_role(row: dict[str, Any]) -> dict[str, Any]:
    pattern = str(row.get("yolo_pattern") or "").strip()
    yolo_bias = str(row.get("yolo_bias") or "neutral").lower()
    recency = str(row.get("yolo_recency") or "unknown").lower()
    age_days = _to_float(row.get("yolo_age_days"))
    conflict = bool(row.get("yolo_direction_conflict"))
    conflict_strength = str(row.get("yolo_conflict_strength") or "none").lower()

    evidence: list[str] = []
    risks: list[str] = []
    if pattern:
        evidence.append(f"{recency or 'unknown'} YOLO: {pattern}")
    if age_days is not None:
        evidence.append(f"YOLO age {int(age_days)}d")
    if conflict:
        risks.append(f"YOLO conflict ({conflict_strength})")

    score = 0.0
    if yolo_bias == "bullish":
        score += 0.35
    elif yolo_bias == "bearish":
        score -= 0.35

    recency_weight = {
        "fresh": 1.0,
        "recent": 0.8,
        "aging": 0.45,
        "stale": 0.2,
    }.get(recency, 0.35)
    score *= recency_weight

    if conflict:
        score *= 0.4 if conflict_strength in {"fresh", "recent"} else 0.7

    if abs(score) < 0.08:
        stance = "neutral"
    else:
        stance = "bullish" if score > 0 else "bearish"

    confidence = _clamp(0.38 + abs(score) * 1.1, 0.28, 0.88)
    return {
        "role": "pattern_yolo",
        "stance": stance,
        "confidence": round(confidence, 2),
        "evidence": evidence[:4],
        "risk_flags": risks[:4],
        "weight": 0.2,
    }


def _risk_role(row: dict[str, Any]) -> dict[str, Any]:
    risks: list[str] = []
    evidence: list[str] = []
    score = 0.0

    actionability = str(row.get("actionability") or "").lower()
    if actionability in {"wait", "watch-only", "watch"}:
        score -= 0.25
        risks.append(f"actionability={actionability}")

    if bool(row.get("yolo_direction_conflict")):
        score -= 0.3
        risks.append("direction conflict")

    yolo_recency = str(row.get("yolo_recency") or "").lower()
    if yolo_recency == "stale":
        score -= 0.18
        risks.append("stale YOLO context")

    stretch = str(row.get("stretch_state") or "normal").lower()
    if stretch in {"extended_up", "extended_down"}:
        score -= 0.12
        risks.append(stretch.replace("_", " "))

    level_context = str(row.get("level_context") or "mid_range").lower()
    if level_context in {"at_support", "at_resistance"}:
        evidence.append(f"decision zone: {level_context}")

    if score <= -0.25:
        stance = "bearish"
    elif score >= 0.18:
        stance = "bullish"
    else:
        stance = "neutral"

    confidence = _clamp(0.45 + abs(score) * 0.7, 0.35, 0.9)
    return {
        "role": "risk_manager",
        "stance": stance,
        "confidence": round(confidence, 2),
        "evidence": evidence[:4],
        "risk_flags": risks[:5],
        "weight": 0.16,
    }


def _aggregate_roles(roles: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    stance_value = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    weighted_total = 0.0
    weighted_abs = 0.0

    for role in roles:
        stance = str(role.get("stance") or "neutral")
        conf = _to_float(role.get("confidence")) or 0.0
        weight = _to_float(role.get("weight")) or 0.0
        weighted_total += stance_value.get(stance, 0.0) * conf * weight
        weighted_abs += abs(conf * weight)

    normalized = weighted_total / weighted_abs if weighted_abs > 0 else 0.0
    if normalized >= 0.2:
        consensus_bias = "bullish"
    elif normalized <= -0.2:
        consensus_bias = "bearish"
    else:
        consensus_bias = "neutral"

    opposing: list[dict[str, Any]] = []
    supporters = 0.0
    total_scored = 0.0
    for role in roles:
        stance = str(role.get("stance") or "neutral")
        conf = _to_float(role.get("confidence")) or 0.0
        if stance == "neutral":
            continue
        total_scored += conf
        if consensus_bias != "neutral" and stance == consensus_bias:
            supporters += conf
        elif consensus_bias != "neutral":
            opposing.append(
                {
                    "role": role.get("role"),
                    "stance": stance,
                    "confidence": round(conf, 2),
                    "note": (role.get("risk_flags") or role.get("evidence") or [""])[0],
                }
            )

    agreement_score = 50.0
    if consensus_bias == "neutral":
        agreement_score = 45.0
    elif total_scored > 0:
        agreement_score = _clamp((supporters / total_scored) * 100.0, 0.0, 100.0)

    state = "watch"
    if abs(normalized) >= 0.55 and agreement_score >= 75.0:
        state = "ready"
    elif abs(normalized) >= 0.3 and agreement_score >= 58.0:
        state = "conditional"

    safety_adjustment: list[str] = []
    if bool(row.get("yolo_direction_conflict")):
        if state == "ready":
            state = "conditional"
        elif state == "conditional":
            state = "watch"
        safety_adjustment.append("yolo_direction_conflict")

    yolo_recency = str(row.get("yolo_recency") or "").lower()
    if yolo_recency == "stale" and state == "ready":
        state = "conditional"
        safety_adjustment.append("stale_yolo")

    risk_role = next((r for r in roles if r.get("role") == "risk_manager"), None)
    risk_flags = (risk_role or {}).get("risk_flags") or []
    if len(risk_flags) >= 2 and state == "ready":
        state = "conditional"
        safety_adjustment.append("risk_manager_flags")

    return {
        "consensus_bias": consensus_bias,
        "consensus_state": state,
        "agreement_score": round(agreement_score, 1),
        "consensus_strength": round(abs(normalized), 3),
        "disagreement_count": len(opposing),
        "primary_disagreements": opposing[:3],
        "safety_adjustment": safety_adjustment,
    }


def build_setup_debate(row: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic multi-angle debate payload for a setup row."""
    trend_role = _trend_role(row)
    momentum_role = _momentum_role(row)
    valuation_role = _valuation_role(row)
    yolo_role = _yolo_role(row)
    risk_role = _risk_role(row)
    roles = [trend_role, momentum_role, valuation_role, yolo_role, risk_role]

    return {
        "version": "v1",
        "mode": "rule",
        "roles": roles,
        "consensus": _aggregate_roles(roles, row),
    }

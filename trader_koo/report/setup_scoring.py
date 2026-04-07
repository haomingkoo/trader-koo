"""Setup scoring, confluence analysis, debate guardrails, and evaluation tracking."""
from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
from typing import Any

from trader_koo.debate_engine import build_setup_debate
from trader_koo.llm_narrative import llm_enabled, llm_max_setups, maybe_rewrite_setup_copy
from trader_koo.report.utils import (
    _clamp,
    _fmt_pct_short,
    _median,
    _round_or_none,
    _setup_tier,
    _to_float,
    table_exists,
)

LOG = logging.getLogger(__name__)

TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY_VALUES


SETUP_EVAL_ENABLED = _as_bool(os.getenv("TRADER_KOO_SETUP_EVAL_ENABLED", "1"))
SETUP_EVAL_TRACK_LIMIT = max(5, int(os.getenv("TRADER_KOO_SETUP_EVAL_TRACK_LIMIT", "40")))
SETUP_EVAL_WINDOW_DAYS = max(30, int(os.getenv("TRADER_KOO_SETUP_EVAL_WINDOW_DAYS", "180")))
SETUP_EVAL_MIN_SAMPLE = max(3, int(os.getenv("TRADER_KOO_SETUP_EVAL_MIN_SAMPLE", "5")))
SETUP_EVAL_HIT_THRESHOLD_PCT = float(os.getenv("TRADER_KOO_SETUP_EVAL_HIT_THRESHOLD_PCT", "0.3"))
DEBATE_ENGINE_ENABLED = _as_bool(os.getenv("TRADER_KOO_DEBATE_ENABLED", "1"))


def _yolo_pattern_bias(pattern: Any) -> str:
    name = str(pattern or "").strip().lower()
    if not name:
        return "neutral"
    if ("bottom" in name) or ("w_bottom" in name):
        return "bullish"
    if ("top" in name) or ("m_head" in name):
        return "bearish"
    if "triangle" in name:
        return "neutral"
    return "neutral"


def _yolo_age_factor(age_days: Any, timeframe: Any) -> float:
    tf = str(timeframe or "").strip().lower()
    if not isinstance(age_days, (int, float)):
        return 0.0
    age = int(max(0, float(age_days)))
    if tf == "weekly":
        if age <= 14:
            return 1.0
        if age <= 35:
            return 0.8
        if age <= 70:
            return 0.45
        if age <= 120:
            return 0.18
        return 0.0
    if age <= 5:
        return 1.0
    if age <= 12:
        return 0.8
    if age <= 25:
        return 0.5
    if age <= 45:
        return 0.2
    return 0.0


def _yolo_recency_label(age_days: Any, timeframe: Any) -> str:
    tf = str(timeframe or "").strip().lower()
    if not isinstance(age_days, (int, float)):
        return "unknown"
    age = int(max(0, float(age_days)))
    if tf == "weekly":
        if age <= 14:
            return "fresh"
        if age <= 35:
            return "recent"
        if age <= 70:
            return "aging"
        return "stale"
    if age <= 5:
        return "fresh"
    if age <= 12:
        return "recent"
    if age <= 25:
        return "aging"
    return "stale"


def _fundamental_context(discount: Any, peg: Any) -> dict[str, Any]:
    long_points = 0
    short_points = 0
    long_notes: list[str] = []
    short_notes: list[str] = []
    if isinstance(discount, (int, float)):
        d = float(discount)
        if d >= 25.0:
            long_points += 3
            long_notes.append("deep discount")
        elif d >= 12.0:
            long_points += 2
            long_notes.append("discounted")
        elif d >= 5.0:
            long_points += 1
            long_notes.append("small discount")
        elif d <= 0.0:
            short_points += 2
            short_notes.append("no discount")
        elif d <= 3.0:
            short_points += 1
            short_notes.append("thin valuation cushion")
    if isinstance(peg, (int, float)) and float(peg) > 0:
        p = float(peg)
        if p <= 0.8:
            long_points += 3
            long_notes.append("low PEG")
        elif p <= 1.5:
            long_points += 2
            long_notes.append("reasonable PEG")
        elif p <= 2.5:
            long_points += 1
        elif p >= 5.0:
            short_points += 3
            short_notes.append("high PEG")
        elif p >= 3.0:
            short_points += 2
            short_notes.append("rich PEG")
    bias = "neutral"
    if long_points >= short_points + 2:
        bias = "bullish"
    elif short_points >= long_points + 2:
        bias = "bearish"
    return {
        "bias": bias,
        "long_points": long_points,
        "short_points": short_points,
        "long_notes": long_notes,
        "short_notes": short_notes,
    }


def _score_setup_from_confluence(row: dict[str, Any]) -> dict[str, Any]:
    yolo_bias = _yolo_pattern_bias(row.get("yolo_pattern"))
    yolo_age_days = row.get("yolo_age_days")
    yolo_timeframe = str(row.get("yolo_timeframe") or "daily")
    candle_bias = str(row.get("candle_bias") or "neutral")
    trend = str(row.get("trend_state") or "mixed")
    level = str(row.get("level_context") or "mid_range")
    stretch = str(row.get("stretch_state") or "normal")
    breakout_state = str(row.get("breakout_state") or "none")
    level_event = str(row.get("level_event") or "none")
    structure_state = str(row.get("structure_state") or "normal")
    ma_signal = str(row.get("ma_signal") or "")
    ma_major_signal = str(row.get("ma_major_signal") or "")
    ma_reclaim_state = str(row.get("ma_reclaim_state") or "")
    recent_gap_state = str(row.get("recent_gap_state") or "")
    recent_gap_days = row.get("recent_gap_days")
    pct_change = float(row.get("pct_change") or 0.0)
    yolo_conf = float(row.get("yolo_confidence") or 0.0)
    candle_conf = float(row.get("candle_confidence") or 0.0)
    fund = _fundamental_context(row.get("discount_pct"), row.get("peg"))
    valuation_bias = str(fund.get("bias") or "neutral")
    near_support = bool(
        row.get("near_52w_low")
        or level in {"at_support", "closer_support"}
        or (isinstance(row.get("pct_to_support"), (int, float)) and float(row.get("pct_to_support")) <= 1.5)
        or (isinstance(row.get("pct_from_20d_low"), (int, float)) and float(row.get("pct_from_20d_low")) <= 2.5)
    )
    near_resistance = bool(
        row.get("near_52w_high")
        or level in {"at_resistance", "closer_resistance"}
        or (isinstance(row.get("pct_to_resistance"), (int, float)) and float(row.get("pct_to_resistance")) <= 1.5)
        or (isinstance(row.get("pct_from_20d_high"), (int, float)) and float(row.get("pct_from_20d_high")) <= 2.5)
    )

    bull_score = 0.0
    bear_score = 0.0
    confirmations_bull = 0
    confirmations_bear = 0
    contradictions_bull = 0
    contradictions_bear = 0

    yolo_age_factor = _yolo_age_factor(yolo_age_days, yolo_timeframe)
    yolo_recency = _yolo_recency_label(yolo_age_days, yolo_timeframe)
    yolo_direction_conflict = False
    yolo_conflict_strength = "none"
    fresh_bear_gap = recent_gap_state == "bear_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2
    fresh_bull_gap = recent_gap_state == "bull_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2
    below_short_mas = bool(
        isinstance(row.get("pct_vs_ma20"), (int, float))
        and isinstance(row.get("pct_vs_ma50"), (int, float))
        and float(row.get("pct_vs_ma20")) < 0.0
        and float(row.get("pct_vs_ma50")) < 0.0
    )
    above_short_mas = bool(
        isinstance(row.get("pct_vs_ma20"), (int, float))
        and isinstance(row.get("pct_vs_ma50"), (int, float))
        and float(row.get("pct_vs_ma20")) > 0.0
        and float(row.get("pct_vs_ma50")) > 0.0
    )

    if valuation_bias == "bullish":
        bull_score += 10.0 + (float(fund.get("long_points") or 0) * 2.0)
        confirmations_bull += 1
        contradictions_bear += 1
    elif valuation_bias == "bearish":
        bear_score += 10.0 + (float(fund.get("short_points") or 0) * 2.0)
        confirmations_bear += 1
        contradictions_bull += 1

    if yolo_bias == "bullish" and yolo_age_factor > 0.0:
        boost = (8.0 + min(8.0, yolo_conf * 10.0)) * yolo_age_factor
        bull_score += boost
        if yolo_age_factor >= 0.5:
            confirmations_bull += 1
        contradictions_bear += 1
    elif yolo_bias == "bearish" and yolo_age_factor > 0.0:
        boost = (8.0 + min(8.0, yolo_conf * 10.0)) * yolo_age_factor
        bear_score += boost
        if yolo_age_factor >= 0.5:
            confirmations_bear += 1
        contradictions_bull += 1

    if candle_bias == "bullish":
        bull_score += 3.0 + min(4.0, candle_conf * 3.0)
        confirmations_bull += 1
        contradictions_bear += 1
    elif candle_bias == "bearish":
        bear_score += 3.0 + min(4.0, candle_conf * 3.0)
        confirmations_bear += 1
        contradictions_bull += 1

    if near_support:
        bull_score += 10.0
        confirmations_bull += 1
        contradictions_bear += 1
    if near_resistance:
        bear_score += 10.0
        confirmations_bear += 1
        contradictions_bull += 1

    if trend == "uptrend":
        bull_score += 4.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif trend == "downtrend":
        bear_score += 4.0
        confirmations_bear += 1
        contradictions_bull += 1

    if level_event == "resistance_breakout":
        bull_score += 8.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif level_event == "support_reclaim":
        bull_score += 6.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif level_event == "support_breakdown":
        bear_score += 8.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif level_event == "resistance_reject":
        bear_score += 6.0
        confirmations_bear += 1
        contradictions_bull += 1

    if ma_reclaim_state == "reclaimed_ma20":
        bull_score += 4.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif ma_reclaim_state == "reclaimed_ma50":
        bull_score += 6.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif ma_reclaim_state == "lost_ma20":
        bear_score += 4.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif ma_reclaim_state == "lost_ma50":
        bear_score += 6.0
        confirmations_bear += 1
        contradictions_bull += 1

    if recent_gap_state == "bull_gap":
        if fresh_bull_gap:
            bull_score += 6.0
            confirmations_bull += 1
            contradictions_bear += 1
        else:
            bull_score += 2.0
    elif recent_gap_state == "bear_gap":
        if fresh_bear_gap:
            bear_score += 6.0
            confirmations_bear += 1
            contradictions_bull += 1
        else:
            bear_score += 2.0

    if fresh_bear_gap:
        bear_score += 4.0
        contradictions_bull += 1
        if below_short_mas:
            bear_score += 6.0
            contradictions_bull += 1
            bull_score -= 3.0
        if isinstance(pct_change, (int, float)) and float(pct_change) <= -3.0:
            bear_score += 4.0
            confirmations_bear += 1
        if ma_signal == "bearish_20_50_cross":
            bear_score += 4.0
            confirmations_bear += 1
        if level_event == "support_breakdown":
            bear_score += 4.0
        if breakout_state == "failed_breakdown_down":
            bull_score += 2.0
    elif fresh_bull_gap:
        bull_score += 4.0
        contradictions_bear += 1
        if above_short_mas:
            bull_score += 6.0
            contradictions_bear += 1
            bear_score -= 3.0
        if isinstance(pct_change, (int, float)) and float(pct_change) >= 3.0:
            bull_score += 4.0
            confirmations_bull += 1
        if ma_signal == "bullish_20_50_cross":
            bull_score += 4.0
            confirmations_bull += 1
        if level_event == "resistance_breakout":
            bull_score += 4.0

    if ma_major_signal == "death_cross":
        bear_score += 5.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif ma_major_signal == "golden_cross":
        bull_score += 5.0
        confirmations_bull += 1
        contradictions_bear += 1

    if breakout_state == "breakout_up":
        bull_score += 8.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif breakout_state == "breakout_down":
        bear_score += 8.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif breakout_state == "failed_breakout_up":
        bear_score += 6.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif breakout_state == "failed_breakdown_down":
        bull_score += 6.0
        confirmations_bull += 1
        contradictions_bear += 1

    if structure_state == "tight_consolidation_high":
        bull_score += 5.0 if trend != "downtrend" else 2.0
        confirmations_bull += 1 if trend != "downtrend" else 0
    elif structure_state == "tight_consolidation_low":
        bear_score += 5.0 if trend != "uptrend" else 2.0
        confirmations_bear += 1 if trend != "uptrend" else 0
    elif structure_state == "tight_consolidation_mid":
        bull_score += 1.5
        bear_score += 1.5
    elif structure_state == "parabolic_up":
        bull_score += 3.5
        confirmations_bull += 1
        contradictions_bull += 1
        contradictions_bear += 1
    elif structure_state == "parabolic_down":
        bear_score += 3.5
        confirmations_bear += 1
        contradictions_bear += 1
        contradictions_bull += 1

    if 0.5 <= pct_change <= 4.0:
        bull_score += 3.0
    elif pct_change > 5.0:
        bull_score -= 4.0
        contradictions_bull += 1
    if -4.0 <= pct_change <= -0.5:
        bear_score += 3.0
    elif pct_change < -5.0:
        bear_score -= 4.0
        contradictions_bear += 1

    if stretch == "extended_up":
        bull_score -= 5.0
        contradictions_bull += 1
        bear_score += 2.0
    elif stretch == "extended_down":
        bear_score -= 5.0
        contradictions_bear += 1
        bull_score += 2.0

    pct_vs_ma20 = row.get("pct_vs_ma20")
    if isinstance(pct_vs_ma20, (int, float)):
        if float(pct_vs_ma20) <= -3.0:
            bear_score += 3.0
            contradictions_bull += 1
        elif float(pct_vs_ma20) >= 3.0:
            bull_score += 3.0
            contradictions_bear += 1
    pct_vs_ma50 = row.get("pct_vs_ma50")
    if isinstance(pct_vs_ma50, (int, float)):
        if float(pct_vs_ma50) <= -3.0:
            bear_score += 2.0
        elif float(pct_vs_ma50) >= 3.0:
            bull_score += 2.0

    rv20 = row.get("realized_vol_20")
    atr = row.get("atr_pct_14")
    if isinstance(rv20, (int, float)):
        if 18.0 <= float(rv20) <= 45.0:
            bull_score += 1.5
            bear_score += 1.5
        elif float(rv20) >= 65.0:
            bull_score -= 2.0
            bear_score -= 2.0
    if isinstance(atr, (int, float)):
        if float(atr) >= 9.0:
            bull_score -= 2.0
            bear_score -= 2.0

    if yolo_age_factor > 0.0 and yolo_age_factor < 0.3:
        if yolo_bias == "bullish":
            bull_score -= 2.0
        elif yolo_bias == "bearish":
            bear_score -= 2.0

    if bull_score >= bear_score + 5.0:
        bias = "bullish"
    elif bear_score >= bull_score + 5.0:
        bias = "bearish"
    else:
        bias = "neutral"
    score_margin = abs(bull_score - bear_score)
    if bias in {"bullish", "bearish"} and yolo_bias in {"bullish", "bearish"} and bias != yolo_bias:
        yolo_direction_conflict = True
        if yolo_recency == "fresh":
            yolo_conflict_strength = "fresh"
            # Fresh opposite YOLO should neutralize directional claims unless the edge is very large.
            if score_margin < 12.0:
                bias = "neutral"
        elif yolo_recency == "recent":
            yolo_conflict_strength = "recent"
            if score_margin < 8.0:
                bias = "neutral"
        elif yolo_recency == "aging":
            yolo_conflict_strength = "aging"
        elif yolo_recency == "stale":
            yolo_conflict_strength = "stale"

    family = "neutral_watch"
    confirmations = 0
    contradictions = 0
    score = 42.0
    if bias == "bullish":
        confirmations = confirmations_bull
        contradictions = contradictions_bull
        if breakout_state in {"breakout_up", "failed_breakdown_down"} or (
            trend == "uptrend" and structure_state in {"tight_consolidation_high", "parabolic_up"}
        ):
            family = "bullish_continuation"
            score = 36.0 + bull_score
        elif near_support or level == "below_support" or row.get("near_52w_low"):
            family = "bullish_reversal"
            score = 35.0 + bull_score
        elif trend == "uptrend" and not near_resistance:
            family = "bullish_continuation"
            score = 33.0 + bull_score
        else:
            family = "bullish_watch"
            score = 28.0 + bull_score
    elif bias == "bearish":
        confirmations = confirmations_bear
        contradictions = contradictions_bear
        if breakout_state in {"breakout_down", "failed_breakout_up"} or (
            trend == "downtrend" and structure_state in {"tight_consolidation_low", "parabolic_down"}
        ):
            family = "bearish_continuation"
            score = 36.0 + bear_score
        elif near_resistance or level == "above_resistance" or row.get("near_52w_high"):
            family = "bearish_reversal"
            score = 35.0 + bear_score
        elif trend == "downtrend" and not near_support:
            family = "bearish_continuation"
            score = 33.0 + bear_score
        else:
            family = "bearish_watch"
            score = 28.0 + bear_score
    else:
        confirmations = max(confirmations_bull, confirmations_bear)
        contradictions = max(contradictions_bull, contradictions_bear)
        score = 25.0 + max(bull_score, bear_score) - abs(bull_score - bear_score)

    if yolo_direction_conflict:
        if yolo_conflict_strength == "fresh":
            score -= 10.0
            contradictions += 2
        elif yolo_conflict_strength == "recent":
            score -= 7.0
            contradictions += 1
        elif yolo_conflict_strength == "aging":
            score -= 4.0
            contradictions += 1
        elif yolo_conflict_strength == "stale":
            score -= 2.0
    if yolo_recency == "stale" and yolo_bias in {"bullish", "bearish"}:
        score -= 3.0
    if (
        valuation_bias == "bullish"
        and fresh_bear_gap
        and candle_bias != "bullish"
    ):
        score -= 6.0
        contradictions += 1
    if (
        valuation_bias == "bearish"
        and fresh_bull_gap
        and candle_bias != "bearish"
    ):
        score -= 6.0
        contradictions += 1

    score -= contradictions * 5.0
    if confirmations == 0:
        score -= 12.0
    elif confirmations == 1:
        score -= 6.0
    score = round(_clamp(score, 0.0, 100.0), 1)

    if bias == "neutral":
        tier = "C" if score >= 60.0 else "D"
    elif confirmations >= 4 and contradictions == 0 and score >= 78.0 and "watch" not in family:
        tier = "A"
    elif confirmations >= 3 and score >= 68.0:
        tier = "B"
    elif score >= 55.0:
        tier = "C"
    else:
        tier = "D"

    if yolo_direction_conflict:
        if bias == "bullish" and family in {"bullish_reversal", "bullish_continuation"}:
            family = "bullish_watch"
        elif bias == "bearish" and family in {"bearish_reversal", "bearish_continuation"}:
            family = "bearish_watch"
        if yolo_conflict_strength == "fresh" and tier in {"A", "B"}:
            tier = "C"
        elif yolo_conflict_strength == "recent" and tier == "A":
            tier = "B"

    return {
        "signal_bias": bias,
        "setup_family": family,
        "score": score,
        "confluence_score": score,
        "setup_tier": tier,
        "confirmation_count": confirmations,
        "contradiction_count": contradictions,
        "valuation_bias": valuation_bias,
        "valuation_notes": ", ".join((fund.get("long_notes") if bias != "bearish" else fund.get("short_notes")) or []),
        "bull_score": round(bull_score, 1),
        "bear_score": round(bear_score, 1),
        "yolo_age_factor": round(yolo_age_factor, 2) if yolo_age_factor else 0.0,
        "yolo_recency": yolo_recency,
        "yolo_bias": yolo_bias or "neutral",
        "yolo_direction_conflict": yolo_direction_conflict,
        "yolo_conflict_strength": yolo_conflict_strength,
        "score_margin": round(score_margin, 1),
    }


def _describe_setup(row: dict[str, Any]) -> dict[str, str]:
    bias = str(row.get("signal_bias") or "neutral")
    family = str(row.get("setup_family") or "neutral_watch")
    trend = str(row.get("trend_state") or "mixed")
    ma_signal = str(row.get("ma_signal") or "")
    ma_major_signal = str(row.get("ma_major_signal") or "")
    ma_reclaim_state = str(row.get("ma_reclaim_state") or "")
    level = str(row.get("level_context") or "mid_range")
    level_event = str(row.get("level_event") or "none")
    stretch = str(row.get("stretch_state") or "normal")
    breakout_state = str(row.get("breakout_state") or "none")
    structure_state = str(row.get("structure_state") or "normal")
    recent_gap_state = str(row.get("recent_gap_state") or "")
    recent_gap_days = row.get("recent_gap_days")
    candle_bias = str(row.get("candle_bias") or "neutral")
    valuation_bias = str(row.get("valuation_bias") or "neutral")
    valuation_notes = str(row.get("valuation_notes") or "").strip()
    pct_from_high = row.get("pct_from_20d_high")
    pct_from_low = row.get("pct_from_20d_low")
    pct_vs_ma20 = row.get("pct_vs_ma20")
    pct_change = row.get("pct_change")
    pattern = str(row.get("yolo_pattern") or "").strip()
    yolo_bias = str(row.get("yolo_bias") or _yolo_pattern_bias(pattern) or "neutral")
    yolo_direction_conflict = bool(row.get("yolo_direction_conflict"))
    if not yolo_direction_conflict and bias in {"bullish", "bearish"} and yolo_bias in {"bullish", "bearish"}:
        yolo_direction_conflict = yolo_bias != bias
    yolo_conflict_strength = str(row.get("yolo_conflict_strength") or "none").strip().lower()
    yolo_age_days = row.get("yolo_age_days")
    yolo_timeframe = str(row.get("yolo_timeframe") or "daily")
    yolo_recency = str(row.get("yolo_recency") or _yolo_recency_label(yolo_age_days, yolo_timeframe))
    yolo_first_seen_asof = row.get("yolo_first_seen_asof")
    yolo_last_seen_asof = row.get("yolo_last_seen_asof")
    yolo_snapshots_seen = row.get("yolo_snapshots_seen")
    yolo_current_streak = row.get("yolo_current_streak")
    support_level = row.get("support_level")
    resistance_level = row.get("resistance_level")
    pct_to_support = row.get("pct_to_support")
    pct_to_resistance = row.get("pct_to_resistance")
    volume_ratio_20 = row.get("volume_ratio_20")

    level_label = {
        "below_support": "trading below nearby support",
        "at_support": "sitting inside real support",
        "closer_support": "trading closer to support than resistance",
        "at_resistance": "pressing into real resistance",
        "closer_resistance": "trading closer to resistance than support",
        "above_resistance": "trading above nearby resistance",
        "mid_range": "trading in the middle of its recent range",
    }.get(level, "trading in the middle of its recent range")
    trend_label = {
        "uptrend": "in an uptrend",
        "downtrend": "in a downtrend",
        "mixed": "in mixed trend structure",
    }.get(trend, "in mixed trend structure")
    stretch_label = {
        "extended_up": "already stretched above trend",
        "extended_down": "already stretched below trend",
        "normal": "not overly stretched",
    }.get(stretch, "not overly stretched")

    family_label = {
        "bullish_reversal": "bullish reversal candidate",
        "bullish_continuation": "bullish continuation candidate",
        "bullish_watch": "bullish watchlist candidate",
        "bearish_reversal": "bearish reversal candidate",
        "bearish_continuation": "bearish continuation candidate",
        "bearish_watch": "bearish watchlist candidate",
        "neutral_watch": "mixed / unconfirmed candidate",
    }.get(family, "mixed / unconfirmed candidate")
    family_short_label = {
        "bullish_reversal": "Bullish reversal",
        "bullish_continuation": "Bullish continuation",
        "bullish_watch": "Bullish watch",
        "bearish_reversal": "Bearish reversal",
        "bearish_continuation": "Bearish continuation",
        "bearish_watch": "Bearish watch",
        "neutral_watch": "Neutral watch",
    }.get(family, "Neutral watch")
    bias_short_label = {
        "bullish": "bullish bias",
        "bearish": "bearish bias",
        "neutral": "neutral bias",
    }.get(bias, "neutral bias")
    trend_short_label = {
        "uptrend": "uptrend",
        "downtrend": "downtrend",
        "mixed": "mixed trend",
    }.get(trend, "mixed trend")
    ma_signal_label = {
        "bearish_20_50_cross": "recent bearish 20/50 cross",
        "bullish_20_50_cross": "recent bullish 20/50 cross",
        "20_below_50": "20 below 50",
        "20_above_50": "20 above 50",
    }.get(ma_signal)
    ma_major_signal_label = {
        "death_cross": "classic death cross (50 below 200)",
        "golden_cross": "classic golden cross (50 above 200)",
        "50_below_200": "50 below 200",
        "50_above_200": "50 above 200",
    }.get(ma_major_signal)
    ma_reclaim_label = {
        "reclaimed_ma20": "reclaimed the 20-day average",
        "reclaimed_ma50": "reclaimed the 50-day average",
        "lost_ma20": "lost the 20-day average",
        "lost_ma50": "lost the 50-day average",
    }.get(ma_reclaim_state)

    observation_parts: list[str] = []
    observation_parts.append(family_label)
    if valuation_notes:
        observation_parts.append(valuation_notes)
    if ma_major_signal == "death_cross":
        observation_parts.append("classic 50/200 death cross is in place")
    elif ma_major_signal == "golden_cross":
        observation_parts.append("classic 50/200 golden cross is in place")
    elif ma_signal == "bearish_20_50_cross":
        observation_parts.append("recent bearish 20/50 crossover")
    elif ma_signal == "bullish_20_50_cross":
        observation_parts.append("recent bullish 20/50 crossover")
    if ma_reclaim_label:
        observation_parts.append(ma_reclaim_label)
    if level_event == "resistance_breakout":
        observation_parts.append("resistance got blown through")
    elif level_event == "support_breakdown":
        observation_parts.append("support got blown through")
    elif level_event == "resistance_reject":
        observation_parts.append("price rejected at resistance")
    elif level_event == "support_reclaim":
        observation_parts.append("price reclaimed broken support")
    elif breakout_state == "breakout_up":
        observation_parts.append("price is already through resistance")
    elif breakout_state == "breakout_down":
        observation_parts.append("price is already below support")
    elif breakout_state == "failed_breakout_up":
        observation_parts.append("recent breakout attempt failed back under resistance")
    elif breakout_state == "failed_breakdown_down":
        observation_parts.append("recent breakdown attempt failed back above support")
    if structure_state == "tight_consolidation_high":
        observation_parts.append("tight consolidation is building just under resistance")
    elif structure_state == "tight_consolidation_low":
        observation_parts.append("tight consolidation is building just above support")
    elif structure_state == "tight_consolidation_mid":
        observation_parts.append("price is compressing in a tight range")
    elif structure_state == "parabolic_up":
        observation_parts.append("move is becoming parabolic / extended")
    elif structure_state == "parabolic_down":
        observation_parts.append("selloff is becoming climactic / stretched")
    if recent_gap_state == "bear_gap":
        if isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            observation_parts.append("fresh bearish gap is still influencing price")
        else:
            observation_parts.append("older bearish gap is still overhead")
    elif recent_gap_state == "bull_gap":
        if isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            observation_parts.append("fresh bullish gap is supporting the move")
        else:
            observation_parts.append("older bullish gap remains in play")
    if pattern:
        if yolo_recency == "fresh":
            observation_parts.append(f"fresh YOLO: {pattern}")
        elif yolo_recency == "recent":
            observation_parts.append(f"recent YOLO: {pattern}")
        elif yolo_recency == "aging":
            observation_parts.append(f"older YOLO context: {pattern}")
        else:
            observation_parts.append(f"stale YOLO context: {pattern}")
    else:
        observation_parts.append("no decisive YOLO")
    if yolo_direction_conflict and pattern:
        conflict_prefix = (
            "fresh" if yolo_conflict_strength == "fresh"
            else ("recent" if yolo_conflict_strength == "recent" else "older")
        )
        observation_parts.append(f"{conflict_prefix} YOLO disagrees with this direction ({yolo_bias} {pattern})")
    if isinstance(yolo_snapshots_seen, int) and yolo_snapshots_seen > 1:
        if isinstance(yolo_current_streak, int) and yolo_current_streak > 1:
            observation_parts.append(f"YOLO has persisted {yolo_current_streak} snapshots")
        else:
            observation_parts.append(f"YOLO has appeared in {yolo_snapshots_seen} retained snapshots")
    if candle_bias == "bullish":
        observation_parts.append("bullish candle confirmation")
    elif candle_bias == "bearish":
        observation_parts.append("bearish candle confirmation")
    else:
        observation_parts.append("no candle confirmation")
    observation_parts.append(trend_label)
    observation_parts.append(level_label)
    observation_parts.append(stretch_label)
    if candle_bias == "bullish":
        observation_parts.append("latest candle bias is supportive")
    elif candle_bias == "bearish":
        observation_parts.append("latest candle bias is conflicting")

    actionability = "watch-only"
    action = "No strong edge yet. Keep it on watch, do not force a trade."

    near_support = (
        level in {"at_support", "closer_support"}
        or (isinstance(pct_to_support, (int, float)) and float(pct_to_support) <= 1.5)
        or (isinstance(pct_from_low, (int, float)) and float(pct_from_low) <= 2.5)
    )
    near_resistance = (
        level in {"at_resistance", "closer_resistance"}
        or (isinstance(pct_to_resistance, (int, float)) and float(pct_to_resistance) <= 1.5)
        or (isinstance(pct_from_high, (int, float)) and float(pct_from_high) <= 2.5)
    )
    large_day = isinstance(pct_change, (int, float)) and abs(float(pct_change)) >= 5.0

    if bias == "bullish":
        if family == "bullish_reversal":
            actionability = "conditional"
            action = "Long only if support holds and the reversal actually confirms. Do not buy a falling knife into earnings or resistance."
        elif family == "bullish_continuation":
            actionability = "conditional"
            action = "Long only on clean continuation through resistance or a disciplined retest. Avoid late chase entries."
        if structure_state == "parabolic_up":
            actionability = "wait"
            action = "Uptrend is still intact, but the move is getting parabolic. Wait for a hold above resistance or a cleaner retest instead of chasing."
        elif breakout_state == "breakout_up":
            actionability = "higher-probability" if trend == "uptrend" else "conditional"
            action = "Breakout / continuation watch. Best entry is a clean hold above resistance or a disciplined retest, not a momentum chase."
        elif structure_state == "tight_consolidation_high":
            actionability = "conditional"
            action = "Compression under resistance. Higher-probability entry is a decisive close through the level or a clean retest after the break."
        if recent_gap_state == "bear_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            actionability = "wait"
            action = "Recent bearish gap is still in control. Wait for price to repair the gap or reclaim short-term structure before treating this as a long."
        if stretch == "extended_up" or large_day:
            actionability = "wait"
            action = "Bullish idea, but do not chase strength. Prefer a pullback or breakout retest."
        elif level == "above_resistance" and trend == "uptrend":
            actionability = "conditional"
            action = "Breakout is already through resistance. Best follow-through entry is a clean hold or retest, not an emotional chase."
        elif family == "bullish_continuation" and trend == "uptrend" and near_resistance:
            actionability = "higher-probability"
            action = "Trend continuation watch. Best if price closes through resistance or retests it cleanly."
        elif family == "bullish_reversal" and near_support and candle_bias != "bearish":
            actionability = "higher-probability"
            action = "Reversal watch at support. Actionable only if support holds and the next candles confirm."
        elif level == "below_support":
            actionability = "wait"
            action = "Bullish pattern is fighting broken support. Stand aside until price reclaims the level."
        elif trend == "downtrend":
            actionability = "conditional"
            action = "Counter-trend bounce only. Wait for trend repair before treating it as a core long."
        else:
            actionability = "conditional"
            action = "Bullish setup is present, but wait for confirmation instead of buying the first signal."
    elif bias == "bearish":
        if family == "bearish_reversal":
            actionability = "conditional"
            action = "Short only if resistance rejection confirms or support breaks. Do not force a short after an already extended flush."
        elif family == "bearish_continuation":
            actionability = "conditional"
            action = "Short only on failed bounces, rejected retests, or clean continuation below support."
        if structure_state == "parabolic_down":
            actionability = "wait"
            action = "Downtrend is still intact, but the move is already stretched lower. Prefer a failed bounce or retest rather than chasing the flush."
        elif breakout_state == "breakout_down":
            actionability = "higher-probability" if trend == "downtrend" else "conditional"
            action = "Breakdown / continuation watch. Best entry is a failed reclaim of broken support or fresh downside follow-through."
        elif structure_state == "tight_consolidation_low":
            actionability = "conditional"
            action = "Compression above support. Higher-probability short only comes if support gives way with follow-through."
        if recent_gap_state == "bull_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            actionability = "wait"
            action = "Recent bullish gap is still defending the move. Wait for the gap to fail before leaning bearish."
        if stretch == "extended_down" or large_day:
            actionability = "wait"
            action = "Avoid chasing the flush. Better setup is a failed bounce or a clean support break."
        elif family == "bearish_continuation" and level == "below_support" and trend == "downtrend":
            actionability = "higher-probability"
            action = "Bearish continuation is already below support. Best entry is failed reclaim or fresh breakdown follow-through."
        elif family == "bearish_reversal" and near_support:
            actionability = "higher-probability"
            action = "Support-failure watch. It is only actionable if support actually breaks with follow-through."
        elif family == "bearish_continuation" and trend == "downtrend" and (near_resistance or candle_bias == "bearish"):
            actionability = "higher-probability"
            action = "Bearish continuation watch. Best entry is rejection near resistance or failed rally."
        elif level == "above_resistance":
            actionability = "wait"
            action = "Bearish idea is fighting price above resistance. Wait for failure back under the level before acting."
        elif trend == "uptrend":
            actionability = "conditional"
            action = "Bearish warning against the trend. Avoid new longs until structure improves."
        else:
            actionability = "conditional"
            action = "Bearish risk is present, but wait for breakdown confirmation rather than front-running it."
    else:
        if near_support and candle_bias == "bullish":
            actionability = "conditional"
            action = "Possible support bounce watch, but there is no strong AI edge yet."
        elif near_resistance and candle_bias == "bearish":
            actionability = "conditional"
            action = "Possible rejection watch near resistance, but the signal is still weak."

    stale_yolo = yolo_recency == "stale"
    aging_yolo = yolo_recency in {"aging", "stale"}
    if stale_yolo and pattern:
        if actionability == "higher-probability":
            actionability = "conditional"
        action = "Old YOLO structure only. Use current candle and level confirmation; do not treat the old box as a fresh trigger."
    elif aging_yolo and pattern and actionability == "higher-probability":
        actionability = "conditional"
        action = "Pattern context is older. Keep it on watch and require fresh confirmation from current price action."
    if yolo_direction_conflict and pattern:
        if yolo_conflict_strength in {"fresh", "recent"}:
            actionability = "wait"
        elif actionability == "higher-probability":
            actionability = "conditional"
        action = (
            f"Directional conflict: latest {yolo_bias} YOLO context disagrees with the current {bias} read. "
            "Treat this as watch-only until candles and levels resolve in one direction."
        )

    risk_notes: list[str] = []
    if stretch == "extended_up":
        risk_notes.append("extended above trend")
    elif stretch == "extended_down":
        risk_notes.append("washed out below trend")
    if isinstance(row.get("realized_vol_20"), (int, float)) and float(row["realized_vol_20"]) >= 60.0:
        risk_notes.append("high volatility")
    if isinstance(row.get("peg"), (int, float)) and float(row["peg"]) >= 2.5:
        risk_notes.append("rich PEG")
    if isinstance(row.get("discount_pct"), (int, float)) and float(row["discount_pct"]) <= 5.0:
        risk_notes.append("little valuation cushion")
    if stale_yolo:
        risk_notes.append("stale YOLO context")
    elif aging_yolo and pattern:
        risk_notes.append("older YOLO context")
    if yolo_direction_conflict:
        if yolo_conflict_strength in {"fresh", "recent"}:
            risk_notes.append("fresh opposite YOLO signal")
        else:
            risk_notes.append("YOLO direction conflict")
    if ma_major_signal == "death_cross":
        risk_notes.append("death cross regime")
    elif ma_signal == "bearish_20_50_cross":
        risk_notes.append("bearish 20/50 crossover")

    if valuation_bias != "neutral" and valuation_bias != bias:
        observation_parts.append("valuation does not fully agree with the direction")
    if bias == "neutral":
        actionability = "watch-only"
        action = "Evidence is mixed. Keep it on watch until price, pattern, and level context line up in one direction."

    observation = ". ".join(part[0].upper() + part[1:] if idx == 0 else part for idx, part in enumerate(observation_parts)) + "."
    level_bits = []
    if isinstance(support_level, (int, float)):
        level_bits.append(f"support {support_level}")
    if isinstance(resistance_level, (int, float)):
        level_bits.append(f"resistance {resistance_level}")
    if isinstance(yolo_age_days, (int, float)) and pattern:
        level_bits.append(f"YOLO age {int(float(yolo_age_days))}d")
    if yolo_direction_conflict and yolo_bias in {"bullish", "bearish"} and pattern:
        level_bits.append(f"YOLO conflict ({yolo_bias})")
    if isinstance(yolo_snapshots_seen, int) and yolo_snapshots_seen > 0 and yolo_first_seen_asof:
        if (
            isinstance(yolo_age_days, (int, float))
            and int(float(yolo_age_days)) > 0
            and int(yolo_snapshots_seen) <= 1
        ):
            level_bits.append(f"retained history starts {yolo_first_seen_asof}")
        elif isinstance(yolo_current_streak, int) and yolo_current_streak > 1:
            level_bits.append(f"YOLO {yolo_current_streak}x since {yolo_first_seen_asof}")
        else:
            level_bits.append(f"YOLO first seen {yolo_first_seen_asof}")
    if ma_signal_label:
        level_bits.append(f"MA {ma_signal_label}")
    if ma_major_signal_label:
        level_bits.append(f"MA {ma_major_signal_label}")
    if ma_reclaim_label:
        level_bits.append(ma_reclaim_label)
    if recent_gap_state:
        recent_gap_label = "fresh" if isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2 else "older"
        level_bits.append(f"{recent_gap_label} {recent_gap_state.replace('_', ' ')}")
    if breakout_state != "none":
        level_bits.append(breakout_state.replace("_", " "))
    if structure_state != "normal":
        level_bits.append(structure_state.replace("_", " "))
    if isinstance(volume_ratio_20, (int, float)):
        level_bits.append(f"vol {float(volume_ratio_20):.2f}x")
    level_suffix = f" | {' / '.join(level_bits)}" if level_bits else ""
    if near_support and near_resistance:
        location_short = "between key levels"
    elif near_support:
        location_short = "near support"
    elif near_resistance:
        location_short = "near resistance"
    else:
        location_short = "mid-range"
    return {
        "signal_bias": bias,
        "observation": observation,
        "actionability": actionability,
        "action": action,
        "risk_note": ", ".join(risk_notes[:3]) if risk_notes else "none",
        "yolo_direction_conflict": yolo_direction_conflict,
        "yolo_conflict_strength": yolo_conflict_strength,
        "technical_read": (
            f"{family_short_label} | {bias_short_label} | {trend_short_label} | "
            f"{location_short} | "
            f"vs MA20 {_fmt_pct_short(pct_vs_ma20)}{level_suffix}"
        ),
    }


def _apply_llm_narrative_overrides(
    setup_rows: list[dict[str, Any]],
    *,
    source: str,
) -> None:
    if not setup_rows:
        return
    if not llm_enabled():
        for row in setup_rows:
            if isinstance(row, dict):
                row.setdefault("narrative_source", "rule")
        return
    max_rows = llm_max_setups()
    if max_rows <= 0:
        for row in setup_rows:
            if isinstance(row, dict):
                row.setdefault("narrative_source", "rule")
        return
    for idx, row in enumerate(setup_rows):
        if not isinstance(row, dict):
            continue
        row.setdefault("narrative_source", "rule")
        if idx >= max_rows:
            continue
        overrides = maybe_rewrite_setup_copy(row, source=source)
        if not overrides:
            continue
        row.update(overrides)
        row["narrative_source"] = "llm"


def _apply_debate_payload(setup_rows: list[dict[str, Any]]) -> None:
    if not DEBATE_ENGINE_ENABLED:
        for row in setup_rows:
            if isinstance(row, dict):
                row.setdefault("debate_v1", {"version": "v1", "mode": "disabled"})
        return
    for row in setup_rows:
        if not isinstance(row, dict):
            continue
        try:
            row["debate_v1"] = build_setup_debate(row)
        except Exception as e:
            ticker = row.get("ticker", "UNKNOWN")
            LOG.warning("Debate engine failed for %s: %s: %s", ticker, type(e).__name__, e)
            row.setdefault("debate_v1", {"version": "v1", "mode": "error"})


def _tier_rank(tier: Any) -> int:
    raw = str(tier or "").strip().upper()
    return {"D": 1, "C": 2, "B": 3, "A": 4}.get(raw, 1)


def _cap_tier(current_tier: Any, max_tier: str) -> str:
    cur = str(current_tier or "").strip().upper() or "D"
    cap = str(max_tier or "").strip().upper() or "D"
    return cap if _tier_rank(cur) > _tier_rank(cap) else cur


def _downgrade_tier(tier: str) -> str:
    """Downgrade tier by one level (A→B, B→C, C→D, D→D)."""
    tier_map = {"A": "B", "B": "C", "C": "D", "D": "D"}
    return tier_map.get(tier.strip().upper(), "D")


def _apply_agreement_tier_adjustment(row: dict[str, Any]) -> None:
    """Apply tier downgrade if agreement score indicates high debate."""
    debate = row.get("debate_v1")
    if not isinstance(debate, dict):
        return

    consensus = debate.get("consensus")
    if not isinstance(consensus, dict):
        return

    agreement_score = _to_float(consensus.get("agreement_score"))
    if agreement_score is None:
        return  # No adjustment if missing

    # Clamp to valid range
    if agreement_score < 0.0 or agreement_score > 100.0:
        original_score = agreement_score
        agreement_score = _clamp(agreement_score, 0.0, 100.0)
        ticker = row.get("ticker", "UNKNOWN")
        print(f"[AGREEMENT] Agreement score {original_score:.1f} clamped to {agreement_score:.1f} for {ticker}")

    # Only downgrade if agreement < 50%
    if agreement_score < 50.0:
        current_tier = str(row.get("setup_tier") or "D").strip().upper()
        downgraded_tier = _downgrade_tier(current_tier)
        row["setup_tier"] = downgraded_tier

        # Log the adjustment
        if downgraded_tier != current_tier:
            ticker = row.get("ticker", "UNKNOWN")
            print(
                f"[AGREEMENT] Tier downgraded {current_tier}→{downgraded_tier} "
                f"for {ticker} due to low agreement ({agreement_score:.1f}%)"
            )


def _apply_debate_guardrails(setup_rows: list[dict[str, Any]]) -> None:
    for row in setup_rows:
        if not isinstance(row, dict):
            continue
        debate = row.get("debate_v1")
        if not isinstance(debate, dict):
            continue
        consensus = debate.get("consensus")
        if not isinstance(consensus, dict):
            continue

        state = str(consensus.get("consensus_state") or "watch").strip().lower()
        bias = str(consensus.get("consensus_bias") or "neutral").strip().lower()
        agreement = _to_float(consensus.get("agreement_score")) or 0.0
        disagreement_count = int(_to_float(consensus.get("disagreement_count")) or 0)
        safety_adj = consensus.get("safety_adjustment")
        if not isinstance(safety_adj, list):
            safety_adj = []

        row["debate_consensus_state"] = state
        row["debate_consensus_bias"] = bias
        row["debate_agreement_score"] = round(agreement, 1)
        row["debate_disagreement_count"] = disagreement_count
        row["debate_safety_adjustment"] = safety_adj

        current_actionability = str(row.get("actionability") or "watch-only").strip().lower()

        if state == "watch":
            row["actionability"] = "watch-only"
            row["setup_tier"] = _cap_tier(row.get("setup_tier"), "C")
            row["action"] = (
                "Watch-only. Multi-angle evidence is mixed; wait for clearer confirmation "
                "from trend, levels, and participation."
            )
        elif state == "conditional":
            if current_actionability in {"higher-probability", "setup_ready", "ready"}:
                row["actionability"] = "conditional"
            row["setup_tier"] = _cap_tier(row.get("setup_tier"), "B")
            if current_actionability in {"higher-probability", "setup_ready", "ready"}:
                row["action"] = (
                    "Conditional setup. Debate disagreement requires confirmation before acting."
                )
        elif state == "ready":
            # Let the debate lift "watch-only" to conditional when agreement is strong,
            # but do not auto-promote to higher-probability here.
            if current_actionability in {"watch-only", "watch", "wait"} and agreement >= 70.0:
                row["actionability"] = "conditional"

        if disagreement_count >= 2 and str(row.get("actionability") or "").strip().lower() in {
            "higher-probability",
            "setup_ready",
            "ready",
        }:
            row["actionability"] = "conditional"

        if bias in {"bullish", "bearish"} and str(row.get("signal_bias") or "neutral").lower() == "neutral":
            row["signal_bias"] = bias

        if safety_adj and str(row.get("actionability") or "").strip().lower() in {"higher-probability", "setup_ready", "ready"}:
            row["actionability"] = "conditional"
            row["action"] = "Conditional setup. Safety guardrails are active; wait for confirmation."

        existing_risk = str(row.get("risk_note") or "").strip()
        debate_risk = f"debate={state} ({agreement:.0f}% agreement)"
        if existing_risk and existing_risk.lower() not in {"none", "-"}:
            if debate_risk not in existing_risk:
                row["risk_note"] = f"{existing_risk}; {debate_risk}"
        else:
            row["risk_note"] = debate_risk


def ensure_setup_call_eval_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS setup_call_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            report_kind TEXT NOT NULL DEFAULT 'daily',
            generated_ts TEXT,
            call_direction TEXT NOT NULL,
            validity_days INTEGER NOT NULL DEFAULT 5,
            valid_target_date TEXT,
            setup_family TEXT,
            setup_tier TEXT,
            signal_bias TEXT,
            actionability TEXT,
            score REAL,
            close_asof REAL NOT NULL,
            yolo_pattern TEXT,
            yolo_recency TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'open',
            evaluated_date TEXT,
            close_evaluated REAL,
            raw_return_pct REAL,
            signed_return_pct REAL,
            direction_hit INTEGER,
            UNIQUE(asof_date, ticker, report_kind)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_setup_call_eval_status ON setup_call_evaluations(status, asof_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_setup_call_eval_family ON setup_call_evaluations(setup_family, call_direction, asof_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_setup_call_eval_tier ON setup_call_evaluations(setup_tier, asof_date)"
    )


def _setup_call_direction(row: dict[str, Any]) -> str:
    family = str(row.get("setup_family") or "").strip().lower()
    bias = str(row.get("signal_bias") or "").strip().lower()
    if family.startswith("bullish") or bias == "bullish":
        return "long"
    if family.startswith("bearish") or bias == "bearish":
        return "short"
    return "neutral"


def _setup_validity_days(row: dict[str, Any]) -> int:
    family = str(row.get("setup_family") or "").strip().lower()
    actionability = str(row.get("actionability") or "").strip().lower()
    yolo_recency = str(row.get("yolo_recency") or "").strip().lower()
    if "continuation" in family:
        days = 5
    elif "reversal" in family:
        days = 7
    elif "watch" in family:
        days = 4
    else:
        days = 3

    if actionability == "higher-probability":
        days += 1
    elif actionability in {"wait", "watch-only"}:
        days = max(3, days - 1)

    if yolo_recency == "stale":
        days = max(2, days - 2)
    elif yolo_recency == "aging":
        days = max(3, days - 1)
    return int(max(2, min(12, days)))


def _persist_setup_call_candidates(
    conn: sqlite3.Connection,
    *,
    generated_ts: str | None,
    report_kind: str,
    asof_date: str | None,
    setup_rows: list[dict[str, Any]],
) -> int:
    if not asof_date:
        return 0
    inserted = 0
    for row in (setup_rows or [])[:SETUP_EVAL_TRACK_LIMIT]:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        direction = _setup_call_direction(row)
        if direction not in {"long", "short"}:
            continue
        close_asof = row.get("close")
        if not isinstance(close_asof, (int, float)) or float(close_asof) <= 0:
            close_row = conn.execute(
                "SELECT CAST(close AS REAL) FROM price_daily WHERE ticker = ? AND date = ? LIMIT 1",
                (ticker, asof_date),
            ).fetchone()
            close_asof = float(close_row[0]) if close_row and close_row[0] is not None else None
        if not isinstance(close_asof, (int, float)) or float(close_asof) <= 0:
            continue
        validity_days = _setup_validity_days(row)
        before_changes = conn.total_changes
        conn.execute(
            """
            INSERT INTO setup_call_evaluations (
                asof_date,
                ticker,
                report_kind,
                generated_ts,
                call_direction,
                validity_days,
                setup_family,
                setup_tier,
                signal_bias,
                actionability,
                score,
                close_asof,
                yolo_pattern,
                yolo_recency,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            ON CONFLICT(asof_date, ticker, report_kind) DO NOTHING
            """,
            (
                asof_date,
                ticker,
                report_kind,
                generated_ts,
                direction,
                int(validity_days),
                row.get("setup_family"),
                row.get("setup_tier"),
                row.get("signal_bias"),
                row.get("actionability"),
                row.get("score"),
                float(close_asof),
                row.get("yolo_pattern"),
                row.get("yolo_recency"),
            ),
        )
        if conn.total_changes > before_changes:
            inserted += 1
    return inserted


def _score_open_setup_call_outcomes(conn: sqlite3.Connection) -> int:
    open_rows = conn.execute(
        """
        SELECT id, ticker, asof_date, call_direction, validity_days, close_asof
        FROM setup_call_evaluations
        WHERE status = 'open'
        ORDER BY asof_date ASC, id ASC
        """
    ).fetchall()
    scored = 0
    for row in open_rows:
        call_id = int(row[0])
        ticker = str(row[1] or "").upper().strip()
        asof_date = str(row[2] or "").strip()
        call_direction = str(row[3] or "").strip().lower()
        validity_days = int(row[4] or 0)
        close_asof = float(row[5] or 0.0)
        if call_direction not in {"long", "short"} or validity_days <= 0 or close_asof <= 0.0:
            conn.execute(
                """
                UPDATE setup_call_evaluations
                SET status = 'invalid',
                    updated_ts = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (call_id,),
            )
            continue

        future_rows = conn.execute(
            """
            SELECT date, CAST(close AS REAL) AS close
            FROM price_daily
            WHERE ticker = ? AND date > ?
            ORDER BY date ASC
            LIMIT ?
            """,
            (ticker, asof_date, int(validity_days)),
        ).fetchall()
        if len(future_rows) < validity_days:
            continue
        eval_date = str(future_rows[-1][0] or "").strip()
        eval_close = float(future_rows[-1][1] or 0.0)
        if not eval_date or eval_close <= 0.0:
            continue
        raw_return_pct = ((eval_close / close_asof) - 1.0) * 100.0
        signed_return_pct = raw_return_pct if call_direction == "long" else (-raw_return_pct)
        hit = 1 if signed_return_pct >= float(SETUP_EVAL_HIT_THRESHOLD_PCT) else 0
        conn.execute(
            """
            UPDATE setup_call_evaluations
            SET status = 'scored',
                valid_target_date = ?,
                evaluated_date = ?,
                close_evaluated = ?,
                raw_return_pct = ?,
                signed_return_pct = ?,
                direction_hit = ?,
                updated_ts = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                eval_date,
                eval_date,
                round(eval_close, 6),
                round(raw_return_pct, 6),
                round(signed_return_pct, 6),
                int(hit),
                call_id,
            ),
        )
        scored += 1
    return scored


def _setup_eval_bucket(
    label: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    calls = len(rows)
    wins = [row for row in rows if int(row.get("direction_hit") or 0) == 1]
    losses = [row for row in rows if int(row.get("direction_hit") or 0) != 1]
    hits = len(wins)
    losses_count = len(losses)
    signed_vals = [float(row.get("signed_return_pct")) for row in rows if isinstance(row.get("signed_return_pct"), (int, float))]
    win_vals = [float(row.get("signed_return_pct")) for row in wins if isinstance(row.get("signed_return_pct"), (int, float))]
    loss_vals = [float(row.get("signed_return_pct")) for row in losses if isinstance(row.get("signed_return_pct"), (int, float))]
    hit_rate = (hits / calls) if calls else None
    loss_rate = (losses_count / calls) if calls else None
    avg_win = (sum(win_vals) / len(win_vals)) if win_vals else None
    avg_loss = (sum(loss_vals) / len(loss_vals)) if loss_vals else None
    expectancy: float | None = None
    if hit_rate is not None and loss_rate is not None:
        expectancy = (hit_rate * float(avg_win or 0.0)) + (loss_rate * float(avg_loss or 0.0))
    pos_sum = sum(v for v in signed_vals if v > 0.0)
    neg_abs_sum = abs(sum(v for v in signed_vals if v < 0.0))
    profit_factor: float | None = None
    if neg_abs_sum > 0.0:
        profit_factor = pos_sum / neg_abs_sum
    elif pos_sum > 0.0 and calls > 0:
        profit_factor = 9.99
    return {
        "label": label,
        "calls": calls,
        "hits": hits,
        "losses": losses_count,
        "hit_rate_pct": _round_or_none((hit_rate * 100.0) if hit_rate is not None else None),
        "loss_rate_pct": _round_or_none((loss_rate * 100.0) if loss_rate is not None else None),
        "avg_signed_return_pct": _round_or_none((sum(signed_vals) / len(signed_vals)) if signed_vals else None),
        "median_signed_return_pct": _round_or_none(_median(signed_vals)),
        "avg_win_return_pct": _round_or_none(avg_win),
        "avg_loss_return_pct": _round_or_none(avg_loss),
        "expectancy_pct": _round_or_none(expectancy),
        "profit_factor": _round_or_none(profit_factor),
    }


def _build_setup_eval_improvement_actions(
    *,
    scored_calls: int,
    min_sample: int,
    hit_threshold_pct: float,
    overall: dict[str, Any],
    by_direction: list[dict[str, Any]],
    by_family: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if scored_calls < max(5, int(min_sample) * 3):
        actions.append(
            {
                "priority": "info",
                "scope": "dataset",
                "reason": (
                    f"Backtest sample is still small ({scored_calls} scored calls). "
                    f"Target at least {max(5, int(min_sample) * 3)} for stable family tuning."
                ),
                "recommendation": "Keep current score weights conservative and collect more outcomes before major threshold changes.",
            }
        )
        return actions

    overall_expectancy = _to_float(overall.get("expectancy_pct"))
    overall_hit = _to_float(overall.get("hit_rate_pct"))
    if overall_expectancy is not None and overall_expectancy <= 0.0:
        actions.append(
            {
                "priority": "high",
                "scope": "global",
                "reason": (
                    f"Overall expectancy is non-positive ({round(overall_expectancy, 2)}%)."
                ),
                "recommendation": (
                    "Tighten setup-ready criteria: require stronger level confirmation and downgrade stale-context candidates to watch."
                ),
            }
        )
    if overall_hit is not None and overall_hit < float(hit_threshold_pct):
        actions.append(
            {
                "priority": "high",
                "scope": "global",
                "reason": (
                    f"Overall hit rate ({round(overall_hit, 2)}%) is below target threshold ({round(hit_threshold_pct, 2)}%)."
                ),
                "recommendation": (
                    "Increase entry confirmation requirements (fresh YOLO + candle/level alignment) before labeling a setup as actionable."
                ),
            }
        )

    dir_map = {
        str(row.get("direction") or "").strip().lower(): row
        for row in by_direction
        if isinstance(row, dict)
    }
    long_row = dir_map.get("long") or {}
    short_row = dir_map.get("short") or {}
    long_calls = int(long_row.get("calls") or 0)
    short_calls = int(short_row.get("calls") or 0)
    long_expectancy = _to_float(long_row.get("expectancy_pct"))
    short_expectancy = _to_float(short_row.get("expectancy_pct"))
    if long_calls >= int(min_sample) and long_expectancy is not None and long_expectancy < 0.0:
        actions.append(
            {
                "priority": "medium",
                "scope": "long_setups",
                "reason": f"Long calls have negative expectancy ({round(long_expectancy, 2)}%) over {long_calls} samples.",
                "recommendation": "Reduce long setup score bonus until trend/level confirmation improves.",
            }
        )
    if short_calls >= int(min_sample) and short_expectancy is not None and short_expectancy < 0.0:
        actions.append(
            {
                "priority": "medium",
                "scope": "short_setups",
                "reason": f"Short calls have negative expectancy ({round(short_expectancy, 2)}%) over {short_calls} samples.",
                "recommendation": "Reduce short setup score bonus until breakdown confirmation improves.",
            }
        )

    candidates = [row for row in by_family if isinstance(row, dict) and int(row.get("calls") or 0) >= int(min_sample)]
    weak = [
        row for row in candidates
        if (
            (_to_float(row.get("hit_rate_pct")) is not None and float(_to_float(row.get("hit_rate_pct")) or 0.0) < float(hit_threshold_pct))
            or (_to_float(row.get("expectancy_pct")) is not None and float(_to_float(row.get("expectancy_pct")) or 0.0) < 0.0)
        )
    ]
    weak.sort(
        key=lambda row: (
            float(_to_float(row.get("expectancy_pct")) or 0.0),
            float(_to_float(row.get("hit_rate_pct")) or 0.0),
            -int(row.get("calls") or 0),
        )
    )
    if weak:
        top_weak = weak[0]
        family = str(top_weak.get("setup_family") or "-")
        direction = str(top_weak.get("call_direction") or "-")
        actions.append(
            {
                "priority": "medium",
                "scope": "family",
                "reason": (
                    f"Weak family detected: {family}:{direction} "
                    f"(hit {top_weak.get('hit_rate_pct')}%, expectancy {top_weak.get('expectancy_pct')}%, calls {top_weak.get('calls')})."
                ),
                "recommendation": (
                    f"Demote {family}:{direction} by default (or require stronger confirmation) until its backtest edge recovers."
                ),
            }
        )

    strong = [
        row for row in candidates
        if (
            (_to_float(row.get("hit_rate_pct")) is not None and float(_to_float(row.get("hit_rate_pct")) or 0.0) >= float(hit_threshold_pct) + 5.0)
            and (_to_float(row.get("expectancy_pct")) is not None and float(_to_float(row.get("expectancy_pct")) or 0.0) > 0.0)
        )
    ]
    strong.sort(
        key=lambda row: (
            float(_to_float(row.get("expectancy_pct")) or 0.0),
            float(_to_float(row.get("hit_rate_pct")) or 0.0),
            int(row.get("calls") or 0),
        ),
        reverse=True,
    )
    if strong:
        top_strong = strong[0]
        family = str(top_strong.get("setup_family") or "-")
        direction = str(top_strong.get("call_direction") or "-")
        actions.append(
            {
                "priority": "low",
                "scope": "family",
                "reason": (
                    f"Strong family detected: {family}:{direction} "
                    f"(hit {top_strong.get('hit_rate_pct')}%, expectancy {top_strong.get('expectancy_pct')}%, calls {top_strong.get('calls')})."
                ),
                "recommendation": (
                    f"Use {family}:{direction} as a baseline and compare weaker families against this quality bar."
                ),
            }
        )

    return actions[:5]


def _summarize_setup_call_evaluations(
    conn: sqlite3.Connection,
    *,
    window_days: int = SETUP_EVAL_WINDOW_DAYS,
    min_sample: int = SETUP_EVAL_MIN_SAMPLE,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    if not table_exists(conn, "setup_call_evaluations"):
        return {}, {}
    cutoff_date = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=max(30, int(window_days)))).isoformat()
    open_calls = int(
        (
            conn.execute("SELECT COUNT(*) FROM setup_call_evaluations WHERE status = 'open'").fetchone() or [0]
        )[0]
        or 0
    )
    scored_rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT
                ticker,
                asof_date,
                setup_family,
                setup_tier,
                call_direction,
                validity_days,
                direction_hit,
                signed_return_pct
            FROM setup_call_evaluations
            WHERE status = 'scored'
              AND asof_date >= ?
            ORDER BY asof_date DESC, id DESC
            """,
            (cutoff_date,),
        ).fetchall()
    ]

    overall = _setup_eval_bucket("overall", scored_rows)
    validity_vals = [int(row.get("validity_days") or 0) for row in scored_rows if int(row.get("validity_days") or 0) > 0]
    overall["avg_validity_days"] = _round_or_none((sum(validity_vals) / len(validity_vals)) if validity_vals else None)

    by_direction: list[dict[str, Any]] = []
    for direction in ("long", "short"):
        group = [row for row in scored_rows if str(row.get("call_direction") or "").strip().lower() == direction]
        bucket = _setup_eval_bucket(direction, group)
        bucket["direction"] = direction
        by_direction.append(bucket)

    by_validity: list[dict[str, Any]] = []
    validity_groups: dict[int, list[dict[str, Any]]] = {}
    for row in scored_rows:
        validity = int(row.get("validity_days") or 0)
        if validity <= 0:
            continue
        validity_groups.setdefault(validity, []).append(row)
    for validity, group in sorted(validity_groups.items(), key=lambda kv: kv[0]):
        bucket = _setup_eval_bucket(f"{validity}d", group)
        bucket["validity_days"] = int(validity)
        by_validity.append(bucket)

    by_tier: list[dict[str, Any]] = []
    for tier in ("A", "B", "C", "D"):
        group = [row for row in scored_rows if str(row.get("setup_tier") or "").strip().upper() == tier]
        bucket = _setup_eval_bucket(tier, group)
        bucket["setup_tier"] = tier
        by_tier.append(bucket)

    family_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scored_rows:
        family = str(row.get("setup_family") or "").strip().lower()
        direction = str(row.get("call_direction") or "").strip().lower()
        if not family or direction not in {"long", "short"}:
            continue
        family_map.setdefault((family, direction), []).append(row)

    by_family: list[dict[str, Any]] = []
    reliability_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in family_map.items():
        family, direction = key
        bucket = _setup_eval_bucket(f"{family}:{direction}", group)
        bucket["setup_family"] = family
        bucket["call_direction"] = direction
        by_family.append(bucket)
        if int(bucket.get("calls") or 0) >= max(1, int(min_sample)):
            reliability_lookup[key] = {
                "hit_rate_pct": bucket.get("hit_rate_pct"),
                "calls": int(bucket.get("calls") or 0),
                "avg_signed_return_pct": bucket.get("avg_signed_return_pct"),
            }
    by_family.sort(
        key=lambda item: (
            int(item.get("calls") or 0),
            float(item.get("hit_rate_pct") or 0.0),
            float(item.get("avg_signed_return_pct") or 0.0),
        ),
        reverse=True,
    )

    top_long = [row for row in by_family if str(row.get("call_direction") or "").lower() == "long"][:8]
    top_short = [row for row in by_family if str(row.get("call_direction") or "").lower() == "short"][:8]
    weak_families = sorted(
        [
            row for row in by_family
            if int(row.get("calls") or 0) >= int(max(1, min_sample))
        ],
        key=lambda row: (
            float(_to_float(row.get("expectancy_pct")) or 0.0),
            float(_to_float(row.get("hit_rate_pct")) or 0.0),
            -int(row.get("calls") or 0),
        ),
    )[:8]

    newest_scored = next(
        (str(row.get("asof_date") or "").strip() for row in scored_rows if str(row.get("asof_date") or "").strip()),
        None,
    )
    summary = {
        "enabled": True,
        "window_days": int(max(30, int(window_days))),
        "min_sample": int(max(1, int(min_sample))),
        "hit_threshold_pct": float(SETUP_EVAL_HIT_THRESHOLD_PCT),
        "scored_calls": int(overall.get("calls") or 0),
        "open_calls": int(open_calls),
        "latest_scored_asof": newest_scored,
        "overall": overall,
        "by_direction": by_direction,
        "by_validity_days": by_validity,
        "by_tier": by_tier,
        "by_family": by_family[:12],
        "top_long_families": top_long,
        "top_short_families": top_short,
        "weak_families": weak_families,
    }
    summary["improvement_actions"] = _build_setup_eval_improvement_actions(
        scored_calls=int(summary.get("scored_calls") or 0),
        min_sample=int(summary.get("min_sample") or 1),
        hit_threshold_pct=float(summary.get("hit_threshold_pct") or 0.0),
        overall=overall,
        by_direction=by_direction,
        by_family=by_family,
    )
    return summary, reliability_lookup


def _setup_eval_score_adjustment(
    stat: dict[str, Any] | None,
    *,
    min_sample: int,
    hit_threshold_pct: float,
) -> float:
    if not stat:
        return 0.0
    calls = int(stat.get("calls") or 0)
    if calls < max(1, int(min_sample)):
        return 0.0
    hit_rate = float(stat.get("hit_rate_pct") or 0.0)
    avg_signed_return = float(stat.get("avg_signed_return_pct") or 0.0)

    # Confidence grows with sample size but is bounded so old data cannot dominate.
    sample_scale = _clamp(calls / float(max(1, int(min_sample) * 4)), 0.35, 1.0)
    hit_component = _clamp((hit_rate - float(hit_threshold_pct)) * 0.28, -8.0, 8.0)
    return_component = _clamp(avg_signed_return * 1.6, -6.0, 6.0)
    adjustment = (hit_component + return_component) * sample_scale
    return round(_clamp(adjustment, -10.0, 10.0), 1)


def _apply_setup_eval_fields(
    setup_rows: list[dict[str, Any]],
    *,
    reliability_lookup: dict[tuple[str, str], dict[str, Any]],
    min_sample: int,
    hit_threshold_pct: float,
) -> dict[str, Any]:
    adjusted_calls = 0
    adjustments: list[float] = []
    for row in setup_rows or []:
        if not isinstance(row, dict):
            continue
        call_direction = _setup_call_direction(row)
        validity_days = _setup_validity_days(row)
        family = str(row.get("setup_family") or "").strip().lower()
        stat = reliability_lookup.get((family, call_direction))
        row["call_direction"] = call_direction
        row["validity_days"] = int(validity_days)
        row["validity_label"] = f"{int(validity_days)} trading day{'s' if int(validity_days) != 1 else ''}"
        row["historical_reliability_pct"] = stat.get("hit_rate_pct") if stat else None
        row["historical_sample_size"] = int(stat.get("calls") or 0) if stat else 0
        row["historical_avg_signed_return_pct"] = stat.get("avg_signed_return_pct") if stat else None
        if stat:
            row["reliability_label"] = (
                f"{_round_or_none(stat.get('hit_rate_pct'))}% hit rate "
                f"({int(stat.get('calls') or 0)} calls)"
            )
        else:
            row["reliability_label"] = "insufficient history"
        raw_score = float(row.get("score") or 0.0)
        adjustment = _setup_eval_score_adjustment(
            stat,
            min_sample=min_sample,
            hit_threshold_pct=hit_threshold_pct,
        )
        adjusted_score = round(_clamp(raw_score + adjustment, 0.0, 100.0), 1)
        row["setup_score_raw"] = round(raw_score, 1)
        row["setup_score_adjustment"] = adjustment
        row["setup_score_adjusted"] = adjusted_score
        if adjustment >= 2.0:
            row["reliability_signal"] = "tailwind"
        elif adjustment <= -2.0:
            row["reliability_signal"] = "headwind"
        else:
            row["reliability_signal"] = "neutral"
        if adjustment != 0.0:
            adjusted_calls += 1
            adjustments.append(adjustment)
            row["score"] = adjusted_score
            row["confluence_score"] = adjusted_score
            row["setup_tier"] = _setup_tier(adjusted_score)
    return {
        "adjusted_calls": int(adjusted_calls),
        "avg_adjustment": _round_or_none((sum(adjustments) / len(adjustments)) if adjustments else 0.0),
        "max_positive_adjustment": _round_or_none(max(adjustments) if adjustments else 0.0),
        "max_negative_adjustment": _round_or_none(min(adjustments) if adjustments else 0.0),
    }


EARNINGS_PROXIMITY_DAYS = 5


def annotate_earnings_proximity(
    setup_rows: list[dict[str, Any]],
    earnings_catalysts: dict[str, Any],
) -> int:
    """Flag setups with earnings within *EARNINGS_PROXIMITY_DAYS* trading days.

    Mutates each setup row in-place, adding:
      - earnings_within_5d (bool)
      - earnings_date (str | None)
      - days_to_earnings (int | None)

    When a ticker has nearby earnings, "earnings within Xd" is appended to
    the existing risk_note so the paper-trade decision pipeline applies its
    event-risk position haircut automatically.

    Returns the number of setups flagged.
    """
    if not setup_rows or not isinstance(earnings_catalysts, dict):
        for row in setup_rows or []:
            row.setdefault("earnings_within_5d", False)
            row.setdefault("earnings_date", None)
            row.setdefault("days_to_earnings", None)
        return 0

    # Build a ticker -> (nearest earnings_date, days_until) lookup from
    # the earnings calendar rows.
    earnings_rows = earnings_catalysts.get("rows") or []
    nearest: dict[str, tuple[str, int]] = {}
    for erow in earnings_rows:
        if not isinstance(erow, dict):
            continue
        ticker = str(erow.get("ticker") or "").upper().strip()
        days_until = erow.get("days_until")
        if not ticker or not isinstance(days_until, (int, float)):
            continue
        days_int = int(days_until)
        if days_int < 0:
            continue
        prev = nearest.get(ticker)
        if prev is None or days_int < prev[1]:
            nearest[ticker] = (
                str(erow.get("earnings_date") or ""),
                days_int,
            )

    flagged = 0
    for row in setup_rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        hit = nearest.get(ticker)
        if hit and hit[1] <= EARNINGS_PROXIMITY_DAYS:
            row["earnings_within_5d"] = True
            row["earnings_date"] = hit[0]
            row["days_to_earnings"] = hit[1]
            # Append to risk_note so paper-trade decision.py picks it up
            existing = str(row.get("risk_note") or "").strip()
            earnings_tag = f"earnings within {hit[1]}d"
            if "earnings" not in existing.lower():
                if existing and existing.lower() != "none":
                    row["risk_note"] = f"{existing}, {earnings_tag}"
                else:
                    row["risk_note"] = earnings_tag
            flagged += 1
        else:
            row["earnings_within_5d"] = False
            row["earnings_date"] = hit[0] if hit else None
            row["days_to_earnings"] = hit[1] if hit else None

    return flagged


def _refresh_setup_eval_surfaces(signals: dict[str, Any]) -> None:
    setup_rows = (signals.get("setup_quality_top") or []) if isinstance(signals, dict) else []
    by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in setup_rows
        if isinstance(row, dict) and row.get("ticker")
    }
    eval_keys = (
        "call_direction",
        "validity_days",
        "validity_label",
        "historical_reliability_pct",
        "historical_sample_size",
        "historical_avg_signed_return_pct",
        "reliability_label",
        "reliability_signal",
        "setup_score_raw",
        "setup_score_adjustment",
        "setup_score_adjusted",
        "score",
        "confluence_score",
        "setup_tier",
        "setup_family",
        "signal_bias",
        "actionability",
        "observation",
        "action",
        "risk_note",
        "technical_read",
        "narrative_source",
    )
    watchlist = signals.get("watchlist_candidates")
    if isinstance(watchlist, list):
        for row in watchlist:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").upper()
            src = by_ticker.get(ticker)
            if not src:
                continue
            for key in eval_keys:
                row[key] = src.get(key)
    setup_lookup = signals.get("setup_quality_lookup")
    if isinstance(setup_lookup, dict):
        for ticker, payload in list(setup_lookup.items()):
            if not isinstance(payload, dict):
                continue
            src = by_ticker.get(str(ticker or "").upper())
            if not src:
                continue
            for key in eval_keys:
                payload[key] = src.get(key)


def _setup_cluster_rows(rows: list[dict[str, Any]], *, score_window: float = 3.0, scan_limit: int = 8) -> list[dict[str, Any]]:
    if not rows:
        return []
    best = rows[0]
    best_score = float(best.get("score") or 0.0)
    best_tier = str(best.get("setup_tier") or "").strip().upper()
    best_family = str(best.get("setup_family") or "").strip().lower()
    cluster: list[dict[str, Any]] = [best]
    for row in rows[1:scan_limit]:
        row_tier = str(row.get("setup_tier") or "").strip().upper()
        row_family = str(row.get("setup_family") or "").strip().lower()
        row_score = float(row.get("score") or 0.0)
        if row_tier != best_tier:
            continue
        if best_family and row_family != best_family:
            continue
        if (best_score - row_score) > score_window:
            continue
        cluster.append(row)
    return cluster


def build_tonight_key_changes(signals: dict[str, Any], yolo_delta: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    breadth = signals.get("market_breadth") or {}
    if breadth:
        adv = breadth.get("advancers", 0)
        dec = breadth.get("decliners", 0)
        pct = breadth.get("pct_advancing")
        avg_move = breadth.get("avg_pct_change")
        regime = "risk-on" if isinstance(pct, (int, float)) and pct >= 55 else "mixed"
        if isinstance(pct, (int, float)) and pct <= 45:
            regime = "risk-off"
        changes.append(
            {
                "slug": "breadth",
                "title": "Breadth Regime",
                "detail": f"{adv} advancing vs {dec} declining ({pct}% advancers, avg move {avg_move}%).",
                "tone": "positive" if regime == "risk-on" else ("negative" if regime == "risk-off" else "neutral"),
            }
        )

    movers_up = signals.get("movers_up_today") or []
    movers_down = signals.get("movers_down_today") or []
    if movers_up or movers_down:
        top_up = movers_up[0] if movers_up else None
        top_down = movers_down[0] if movers_down else None
        if top_up and top_down:
            detail = (
                f"Leader: {top_up.get('ticker')} {top_up.get('pct_change')}% | "
                f"Laggard: {top_down.get('ticker')} {top_down.get('pct_change')}%."
            )
        elif top_up:
            detail = f"Strongest upside move: {top_up.get('ticker')} {top_up.get('pct_change')}%."
        else:
            detail = f"Weakest move: {top_down.get('ticker')} {top_down.get('pct_change')}%."
        changes.append(
            {
                "slug": "movers",
                "title": "Largest Price Moves",
                "detail": detail,
                "tone": "neutral",
            }
        )

    new_count = int(yolo_delta.get("new_count") or 0)
    lost_count = int(yolo_delta.get("lost_count") or 0)
    if new_count or lost_count:
        new_patterns = yolo_delta.get("new_patterns") or []
        lost_patterns = yolo_delta.get("lost_patterns") or []
        top_new = new_patterns[0] if new_patterns else None
        top_lost = lost_patterns[0] if lost_patterns else None
        yolo_parts = [f"+{new_count} new", f"-{lost_count} lost"]
        if top_new:
            yolo_parts.append(
                f"new highlight: {top_new.get('ticker')} {top_new.get('pattern')} ({top_new.get('confidence')})"
            )
        if top_lost:
            yolo_parts.append(
                f"invalidated/completed: {top_lost.get('ticker')} {top_lost.get('pattern')} ({top_lost.get('confidence')})"
            )
        changes.append(
            {
                "slug": "yolo_delta",
                "title": "YOLO Pattern Churn",
                "detail": " • ".join(yolo_parts),
                "tone": "positive" if new_count > lost_count else ("negative" if lost_count > new_count else "neutral"),
            }
        )

    sector_rows = signals.get("sector_heatmap") or []
    if sector_rows:
        top_sector = sector_rows[0]
        bottom_sector = sector_rows[-1]
        changes.append(
            {
                "slug": "sectors",
                "title": "Sector Rotation",
                "detail": (
                    f"Leader: {top_sector.get('sector')} ({top_sector.get('avg_pct_change')}%) | "
                    f"Laggard: {bottom_sector.get('sector')} ({bottom_sector.get('avg_pct_change')}%)."
                ),
                "tone": "neutral",
            }
        )

    setup_rows = signals.get("setup_quality_top") or []
    if setup_rows:
        best = setup_rows[0]
        setup_cluster = _setup_cluster_rows(setup_rows)
        cluster_family = str(best.get("setup_family") or "").replace("_", " ").strip()
        cluster_label = ", ".join(
            f"{row.get('ticker')} {row.get('score')} ({row.get('setup_tier')})"
            for row in setup_cluster[:3]
        )
        changes.append(
            {
                "slug": "setup",
                "title": "Top Setup Cluster" if len(setup_cluster) > 1 else "Top Setup Candidate",
                "detail": (
                    (
                        f"Leaders: {cluster_label}. "
                        + (f"Shared setup family: {cluster_family}. " if cluster_family else "")
                        if len(setup_cluster) > 1
                        else ""
                    )
                    + f"Highest-rated: {best.get('ticker')} scored {best.get('score')} ({best.get('setup_tier')}) "
                    f"with move {best.get('pct_change')}%, discount {best.get('discount_pct')}%, "
                    f"PEG {best.get('peg')}, ATR {best.get('atr_pct_14') if best.get('atr_pct_14') is not None else '-'}%. "
                    f"Read: {best.get('observation') or 'no technical read yet'} "
                    f"Action: {best.get('action') or 'watch only'}"
                ),
                "tone": "positive",
            }
        )

    while len(changes) < 5:
        changes.append(
            {
                "slug": f"placeholder_{len(changes) + 1}",
                "title": "Signal",
                "detail": "No material change detected for this slot.",
                "tone": "neutral",
            }
        )
    return changes[:5]


def build_no_trade_conditions(report: dict[str, Any]) -> dict[str, Any]:
    warnings = {str(w) for w in (report.get("warnings") or [])}
    signals = report.get("signals") or {}
    breadth = signals.get("market_breadth") or {}
    yolo = report.get("yolo") or {}
    delta_daily = yolo.get("delta_daily") or yolo.get("delta") or {}
    session = report.get("market_session") or {}

    conditions: list[dict[str, Any]] = []

    def add_condition(code: str, severity: str, reason: str) -> None:
        conditions.append({"code": code, "severity": severity, "reason": reason})

    if "price_data_stale" in warnings or "price_data_missing" in warnings:
        add_condition("stale_price_data", "hard", "Price data is stale/missing.")
    if "latest_ingest_run_failed" in warnings:
        add_condition("ingest_failed", "hard", "Latest ingest run failed.")
    if "yolo_data_missing" in warnings:
        add_condition("yolo_missing", "hard", "YOLO detections are missing.")
    if "yolo_data_stale" in warnings:
        add_condition("yolo_stale", "soft", "YOLO detections are stale.")
    if bool(session.get("is_holiday")):
        add_condition(
            "market_holiday",
            "hard",
            f"US market holiday: {session.get('holiday_name') or 'closed'}.",
        )
    if bool(session.get("is_early_close")):
        add_condition(
            "early_close_session",
            "soft",
            f"Early close session: {session.get('early_close_name') or 'shortened day'}.",
        )

    pct_adv = breadth.get("pct_advancing")
    avg_move = breadth.get("avg_pct_change")
    large_moves = breadth.get("large_move_count")
    if (
        isinstance(pct_adv, (int, float))
        and isinstance(avg_move, (int, float))
        and isinstance(large_moves, int)
        and 45.0 <= float(pct_adv) <= 55.0
        and abs(float(avg_move)) <= 0.35
        and int(large_moves) <= 15
    ):
        add_condition(
            "chop_regime",
            "soft",
            (
                "Breadth is mixed and average move is muted "
                f"(adv={pct_adv}%, avg={avg_move}%, large_moves={large_moves})."
            ),
        )

    new_daily = int(delta_daily.get("new_count") or 0)
    lost_daily = int(delta_daily.get("lost_count") or 0)
    if lost_daily >= max(150, new_daily * 8):
        add_condition(
            "high_pattern_churn",
            "soft",
            f"Daily YOLO churn is high (+{new_daily} new / -{lost_daily} lost).",
        )

    hard_count = sum(1 for c in conditions if c.get("severity") == "hard")
    soft_count = sum(1 for c in conditions if c.get("severity") == "soft")
    trade_mode = "blocked" if hard_count > 0 else ("caution" if soft_count > 0 else "normal")
    return {
        "trade_mode": trade_mode,
        "hard_blocks": hard_count,
        "soft_flags": soft_count,
        "conditions": conditions,
    }

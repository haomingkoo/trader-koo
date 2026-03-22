"""Chart dashboard payload builder.

Constructs the full ``/api/dashboard/{ticker}`` response, including
technical analysis, pattern overlays, chart commentary, and
YOLO pattern enrichment. Depends on database.py for raw queries
and report_loader.py for report-snapshot overrides.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import HTTPException

from trader_koo.catalyst_data import get_ticker_earnings_markers
from trader_koo.cv.compare import HybridCVCompareConfig, compare_hybrid_vs_cv
from trader_koo.cv.proxy_patterns import CVProxyConfig, detect_cv_proxy_patterns
from trader_koo.features.candle_patterns import CandlePatternConfig, detect_candlestick_patterns
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.llm_narrative import llm_status, maybe_rewrite_setup_copy
from trader_koo.scripts.generate_daily_report import (
    _describe_setup as _report_describe_setup,
    _score_setup_from_confluence as _report_score_setup_from_confluence,
)
from trader_koo.structure.gaps import GapConfig, detect_gaps, select_gaps_for_display
from trader_koo.structure.hybrid_patterns import HybridPatternConfig, score_hybrid_patterns
from trader_koo.structure.levels import (
    LevelConfig,
    add_fallback_levels,
    build_levels_from_pivots,
    select_target_levels,
)
from trader_koo.structure.patterns import PatternConfig, detect_patterns
from trader_koo.structure.trendlines import TrendlineConfig, detect_trendlines

from trader_koo.backend.services.database import (
    get_latest_fundamentals,
    get_latest_options_summary,
    get_price_df,
    get_yolo_audit,
    get_yolo_patterns,
)
from trader_koo.backend.services.report_loader import latest_report_hmm_for_ticker
from trader_koo.structure.hmm_regime import predict_regimes as hmm_predict_regimes
from trader_koo.backend.services.market_data import get_data_sources
from trader_koo.backend.services.report_loader import latest_report_setup_for_ticker

LOG = logging.getLogger("trader_koo.services.chart_builder")

# Feature/pattern configuration singletons
FEATURE_CFG = FeatureConfig()
LEVEL_CFG = LevelConfig()
GAP_CFG = GapConfig()
TREND_CFG = TrendlineConfig()
PATTERN_CFG = PatternConfig()
CANDLE_CFG = CandlePatternConfig()
HYBRID_PATTERN_CFG = HybridPatternConfig()
CV_PROXY_CFG = CVProxyConfig()
HYBRID_CV_CMP_CFG = HybridCVCompareConfig()

_MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")
try:
    _MARKET_TZ = ZoneInfo(_MARKET_TZ_NAME)
except Exception:
    _MARKET_TZ = dt.timezone.utc


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of dicts, replacing NaN with empty string."""
    if df is None or df.empty:
        return []
    return df.fillna("").to_dict(orient="records")


# ---------------------------------------------------------------------------
# Pattern overlays
# ---------------------------------------------------------------------------

def build_pattern_overlays(
    patterns: pd.DataFrame,
    hybrid_patterns: pd.DataFrame,
    cv_proxy_patterns: pd.DataFrame,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Merge rule-based and CV-proxy patterns into a single overlay table."""
    cols = [
        "source",
        "class_name",
        "status",
        "confidence",
        "start_date",
        "end_date",
        "x0_date",
        "x1_date",
        "y0",
        "y1",
        "y0b",
        "y1b",
        "notes",
    ]
    empty = pd.DataFrame(columns=cols)

    hp = hybrid_patterns.copy() if hybrid_patterns is not None else pd.DataFrame()
    if not hp.empty:
        hp["hybrid_confidence"] = pd.to_numeric(hp.get("hybrid_confidence"), errors="coerce")
        hp = hp.sort_values("hybrid_confidence", ascending=False).drop_duplicates(subset=["pattern"])
        hp = hp[["pattern", "hybrid_confidence"]]
    else:
        hp = pd.DataFrame(columns=["pattern", "hybrid_confidence"])

    rule_df = patterns.copy() if patterns is not None else pd.DataFrame()
    if not rule_df.empty:
        rule_df = rule_df.merge(hp, on="pattern", how="left")
        base_conf = pd.to_numeric(rule_df.get("confidence"), errors="coerce")
        hybrid_conf = pd.to_numeric(rule_df.get("hybrid_confidence"), errors="coerce")
        rule_df["confidence"] = hybrid_conf.where(hybrid_conf.notna(), base_conf)
        rule_df["source"] = rule_df["hybrid_confidence"].apply(
            lambda v: "hybrid_rule" if pd.notna(v) else "rule"
        )
        rule_df = rule_df.rename(columns={"pattern": "class_name"})
        rule_df = rule_df[
            [
                "source",
                "class_name",
                "status",
                "confidence",
                "start_date",
                "end_date",
                "x0_date",
                "x1_date",
                "y0",
                "y1",
                "y0b",
                "y1b",
                "notes",
            ]
        ]
    else:
        rule_df = empty.copy()

    cv_df = cv_proxy_patterns.copy() if cv_proxy_patterns is not None else pd.DataFrame()
    if not cv_df.empty:
        cv_df["source"] = "cv_proxy"
        cv_df["confidence"] = pd.to_numeric(cv_df.get("cv_confidence"), errors="coerce")
        cv_df = cv_df.rename(columns={"pattern": "class_name"})
        cv_df = cv_df[
            [
                "source",
                "class_name",
                "status",
                "confidence",
                "start_date",
                "end_date",
                "x0_date",
                "x1_date",
                "y0",
                "y1",
                "y0b",
                "y1b",
                "notes",
            ]
        ]
    else:
        cv_df = empty.copy()

    out = pd.concat([rule_df, cv_df], ignore_index=True)
    if out.empty:
        return empty

    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    for c in ["y0", "y1", "y0b", "y1b"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["x0_date", "x1_date", "y0", "y1", "y0b", "y1b"])
    out = out[(out["x0_date"].astype(str).str.len() > 0) & (out["x1_date"].astype(str).str.len() > 0)]
    if out.empty:
        return empty

    out = out.sort_values("confidence", ascending=False)
    out = out.drop_duplicates(subset=["source", "class_name", "start_date", "end_date"]).head(max_rows)
    out["confidence"] = out["confidence"].round(2)
    return out[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Technical context
# ---------------------------------------------------------------------------

def _build_chart_technical_context(
    model: pd.DataFrame,
    levels: pd.DataFrame,
) -> dict[str, Any]:
    """Derive technical-analysis context fields from price + level data."""
    closes = pd.to_numeric(model.get("close"), errors="coerce")
    highs = pd.to_numeric(model.get("high"), errors="coerce")
    lows = pd.to_numeric(model.get("low"), errors="coerce")
    volumes = pd.to_numeric(model.get("volume"), errors="coerce").fillna(0.0)
    if closes.empty:
        return {}

    close_now = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else close_now
    high_now = float(highs.iloc[-1])
    low_now = float(lows.iloc[-1])
    ma20 = float(pd.to_numeric(model.get("ma20"), errors="coerce").iloc[-1]) if "ma20" in model.columns and pd.notna(model["ma20"].iloc[-1]) else None
    ma50 = float(pd.to_numeric(model.get("ma50"), errors="coerce").iloc[-1]) if "ma50" in model.columns and pd.notna(model["ma50"].iloc[-1]) else None
    ma100 = float(pd.to_numeric(model.get("ma100"), errors="coerce").iloc[-1]) if "ma100" in model.columns and pd.notna(model["ma100"].iloc[-1]) else None
    ma200 = float(pd.to_numeric(model.get("ma200"), errors="coerce").iloc[-1]) if "ma200" in model.columns and pd.notna(model["ma200"].iloc[-1]) else None
    prev_ma20 = float(pd.to_numeric(model.get("ma20"), errors="coerce").iloc[-2]) if "ma20" in model.columns and len(model) >= 2 and pd.notna(model["ma20"].iloc[-2]) else None
    prev_ma50 = float(pd.to_numeric(model.get("ma50"), errors="coerce").iloc[-2]) if "ma50" in model.columns and len(model) >= 2 and pd.notna(model["ma50"].iloc[-2]) else None
    prev_ma200 = float(pd.to_numeric(model.get("ma200"), errors="coerce").iloc[-2]) if "ma200" in model.columns and len(model) >= 2 and pd.notna(model["ma200"].iloc[-2]) else None
    recent_high_20 = float(highs.tail(20).max()) if len(highs) >= 20 else None
    recent_low_20 = float(lows.tail(20).min()) if len(lows) >= 20 else None
    avg_volume_20 = float(volumes.tail(20).mean()) if len(volumes) >= 20 else None
    volume_ratio_20 = (float(volumes.iloc[-1]) / avg_volume_20) if avg_volume_20 and avg_volume_20 > 0 else None
    pct_vs_ma20 = ((close_now / ma20) - 1.0) * 100.0 if ma20 and ma20 > 0 else None
    pct_vs_ma50 = ((close_now / ma50) - 1.0) * 100.0 if ma50 and ma50 > 0 else None
    pct_from_20d_high = (
        ((recent_high_20 - close_now) / recent_high_20) * 100.0
        if recent_high_20 and recent_high_20 > 0
        else None
    )
    pct_from_20d_low = (
        ((close_now - recent_low_20) / recent_low_20) * 100.0
        if recent_low_20 and recent_low_20 > 0
        else None
    )
    recent_range_pct_10 = (
        ((float(highs.tail(10).max()) - float(lows.tail(10).min())) / close_now) * 100.0
        if len(highs) >= 10 and close_now > 0
        else None
    )
    recent_range_pct_20 = (
        ((float(highs.tail(20).max()) - float(lows.tail(20).min())) / close_now) * 100.0
        if len(highs) >= 20 and close_now > 0
        else None
    )

    trend_state = "mixed"
    if ma20 is not None and ma50 is not None and close_now > ma20 > ma50:
        trend_state = "uptrend"
    elif ma20 is not None and ma50 is not None and close_now < ma20 < ma50:
        trend_state = "downtrend"

    ma_signal = None
    if prev_ma20 is not None and prev_ma50 is not None and ma20 is not None and ma50 is not None:
        if prev_ma20 >= prev_ma50 and ma20 < ma50:
            ma_signal = "bearish_20_50_cross"
        elif prev_ma20 <= prev_ma50 and ma20 > ma50:
            ma_signal = "bullish_20_50_cross"
        elif ma20 < ma50:
            ma_signal = "20_below_50"
        elif ma20 > ma50:
            ma_signal = "20_above_50"

    ma_major_signal = None
    if prev_ma50 is not None and prev_ma200 is not None and ma50 is not None and ma200 is not None:
        if prev_ma50 >= prev_ma200 and ma50 < ma200:
            ma_major_signal = "death_cross"
        elif prev_ma50 <= prev_ma200 and ma50 > ma200:
            ma_major_signal = "golden_cross"
        elif ma50 < ma200:
            ma_major_signal = "50_below_200"
        elif ma50 > ma200:
            ma_major_signal = "50_above_200"

    ma_reclaim_state = None
    if prev_ma20 is not None and ma20 is not None:
        if prev_close <= prev_ma20 and close_now > ma20:
            ma_reclaim_state = "reclaimed_ma20"
        elif prev_close >= prev_ma20 and close_now < ma20:
            ma_reclaim_state = "lost_ma20"
    if prev_ma50 is not None and ma50 is not None:
        if prev_close <= prev_ma50 and close_now > ma50:
            ma_reclaim_state = "reclaimed_ma50"
        elif prev_close >= prev_ma50 and close_now < ma50:
            ma_reclaim_state = "lost_ma50"

    recent_gap_state = None
    recent_gap_days = None
    if len(model) >= 2:
        gap_rows = model[["high", "low"]].copy()
        gap_rows["high"] = pd.to_numeric(gap_rows["high"], errors="coerce")
        gap_rows["low"] = pd.to_numeric(gap_rows["low"], errors="coerce")
        start_idx = max(1, len(gap_rows) - 4)
        for idx in range(len(gap_rows) - 1, start_idx - 1, -1):
            prev_high = float(gap_rows.iloc[idx - 1]["high"])
            prev_low = float(gap_rows.iloc[idx - 1]["low"])
            bar_high = float(gap_rows.iloc[idx]["high"])
            bar_low = float(gap_rows.iloc[idx]["low"])
            if bar_low > prev_high:
                recent_gap_state = "bull_gap"
                recent_gap_days = len(gap_rows) - 1 - idx
                break
            if bar_high < prev_low:
                recent_gap_state = "bear_gap"
                recent_gap_days = len(gap_rows) - 1 - idx
                break

    level_context = "mid_range"
    if isinstance(pct_from_20d_low, (int, float)) and float(pct_from_20d_low) <= 2.5:
        level_context = "at_support"
    elif isinstance(pct_from_20d_high, (int, float)) and float(pct_from_20d_high) <= 2.5:
        level_context = "at_resistance"

    stretch_state = "normal"
    if isinstance(pct_vs_ma20, (int, float)):
        if float(pct_vs_ma20) >= 8.0:
            stretch_state = "extended_up"
        elif float(pct_vs_ma20) <= -8.0:
            stretch_state = "extended_down"

    support_level = None
    support_zone_low = None
    support_zone_high = None
    support_tier = None
    support_touches = None
    resistance_level = None
    resistance_zone_low = None
    resistance_zone_high = None
    resistance_tier = None
    resistance_touches = None
    pct_to_support = None
    pct_to_resistance = None
    range_position = None
    breakout_state = "none"
    level_event = "none"
    structure_state = "normal"

    def _pick_level(side: str) -> dict[str, Any] | None:
        if levels is None or levels.empty:
            return None
        pool = levels[levels["type"] == side].copy()
        if pool.empty:
            return None
        if side == "support":
            preferred = pool[pool["level"] <= close_now].sort_values("level", ascending=False)
        else:
            preferred = pool[pool["level"] >= close_now].sort_values("level", ascending=True)
        if preferred.empty:
            preferred = pool.sort_values(["dist", "touches", "recency_score"], ascending=[True, False, False])
        return preferred.iloc[0].to_dict() if not preferred.empty else None

    support = _pick_level("support")
    resistance = _pick_level("resistance")
    if support:
        support_level = round(float(support.get("level") or 0.0), 2)
        support_zone_low = round(float(support.get("zone_low") or 0.0), 2)
        support_zone_high = round(float(support.get("zone_high") or 0.0), 2)
        support_tier = str(support.get("tier") or "")
        support_touches = int(support.get("touches") or 0)
        if support_level > 0:
            pct_to_support = round(((close_now - support_level) / support_level) * 100.0, 2)
    if resistance:
        resistance_level = round(float(resistance.get("level") or 0.0), 2)
        resistance_zone_low = round(float(resistance.get("zone_low") or 0.0), 2)
        resistance_zone_high = round(float(resistance.get("zone_high") or 0.0), 2)
        resistance_tier = str(resistance.get("tier") or "")
        resistance_touches = int(resistance.get("touches") or 0)
        if resistance_level > 0:
            pct_to_resistance = round(((resistance_level - close_now) / resistance_level) * 100.0, 2)
    if (
        isinstance(support_level, (int, float))
        and isinstance(resistance_level, (int, float))
        and resistance_level > support_level
    ):
        range_position = round((close_now - support_level) / max(resistance_level - support_level, 0.01), 3)
    if isinstance(support_zone_low, (int, float)) and close_now < float(support_zone_low):
        level_context = "below_support"
    elif isinstance(resistance_zone_high, (int, float)) and close_now > float(resistance_zone_high):
        level_context = "above_resistance"
    elif isinstance(support_zone_high, (int, float)) and close_now <= float(support_zone_high):
        level_context = "at_support"
    elif isinstance(resistance_zone_low, (int, float)) and close_now >= float(resistance_zone_low):
        level_context = "at_resistance"
    elif isinstance(range_position, (int, float)):
        if float(range_position) <= 0.35:
            level_context = "closer_support"
        elif float(range_position) >= 0.65:
            level_context = "closer_resistance"
        else:
            level_context = "mid_range"

    if isinstance(resistance_zone_high, (int, float)):
        rzh = float(resistance_zone_high)
        if close_now > rzh:
            breakout_state = "breakout_up"
        elif high_now > rzh and close_now <= rzh:
            breakout_state = "bull_trap"
    if isinstance(support_zone_low, (int, float)):
        szl = float(support_zone_low)
        if close_now < szl:
            breakout_state = "breakout_down"
        elif low_now < szl and close_now >= szl and breakout_state == "none":
            breakout_state = "bear_trap"
    if breakout_state == "breakout_up":
        level_event = "resistance_breakout"
    elif breakout_state == "breakout_down":
        level_event = "support_breakdown"
    elif breakout_state == "bull_trap":
        level_event = "resistance_reject"
    elif breakout_state == "bear_trap":
        level_event = "support_reclaim"

    if (
        isinstance(recent_range_pct_10, (int, float))
        and isinstance(recent_range_pct_20, (int, float))
        and float(recent_range_pct_10) <= 7.0
        and float(recent_range_pct_20) <= 12.0
    ):
        if (
            (isinstance(range_position, (int, float)) and float(range_position) >= 0.58)
            or (isinstance(resistance_touches, int) and resistance_touches >= 2)
        ):
            structure_state = "tight_consolidation_high"
        elif (
            (isinstance(range_position, (int, float)) and float(range_position) <= 0.42)
            or (isinstance(support_touches, int) and support_touches >= 2)
        ):
            structure_state = "tight_consolidation_low"
        else:
            structure_state = "tight_consolidation_mid"

    if isinstance(pct_vs_ma20, (int, float)) and trend_state == "uptrend" and float(pct_vs_ma20) >= 7.5:
        if breakout_state == "breakout_up" or (
            isinstance(pct_from_20d_high, (int, float)) and float(pct_from_20d_high) <= 1.5
        ):
            structure_state = "parabolic_up"
    elif isinstance(pct_vs_ma20, (int, float)) and trend_state == "downtrend" and float(pct_vs_ma20) <= -7.5:
        if breakout_state == "breakout_down" or (
            isinstance(pct_from_20d_low, (int, float)) and float(pct_from_20d_low) <= 1.5
        ):
            structure_state = "parabolic_down"

    returns = closes.pct_change().dropna().tail(20)
    realized_vol_20 = None
    if len(returns) >= 5:
        rv = float(returns.std(ddof=0) * (252.0 ** 0.5) * 100.0)
        realized_vol_20 = round(rv, 2) if rv > 0 else None
    bb_width_20 = None
    if len(closes) >= 20:
        win = closes.tail(20)
        mean_20 = float(win.mean())
        sd_20 = float(win.std(ddof=0))
        if mean_20 > 0 and sd_20 >= 0:
            bb_width_20 = round(((4.0 * sd_20) / mean_20) * 100.0, 2)

    atr_pct_14 = None
    if "atr_pct" in model.columns and pd.notna(model["atr_pct"].iloc[-1]):
        atr_pct_14 = round(float(model["atr_pct"].iloc[-1]), 2)

    return {
        "close": round(close_now, 2),
        "ma20": round(ma20, 2) if ma20 is not None else None,
        "ma50": round(ma50, 2) if ma50 is not None else None,
        "ma100": round(ma100, 2) if ma100 is not None else None,
        "ma200": round(ma200, 2) if ma200 is not None else None,
        "avg_volume_20": round(avg_volume_20, 2) if avg_volume_20 is not None else None,
        "volume_ratio_20": round(volume_ratio_20, 2) if volume_ratio_20 is not None else None,
        "pct_vs_ma20": round(pct_vs_ma20, 2) if pct_vs_ma20 is not None else None,
        "pct_vs_ma50": round(pct_vs_ma50, 2) if pct_vs_ma50 is not None else None,
        "pct_from_20d_high": round(pct_from_20d_high, 2) if pct_from_20d_high is not None else None,
        "pct_from_20d_low": round(pct_from_20d_low, 2) if pct_from_20d_low is not None else None,
        "recent_range_pct_10": round(recent_range_pct_10, 2) if recent_range_pct_10 is not None else None,
        "recent_range_pct_20": round(recent_range_pct_20, 2) if recent_range_pct_20 is not None else None,
        "trend_state": trend_state,
        "ma_signal": ma_signal,
        "ma_major_signal": ma_major_signal,
        "ma_reclaim_state": ma_reclaim_state,
        "level_context": level_context,
        "stretch_state": stretch_state,
        "breakout_state": breakout_state,
        "level_event": level_event,
        "structure_state": structure_state,
        "recent_gap_state": recent_gap_state,
        "recent_gap_days": recent_gap_days,
        "support_level": support_level,
        "support_zone_low": support_zone_low,
        "support_zone_high": support_zone_high,
        "support_tier": support_tier,
        "support_touches": support_touches,
        "resistance_level": resistance_level,
        "resistance_zone_low": resistance_zone_low,
        "resistance_zone_high": resistance_zone_high,
        "resistance_tier": resistance_tier,
        "resistance_touches": resistance_touches,
        "pct_to_support": pct_to_support,
        "pct_to_resistance": pct_to_resistance,
        "range_position": range_position,
        "atr_pct_14": atr_pct_14,
        "realized_vol_20": realized_vol_20,
        "bb_width_20": bb_width_20,
    }


# ---------------------------------------------------------------------------
# Candle signal picker
# ---------------------------------------------------------------------------

def _pick_chart_candle_signal(
    candle_patterns: pd.DataFrame,
    asof_date: str,
) -> dict[str, Any]:
    if candle_patterns is None or candle_patterns.empty or not asof_date:
        return {}
    rows = candle_patterns.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest = rows[rows["date"] == asof_date].copy()
    if latest.empty:
        return {}
    latest["confidence"] = pd.to_numeric(latest.get("confidence"), errors="coerce").fillna(0.0)
    latest = latest.sort_values("confidence", ascending=False)
    top = latest.iloc[0].to_dict()
    return {
        "candle_pattern": top.get("pattern"),
        "candle_bias": top.get("bias") or "neutral",
        "candle_confidence": round(float(top.get("confidence") or 0.0), 2) if top.get("confidence") is not None else None,
    }


# ---------------------------------------------------------------------------
# Chart commentary
# ---------------------------------------------------------------------------

def _build_chart_commentary_payload(
    *,
    ticker: str,
    fund: dict[str, Any],
    model: pd.DataFrame,
    levels: pd.DataFrame,
    candle_patterns: pd.DataFrame,
    yolo_patterns: list[dict[str, Any]],
    yolo_audit: list[dict[str, Any]],
    setup_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``chart_commentary`` block for a dashboard payload."""
    if model.empty:
        return {}
    asof_date = str(pd.to_datetime(model["date"].iloc[-1], errors="coerce").strftime("%Y-%m-%d"))
    close_now = float(pd.to_numeric(model["close"], errors="coerce").iloc[-1])
    prev_close = float(pd.to_numeric(model["close"], errors="coerce").iloc[-2]) if len(model) >= 2 else close_now
    pct_change = (((close_now - prev_close) / prev_close) * 100.0) if prev_close > 0 else 0.0
    tech = _build_chart_technical_context(model, levels)
    candle = _pick_chart_candle_signal(candle_patterns, asof_date)
    primary_yolo = yolo_patterns[0] if yolo_patterns else None
    raw_json = None
    sector = "Unknown"
    industry = None
    if fund.get("raw_json"):
        try:
            raw_json = json.loads(str(fund.get("raw_json") or ""))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            LOG.debug("Failed to parse fund raw_json for %s: %s", ticker, exc)
            raw_json = None
    if isinstance(raw_json, dict):
        sector = str(raw_json.get("Sector") or raw_json.get("sector") or "Unknown").strip() or "Unknown"
        industry = str(raw_json.get("Industry") or raw_json.get("industry") or "").strip() or None

    row: dict[str, Any] = {
        "ticker": ticker,
        "score": 0.0,
        "confluence_score": 0.0,
        "setup_tier": "D",
        "sector": sector,
        "industry": industry,
        "pct_change": round(pct_change, 2),
        "discount_pct": fund.get("discount_pct"),
        "peg": fund.get("peg"),
        "near_52w_high": False,
        "near_52w_low": False,
        "yolo_pattern": primary_yolo.get("pattern") if primary_yolo else None,
        "yolo_confidence": primary_yolo.get("confidence") if primary_yolo else None,
        "yolo_age_days": primary_yolo.get("age_days") if primary_yolo else None,
        "yolo_timeframe": primary_yolo.get("timeframe") if primary_yolo else None,
        "yolo_first_seen_asof": primary_yolo.get("first_seen_asof") if primary_yolo else None,
        "yolo_last_seen_asof": primary_yolo.get("last_seen_asof") if primary_yolo else None,
        "yolo_snapshots_seen": primary_yolo.get("snapshots_seen") if primary_yolo else None,
        "yolo_current_streak": primary_yolo.get("current_streak") if primary_yolo else None,
        "yolo_signal_role": primary_yolo.get("signal_role") if primary_yolo else None,
        "candle_pattern": candle.get("candle_pattern"),
        "candle_bias": candle.get("candle_bias") or "neutral",
        "candle_confidence": candle.get("candle_confidence"),
    }
    row.update(tech)
    # Only recompute tier/narrative from scratch if there's NO report snapshot.
    # When a snapshot exists, the report's tier is authoritative (computed at
    # 22:00 UTC with full market data).  Recomputing live with partial/stale
    # data causes tier mismatches (e.g., report says A, live says C).
    if not (isinstance(setup_override, dict) and setup_override):
        row.update(_report_score_setup_from_confluence(row))
        row.update(_report_describe_setup(row))
    baseline_narrative = {
        "observation": row.get("observation"),
        "action": row.get("action"),
        "risk_note": row.get("risk_note"),
        "technical_read": row.get("technical_read"),
    }
    prompt_instruction = (
        "Summarize the chart in plain English using the freshest active YOLO pattern first, "
        "then recent persistent YOLO context, then any stale historical context. "
        "Explain trend, level location, candle confirmation, breakout/compression state, "
        "and whether the setup is actionable now or only on confirmation."
    )
    llm_meta: dict[str, Any] = {}
    try:
        llm_meta = llm_status()
    except Exception as exc:
        LOG.warning("Failed to get LLM status for chart commentary: %s", exc)
        llm_meta = {"error": str(exc)}
    row["narrative_source"] = "rule"
    if isinstance(setup_override, dict) and setup_override:
        override = dict(setup_override)
        if override.get("setup_tier") in (None, "") and override.get("tier") not in (None, ""):
            override["setup_tier"] = override.get("tier")
        if override.get("score") in (None, "") and override.get("setup_score") not in (None, ""):
            override["score"] = override.get("setup_score")
        if override.get("setup_family") in (None, "") and override.get("setup") not in (None, ""):
            override["setup_family"] = override.get("setup")
        if override.get("signal_bias") in (None, "") and override.get("bias") not in (None, ""):
            override["signal_bias"] = override.get("bias")
        if override.get("actionability") in (None, "") and override.get("state") not in (None, ""):
            override["actionability"] = override.get("state")
        if override.get("observation") in (None, "") and override.get("what_it_is") not in (None, ""):
            override["observation"] = override.get("what_it_is")
        if override.get("action") in (None, "") and override.get("next_step") not in (None, ""):
            override["action"] = override.get("next_step")
        if override.get("risk_note") in (None, "") and override.get("risk") not in (None, ""):
            override["risk_note"] = override.get("risk")
        if override.get("technical_read") in (None, "") and override.get("technical_context") not in (None, ""):
            override["technical_read"] = override.get("technical_context")
        if override.get("yolo_signal_role") in (None, "") and override.get("yolo_role") not in (None, ""):
            override["yolo_signal_role"] = override.get("yolo_role")
        for key in (
            "score",
            "confluence_score",
            "setup_tier",
            "setup_family",
            "signal_bias",
            "actionability",
            "action",
            "risk_note",
            "observation",
            "technical_read",
            "yolo_bias",
            "yolo_pattern",
            "yolo_confidence",
            "yolo_age_days",
            "yolo_timeframe",
            "yolo_first_seen_asof",
            "yolo_last_seen_asof",
            "yolo_snapshots_seen",
            "yolo_current_streak",
            "yolo_signal_role",
            "yolo_role",
            "yolo_recency",
            "yolo_direction_conflict",
            "yolo_conflict_strength",
            "yolo_confirmation_trend",
            "yolo_lifecycle_state",
            "yolo_latest_close_in_pattern",
            "breakout_state",
            "debate_v1",
            "debate_consensus_state",
            "debate_consensus_bias",
            "debate_agreement_score",
            "debate_disagreement_count",
            "debate_safety_adjustment",
        ):
            if key in override and override.get(key) is not None:
                row[key] = override.get(key)
        row["narrative_source"] = "report_snapshot"
        # Re-generate narrative text AFTER applying overrides so the
        # observation/action/risk reflect the snapshot tier, not the
        # independently recomputed tier.  This is the single-source-of-truth
        # fix: badges AND narrative now agree with the report.
        row.update(_report_describe_setup(row))
    else:
        llm_overrides = maybe_rewrite_setup_copy(row, source="chart_commentary")
        if llm_overrides:
            row.update(llm_overrides)
            row["narrative_source"] = "llm"
    final_narrative = {
        "observation": row.get("observation"),
        "action": row.get("action"),
        "risk_note": row.get("risk_note"),
        "technical_read": row.get("technical_read"),
    }

    primary_audit = yolo_audit[0] if yolo_audit else None
    current_active = [item for item in yolo_audit if bool(item.get("active_now"))]
    fresh_active = [item for item in current_active if str(item.get("signal_role") or "") in {"primary", "secondary"}]
    commentary_summary = {
        "latest_actionable_yolo": primary_yolo,
        "recent_persisting_yolo": fresh_active[:3],
        "historical_yolo_context": [item for item in yolo_audit if not bool(item.get("active_now"))][:3],
        "primary_audit_row": primary_audit,
    }
    yolo_role = (
        row.get("yolo_role")
        or row.get("yolo_signal_role")
        or (primary_audit.get("signal_role") if isinstance(primary_audit, dict) else None)
        or (primary_yolo.get("signal_role") if primary_yolo else None)
        or "none"
    )
    yolo_recency = (
        row.get("yolo_recency")
        or (primary_audit.get("yolo_recency") if isinstance(primary_audit, dict) else None)
        or (primary_yolo.get("yolo_recency") if primary_yolo else None)
        or "none"
    )
    yolo_confirmation_trend = (
        row.get("yolo_confirmation_trend")
        or (primary_audit.get("confirmation_trend") if isinstance(primary_audit, dict) else None)
        or "unknown"
    )
    yolo_lifecycle_state = (
        row.get("yolo_lifecycle_state")
        or (primary_audit.get("lifecycle_state") if isinstance(primary_audit, dict) else None)
        or "unknown"
    )
    yolo_latest_close_in_pattern = row.get("yolo_latest_close_in_pattern")
    if yolo_latest_close_in_pattern is None and isinstance(primary_audit, dict):
        yolo_latest_close_in_pattern = primary_audit.get("latest_close_in_pattern")
    llm_trace = {
        "narrative_source": row.get("narrative_source"),
        "baseline": baseline_narrative,
        "final": final_narrative,
        "changed_fields": {
            "observation": baseline_narrative.get("observation") != final_narrative.get("observation"),
            "action": baseline_narrative.get("action") != final_narrative.get("action"),
            "risk_note": baseline_narrative.get("risk_note") != final_narrative.get("risk_note"),
            "technical_read": baseline_narrative.get("technical_read") != final_narrative.get("technical_read"),
        },
        "llm_status": {
            "enabled": bool(llm_meta.get("enabled")),
            "ready": bool(llm_meta.get("ready")),
            "runtime_disabled": bool(llm_meta.get("runtime_disabled")),
            "runtime_disabled_remaining_sec": llm_meta.get("runtime_disabled_remaining_sec"),
            "provider": llm_meta.get("provider"),
            "temperature": llm_meta.get("temperature"),
            "max_tokens": llm_meta.get("max_tokens"),
            "timeout_sec": llm_meta.get("timeout_sec"),
            "api_version": llm_meta.get("api_version"),
        },
        "rewrite_instruction": prompt_instruction,
        "input_facts": {
            "setup_family": row.get("setup_family"),
            "signal_bias": row.get("signal_bias"),
            "trend_state": row.get("trend_state"),
            "level_context": row.get("level_context"),
            "breakout_state": row.get("breakout_state"),
            "structure_state": row.get("structure_state"),
            "candle_bias": row.get("candle_bias"),
            "yolo_pattern": row.get("yolo_pattern"),
            "yolo_bias": row.get("yolo_bias"),
            "yolo_recency": row.get("yolo_recency"),
            "yolo_age_days": row.get("yolo_age_days"),
            "yolo_direction_conflict": row.get("yolo_direction_conflict"),
            "yolo_conflict_strength": row.get("yolo_conflict_strength"),
        },
    }
    return {
        "ticker": ticker,
        "asof": asof_date,
        "score": row.get("score"),
        "setup_tier": row.get("setup_tier"),
        "signal_bias": row.get("signal_bias"),
        "setup_family": row.get("setup_family"),
        "observation": row.get("observation"),
        "actionability": row.get("actionability"),
        "action": row.get("action"),
        "risk_note": row.get("risk_note"),
        "technical_read": row.get("technical_read"),
        "narrative_source": row.get("narrative_source"),
        "primary_yolo_role": yolo_role,
        "primary_yolo_recency": yolo_recency,
        "primary_yolo_confirmation_trend": yolo_confirmation_trend,
        "primary_yolo_lifecycle_state": yolo_lifecycle_state,
        "primary_yolo_latest_close_in_pattern": bool(yolo_latest_close_in_pattern) if yolo_latest_close_in_pattern is not None else None,
        "yolo_bias": row.get("yolo_bias"),
        "yolo_direction_conflict": bool(row.get("yolo_direction_conflict")),
        "yolo_conflict_strength": row.get("yolo_conflict_strength"),
        "debate_v1": row.get("debate_v1"),
        "debate_consensus_state": row.get("debate_consensus_state"),
        "debate_consensus_bias": row.get("debate_consensus_bias"),
        "debate_agreement_score": row.get("debate_agreement_score"),
        "debate_disagreement_count": row.get("debate_disagreement_count"),
        "debate_safety_adjustment": row.get("debate_safety_adjustment"),
        "commentary_context": {
            "ticker": ticker,
            "asof": asof_date,
            "sector": sector,
            "industry": industry,
            "latest_close": close_now,
            "pct_change": round(pct_change, 2),
            "fundamentals": {
                "discount_pct": fund.get("discount_pct"),
                "peg": fund.get("peg"),
            },
            "technical": tech,
            "candle": candle,
            "yolo": commentary_summary,
        },
        "llm_ready_prompt": prompt_instruction,
        "narrative_trace": llm_trace,
    }


# ---------------------------------------------------------------------------
# Data freshness
# ---------------------------------------------------------------------------

_STALE_THRESHOLD_HOURS = 36  # accounts for weekends + buffer


def _compute_data_freshness(
    conn: sqlite3.Connection,
    ticker: str,
) -> dict[str, Any]:
    """Return freshness metadata for the latest price_daily row."""
    row = conn.execute(
        "SELECT MAX(date) FROM price_daily WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    latest_date_str = row[0] if row else None
    if not latest_date_str:
        return {
            "latest_price_date": None,
            "age_hours": None,
            "is_stale": True,
        }
    try:
        latest_date = dt.date.fromisoformat(str(latest_date_str)[:10])
    except ValueError:
        LOG.warning("Failed to parse latest price date=%s for %s", latest_date_str, ticker)
        return {
            "latest_price_date": str(latest_date_str),
            "age_hours": None,
            "is_stale": True,
        }
    # Combine with end-of-day (16:00 ET / 21:00 UTC) for age calculation
    latest_dt = dt.datetime.combine(latest_date, dt.time(21, 0), tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    age_hours = round((now - latest_dt).total_seconds() / 3600.0, 1)
    return {
        "latest_price_date": latest_date.isoformat(),
        "age_hours": age_hours,
        "is_stale": age_hours > _STALE_THRESHOLD_HOURS,
    }


# ---------------------------------------------------------------------------
# Shared model preparation (used by both quick and full builders)
# ---------------------------------------------------------------------------

def _prepare_model_and_features(
    conn: sqlite3.Connection,
    ticker: str,
    months: int,
) -> tuple[
    pd.DataFrame,   # prices (full)
    pd.DataFrame,   # model (features + pivots)
    pd.DataFrame,   # chart_rows (view window, date-formatted)
    pd.DataFrame,   # levels
    pd.DataFrame,   # gaps
    pd.DataFrame,   # trendlines
    pd.DataFrame,   # patterns
    pd.DataFrame,   # candle_patterns
    pd.DataFrame,   # hybrid_patterns
    pd.DataFrame,   # cv_proxy_patterns
    pd.DataFrame,   # hybrid_cv_compare
    pd.DataFrame,   # pattern_overlays
    dict[str, Any],  # fund
]:
    """Compute price model, features, and technical structure.

    Returns all intermediate artifacts needed by both the quick and
    full dashboard builders so the heavy DataFrame work runs once.
    """
    fund = get_latest_fundamentals(conn, ticker)
    prices = get_price_df(conn, ticker)
    if prices.empty:
        raise HTTPException(
            status_code=404, detail=f"No price data for {ticker}"
        )

    max_date = prices["date"].max()
    if months <= 0:
        calc_cutoff = prices["date"].min()
        view_cutoff = prices["date"].min()
    else:
        calc_cutoff = max_date - pd.DateOffset(months=max(6, months * 2))
        view_cutoff = max_date - pd.DateOffset(months=max(1, months))

    model_prices = prices[prices["date"] >= calc_cutoff].reset_index(
        drop=True
    )
    model = add_basic_features(model_prices, FEATURE_CFG)
    model = compute_pivots(model, left=3, right=3)

    last_close = float(model["close"].iloc[-1])
    levels_raw = build_levels_from_pivots(model, LEVEL_CFG)
    levels = select_target_levels(levels_raw, last_close, LEVEL_CFG)
    levels = add_fallback_levels(model, levels, last_close, LEVEL_CFG)

    gaps = select_gaps_for_display(
        detect_gaps(model),
        last_close=last_close,
        asof=max_date,
        cfg=GAP_CFG,
    )
    trendlines = detect_trendlines(
        model, last_close=last_close, cfg=TREND_CFG
    )
    patterns = detect_patterns(model, cfg=PATTERN_CFG)
    candle_patterns = detect_candlestick_patterns(model, cfg=CANDLE_CFG)
    hybrid_patterns = score_hybrid_patterns(
        model, patterns, candle_patterns, HYBRID_PATTERN_CFG
    )
    cv_proxy_patterns = detect_cv_proxy_patterns(model, cfg=CV_PROXY_CFG)
    hybrid_cv_compare = compare_hybrid_vs_cv(
        hybrid_patterns, cv_proxy_patterns, HYBRID_CV_CMP_CFG
    )
    pattern_overlays = build_pattern_overlays(
        patterns=patterns,
        hybrid_patterns=hybrid_patterns,
        cv_proxy_patterns=cv_proxy_patterns,
        max_rows=10,
    )

    chart_rows = model[model["date"] >= view_cutoff].copy()
    for col in ["date"]:
        chart_rows[col] = (
            pd.to_datetime(chart_rows[col], errors="coerce")
            .dt.strftime("%Y-%m-%d")
        )

    chart_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma20",
        "ma50",
        "ma100",
        "ma200",
        "ema20",
        "ema50",
        "ema100",
        "ema200",
        "atr",
        "atr_pct",
    ]
    chart_rows = chart_rows[
        [c for c in chart_cols if c in chart_rows.columns]
    ].copy()

    return (
        prices,
        model,
        chart_rows,
        levels,
        gaps,
        trendlines,
        patterns,
        candle_patterns,
        hybrid_patterns,
        cv_proxy_patterns,
        hybrid_cv_compare,
        pattern_overlays,
        fund,
    )


# ---------------------------------------------------------------------------
# Quick dashboard builder (fast path — no LLM / HMM)
# ---------------------------------------------------------------------------

def build_dashboard_quick_payload(
    conn: sqlite3.Connection,
    ticker: str,
    months: int,
) -> dict[str, Any]:
    """Build the fast-path dashboard payload (DB data + technical analysis).

    Skips chart commentary (LLM/debate) and HMM regime detection so the
    frontend can render the chart immediately.
    """
    ticker = ticker.upper().strip()
    (
        _prices,
        _model,
        chart_rows,
        levels,
        gaps,
        trendlines,
        patterns,
        candle_patterns,
        hybrid_patterns,
        cv_proxy_patterns,
        hybrid_cv_compare,
        pattern_overlays,
        fund,
    ) = _prepare_model_and_features(conn, ticker, months)

    market_date = dt.datetime.now(_MARKET_TZ).date()
    earnings_markers = get_ticker_earnings_markers(
        conn,
        ticker=ticker,
        market_date=market_date,
        forward_days=120,
        max_markers=3,
    )
    yolo_pats = get_yolo_patterns(conn, ticker)
    yolo_aud = get_yolo_audit(conn, ticker, limit=14)
    data_freshness = _compute_data_freshness(conn, ticker)

    return {
        "ticker": ticker,
        "asof": chart_rows["date"].iloc[-1],
        "fundamentals": fund,
        "options_summary": get_latest_options_summary(conn, ticker),
        "chart": _serialize_df(chart_rows),
        "levels": _serialize_df(levels),
        "gaps": _serialize_df(gaps),
        "trendlines": _serialize_df(trendlines),
        "patterns": _serialize_df(patterns),
        "candlestick_patterns": _serialize_df(candle_patterns),
        "hybrid_patterns": _serialize_df(hybrid_patterns),
        "cv_proxy_patterns": _serialize_df(cv_proxy_patterns),
        "hybrid_cv_compare": _serialize_df(hybrid_cv_compare),
        "pattern_overlays": _serialize_df(pattern_overlays),
        "yolo_patterns": yolo_pats,
        "yolo_audit": yolo_aud,
        "earnings_markers": earnings_markers,
        "data_sources": get_data_sources(conn, ticker),
        "data_freshness": data_freshness,
        "meta": {
            "schema": ["date", "open", "high", "low", "close", "volume"],
            "config": {
                "level": LEVEL_CFG.__dict__,
                "gap": GAP_CFG.__dict__,
                "trendline": TREND_CFG.__dict__,
                "pattern": PATTERN_CFG.__dict__,
                "candlestick_pattern": CANDLE_CFG.__dict__,
                "hybrid_pattern": HYBRID_PATTERN_CFG.__dict__,
                "cv_proxy_pattern": CV_PROXY_CFG.__dict__,
                "hybrid_cv_compare": HYBRID_CV_CMP_CFG.__dict__,
            },
        },
    }


# ---------------------------------------------------------------------------
# Commentary-only builder (slow path — LLM + debate + HMM)
# ---------------------------------------------------------------------------

def build_commentary_payload(
    conn: sqlite3.Connection,
    ticker: str,
    months: int,
    *,
    report_dir: Path,
    report_generated_ts: str | None = None,
) -> dict[str, Any]:
    """Build the slow-path commentary payload (LLM narrative + HMM regime).

    Designed to be called after the quick endpoint has already returned
    chart data to the frontend.
    """
    ticker = ticker.upper().strip()
    (
        prices,
        model,
        _chart_rows,
        levels,
        _gaps,
        _trendlines,
        _patterns,
        candle_patterns,
        _hybrid_patterns,
        _cv_proxy_patterns,
        _hybrid_cv_compare,
        _pattern_overlays,
        fund,
    ) = _prepare_model_and_features(conn, ticker, months)

    yolo_pats = get_yolo_patterns(conn, ticker)
    yolo_aud = get_yolo_audit(conn, ticker, limit=14)
    setup_override = latest_report_setup_for_ticker(
        report_dir,
        ticker,
        generated_ts=report_generated_ts,
    )
    chart_commentary = _build_chart_commentary_payload(
        ticker=ticker,
        fund=fund,
        model=model,
        levels=levels,
        candle_patterns=candle_patterns,
        yolo_patterns=yolo_pats,
        yolo_audit=yolo_aud,
        setup_override=setup_override,
    )

    # HMM regime — use cached from nightly report, fall back to live compute
    hmm_regime = latest_report_hmm_for_ticker(report_dir, ticker, generated_ts=report_generated_ts)
    if hmm_regime is None:
        try:
            hmm_regime = hmm_predict_regimes(prices, ticker=ticker)
        except Exception as exc:
            LOG.warning("HMM regime detection failed for %s: %s", ticker, exc)

    return {
        "ticker": ticker,
        "chart_commentary": chart_commentary,
        "hmm_regime": hmm_regime,
        "report_generated_ts": report_generated_ts,
    }


# ---------------------------------------------------------------------------
# Main dashboard builder (backward-compatible full payload)
# ---------------------------------------------------------------------------

def build_dashboard_payload(
    conn: sqlite3.Connection,
    ticker: str,
    months: int,
    *,
    report_dir: Path,
    report_generated_ts: str | None = None,
) -> dict[str, Any]:
    """Build the complete ``/api/dashboard/{ticker}`` response payload.

    Parameters
    ----------
    conn:
        Open SQLite connection with Row factory.
    ticker:
        Ticker symbol (uppercased internally).
    months:
        Chart lookback in months.
    report_dir:
        Path to report JSON directory.
    report_generated_ts:
        Optional pinned report timestamp for snapshot override.
    """
    ticker = ticker.upper().strip()
    (
        prices,
        model,
        chart_rows,
        levels,
        gaps,
        trendlines,
        patterns,
        candle_patterns,
        hybrid_patterns,
        cv_proxy_patterns,
        hybrid_cv_compare,
        pattern_overlays,
        fund,
    ) = _prepare_model_and_features(conn, ticker, months)

    market_date = dt.datetime.now(_MARKET_TZ).date()
    earnings_markers = get_ticker_earnings_markers(
        conn,
        ticker=ticker,
        market_date=market_date,
        forward_days=120,
        max_markers=3,
    )
    yolo_pats = get_yolo_patterns(conn, ticker)
    yolo_aud = get_yolo_audit(conn, ticker, limit=14)
    setup_override = latest_report_setup_for_ticker(
        report_dir,
        ticker,
        generated_ts=report_generated_ts,
    )
    chart_commentary = _build_chart_commentary_payload(
        ticker=ticker,
        fund=fund,
        model=model,
        levels=levels,
        candle_patterns=candle_patterns,
        yolo_patterns=yolo_pats,
        yolo_audit=yolo_aud,
        setup_override=setup_override,
    )

    # HMM regime — use cached from nightly report, fall back to live compute
    hmm_regime = latest_report_hmm_for_ticker(report_dir, ticker, generated_ts=report_generated_ts)
    if hmm_regime is None:
        try:
            hmm_regime = hmm_predict_regimes(prices, ticker=ticker)
        except Exception as exc:
            LOG.warning("HMM regime detection failed for %s: %s", ticker, exc)

    # Data freshness indicator
    data_freshness = _compute_data_freshness(conn, ticker)

    return {
        "ticker": ticker,
        "asof": chart_rows["date"].iloc[-1],
        "fundamentals": fund,
        "options_summary": get_latest_options_summary(conn, ticker),
        "chart": _serialize_df(chart_rows),
        "levels": _serialize_df(levels),
        "gaps": _serialize_df(gaps),
        "trendlines": _serialize_df(trendlines),
        "patterns": _serialize_df(patterns),
        "candlestick_patterns": _serialize_df(candle_patterns),
        "hybrid_patterns": _serialize_df(hybrid_patterns),
        "cv_proxy_patterns": _serialize_df(cv_proxy_patterns),
        "hybrid_cv_compare": _serialize_df(hybrid_cv_compare),
        "pattern_overlays": _serialize_df(pattern_overlays),
        "yolo_patterns": yolo_pats,
        "yolo_audit": yolo_aud,
        "chart_commentary": chart_commentary,
        "hmm_regime": hmm_regime,
        "earnings_markers": earnings_markers,
        "report_generated_ts": report_generated_ts,
        "data_sources": get_data_sources(conn, ticker),
        "data_freshness": data_freshness,
        "meta": {
            "schema": ["date", "open", "high", "low", "close", "volume"],
            "config": {
                "level": LEVEL_CFG.__dict__,
                "gap": GAP_CFG.__dict__,
                "trendline": TREND_CFG.__dict__,
                "pattern": PATTERN_CFG.__dict__,
                "candlestick_pattern": CANDLE_CFG.__dict__,
                "hybrid_pattern": HYBRID_PATTERN_CFG.__dict__,
                "cv_proxy_pattern": CV_PROXY_CFG.__dict__,
                "hybrid_cv_compare": HYBRID_CV_CMP_CFG.__dict__,
            },
        },
    }

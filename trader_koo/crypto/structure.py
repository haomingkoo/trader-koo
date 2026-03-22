"""Crypto structure analysis: levels, trendlines, and intraday regime context."""
from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger("trader_koo.crypto.structure")

import numpy as np
import pandas as pd

from trader_koo.crypto.models import CryptoBar
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.structure.hmm_regime import predict_regimes, predict_directional_regimes
from trader_koo.structure.levels import (
    LevelConfig,
    add_fallback_levels,
    build_levels_from_pivots,
    select_target_levels,
)
from trader_koo.structure.trendlines import TrendlineConfig, detect_trendlines

FEATURE_CFG = FeatureConfig()
LEVEL_CFG = LevelConfig(
    level_tol_atr=0.8,
    zone_half_width_atr=0.45,
    min_zone_width=0.0001,
    primary_each_side=3,
    secondary_each_side=2,
    max_dist_primary_pct=0.15,
    max_dist_secondary_pct=0.35,
    near_side_tolerance_pct=0.025,
    fallback_lookback_bars=240,
)
TREND_CFG = TrendlineConfig(
    lookback_bars=240,
    min_points=3,
    max_lines_per_side=2,
    touch_tol_atr=0.9,
    max_slope_pct_per_day=0.5,
    recency_half_life_days=7,
)
_PIVOT_WINDOWS: dict[str, tuple[int, int]] = {
    "1m": (4, 4),
    "5m": (4, 4),
    "15m": (4, 4),
    "30m": (4, 4),
    "1h": (3, 3),
    "2h": (3, 3),
    "4h": (3, 3),
    "6h": (3, 3),
    "12h": (2, 2),
    "1d": (2, 2),
    "1w": (2, 2),
}


def _bars_to_frame(bars: list[CryptoBar]) -> pd.DataFrame:
    rows = [
        {
            "date": pd.Timestamp(bar.timestamp),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        for bar in bars
    ]
    return pd.DataFrame(rows)


def _serialize_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out.replace({np.nan: None})
    return out.to_dict(orient="records")


def _first_level(levels: pd.DataFrame, side: str) -> dict[str, Any] | None:
    if levels.empty:
        return None
    side_levels = levels[levels["type"] == side]
    if side_levels.empty:
        return None
    return side_levels.iloc[0].to_dict()


def _build_context(model: pd.DataFrame, levels: pd.DataFrame) -> dict[str, Any]:
    closes = pd.to_numeric(model["close"], errors="coerce")
    latest_close = float(closes.iloc[-1])
    ma20 = float(model["ma20"].iloc[-1]) if "ma20" in model.columns and pd.notna(model["ma20"].iloc[-1]) else None
    ma50 = float(model["ma50"].iloc[-1]) if "ma50" in model.columns and pd.notna(model["ma50"].iloc[-1]) else None
    atr_pct = (
        round(float(model["atr_pct"].iloc[-1]) * 100.0, 2)
        if "atr_pct" in model.columns and pd.notna(model["atr_pct"].iloc[-1])
        else None
    )
    momentum_20 = None
    if len(closes) >= 21 and float(closes.iloc[-21]) != 0:
        momentum_20 = round(((latest_close / float(closes.iloc[-21])) - 1.0) * 100.0, 2)
    realized_vol = None
    if len(closes) >= 21:
        rets = closes.pct_change().dropna().tail(20)
        if not rets.empty:
            realized_vol = round(float(rets.std(ddof=0)) * 100.0, 2)

    support = _first_level(levels, "support")
    resistance = _first_level(levels, "resistance")
    support_level = float(support["level"]) if support and support.get("level") is not None else None
    resistance_level = float(resistance["level"]) if resistance and resistance.get("level") is not None else None
    support_zone_low = float(support["zone_low"]) if support and support.get("zone_low") is not None else None
    support_zone_high = float(support["zone_high"]) if support and support.get("zone_high") is not None else None
    resistance_zone_low = float(resistance["zone_low"]) if resistance and resistance.get("zone_low") is not None else None
    resistance_zone_high = float(resistance["zone_high"]) if resistance and resistance.get("zone_high") is not None else None

    pct_to_support = (
        round(((latest_close - support_level) / support_level) * 100.0, 2)
        if isinstance(support_level, float) and support_level > 0
        else None
    )
    pct_to_resistance = (
        round(((resistance_level - latest_close) / resistance_level) * 100.0, 2)
        if isinstance(resistance_level, float) and resistance_level > 0
        else None
    )
    range_position = None
    if (
        isinstance(support_level, float)
        and isinstance(resistance_level, float)
        and resistance_level > support_level
    ):
        range_position = round(
            (latest_close - support_level) / max(resistance_level - support_level, 0.000001),
            3,
        )

    if isinstance(support_zone_low, float) and latest_close < support_zone_low:
        level_context = "below_support"
    elif isinstance(resistance_zone_high, float) and latest_close > resistance_zone_high:
        level_context = "above_resistance"
    elif isinstance(support_zone_high, float) and latest_close <= support_zone_high:
        level_context = "at_support"
    elif isinstance(resistance_zone_low, float) and latest_close >= resistance_zone_low:
        level_context = "at_resistance"
    elif isinstance(pct_to_support, float) and isinstance(pct_to_resistance, float):
        level_context = "closer_support" if pct_to_support <= pct_to_resistance else "closer_resistance"
    else:
        level_context = "mid_range"

    if isinstance(ma20, float) and isinstance(ma50, float):
        if latest_close > ma20 > ma50:
            ma_trend = "bullish"
        elif latest_close < ma20 < ma50:
            ma_trend = "bearish"
        else:
            ma_trend = "mixed"
    else:
        ma_trend = "unknown"

    return {
        "latest_close": round(latest_close, 6),
        "support_level": round(support_level, 6) if support_level is not None else None,
        "support_zone_low": round(support_zone_low, 6) if support_zone_low is not None else None,
        "support_zone_high": round(support_zone_high, 6) if support_zone_high is not None else None,
        "resistance_level": round(resistance_level, 6) if resistance_level is not None else None,
        "resistance_zone_low": round(resistance_zone_low, 6) if resistance_zone_low is not None else None,
        "resistance_zone_high": round(resistance_zone_high, 6) if resistance_zone_high is not None else None,
        "pct_to_support": pct_to_support,
        "pct_to_resistance": pct_to_resistance,
        "range_position": range_position,
        "level_context": level_context,
        "ma_trend": ma_trend,
        "ma20": round(ma20, 6) if ma20 is not None else None,
        "ma50": round(ma50, 6) if ma50 is not None else None,
        "atr_pct": atr_pct,
        "momentum_20": momentum_20,
        "realized_vol_20": realized_vol,
    }


def build_crypto_structure(
    symbol: str,
    bars: list[CryptoBar],
    *,
    interval: str = "1m",
    include_hmm: bool = True,
) -> dict[str, Any]:
    if not bars:
        return {
            "symbol": symbol,
            "interval": interval,
            "bar_count": 0,
            "levels": [],
            "trendlines": [],
            "hmm_regime": None,
            "hmm_directional": None,
            "context": {
                "latest_close": None,
                "support_level": None,
                "support_zone_low": None,
                "support_zone_high": None,
                "resistance_level": None,
                "resistance_zone_low": None,
                "resistance_zone_high": None,
                "pct_to_support": None,
                "pct_to_resistance": None,
                "range_position": None,
                "level_context": "unavailable",
                "ma_trend": "unknown",
                "ma20": None,
                "ma50": None,
                "atr_pct": None,
                "momentum_20": None,
                "realized_vol_20": None,
            },
        }

    prices = _bars_to_frame(bars).sort_values("date").reset_index(drop=True)
    model = add_basic_features(prices, FEATURE_CFG)
    left, right = _PIVOT_WINDOWS.get(interval, (3, 3))
    model = compute_pivots(model, left=left, right=right)

    last_close = float(model["close"].iloc[-1])
    levels_raw = build_levels_from_pivots(model, LEVEL_CFG)
    levels = select_target_levels(levels_raw, last_close, LEVEL_CFG)
    levels = add_fallback_levels(model, levels, last_close, LEVEL_CFG)
    trendlines = detect_trendlines(model, last_close=last_close, cfg=TREND_CFG)

    hmm_regime = None
    hmm_directional = None
    if include_hmm:
        hmm_input = prices[["date", "open", "high", "low", "close", "volume"]].copy()
        hmm_regime = predict_regimes(
            hmm_input,
            lookback_days=min(len(hmm_input), 720),
            ticker=f"crypto:{symbol}:{interval}",
        )
        try:
            hmm_directional = predict_directional_regimes(
                hmm_input,
                lookback_days=min(len(hmm_input), 720),
                ticker=f"crypto:{symbol}:{interval}",
            )
        except Exception as exc:
            LOG.debug("Directional HMM failed for %s: %s", symbol, exc)

    return {
        "symbol": symbol,
        "interval": interval,
        "bar_count": int(len(model)),
        "levels": _serialize_df(levels),
        "trendlines": _serialize_df(trendlines),
        "hmm_regime": hmm_regime,
        "hmm_directional": hmm_directional,
        "context": _build_context(model, levels),
    }

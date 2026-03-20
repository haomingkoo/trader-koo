"""Market context: volatility inputs, regime analysis, technical context per ticker."""
from __future__ import annotations

import logging
import math
import os
import sqlite3
from collections import defaultdict
from typing import Any

import pandas as pd

from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.structure.levels import (
    LevelConfig,
    add_fallback_levels,
    build_levels_from_pivots,
    select_target_levels,
)
from trader_koo.structure.vix_analysis import (
    calculate_compression_thresholds,
    calculate_term_structure,
    calculate_vix_percentile,
    detect_compression_signal,
    format_compression_thresholds_display,
    get_percentile_color,
    should_show_volatility_warning,
)
from trader_koo.structure.vix_patterns import (
    VIXTrapReclaimConfig,
    detect_vix_trap_reclaim_patterns,
    get_pattern_glossary,
)
from trader_koo.report.utils import (
    _clamp,
    _percentile_rank,
    _stdev,
    table_exists,
)

LOG = logging.getLogger(__name__)

# Module-level warning accumulator — set by generator before calling these functions.
_report_warnings: list[str] = []

REPORT_FEATURE_CFG = FeatureConfig()
REPORT_LEVEL_CFG = LevelConfig()


def _fetch_symbol_ohlcv(
    conn: sqlite3.Connection, ticker: str, limit: int = 120
) -> list[dict[str, float | str]]:
    rows = conn.execute(
        """
        SELECT date, CAST(open AS REAL), CAST(high AS REAL), CAST(low AS REAL), CAST(close AS REAL), CAST(volume AS REAL)
        FROM price_daily
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, int(max(10, limit))),
    ).fetchall()
    out: list[dict[str, float | str]] = []
    for r in reversed(rows):
        try:
            close_v = float(r[4])
        except (TypeError, ValueError):
            continue
        if close_v <= 0:
            continue
        try:
            open_v = float(r[1])
            high_v = float(r[2])
            low_v = float(r[3])
            vol_v = float(r[5] or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "date": str(r[0]),
                "open": open_v,
                "high": high_v,
                "low": low_v,
                "close": close_v,
                "volume": vol_v,
            }
        )
    return out


def _fetch_volatility_inputs(conn: sqlite3.Connection) -> tuple[dict[str, dict[str, float | None]], dict[str, Any]]:
    """Build per-ticker volatility features and a market volatility context."""
    by_ticker: dict[str, dict[str, float | None]] = {}
    market_ctx: dict[str, Any] = {}
    if not table_exists(conn, "price_daily"):
        return by_ticker, market_ctx

    rows = conn.execute(
        """
        SELECT ticker, date, CAST(high AS REAL), CAST(low AS REAL), CAST(close AS REAL)
        FROM (
            SELECT ticker, date, high, low, close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM price_daily
        )
        WHERE rn <= 40
        ORDER BY ticker, date
        """
    ).fetchall()
    bucket: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        ticker = str(r[0] or "").upper().strip()
        if not ticker:
            continue
        high = r[2]
        low = r[3]
        close = r[4]
        if high is None or low is None or close is None:
            continue
        try:
            h = float(high)
            l = float(low)
            c = float(close)
        except (TypeError, ValueError):
            continue
        if not (h > 0 and l > 0 and c > 0):
            continue
        bucket[ticker].append((h, l, c))

    for ticker, bars in bucket.items():
        if len(bars) < 3:
            continue
        highs = [b[0] for b in bars]
        lows = [b[1] for b in bars]
        closes = [b[2] for b in bars]

        returns: list[float] = []
        trs: list[float] = []
        for i in range(1, len(closes)):
            prev_close = closes[i - 1]
            close = closes[i]
            if prev_close > 0:
                returns.append((close / prev_close) - 1.0)
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev_close),
                abs(lows[i] - prev_close),
            )
            trs.append(tr)

        atr_pct_14: float | None = None
        if len(trs) >= 14 and closes[-1] > 0:
            atr_14 = sum(trs[-14:]) / 14.0
            atr_pct_14 = (atr_14 / closes[-1]) * 100.0

        realized_vol_20: float | None = None
        if len(returns) >= 20:
            ret_sd = _stdev(returns[-20:])
            if ret_sd is not None:
                realized_vol_20 = ret_sd * math.sqrt(252.0) * 100.0

        bb_width_20: float | None = None
        if len(closes) >= 20:
            win = closes[-20:]
            mean_20 = sum(win) / 20.0
            sd_20 = _stdev(win)
            if mean_20 > 0 and sd_20 is not None:
                bb_width_20 = ((4.0 * sd_20) / mean_20) * 100.0

        by_ticker[ticker] = {
            "atr_pct_14": round(atr_pct_14, 2) if atr_pct_14 is not None else None,
            "realized_vol_20": round(realized_vol_20, 2) if realized_vol_20 is not None else None,
            "bb_width_20": round(bb_width_20, 2) if bb_width_20 is not None else None,
        }

    vix_rows = conn.execute(
        """
        SELECT CAST(close AS REAL)
        FROM price_daily
        WHERE ticker = '^VIX' AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT 252
        """
    ).fetchall()
    vix_vals = []
    for r in vix_rows:
        try:
            v = float(r[0])
        except (TypeError, ValueError):
            continue
        if v > 0:
            vix_vals.append(v)
    if vix_vals:
        vix_now = vix_vals[0]
        rank_le = sum(1 for v in vix_vals if v <= vix_now)
        vix_pctile = (rank_le / len(vix_vals)) * 100.0
        market_ctx = {
            "vix_close": round(vix_now, 2),
            "vix_percentile_1y": round(vix_pctile, 1),
            "vix_points": len(vix_vals),
        }

    return by_ticker, market_ctx


def _build_regime_llm_commentary(regime: dict[str, Any]) -> dict[str, Any]:
    safe_regime = regime if isinstance(regime, dict) else {}
    vix = safe_regime.get("vix") if isinstance(safe_regime.get("vix"), dict) else {}
    health = safe_regime.get("health") if isinstance(safe_regime.get("health"), dict) else {}
    summary = str(safe_regime.get("summary") or "").strip() or "Regime context is unavailable for this snapshot."
    health_state = str(health.get("state") or "neutral").strip().lower()
    warnings = [str(w).strip() for w in (health.get("warnings") or []) if str(w).strip()]
    risk_note = warnings[0] if warnings else "context_only_signal"
    asof_date = str(safe_regime.get("asof_date") or "").strip() or None

    action = "No urgency."
    if health_state == "risk_on":
        action = "Backdrop supports continuation setups and risk-on positioning."
    elif health_state == "risk_off":
        action = "Reduce exposure; favour hedged or defensive positioning."

    source = "regime_context"
    if asof_date:
        source = f"regime_context_{asof_date}"

    row: dict[str, Any] = {}
    try:
        from trader_koo.llm_narrative import llm_enabled, maybe_rewrite_setup_copy
        if llm_enabled():
            row = maybe_rewrite_setup_copy(
                {
                    "observation": summary,
                    "action": action,
                    "risk_note": risk_note,
                    "signal_bias": health_state.replace("_", " "),
                    "setup_family": "regime_context",
                    "ticker": "^VIX",
                    "score": float(health.get("score") or 50.0),
                },
                source=source,
            ) or {}
    except Exception:
        row = {}

    return {
        "source": source,
        "observation": str(row.get("observation") or summary),
        "action": str(row.get("action") or action),
        "risk_note": str(row.get("risk_note") or risk_note or "context_only_signal"),
    }


def _build_regime_context(conn: sqlite3.Connection) -> dict[str, Any]:
    regime: dict[str, Any] = {
        "context_only": True,
        "asof_date": None,
        "summary": "",
        "vix": {},
        "ma_matrix": [],
        "comparison": {},
        "participation": [],
        "overall": {},
        "health": {},
        "timeframes": [],
        "levels": [],
    }
    if not table_exists(conn, "price_daily"):
        return regime

    # --- VIX structure context ---
    vix_rows = _fetch_symbol_ohlcv(conn, "^VIX", limit=280)
    vix_latest = None
    if len(vix_rows) >= 55:
        closes = [float(r["close"]) for r in vix_rows]
        latest = closes[-1]
        prev = closes[-2]
        vix_latest = latest
        ma20 = sum(closes[-20:]) / 20.0
        ma50 = sum(closes[-50:]) / 50.0
        ma100 = sum(closes[-100:]) / 100.0 if len(closes) >= 100 else None
        pct_vs_ma20 = ((latest / ma20) - 1.0) * 100.0 if ma20 > 0 else None
        pct_vs_ma50 = ((latest / ma50) - 1.0) * 100.0 if ma50 > 0 else None
        pct_vs_ma100 = ((latest / ma100) - 1.0) * 100.0 if isinstance(ma100, (int, float)) and ma100 > 0 else None

        if latest >= ma20 and latest >= ma50:
            ma_state = "above_ma20_ma50"
        elif latest >= ma20 and latest < ma50:
            ma_state = "above_ma20_below_ma50"
        elif latest < ma20 and latest >= ma50:
            ma_state = "below_ma20_above_ma50"
        else:
            ma_state = "below_ma20_ma50"

        ma_cross_state = "ma20_above_ma50" if ma20 > ma50 else "ma20_below_ma50"
        ma20_vs_ma50 = ((ma20 / ma50) - 1.0) * 100.0 if ma50 > 0 else None
        ma50_vs_ma100 = ((ma50 / ma100) - 1.0) * 100.0 if isinstance(ma100, (int, float)) and ma100 > 0 else None

        bb_width_series: list[float] = []
        for idx in range(19, len(closes)):
            win = closes[idx - 19 : idx + 1]
            mean_20 = sum(win) / 20.0
            sd_20 = _stdev(win)
            if mean_20 > 0 and sd_20 is not None:
                bb_width_series.append(((4.0 * sd_20) / mean_20) * 100.0)
        bb_width_20 = bb_width_series[-1] if bb_width_series else None
        bb_width_pctile = _percentile_rank(bb_width_series, bb_width_20)

        # Calculate adaptive compression thresholds (Requirements 13.1-13.4)
        compression_thresholds = calculate_compression_thresholds(conn)

        # Detect compression using adaptive thresholds (Requirement 13.6)
        compression_state, compression_labeled = detect_compression_signal(
            bb_width_pctile, compression_thresholds
        )

        # 10-day range break context (excluding latest bar for reference range).
        prev_window = closes[-11:-1] if len(closes) >= 11 else closes[:-1]
        breakout_state = "inside_range"
        if prev_window:
            hi_prev = max(prev_window)
            lo_prev = min(prev_window)
            if latest >= (hi_prev * 1.002):
                breakout_state = "range_breakout_up"
            elif latest <= (lo_prev * 0.998):
                breakout_state = "range_breakdown_down"

        if ma_state == "above_ma20_ma50" and breakout_state == "range_breakout_up":
            risk_state = "risk_off_pressure_rising"
        elif ma_state == "below_ma20_ma50" and breakout_state == "range_breakdown_down":
            risk_state = "risk_on_relief"
        elif compression_state == "compression":
            risk_state = "coiled_regime"
        else:
            risk_state = "mixed_regime"

        # Calculate term structure with fallback (Requirements 9.1-9.6)
        term_structure = calculate_term_structure(conn)

        # Extract values for backward compatibility
        vix3m_latest = term_structure.vix_3m or term_structure.vix_6m
        term_structure_ratio = None
        term_structure_state = "unavailable"

        if term_structure.source != "unavailable":
            if term_structure.vix_3m:
                term_structure_ratio = term_structure.vix_3m / latest
            elif term_structure.vix_6m:
                term_structure_ratio = term_structure.vix_6m / latest

            if term_structure_ratio:
                if term_structure_ratio >= 1.03:
                    term_structure_state = "contango"
                elif term_structure_ratio <= 0.97:
                    term_structure_state = "backwardation"
                else:
                    term_structure_state = "flat"

        # Multi-timeframe VIX structure using rolling windows on daily bars.
        tf_rows: list[dict[str, Any]] = []
        for label, bars in (("1W", 5), ("2W", 10), ("1M", 21), ("3M", 63)):
            if len(closes) < (bars + 1):
                continue
            start = closes[-(bars + 1)]
            if start <= 0:
                continue
            change_pct = ((latest / start) - 1.0) * 100.0
            win = closes[-bars:]
            hi = max(win)
            lo = min(win)
            range_pos = ((latest - lo) / max(hi - lo, 1e-9)) * 100.0
            if change_pct >= 8.0:
                structure = "vol_expansion"
            elif change_pct <= -8.0:
                structure = "vol_compression"
            else:
                structure = "rangebound"
            if range_pos >= 80.0:
                location = "near_range_high"
            elif range_pos <= 20.0:
                location = "near_range_low"
            else:
                location = "mid_range"
            tf_rows.append(
                {
                    "timeframe": label,
                    "lookback_days": bars,
                    "change_pct": round(change_pct, 2),
                    "range_high": round(hi, 2),
                    "range_low": round(lo, 2),
                    "range_position_pct": round(range_pos, 1),
                    "structure": structure,
                    "location": location,
                }
            )
        regime["timeframes"] = tf_rows

        # VIX support/resistance using the same level engine as chart mode.
        levels_out: list[dict[str, Any]] = []
        level_source = "shared_level_engine"
        try:
            model = pd.DataFrame(vix_rows).copy()
            if len(model) >= REPORT_FEATURE_CFG.min_bars:
                model = add_basic_features(model, REPORT_FEATURE_CFG)
                model = compute_pivots(model, REPORT_FEATURE_CFG)
                levels_raw = build_levels_from_pivots(model, REPORT_LEVEL_CFG)
                levels = select_target_levels(levels_raw, float(latest), REPORT_LEVEL_CFG)
                levels = add_fallback_levels(model, levels, float(latest), REPORT_LEVEL_CFG)
                if levels is not None and not levels.empty:
                    level_pool = levels.copy()

                    def _ordered_side(side: str) -> pd.DataFrame:
                        pool = level_pool[level_pool["type"] == side].copy()
                        if pool.empty:
                            return pool
                        if side == "support":
                            near = pool[pool["level"] <= latest].sort_values("level", ascending=False)
                            far = pool[pool["level"] > latest].sort_values("level", ascending=True)
                        else:
                            near = pool[pool["level"] >= latest].sort_values("level", ascending=True)
                            far = pool[pool["level"] < latest].sort_values("level", ascending=False)
                        return pd.concat([near, far], ignore_index=True).head(3)

                    selected = pd.concat(
                        [_ordered_side("support"), _ordered_side("resistance")],
                        ignore_index=True,
                    )
                    for _, lv in selected.iterrows():
                        level_v = float(lv.get("level") or 0.0)
                        if level_v <= 0:
                            continue
                        side = str(lv.get("type") or "")
                        if side == "support":
                            dist_pct = ((latest - level_v) / level_v) * 100.0
                        else:
                            dist_pct = ((level_v - latest) / level_v) * 100.0
                        levels_out.append(
                            {
                                "type": side,
                                "level": round(level_v, 2),
                                "zone_low": round(float(lv.get("zone_low") or level_v), 2),
                                "zone_high": round(float(lv.get("zone_high") or level_v), 2),
                                "tier": str(lv.get("tier") or ""),
                                "touches": int(lv.get("touches") or 0),
                                "distance_pct": round(dist_pct, 2),
                                "last_touch_date": str(lv.get("last_touch_date") or ""),
                                "source": str(lv.get("source") or "pivot_cluster"),  # Requirement 10.1, 10.3
                            }
                        )
        except Exception as exc:
            LOG.error("Level computation failed for ticker: %s", exc, exc_info=True)
            _report_warnings.append("levels_computation_failed")
            levels_out = []
        if not levels_out:
            level_source = "rolling_window_fallback"
            closes_for_levels = closes
            dates_for_levels = [str(r.get("date") or "") for r in vix_rows]
            candidates: list[tuple[str, float, str]] = []
            for bars, tier in ((21, "primary"), (63, "secondary"), (126, "secondary")):
                if len(closes_for_levels) < bars:
                    continue
                win = closes_for_levels[-bars:]
                candidates.append(("support", float(min(win)), tier))
                candidates.append(("resistance", float(max(win)), tier))
            dedup: dict[tuple[str, int], tuple[str, float, str]] = {}
            for side, lv, tier in candidates:
                key = (side, int(round(lv * 100)))
                dedup[key] = (side, lv, tier)

            def _ordered_fallback(side: str) -> list[tuple[str, float, str]]:
                pool = [item for item in dedup.values() if item[0] == side]
                if side == "support":
                    near = sorted([p for p in pool if p[1] <= latest], key=lambda x: x[1], reverse=True)
                    far = sorted([p for p in pool if p[1] > latest], key=lambda x: x[1])
                else:
                    near = sorted([p for p in pool if p[1] >= latest], key=lambda x: x[1])
                    far = sorted([p for p in pool if p[1] < latest], key=lambda x: x[1], reverse=True)
                return (near + far)[:3]

            for side, level_v, tier in _ordered_fallback("support") + _ordered_fallback("resistance"):
                if level_v <= 0:
                    continue
                zone_pad = max(level_v * 0.003, 0.05)
                zone_low = level_v - zone_pad
                zone_high = level_v + zone_pad
                if side == "support":
                    dist_pct = ((latest - level_v) / level_v) * 100.0
                else:
                    dist_pct = ((level_v - latest) / level_v) * 100.0
                last_touch_idx = None
                for idx in range(len(closes_for_levels) - 1, -1, -1):
                    cv = closes_for_levels[idx]
                    if abs(cv - level_v) / max(level_v, 1e-9) <= 0.004:
                        last_touch_idx = idx
                        break
                levels_out.append(
                    {
                        "type": side,
                        "level": round(level_v, 2),
                        "zone_low": round(zone_low, 2),
                        "zone_high": round(zone_high, 2),
                        "tier": tier,
                        "touches": 1 if last_touch_idx is not None else 0,
                        "distance_pct": round(dist_pct, 2),
                        "last_touch_date": dates_for_levels[last_touch_idx] if last_touch_idx is not None else "",
                        "source": "fallback",  # Requirement 10.1, 10.3
                    }
                )
        regime["levels"] = levels_out

        # Detect VIX trap/reclaim patterns
        trap_reclaim_patterns: list[dict[str, Any]] = []
        try:
            vix_df = pd.DataFrame(vix_rows)
            if not vix_df.empty and levels_out:
                patterns = detect_vix_trap_reclaim_patterns(
                    vix_df, levels_out, VIXTrapReclaimConfig()
                )
                for pattern in patterns:
                    trap_reclaim_patterns.append(
                        {
                            "pattern_type": pattern.pattern_type,
                            "date": pattern.date,
                            "price": round(pattern.price, 2),
                            "level": round(pattern.level, 2),
                            "confidence": round(pattern.confidence, 2),
                            "explanation": pattern.explanation,
                            "bars_to_reversal": pattern.bars_to_reversal,
                            "volume_factor": round(pattern.volume_factor, 2) if pattern.volume_factor else None,
                            "reversal_speed_factor": round(pattern.reversal_speed_factor, 2) if pattern.reversal_speed_factor else None,
                        }
                    )
        except Exception as e:
            LOG.error("VIX trap/reclaim pattern detection failed: %s", e, exc_info=True)
            _report_warnings.append("vix_trap_reclaim_detection_failed")

        regime["trap_reclaim_patterns"] = trap_reclaim_patterns

        # Calculate VIX percentile (Requirement 11.1)
        vix_percentile = calculate_vix_percentile(conn, window_days=252)
        vix_percentile_color = get_percentile_color(vix_percentile)
        vix_percentile_warning = should_show_volatility_warning(vix_percentile)

        regime["asof_date"] = str(vix_rows[-1]["date"])
        regime["vix"] = {
            "ticker": "^VIX",
            "close": round(latest, 2),
            "change_pct_1d": round(((latest / prev) - 1.0) * 100.0, 2) if prev > 0 else None,
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "ma100": round(ma100, 2) if isinstance(ma100, (int, float)) else None,
            "pct_vs_ma20": round(pct_vs_ma20, 2) if pct_vs_ma20 is not None else None,
            "pct_vs_ma50": round(pct_vs_ma50, 2) if pct_vs_ma50 is not None else None,
            "pct_vs_ma100": round(pct_vs_ma100, 2) if pct_vs_ma100 is not None else None,
            "ma_state": ma_state,
            "ma_cross_state": ma_cross_state,
            "bb_width_20": round(bb_width_20, 2) if isinstance(bb_width_20, (int, float)) else None,
            "bb_width_pctile_lookback": round(bb_width_pctile, 1)
            if isinstance(bb_width_pctile, (int, float))
            else None,
            "compression_state": compression_state,
            "compression_labeled": compression_labeled,  # Requirement 13.6
            "compression_thresholds": compression_thresholds.to_dict(),  # Requirements 13.1-13.5
            "breakout_state": breakout_state,
            "risk_state": risk_state,
            "term_structure_ratio": round(term_structure_ratio, 3)
            if isinstance(term_structure_ratio, (int, float))
            else None,
            "term_structure_state": term_structure_state,
            "term_structure_source": term_structure.source,  # Requirement 9.3
            "term_structure_timestamp": term_structure.timestamp.isoformat(),  # Requirement 9.6
            "vix3m_close": round(vix3m_latest, 2) if isinstance(vix3m_latest, (int, float)) else None,
            "level_source": level_source,
            "percentile": round(vix_percentile, 1) if vix_percentile is not None else None,  # Requirement 11.6
            "percentile_color": vix_percentile_color,  # Requirement 11.3
            "percentile_warning": vix_percentile_warning,  # Requirement 11.5
        }
        regime["ma_matrix"] = [
            {
                "metric": "Close vs MA20",
                "value_pct": round(pct_vs_ma20, 2) if isinstance(pct_vs_ma20, (int, float)) else None,
                "state": "above_ma20" if isinstance(pct_vs_ma20, (int, float)) and pct_vs_ma20 >= 0 else "below_ma20",
                "risk_read": "risk_off_pressure" if isinstance(pct_vs_ma20, (int, float)) and pct_vs_ma20 >= 0 else "risk_on_relief",
            },
            {
                "metric": "Close vs MA50",
                "value_pct": round(pct_vs_ma50, 2) if isinstance(pct_vs_ma50, (int, float)) else None,
                "state": "above_ma50" if isinstance(pct_vs_ma50, (int, float)) and pct_vs_ma50 >= 0 else "below_ma50",
                "risk_read": "risk_off_pressure" if isinstance(pct_vs_ma50, (int, float)) and pct_vs_ma50 >= 0 else "risk_on_relief",
            },
            {
                "metric": "Close vs MA100",
                "value_pct": round(pct_vs_ma100, 2) if isinstance(pct_vs_ma100, (int, float)) else None,
                "state": "above_ma100" if isinstance(pct_vs_ma100, (int, float)) and pct_vs_ma100 >= 0 else "below_ma100",
                "risk_read": "risk_off_pressure" if isinstance(pct_vs_ma100, (int, float)) and pct_vs_ma100 >= 0 else "risk_on_relief",
            },
            {
                "metric": "MA20 vs MA50",
                "value_pct": round(ma20_vs_ma50, 2) if isinstance(ma20_vs_ma50, (int, float)) else None,
                "state": "ma20_above_ma50" if ma20 > ma50 else "ma20_below_ma50",
                "risk_read": "short_term_stress_up" if ma20 > ma50 else "short_term_stress_down",
            },
            {
                "metric": "MA50 vs MA100",
                "value_pct": round(ma50_vs_ma100, 2) if isinstance(ma50_vs_ma100, (int, float)) else None,
                "state": "ma50_above_ma100"
                if isinstance(ma50_vs_ma100, (int, float)) and ma50_vs_ma100 >= 0
                else "ma50_below_ma100",
                "risk_read": "intermediate_stress_up"
                if isinstance(ma50_vs_ma100, (int, float)) and ma50_vs_ma100 >= 0
                else "intermediate_stress_down",
            },
        ]

        # Benchmark comparison window for regime context: VIX + SPX/DJI/NDX.
        benchmark_candidates = {
            "spx": ("^GSPC", ".SPX", "SPX"),
            "dji": ("^DJI", ".DJI", "DJI"),
            "ndx": ("^NDX", ".NDX", "NDX", "QQQ"),
        }
        benchmark_maps: dict[str, dict[str, float]] = {}
        benchmark_symbol_used: dict[str, str] = {"vix": "^VIX"}
        for key, symbols in benchmark_candidates.items():
            rows_best: list[dict[str, float | str]] = []
            symbol_best = None
            for symbol in symbols:
                rows = _fetch_symbol_ohlcv(conn, symbol, limit=160)
                if len(rows) > len(rows_best):
                    rows_best = rows
                    symbol_best = symbol
            if rows_best:
                benchmark_maps[key] = {str(r["date"]): float(r["close"]) for r in rows_best if float(r["close"]) > 0}
                benchmark_symbol_used[key] = str(symbol_best or "")

        vix_map = {str(r["date"]): float(r["close"]) for r in vix_rows if float(r["close"]) > 0}
        date_pool = sorted(vix_map.keys())[-90:]
        comparison_rows: list[dict[str, Any]] = []
        if date_pool:
            bases: dict[str, float] = {}
            for name in ("vix", "spx", "dji", "ndx"):
                source_map = vix_map if name == "vix" else benchmark_maps.get(name, {})
                base_v = next((source_map.get(d) for d in date_pool if isinstance(source_map.get(d), (int, float))), None)
                if isinstance(base_v, (int, float)) and base_v > 0:
                    bases[name] = float(base_v)
            for d in date_pool:
                row_cmp: dict[str, Any] = {"date": d}
                for name in ("vix", "spx", "dji", "ndx"):
                    source_map = vix_map if name == "vix" else benchmark_maps.get(name, {})
                    raw_v = source_map.get(d)
                    row_cmp[f"{name}_close"] = round(float(raw_v), 2) if isinstance(raw_v, (int, float)) else None
                    base_v = bases.get(name)
                    row_cmp[f"{name}_idx"] = (
                        round((float(raw_v) / base_v) * 100.0, 2)
                        if isinstance(raw_v, (int, float)) and isinstance(base_v, (int, float)) and base_v > 0
                        else None
                    )
                comparison_rows.append(row_cmp)
        regime["comparison"] = {
            "window_days": len(comparison_rows),
            "symbols": benchmark_symbol_used,
            "series": comparison_rows,
        }

    # --- Participation / distribution context for core market proxies ---
    participation_rows: list[dict[str, Any]] = []
    for symbol in ("SPY", "QQQ"):
        sym_rows = _fetch_symbol_ohlcv(conn, symbol, limit=45)
        window = 20
        if len(sym_rows) < window + 1:
            continue
        closes = [float(r["close"]) for r in sym_rows[-(window + 1) :]]
        vols = [float(r["volume"]) for r in sym_rows[-(window + 1) :]]
        up_days = 0
        down_days = 0
        up_vol = 0.0
        down_vol = 0.0
        day_changes: list[tuple[float, float]] = []
        for idx in range(1, len(closes)):
            prev_c = closes[idx - 1]
            cur_c = closes[idx]
            vol = vols[idx]
            if prev_c <= 0:
                continue
            ret = ((cur_c / prev_c) - 1.0) * 100.0
            day_changes.append((ret, vol))
            if ret > 0:
                up_days += 1
                up_vol += vol
            elif ret < 0:
                down_days += 1
                down_vol += vol

        total_signed_vol = up_vol + down_vol
        up_share = (up_vol / total_signed_vol) * 100.0 if total_signed_vol > 0 else None
        down_share = (down_vol / total_signed_vol) * 100.0 if total_signed_vol > 0 else None
        vol_ratio = (up_vol / down_vol) if down_vol > 0 else None

        avg_vol = sum(v for _, v in day_changes) / len(day_changes) if day_changes else 0.0
        heavy_up_days = sum(1 for ret, vol in day_changes if ret > 0 and vol >= avg_vol)
        heavy_down_days = sum(1 for ret, vol in day_changes if ret < 0 and vol >= avg_vol)

        if isinstance(vol_ratio, (int, float)) and vol_ratio >= 1.15 and heavy_up_days >= heavy_down_days:
            bias = "accumulation"
        elif isinstance(vol_ratio, (int, float)) and vol_ratio <= 0.85 and heavy_down_days >= heavy_up_days:
            bias = "distribution"
        else:
            bias = "balanced"

        participation_rows.append(
            {
                "symbol": symbol,
                "window_days": window,
                "up_days": up_days,
                "down_days": down_days,
                "up_volume_share_pct": round(up_share, 2) if up_share is not None else None,
                "down_volume_share_pct": round(down_share, 2) if down_share is not None else None,
                "up_down_volume_ratio": round(vol_ratio, 3) if isinstance(vol_ratio, (int, float)) else None,
                "heavy_up_days": int(heavy_up_days),
                "heavy_down_days": int(heavy_down_days),
                "bias": bias,
            }
        )

    regime["participation"] = participation_rows
    if participation_rows:
        accum = sum(1 for row in participation_rows if row.get("bias") == "accumulation")
        dist = sum(1 for row in participation_rows if row.get("bias") == "distribution")
        avg_up_share = [
            float(row.get("up_volume_share_pct"))
            for row in participation_rows
            if isinstance(row.get("up_volume_share_pct"), (int, float))
        ]
        if accum > dist:
            overall_participation = "accumulation_bias"
        elif dist > accum:
            overall_participation = "distribution_bias"
        else:
            overall_participation = "balanced_bias"
        regime["overall"] = {
            "participation_bias": overall_participation,
            "accumulation_symbols": accum,
            "distribution_symbols": dist,
            "total_symbols": len(participation_rows),
            "avg_up_volume_share_pct": round(sum(avg_up_share) / len(avg_up_share), 2) if avg_up_share else None,
        }
    else:
        regime["overall"] = {"participation_bias": "unknown"}

    # --- Overall market health score (0-100, context-only) ---
    vix_info = regime.get("vix", {}) if isinstance(regime.get("vix"), dict) else {}
    participation_bias = str((regime.get("overall") or {}).get("participation_bias") or "unknown")
    health_score = 50.0
    drivers: list[str] = []
    warnings: list[str] = []

    # VIX percentile as primary factor (Requirement 11.4)
    vix_percentile = vix_info.get("percentile")
    if isinstance(vix_percentile, (int, float)):
        if vix_percentile < 30:
            health_score += 20.0
            drivers.append(f"VIX percentile at {vix_percentile:.1f}% (historically low volatility)")
        elif vix_percentile < 50:
            health_score += 10.0
            drivers.append(f"VIX percentile at {vix_percentile:.1f}% (below median volatility)")
        elif vix_percentile < 70:
            health_score -= 10.0
            warnings.append(f"VIX percentile at {vix_percentile:.1f}% (above median volatility)")
        else:
            health_score -= 20.0
            warnings.append(f"VIX percentile at {vix_percentile:.1f}% (historically high volatility)")

        if vix_info.get("percentile_warning"):
            warnings.append("⚠️ ELEVATED VOLATILITY: VIX percentile exceeds 80%")

    ma_state = str(vix_info.get("ma_state") or "")
    if ma_state == "below_ma20_ma50":
        health_score += 15.0
        drivers.append("VIX below MA20 and MA50 (risk-on backdrop)")
    elif ma_state == "above_ma20_ma50":
        health_score -= 15.0
        warnings.append("VIX above MA20 and MA50 (risk-off pressure)")
    elif ma_state == "below_ma20_above_ma50":
        health_score += 6.0
        drivers.append("VIX below MA20 while still above MA50")
    elif ma_state == "above_ma20_below_ma50":
        health_score -= 6.0
        warnings.append("VIX above MA20 while below MA50 (short-term stress)")

    breakout_state = str(vix_info.get("breakout_state") or "")
    if breakout_state == "range_breakdown_down":
        health_score += 10.0
        drivers.append("VIX broke down from its recent range")
    elif breakout_state == "range_breakout_up":
        health_score -= 10.0
        warnings.append("VIX broke up from its recent range")

    compression_state = str(vix_info.get("compression_state") or "")
    if compression_state == "compression":
        health_score += 4.0
        drivers.append("VIX volatility is compressed (coiled regime)")
    elif compression_state == "expansion":
        health_score -= 4.0
        warnings.append("VIX volatility is expanding")

    change_1d = vix_info.get("change_pct_1d")
    if isinstance(change_1d, (int, float)):
        if change_1d <= -2.0:
            health_score += 6.0
            drivers.append("VIX fell sharply vs prior session")
        elif change_1d >= 2.0:
            health_score -= 6.0
            warnings.append("VIX rose sharply vs prior session")

    term_state = str(vix_info.get("term_structure_state") or "")
    if term_state == "contango":
        health_score += 8.0
        drivers.append("VIX term structure is in contango")
    elif term_state == "backwardation":
        health_score -= 8.0
        warnings.append("VIX term structure is in backwardation")

    if participation_bias == "accumulation_bias":
        health_score += 12.0
        drivers.append("SPY/QQQ volume profile shows accumulation")
    elif participation_bias == "distribution_bias":
        health_score -= 12.0
        warnings.append("SPY/QQQ volume profile shows distribution")

    health_score = _clamp(health_score, 0.0, 100.0)
    if health_score >= 65.0:
        health_state = "risk_on"
    elif health_score <= 35.0:
        health_state = "risk_off"
    else:
        health_state = "neutral"

    regime["health"] = {
        "score": round(health_score, 1),
        "state": health_state,
        "confidence": round(abs(health_score - 50.0) * 2.0, 1),
        "drivers": drivers[:6],
        "warnings": warnings[:6],
    }

    if isinstance(vix_latest, (int, float)):
        summary_prefix = (
            f"Market health {regime['health']['score']}/100 ({health_state.replace('_', ' ')})"
        )
        vix_risk = str(vix_info.get("risk_state") or "mixed_regime").replace("_", " ")
        participation_state = participation_bias.replace("_", " ")
        regime["summary"] = (
            f"{summary_prefix}. VIX regime: {vix_risk}; participation: {participation_state}. "
            "Use this as backdrop for sizing and confirmation, not as a standalone trigger."
        )
    else:
        regime["summary"] = "Regime context unavailable (insufficient VIX history)."
    regime["llm_commentary"] = _build_regime_llm_commentary(regime)
    return regime


def _fetch_technical_context(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Build report interpretation context using the same level engine as the dashboard."""
    by_ticker: dict[str, dict[str, Any]] = {}
    if not table_exists(conn, "price_daily"):
        return by_ticker

    rows = conn.execute(
        """
        SELECT ticker, date, CAST(open AS REAL), CAST(high AS REAL), CAST(low AS REAL), CAST(close AS REAL), CAST(volume AS REAL)
        FROM (
            SELECT ticker, date, open, high, low, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM price_daily
        )
        WHERE rn <= 320
        ORDER BY ticker, date
        """
    ).fetchall()
    bucket: dict[str, list[tuple[str, float, float, float, float, float]]] = defaultdict(list)
    for r in rows:
        ticker = str(r[0] or "").upper().strip()
        if not ticker:
            continue
        try:
            open_v = float(r[2])
            high_v = float(r[3])
            low_v = float(r[4])
            close_v = float(r[5])
            volume_v = float(r[6] or 0.0)
        except (TypeError, ValueError):
            continue
        if min(open_v, high_v, low_v, close_v) <= 0:
            continue
        bucket[ticker].append((str(r[1]), open_v, high_v, low_v, close_v, volume_v))

    for ticker, bars in bucket.items():
        if len(bars) < 60:
            continue
        closes = [b[4] for b in bars]
        highs = [b[2] for b in bars]
        lows = [b[3] for b in bars]
        volumes = [b[5] for b in bars]
        close_now = closes[-1]
        prev_close = closes[-2] if len(closes) >= 2 else close_now
        high_now = highs[-1]
        low_now = lows[-1]
        ma20 = sum(closes[-20:]) / 20.0 if len(closes) >= 20 else None
        ma50 = sum(closes[-50:]) / 50.0 if len(closes) >= 50 else None
        ma100 = sum(closes[-100:]) / 100.0 if len(closes) >= 100 else None
        ma200 = sum(closes[-200:]) / 200.0 if len(closes) >= 200 else None
        prev_ma20 = sum(closes[-21:-1]) / 20.0 if len(closes) >= 21 else None
        prev_ma50 = sum(closes[-51:-1]) / 50.0 if len(closes) >= 51 else None
        prev_ma200 = sum(closes[-201:-1]) / 200.0 if len(closes) >= 201 else None
        recent_high_20 = max(highs[-20:]) if len(highs) >= 20 else None
        recent_low_20 = min(lows[-20:]) if len(lows) >= 20 else None
        recent_high_prev_20 = max(highs[-21:-1]) if len(highs) >= 21 else None
        recent_low_prev_20 = min(lows[-21:-1]) if len(lows) >= 21 else None
        avg_volume_20 = (sum(volumes[-20:]) / 20.0) if len(volumes) >= 20 else None
        volume_ratio_20 = (volumes[-1] / avg_volume_20) if avg_volume_20 and avg_volume_20 > 0 else None
        recent_range_pct_10 = (
            ((max(highs[-10:]) - min(lows[-10:])) / close_now) * 100.0
            if len(highs) >= 10 and close_now > 0
            else None
        )
        recent_range_pct_20 = (
            ((max(highs[-20:]) - min(lows[-20:])) / close_now) * 100.0
            if len(highs) >= 20 and close_now > 0
            else None
        )

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
        if len(bars) >= 2:
            start_idx = max(1, len(bars) - 4)
            for idx in range(len(bars) - 1, start_idx - 1, -1):
                prev_high = highs[idx - 1]
                prev_low = lows[idx - 1]
                bar_high = highs[idx]
                bar_low = lows[idx]
                if bar_low > prev_high:
                    recent_gap_state = "bull_gap"
                    recent_gap_days = len(bars) - 1 - idx
                    break
                if bar_high < prev_low:
                    recent_gap_state = "bear_gap"
                    recent_gap_days = len(bars) - 1 - idx
                    break

        level_context = "mid_range"
        if isinstance(pct_from_20d_low, (int, float)) and pct_from_20d_low <= 2.5:
            level_context = "at_support"
        elif isinstance(pct_from_20d_high, (int, float)) and pct_from_20d_high <= 2.5:
            level_context = "at_resistance"

        stretch_state = "normal"
        if isinstance(pct_vs_ma20, (int, float)):
            if pct_vs_ma20 >= 8.0:
                stretch_state = "extended_up"
            elif pct_vs_ma20 <= -8.0:
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

        try:
            frame = pd.DataFrame(
                bars,
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).reset_index(drop=True)
            if len(frame) >= 60:
                model = add_basic_features(frame.copy(), REPORT_FEATURE_CFG)
                model = compute_pivots(model, left=3, right=3)
                levels_raw = build_levels_from_pivots(model, REPORT_LEVEL_CFG)
                levels = select_target_levels(levels_raw, float(close_now), REPORT_LEVEL_CFG)
                levels = add_fallback_levels(model, levels, float(close_now), REPORT_LEVEL_CFG)

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
                        preferred = pool.sort_values(
                            ["dist", "touches", "recency_score"],
                            ascending=[True, False, False],
                        )
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
                    range_position = round(
                        (close_now - support_level) / max(resistance_level - support_level, 0.01),
                        3,
                    )
                if (
                    isinstance(support_zone_low, (int, float))
                    and close_now < float(support_zone_low)
                ):
                    level_context = "below_support"
                elif (
                    isinstance(resistance_zone_high, (int, float))
                    and close_now > float(resistance_zone_high)
                ):
                    level_context = "above_resistance"
                elif (
                    isinstance(support_zone_high, (int, float))
                    and close_now <= float(support_zone_high)
                ):
                    level_context = "at_support"
                elif (
                    isinstance(resistance_zone_low, (int, float))
                    and close_now >= float(resistance_zone_low)
                ):
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
                        breakout_state = "failed_breakout_up"
                if isinstance(support_zone_low, (int, float)):
                    szl = float(support_zone_low)
                    if close_now < szl:
                        breakout_state = "breakout_down"
                    elif low_now < szl and close_now >= szl and breakout_state == "none":
                        breakout_state = "failed_breakdown_down"
                if breakout_state == "breakout_up":
                    level_event = "resistance_breakout"
                elif breakout_state == "breakout_down":
                    level_event = "support_breakdown"
                elif breakout_state == "failed_breakout_up":
                    level_event = "resistance_reject"
                elif breakout_state == "failed_breakdown_down":
                    level_event = "support_reclaim"
        except Exception as exc:
            LOG.warning("Level context/breakout detection failed for %s: %s", ticker, exc)

        if (
            isinstance(recent_range_pct_10, (int, float))
            and isinstance(recent_range_pct_20, (int, float))
            and recent_range_pct_10 <= 7.0
            and recent_range_pct_20 <= 12.0
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

        by_ticker[ticker] = {
            "close": round(close_now, 2),
            "ma20": round(ma20, 2) if ma20 is not None else None,
            "ma50": round(ma50, 2) if ma50 is not None else None,
            "ma100": round(ma100, 2) if ma100 is not None else None,
            "ma200": round(ma200, 2) if ma200 is not None else None,
            "avg_volume_20": round(avg_volume_20, 2) if avg_volume_20 is not None else None,
            "volume_ratio_20": round(volume_ratio_20, 2) if volume_ratio_20 is not None else None,
            "pct_vs_ma20": round(pct_vs_ma20, 2) if pct_vs_ma20 is not None else None,
            "pct_vs_ma50": round(pct_vs_ma50, 2) if pct_vs_ma50 is not None else None,
            "recent_high_20": round(recent_high_20, 2) if recent_high_20 is not None else None,
            "recent_low_20": round(recent_low_20, 2) if recent_low_20 is not None else None,
            "recent_range_pct_10": round(recent_range_pct_10, 2) if recent_range_pct_10 is not None else None,
            "recent_range_pct_20": round(recent_range_pct_20, 2) if recent_range_pct_20 is not None else None,
            "pct_from_20d_high": round(pct_from_20d_high, 2) if pct_from_20d_high is not None else None,
            "pct_from_20d_low": round(pct_from_20d_low, 2) if pct_from_20d_low is not None else None,
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
        }
    return by_ticker

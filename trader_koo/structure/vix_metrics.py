"""Advanced VIX metrics for the dashboard.

Computes term-structure ratios, realized-vs-implied vol spread,
percentile context, spike detection, position sizing recommendations,
and gauge-zone classification from price data in the DB.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gauge zone definitions (VIX level -> zone/color)
# ---------------------------------------------------------------------------

_GAUGE_ZONES: list[tuple[float, float, str, str]] = [
    (0, 12, "complacency", "#00c853"),
    (12, 16, "calm", "#4caf50"),
    (16, 20, "normal", "#c0ca33"),
    (20, 25, "caution", "#fdd835"),
    (25, 30, "stress", "#ff9800"),
    (30, 40, "fear", "#f44336"),
    (40, 60, "extreme_fear", "#b71c1c"),
    (60, 80, "panic", "#212121"),
]


def _gauge_for_vix(vix_close: float) -> tuple[str, str]:
    """Return (zone, hex_color) for a VIX level."""
    for lo, hi, zone, color in _GAUGE_ZONES:
        if vix_close < hi:
            return zone, color
    return "panic", "#212121"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_close_series(
    conn: sqlite3.Connection,
    ticker: str,
    limit: int,
) -> list[float]:
    """Fetch the most recent *limit* close prices (newest first)."""
    rows = conn.execute(
        """
        SELECT CAST(close AS REAL)
        FROM price_daily
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, limit),
    ).fetchall()
    return [float(r[0]) for r in rows if r[0] is not None]


def _fetch_latest_close(conn: sqlite3.Connection, ticker: str) -> float | None:
    """Fetch latest close for *ticker*."""
    closes = _fetch_close_series(conn, ticker, 1)
    return closes[0] if closes else None


def _fetch_prev_close(conn: sqlite3.Connection, ticker: str) -> float | None:
    """Fetch the second-most-recent close for *ticker*."""
    closes = _fetch_close_series(conn, ticker, 2)
    return closes[1] if len(closes) >= 2 else None


def _realized_vol_annualized(
    closes: list[float],
    window: int = 20,
) -> float | None:
    """Compute annualized realized volatility from close prices.

    *closes* must be ordered newest-first.  We need at least
    ``window + 1`` prices to compute ``window`` log-returns.
    """
    if len(closes) < window + 1:
        return None
    # Use the first (window+1) prices (most recent window days)
    subset = closes[: window + 1]
    log_returns = [
        math.log(subset[i] / subset[i + 1])
        for i in range(len(subset) - 1)
        if subset[i + 1] > 0
    ]
    if len(log_returns) < window:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252) * 100  # annualized, percent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_vix_metrics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Compute advanced VIX metrics for the dashboard.

    All fields are explicitly null when data is unavailable --
    no hidden fallbacks.
    """
    vix_close = _fetch_latest_close(conn, "^VIX")
    if vix_close is None:
        logger.warning("VIX close unavailable -- returning null metrics")
        return _null_metrics()

    # ------------------------------------------------------------------
    # 1. VIX / VIX3M term-structure ratio
    # ------------------------------------------------------------------
    vix3m = (
        _fetch_latest_close(conn, "^VIX3M")
        or _fetch_latest_close(conn, "VIX3M")
    )
    vix_vix3m_ratio: float | None = None
    term_structure_signal = "unavailable"
    if vix3m and vix3m > 0:
        vix_vix3m_ratio = round(vix_close / vix3m, 4)
        if vix_vix3m_ratio < 0.9:
            term_structure_signal = "complacent"
        elif vix_vix3m_ratio <= 1.0:
            term_structure_signal = "normal"
        elif vix_vix3m_ratio <= 1.1:
            term_structure_signal = "elevated"
        else:
            term_structure_signal = "fear"

    # ------------------------------------------------------------------
    # 2. Realized vs Implied Vol spread
    # ------------------------------------------------------------------
    spx_closes = _fetch_close_series(conn, "^GSPC", 30)
    realized_vol_20d = _realized_vol_annualized(spx_closes, window=20)
    vol_risk_premium: float | None = None
    vol_premium_signal = "unavailable"
    if realized_vol_20d is not None:
        vol_risk_premium = round(vix_close - realized_vol_20d, 2)
        if vol_risk_premium < 0:
            vol_premium_signal = "cheap_vol"
        elif vol_risk_premium <= 5:
            vol_premium_signal = "normal"
        else:
            vol_premium_signal = "expensive_vol"

    # ------------------------------------------------------------------
    # 3. VIX percentile (252-day)
    # ------------------------------------------------------------------
    vix_closes = _fetch_close_series(conn, "^VIX", 253)
    vix_percentile_252d: float | None = None
    percentile_zone = "unavailable"
    above_80th_pctile = False
    if len(vix_closes) >= 2:
        current = vix_closes[0]
        below_count = sum(1 for c in vix_closes if c < current)
        vix_percentile_252d = round((below_count / len(vix_closes)) * 100, 1)
        if vix_percentile_252d < 10:
            percentile_zone = "extreme_low"
        elif vix_percentile_252d < 30:
            percentile_zone = "low"
        elif vix_percentile_252d < 70:
            percentile_zone = "normal"
        elif vix_percentile_252d < 90:
            percentile_zone = "elevated"
        else:
            percentile_zone = "extreme_high"
        above_80th_pctile = vix_percentile_252d >= 80

    # ------------------------------------------------------------------
    # 4. Spike detection
    # ------------------------------------------------------------------
    vix_prev = _fetch_prev_close(conn, "^VIX")
    vix_daily_change_pct: float | None = None
    is_spike = False
    spike_magnitude: str | None = None
    if vix_prev is not None and vix_prev > 0:
        vix_daily_change_pct = round(
            ((vix_close - vix_prev) / vix_prev) * 100, 2,
        )
        abs_change = abs(vix_daily_change_pct)
        if abs_change > 15:
            is_spike = True
            if abs_change > 40:
                spike_magnitude = "extreme"
            elif abs_change > 25:
                spike_magnitude = "large"
            else:
                spike_magnitude = "moderate"

    # ------------------------------------------------------------------
    # 5. Position-sizing recommendation
    # ------------------------------------------------------------------
    recommended_position_pct, sizing_reason = _position_sizing(
        vix_close, vix_percentile_252d, is_spike,
    )

    # ------------------------------------------------------------------
    # 6. Gauge zone
    # ------------------------------------------------------------------
    gauge_zone, gauge_color = _gauge_for_vix(vix_close)

    return {
        "vix_vix3m_ratio": vix_vix3m_ratio,
        "term_structure_signal": term_structure_signal,
        "realized_vol_20d": (
            round(realized_vol_20d, 2) if realized_vol_20d is not None else None
        ),
        "vol_risk_premium": vol_risk_premium,
        "vol_premium_signal": vol_premium_signal,
        "vix_percentile_252d": vix_percentile_252d,
        "percentile_zone": percentile_zone,
        "above_80th_pctile": above_80th_pctile,
        "vix_daily_change_pct": vix_daily_change_pct,
        "is_spike": is_spike,
        "spike_magnitude": spike_magnitude,
        "recommended_position_pct": recommended_position_pct,
        "sizing_reason": sizing_reason,
        "gauge_zone": gauge_zone,
        "gauge_color": gauge_color,
    }


# ---------------------------------------------------------------------------
# Position sizing logic
# ---------------------------------------------------------------------------

def _position_sizing(
    vix_close: float,
    percentile: float | None,
    is_spike: bool,
) -> tuple[int, str]:
    """Return (recommended_pct, reason) based on VIX regime."""
    if is_spike:
        return 25, "VIX spike detected -- reduce exposure until volatility stabilizes"
    if vix_close >= 40:
        return 25, "VIX above 40 (extreme fear) -- minimal position sizing"
    if vix_close >= 30:
        return 50, "VIX 30-40 (fear zone) -- half position sizing"
    if percentile is not None and percentile >= 80:
        return 50, "VIX above 80th percentile -- historically elevated, reduce sizing"
    if vix_close >= 25:
        return 75, "VIX 25-30 (stress zone) -- moderate caution"
    if vix_close < 12:
        return 75, "VIX below 12 (complacency) -- potential snap-back risk"
    return 100, "VIX in normal range -- full position sizing"


# ---------------------------------------------------------------------------
# Null-metrics fallback (explicit nulls, no hidden defaults)
# ---------------------------------------------------------------------------

def _null_metrics() -> dict[str, Any]:
    """Return a metrics dict with all nullable fields set to None."""
    return {
        "vix_vix3m_ratio": None,
        "term_structure_signal": "unavailable",
        "realized_vol_20d": None,
        "vol_risk_premium": None,
        "vol_premium_signal": "unavailable",
        "vix_percentile_252d": None,
        "percentile_zone": "unavailable",
        "above_80th_pctile": False,
        "vix_daily_change_pct": None,
        "is_spike": False,
        "spike_magnitude": None,
        "recommended_position_pct": 25,
        "sizing_reason": "VIX data unavailable -- defaulting to minimal sizing",
        "gauge_zone": "unavailable",
        "gauge_color": "#757575",
    }

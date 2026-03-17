"""Technical indicators computed from buffered crypto bars.

All functions return ``None`` when there is insufficient data
rather than silently falling back to a default.
"""
from __future__ import annotations

import logging
import math

from trader_koo.crypto.models import CryptoBar

LOG = logging.getLogger("trader_koo.crypto.indicators")


def compute_sma(closes: list[float], period: int) -> float | None:
    """Simple Moving Average over the last *period* closes."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    return sum(window) / period


def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Relative Strength Index (Wilder smoothing).

    Requires at least ``period + 1`` data points.
    """
    if len(closes) < period + 1:
        return None

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Seed: average gain/loss over first `period` deltas
    gains = [max(d, 0.0) for d in deltas[:period]]
    losses = [abs(min(d, 0.0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing for the rest
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + abs(min(d, 0.0))) / period

    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(values: list[float], period: int) -> list[float]:
    """Compute EMA series. Returns list of same length with leading NaNs."""
    if len(values) < period:
        return [float("nan")] * len(values)
    multiplier = 2.0 / (period + 1)
    result: list[float] = [float("nan")] * (period - 1)
    # Seed with SMA of first `period` values
    seed = sum(values[:period]) / period
    result.append(seed)
    for val in values[period:]:
        seed = (val - seed) * multiplier + seed
        result.append(seed)
    return result


def compute_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict[str, float | None]:
    """MACD line, signal line, and histogram."""
    if len(closes) < slow + signal_period:
        return {"macd": None, "signal": None, "histogram": None}

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)

    # MACD line = fast EMA - slow EMA (only where both are valid)
    macd_line: list[float] = []
    for f, s in zip(ema_fast, ema_slow):
        if math.isnan(f) or math.isnan(s):
            continue
        macd_line.append(f - s)

    if len(macd_line) < signal_period:
        return {"macd": None, "signal": None, "histogram": None}

    signal_line = _ema(macd_line, signal_period)

    macd_val = macd_line[-1]
    signal_val = signal_line[-1] if not math.isnan(signal_line[-1]) else None
    histogram = (macd_val - signal_val) if signal_val is not None else None

    return {
        "macd": round(macd_val, 6),
        "signal": round(signal_val, 6) if signal_val is not None else None,
        "histogram": round(histogram, 6) if histogram is not None else None,
    }


def compute_bollinger(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, float | None]:
    """Bollinger Bands: upper, middle (SMA), lower, and width."""
    if len(closes) < period:
        return {"upper": None, "middle": None, "lower": None, "width": None}

    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    sd = math.sqrt(variance)
    upper = middle + std_dev * sd
    lower = middle - std_dev * sd
    width = (upper - lower) / middle if middle != 0 else None

    return {
        "upper": round(upper, 6),
        "middle": round(middle, 6),
        "lower": round(lower, 6),
        "width": round(width, 6) if width is not None else None,
    }


def compute_vwap(bars: list[CryptoBar]) -> float | None:
    """Volume-weighted average price over the provided bars.

    Uses the typical price (H+L+C)/3 per bar.
    """
    if not bars:
        return None

    total_vp = 0.0
    total_vol = 0.0
    for bar in bars:
        typical = (bar.high + bar.low + bar.close) / 3.0
        total_vp += typical * bar.volume
        total_vol += bar.volume

    if total_vol == 0.0:
        return None
    return round(total_vp / total_vol, 6)


def compute_all_indicators(
    bars: list[CryptoBar],
) -> dict:
    """Compute all indicators from a list of bars.

    Returns a dict matching the CryptoIndicators API shape.
    """
    closes = [b.close for b in bars]
    return {
        "sma_20": round(v, 6) if (v := compute_sma(closes, 20)) is not None else None,
        "sma_50": round(v, 6) if (v := compute_sma(closes, 50)) is not None else None,
        "rsi_14": round(v, 4) if (v := compute_rsi(closes, 14)) is not None else None,
        "macd": compute_macd(closes),
        "bollinger": compute_bollinger(closes),
        "vwap": compute_vwap(bars),
    }

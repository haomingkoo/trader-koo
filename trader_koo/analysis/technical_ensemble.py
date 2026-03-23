"""Multi-strategy technical ensemble inspired by ai-hedge-fund.

Five sub-strategies vote independently, each producing a SignalOutput.
Weighted combination determines overall bias and confidence.

Strategy weights (sum to 1.0):
    Trend following:     0.25
    Mean reversion:      0.20
    Momentum:            0.25
    Volatility regime:   0.15
    Statistical arb:     0.15
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from trader_koo.signals.types import SignalOutput, aggregate_signals

LOG = logging.getLogger(__name__)

STRATEGY_WEIGHTS = {
    "trend": 0.25,
    "mean_reversion": 0.20,
    "momentum": 0.25,
    "volatility": 0.15,
    "stat_arb": 0.15,
}


# ---------------------------------------------------------------------------
# Helper indicators
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average Directional Index (Wilder smoothing)."""
    if len(close) < period + 1:
        return 0.0
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return float(adx_val.iloc[-1]) if not adx_val.empty and pd.notna(adx_val.iloc[-1]) else 0.0


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    last_gain = float(gain.iloc[-1])
    last_loss = float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return round(100 - (100 / (1 + rs)), 2)


def _hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Hurst exponent via rescaled range (R/S) method.

    H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending.
    """
    if len(series) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag + 1)
    rs_values = []
    for lag in lags:
        chunks = [series.iloc[i:i + lag].values for i in range(0, len(series) - lag, lag)]
        if not chunks:
            continue
        rs_list = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = chunk.mean()
            deviations = chunk - mean_c
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))
    if len(rs_values) < 3:
        return 0.5
    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


# ---------------------------------------------------------------------------
# Sub-strategies
# ---------------------------------------------------------------------------

def _trend_strategy(df: pd.DataFrame) -> SignalOutput:
    """EMA crossover (12/26) + ADX trend strength filter."""
    close = df["close"].astype(float)
    ema12 = _ema(close, 12).iloc[-1]
    ema26 = _ema(close, 26).iloc[-1]
    adx_val = _adx(
        df["high"].astype(float), df["low"].astype(float), close, 14,
    )

    trending = adx_val > 25
    if ema12 > ema26:
        bias = "bullish"
        conf = min(80, 40 + adx_val) if trending else 30
    elif ema12 < ema26:
        bias = "bearish"
        conf = min(80, 40 + adx_val) if trending else 30
    else:
        bias = "neutral"
        conf = 20

    return SignalOutput(
        signal_type="trend",
        bias=bias,
        confidence=round(conf, 1),
        reasoning=f"EMA12={'>' if ema12 > ema26 else '<'}EMA26, ADX={adx_val:.1f}",
        weight=STRATEGY_WEIGHTS["trend"],
    )


def _mean_reversion_strategy(df: pd.DataFrame) -> SignalOutput:
    """Z-score + Bollinger Bands + RSI for oversold/overbought."""
    close = df["close"].astype(float)
    rsi_val = _rsi(close, 14)

    # Bollinger z-score
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    if pd.notna(sma20.iloc[-1]) and std20.iloc[-1] > 0:
        z_score = float((close.iloc[-1] - sma20.iloc[-1]) / std20.iloc[-1])
    else:
        z_score = 0.0

    # Oversold: z < -2 AND RSI < 30
    if z_score < -2 and rsi_val < 30:
        bias = "bullish"
        conf = min(85, 50 + abs(z_score) * 10 + (30 - rsi_val))
    elif z_score > 2 and rsi_val > 70:
        bias = "bearish"
        conf = min(85, 50 + abs(z_score) * 10 + (rsi_val - 70))
    elif z_score < -1 and rsi_val < 40:
        bias = "bullish"
        conf = 40
    elif z_score > 1 and rsi_val > 60:
        bias = "bearish"
        conf = 40
    else:
        bias = "neutral"
        conf = 20

    return SignalOutput(
        signal_type="mean_reversion",
        bias=bias,
        confidence=round(conf, 1),
        reasoning=f"z-score={z_score:.2f}, RSI={rsi_val:.1f}",
        weight=STRATEGY_WEIGHTS["mean_reversion"],
    )


def _momentum_strategy(df: pd.DataFrame) -> SignalOutput:
    """Multi-timeframe momentum (5d, 21d, 63d returns)."""
    close = df["close"].astype(float)
    n = len(close)

    ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if n >= 6 else 0
    ret_21d = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if n >= 22 else 0
    ret_63d = float((close.iloc[-1] / close.iloc[-64] - 1) * 100) if n >= 64 else 0

    bull_count = sum(1 for r in [ret_5d, ret_21d, ret_63d] if r > 0)
    bear_count = sum(1 for r in [ret_5d, ret_21d, ret_63d] if r < 0)
    avg_ret = (ret_5d + ret_21d + ret_63d) / 3

    if bull_count == 3:
        bias = "bullish"
        conf = min(80, 50 + abs(avg_ret) * 2)
    elif bear_count == 3:
        bias = "bearish"
        conf = min(80, 50 + abs(avg_ret) * 2)
    elif bull_count >= 2:
        bias = "bullish"
        conf = 40
    elif bear_count >= 2:
        bias = "bearish"
        conf = 40
    else:
        bias = "neutral"
        conf = 20

    return SignalOutput(
        signal_type="momentum",
        bias=bias,
        confidence=round(conf, 1),
        reasoning=f"5d={ret_5d:+.1f}%, 21d={ret_21d:+.1f}%, 63d={ret_63d:+.1f}%",
        weight=STRATEGY_WEIGHTS["momentum"],
    )


def _volatility_strategy(
    df: pd.DataFrame, vix_level: float | None = None,
) -> SignalOutput:
    """Volatility regime + ATR expansion/contraction."""
    close = df["close"].astype(float)
    atr_14 = (df["high"].astype(float) - df["low"].astype(float)).rolling(14).mean()
    atr_current = float(atr_14.iloc[-1]) if pd.notna(atr_14.iloc[-1]) else 0
    atr_ma = float(atr_14.rolling(50).mean().iloc[-1]) if len(atr_14) >= 50 and pd.notna(atr_14.rolling(50).mean().iloc[-1]) else atr_current

    # ATR expanding = volatile, contracting = compression (energy building)
    if atr_ma > 0:
        atr_ratio = atr_current / atr_ma
    else:
        atr_ratio = 1.0

    # High vol = reduce confidence in any direction
    if vix_level is not None and vix_level > 30:
        bias = "bearish"
        conf = min(70, 40 + (vix_level - 30))
        reasoning = f"Extreme VIX={vix_level:.1f}, ATR ratio={atr_ratio:.2f}"
    elif atr_ratio > 1.5:
        bias = "neutral"
        conf = 30
        reasoning = f"ATR expanding ({atr_ratio:.2f}x), volatile"
    elif atr_ratio < 0.7:
        bias = "neutral"
        conf = 50
        reasoning = f"ATR compressing ({atr_ratio:.2f}x), energy building"
    else:
        bias = "neutral"
        conf = 20
        reasoning = f"Normal volatility, ATR ratio={atr_ratio:.2f}"

    return SignalOutput(
        signal_type="volatility",
        bias=bias,
        confidence=round(conf, 1),
        reasoning=reasoning,
        weight=STRATEGY_WEIGHTS["volatility"],
    )


def _stat_arb_strategy(df: pd.DataFrame) -> SignalOutput:
    """Hurst exponent + return skewness."""
    close = df["close"].astype(float)
    returns = close.pct_change().dropna()

    hurst = _hurst_exponent(returns, max_lag=20)
    skew = float(returns.iloc[-63:].skew()) if len(returns) >= 63 else 0.0

    # Hurst < 0.5 = mean-reverting (buy dips), > 0.5 = trending (follow)
    if hurst < 0.4:
        regime = "mean_reverting"
        # In mean-reverting regime, negative skew = buy opportunity
        if skew < -0.5:
            bias = "bullish"
            conf = min(70, 40 + abs(skew) * 15)
        elif skew > 0.5:
            bias = "bearish"
            conf = min(70, 40 + abs(skew) * 15)
        else:
            bias = "neutral"
            conf = 30
    elif hurst > 0.6:
        regime = "trending"
        # In trending regime, follow the recent direction
        ret_10d = float((close.iloc[-1] / close.iloc[-11] - 1)) if len(close) >= 11 else 0
        if ret_10d > 0:
            bias = "bullish"
            conf = min(70, 40 + hurst * 30)
        elif ret_10d < 0:
            bias = "bearish"
            conf = min(70, 40 + hurst * 30)
        else:
            bias = "neutral"
            conf = 25
    else:
        regime = "random_walk"
        bias = "neutral"
        conf = 15

    return SignalOutput(
        signal_type="stat_arb",
        bias=bias,
        confidence=round(conf, 1),
        reasoning=f"Hurst={hurst:.3f} ({regime}), skew={skew:.2f}",
        weight=STRATEGY_WEIGHTS["stat_arb"],
    )


# ---------------------------------------------------------------------------
# Main ensemble
# ---------------------------------------------------------------------------

def compute_technical_ensemble(
    df: pd.DataFrame,
    vix_level: float | None = None,
) -> dict:
    """Run all 5 sub-strategies and return weighted aggregate.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: date, open, high, low, close, volume.
        At least 64 rows recommended for full feature set.
    vix_level : float | None
        Current VIX level (used by volatility strategy).

    Returns
    -------
    dict with keys:
        signals: list of SignalOutput (one per strategy)
        aggregate: dict from aggregate_signals()
        strategies: dict of per-strategy details
    """
    if df is None or len(df) < 20:
        return {"signals": [], "aggregate": aggregate_signals([]), "strategies": {}}

    signals: list[SignalOutput] = []
    strategies: dict[str, dict] = {}

    for name, fn in [
        ("trend", lambda: _trend_strategy(df)),
        ("mean_reversion", lambda: _mean_reversion_strategy(df)),
        ("momentum", lambda: _momentum_strategy(df)),
        ("volatility", lambda: _volatility_strategy(df, vix_level)),
        ("stat_arb", lambda: _stat_arb_strategy(df)),
    ]:
        try:
            signal = fn()
            signals.append(signal)
            strategies[name] = {
                "bias": signal.bias,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
            }
        except Exception as exc:
            LOG.warning("Ensemble strategy '%s' failed: %s", name, exc)
            strategies[name] = {"bias": "neutral", "confidence": 0, "error": str(exc)}

    return {
        "signals": signals,
        "aggregate": aggregate_signals(signals),
        "strategies": strategies,
    }

"""Tests for the multi-strategy technical ensemble."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader_koo.analysis.technical_ensemble import (
    _adx,
    _hurst_exponent,
    _rsi,
    compute_technical_ensemble,
)
from trader_koo.signals.types import SignalOutput, aggregate_signals


# ---------------------------------------------------------------------------
# Helper: generate OHLCV dataframe
# ---------------------------------------------------------------------------

def _make_ohlcv(closes: list[float], spread_pct: float = 1.0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    highs = closes_arr * (1 + spread_pct / 100)
    lows = closes_arr * (1 - spread_pct / 100)
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="B"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes_arr,
        "volume": np.full(n, 1_000_000),
    })


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_overbought(self):
        # Accelerating uptrend with occasional dips -> RSI > 60
        np.random.seed(99)
        prices = [100.0]
        for _ in range(49):
            prices.append(prices[-1] * (1 + np.random.uniform(0.005, 0.03)))
        close = pd.Series(prices, dtype=float)
        assert _rsi(close, 14) > 60

    def test_oversold(self):
        # Steadily falling prices -> RSI < 30
        prices = [200 - i * 2 for i in range(30)]
        close = pd.Series(prices, dtype=float)
        assert _rsi(close, 14) < 30

    def test_insufficient_data(self):
        close = pd.Series([100, 101, 102], dtype=float)
        assert _rsi(close, 14) == 50.0


class TestADX:
    def test_trending_market(self):
        # Strong uptrend -> ADX > 25
        n = 60
        close = pd.Series([100 + i * 1.5 for i in range(n)], dtype=float)
        high = close * 1.01
        low = close * 0.99
        assert _adx(high, low, close, 14) > 20

    def test_insufficient_data(self):
        close = pd.Series([100, 101], dtype=float)
        high = close * 1.01
        low = close * 0.99
        assert _adx(high, low, close, 14) == 0.0


class TestHurst:
    def test_random_walk(self):
        np.random.seed(42)
        series = pd.Series(np.random.randn(500).cumsum())
        h = _hurst_exponent(series.pct_change().dropna(), max_lag=20)
        # Random walk should be near 0.5 (+/- 0.25)
        assert 0.25 < h < 0.80

    def test_insufficient_data(self):
        series = pd.Series([0.01, -0.01, 0.02])
        assert _hurst_exponent(series, max_lag=20) == 0.5


# ---------------------------------------------------------------------------
# Signal types tests
# ---------------------------------------------------------------------------

class TestSignalOutput:
    def test_creation(self):
        s = SignalOutput("trend", "bullish", 75.0, "EMA12>EMA26", 0.25)
        assert s.bias == "bullish"
        assert s.confidence == 75.0
        assert s.weight == 0.25


class TestAggregateSignals:
    def test_all_bullish(self):
        signals = [
            SignalOutput("a", "bullish", 80, "test", 1.0),
            SignalOutput("b", "bullish", 60, "test", 1.0),
        ]
        result = aggregate_signals(signals)
        assert result["bias"] == "bullish"
        assert result["bull_score"] > 0
        assert result["bear_score"] == 0
        assert result["agreement_pct"] == 100.0

    def test_mixed_signals(self):
        signals = [
            SignalOutput("a", "bullish", 80, "test", 1.0),
            SignalOutput("b", "bearish", 80, "test", 1.0),
        ]
        result = aggregate_signals(signals)
        assert result["bias"] == "neutral"
        assert result["agreement_pct"] == 50.0

    def test_empty_signals(self):
        result = aggregate_signals([])
        assert result["bias"] == "neutral"
        assert result["signal_count"] == 0


# ---------------------------------------------------------------------------
# Ensemble integration tests
# ---------------------------------------------------------------------------

class TestComputeEnsemble:
    def test_uptrend_produces_bullish(self):
        # Strong uptrend: 100 -> 200 over 100 days
        closes = [100 + i for i in range(100)]
        df = _make_ohlcv(closes)
        result = compute_technical_ensemble(df, vix_level=18.0)

        assert len(result["signals"]) == 5
        assert result["aggregate"]["signal_count"] == 5
        # Most strategies should agree on bullish
        bull_count = sum(1 for s in result["signals"] if s.bias == "bullish")
        assert bull_count >= 2  # at least trend + momentum should agree

    def test_downtrend_produces_bearish(self):
        closes = [200 - i for i in range(100)]
        df = _make_ohlcv(closes)
        result = compute_technical_ensemble(df, vix_level=25.0)

        bear_count = sum(1 for s in result["signals"] if s.bias == "bearish")
        assert bear_count >= 2

    def test_insufficient_data_returns_empty(self):
        df = _make_ohlcv([100, 101, 102])
        result = compute_technical_ensemble(df)
        assert result["signals"] == []

    def test_strategies_dict_populated(self):
        closes = [100 + i * 0.5 for i in range(100)]
        df = _make_ohlcv(closes)
        result = compute_technical_ensemble(df)

        for name in ["trend", "mean_reversion", "momentum", "volatility", "stat_arb"]:
            assert name in result["strategies"]

from __future__ import annotations

import datetime as dt
import sqlite3

from trader_koo.crypto.market_insights import (
    build_btc_spy_correlation,
    build_crypto_market_structure,
)
from trader_koo.crypto.models import CryptoBar


def _conn_with_spy() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    base = dt.date(2026, 3, 17)
    for idx in range(30):
        date_str = (base - dt.timedelta(days=idx)).isoformat()
        close = 560.0 + idx * 2.0
        conn.execute(
            "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
            ("SPY", date_str, close),
        )
    conn.commit()
    return conn


def _btc_daily_bars(count: int = 30) -> list[CryptoBar]:
    base = dt.datetime(2026, 2, 16, 0, 0, tzinfo=dt.timezone.utc)
    bars: list[CryptoBar] = []
    for idx in range(count):
        close = 90000.0 + idx * 750.0
        ts = base + dt.timedelta(days=idx)
        bars.append(
            CryptoBar(
                symbol="BTC-USD",
                timestamp=ts,
                interval="1d",
                open=close - 200.0,
                high=close + 400.0,
                low=close - 450.0,
                close=close,
                volume=1000.0 + idx * 10.0,
            )
        )
    return bars


def test_build_btc_spy_correlation_returns_metrics():
    conn = _conn_with_spy()
    try:
        payload = build_btc_spy_correlation(
            conn,
            asset_symbol="BTC-USD",
            benchmark_symbol="SPY",
            asset_bars=_btc_daily_bars(),
        )
    finally:
        conn.close()

    assert payload["ok"] is True
    assert payload["symbol"] == "BTC-USD"
    assert payload["benchmark"] == "SPY"
    assert payload["sample_size"] >= 20
    assert payload["windows"]["20d"]["correlation"] is not None
    assert payload["windows"]["20d"]["asset_return_pct"] is not None
    assert payload["aligned_history"]


def test_build_btc_spy_correlation_handles_insufficient_overlap():
    conn = _conn_with_spy()
    bars = _btc_daily_bars(count=4)
    try:
        payload = build_btc_spy_correlation(
            conn,
            asset_symbol="BTC-USD",
            benchmark_symbol="SPY",
            asset_bars=bars,
        )
    finally:
        conn.close()

    assert payload["ok"] is False
    assert payload["relationship_label"] == "insufficient overlap"
    assert payload["windows"]["20d"]["correlation"] is None


def test_build_crypto_market_structure_summarizes_symbols():
    summaries = {
        "BTC-USD": {"price": 93000.0, "change_pct_24h": 3.2},
        "ETH-USD": {"price": 3400.0, "change_pct_24h": -1.4},
    }
    structures = [
        {
            "symbol": "BTC-USD",
            "context": {
                "latest_close": 93000.0,
                "level_context": "at_resistance",
                "ma_trend": "bullish",
                "support_level": 88000.0,
                "resistance_level": 94000.0,
                "pct_to_support": 5.68,
                "pct_to_resistance": 1.06,
                "atr_pct": 3.8,
                "momentum_20": 11.2,
                "realized_vol_20": 2.9,
                "range_position": 0.86,
            },
            "hmm_regime": None,
        },
        {
            "symbol": "ETH-USD",
            "context": {
                "latest_close": 3400.0,
                "level_context": "at_support",
                "ma_trend": "mixed",
                "support_level": 3325.0,
                "resistance_level": 3575.0,
                "pct_to_support": 2.26,
                "pct_to_resistance": 4.9,
                "atr_pct": 2.1,
                "momentum_20": -3.5,
                "realized_vol_20": 2.0,
                "range_position": 0.3,
            },
            "hmm_regime": None,
        },
    ]

    payload = build_crypto_market_structure(
        interval="1h",
        summaries=summaries,
        structures=structures,
    )

    assert payload["ok"] is True
    assert payload["overview"]["tracked_symbols"] == 2
    assert payload["overview"]["bullish_trend_count"] == 1
    assert payload["overview"]["at_support_count"] == 1
    assert payload["leaders"][0]["symbol"] == "BTC-USD"
    assert payload["laggards"][0]["symbol"] == "ETH-USD"

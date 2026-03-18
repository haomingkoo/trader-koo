from __future__ import annotations

import datetime as dt

from trader_koo.crypto.models import CryptoBar
from trader_koo.crypto.service import _aggregate_bars, _backfill_target_limit, get_crypto_history


def _bar(minute: int, open_: float, high: float, low: float, close: float, volume: float) -> CryptoBar:
    return CryptoBar(
        symbol="BTC-USD",
        timestamp=dt.datetime(2026, 3, 17, 0, minute, tzinfo=dt.timezone.utc),
        interval="1m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def test_aggregate_bars_to_5m_ohlcv():
    bars = [
        _bar(0, 100.0, 101.0, 99.5, 100.5, 10.0),
        _bar(1, 100.5, 102.0, 100.0, 101.5, 11.0),
        _bar(2, 101.5, 103.0, 101.0, 102.0, 12.0),
        _bar(3, 102.0, 103.5, 101.8, 103.0, 13.0),
        _bar(4, 103.0, 104.0, 102.5, 103.5, 14.0),
        _bar(5, 103.5, 104.2, 103.0, 104.0, 15.0),
    ]

    aggregated = _aggregate_bars(bars, "5m")

    assert len(aggregated) == 2
    first = aggregated[0]
    assert first.interval == "5m"
    assert first.open == 100.0
    assert first.high == 104.0
    assert first.low == 99.5
    assert first.close == 103.5
    assert first.volume == 60.0

    second = aggregated[1]
    assert second.open == 103.5
    assert second.close == 104.0


def test_backfill_target_limit_prefers_deeper_defaults():
    assert _backfill_target_limit("1d", 90) == 1825
    assert _backfill_target_limit("1w", 40) == 260
    assert _backfill_target_limit("1h", 240) == 2160
    assert _backfill_target_limit("30m", 500) == 2160
    assert _backfill_target_limit("1m", 720) == 10080
    assert _backfill_target_limit("1m", 12000) == 12000


def test_get_crypto_history_uses_native_interval_backfill(monkeypatch):
    bars = [
        CryptoBar(
            symbol="BTC-USD",
            timestamp=dt.datetime(2026, 3, 17, 0, 0, tzinfo=dt.timezone.utc) + dt.timedelta(days=idx),
            interval="1d",
            open=100.0 + idx,
            high=101.0 + idx,
            low=99.0 + idx,
            close=100.5 + idx,
            volume=10.0 + idx,
        )
        for idx in range(5)
    ]
    calls: list[tuple[str, int, str]] = []

    def fake_load(symbol: str, limit: int, interval: str = "1m") -> list[CryptoBar]:
        calls.append((symbol, limit, interval))
        if len(calls) == 1:
            return []
        return bars

    monkeypatch.setattr("trader_koo.crypto.service._load_recent_bars_from_db", fake_load)
    monkeypatch.setattr(
        "trader_koo.crypto.service._backfill_history",
        lambda symbol, interval, limit: bars,
    )

    result = get_crypto_history("BTC-USD", interval="1d", limit=5)

    assert len(result) == 5
    assert result[0].interval == "1d"
    assert calls[0][2] == "1d"


def test_get_crypto_history_supports_weekly_native_backfill(monkeypatch):
    bars = [
        CryptoBar(
            symbol="BTC-USD",
            timestamp=dt.datetime(2025, 1, 6, tzinfo=dt.timezone.utc) + dt.timedelta(weeks=idx),
            interval="1w",
            open=100.0 + idx,
            high=101.0 + idx,
            low=99.0 + idx,
            close=100.5 + idx,
            volume=10.0 + idx,
        )
        for idx in range(8)
    ]
    calls: list[tuple[str, int, str]] = []

    def fake_load(symbol: str, limit: int, interval: str = "1m") -> list[CryptoBar]:
        calls.append((symbol, limit, interval))
        if interval == "1w" and len(calls) > 1:
            return bars
        return []

    monkeypatch.setattr("trader_koo.crypto.service._load_recent_bars_from_db", fake_load)
    monkeypatch.setattr(
        "trader_koo.crypto.service._backfill_history",
        lambda symbol, interval, limit: bars,
    )

    result = get_crypto_history("BTC-USD", interval="1w", limit=8)

    assert len(result) == 8
    assert result[0].interval == "1w"
    assert calls[0][2] == "1w"


def test_get_crypto_history_refreshes_stale_native_interval_history(monkeypatch):
    stale_bars = [
        CryptoBar(
            symbol="BTC-USD",
            timestamp=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(hours=idx),
            interval="1h",
            open=100.0 + idx,
            high=101.0 + idx,
            low=99.0 + idx,
            close=100.5 + idx,
            volume=10.0 + idx,
        )
        for idx in range(2160)
    ]
    refreshed_tail = [
        CryptoBar(
            symbol="BTC-USD",
            timestamp=dt.datetime(2026, 3, 17, 0, 0, tzinfo=dt.timezone.utc) + dt.timedelta(hours=idx),
            interval="1h",
            open=500.0 + idx,
            high=501.0 + idx,
            low=499.0 + idx,
            close=500.5 + idx,
            volume=50.0 + idx,
        )
        for idx in range(24)
    ]
    refresh_calls: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        "trader_koo.crypto.service._load_recent_bars_from_db",
        lambda symbol, limit, interval="1m": stale_bars if interval == "1h" else [],
    )
    monkeypatch.setattr(
        "trader_koo.crypto.service._backfill_history",
        lambda symbol, interval, limit: (
            refresh_calls.append((symbol, interval, limit)) or refreshed_tail
        ),
    )
    monkeypatch.setattr("trader_koo.crypto.service._client", None)

    result = get_crypto_history("BTC-USD", interval="1h", limit=48)

    assert refresh_calls, "stale native 1h history should trigger a refresh"
    assert refresh_calls[0][1] == "1h"
    assert result[-1].timestamp == refreshed_tail[-1].timestamp

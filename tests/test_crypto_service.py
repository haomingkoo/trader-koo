from __future__ import annotations

import datetime as dt

from trader_koo.crypto.models import CryptoBar
from trader_koo.crypto.service import _aggregate_bars


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

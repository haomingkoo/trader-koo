from __future__ import annotations

import datetime as dt

from trader_koo.crypto.aggregator import CandleAggregator
from trader_koo.crypto.models import CryptoBar


def _bar(
    minute: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> CryptoBar:
    return CryptoBar(
        symbol="BTC-USD",
        timestamp=dt.datetime(2026, 3, 18, 0, minute, tzinfo=dt.timezone.utc),
        interval="1m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def test_forming_candle_replaces_same_minute_state_instead_of_double_counting_volume():
    agg = CandleAggregator()

    agg.on_bar_update(
        _bar(0, open_=100.0, high=101.0, low=99.5, close=100.5, volume=2.0),
    )
    agg.on_bar_update(
        _bar(0, open_=100.0, high=102.0, low=99.5, close=101.5, volume=5.0),
    )
    agg.on_bar_update(
        _bar(1, open_=101.5, high=103.0, low=101.0, close=102.0, volume=3.0),
    )

    forming = agg.get_forming("BTC-USD", "5m")

    assert forming is not None
    assert forming["open"] == 100.0
    assert forming["high"] == 103.0
    assert forming["low"] == 99.5
    assert forming["close"] == 102.0
    assert forming["volume"] == 8.0


def test_on_candle_close_finalizes_interval_with_latest_1m_bar_state():
    finalized: list[CryptoBar] = []
    agg = CandleAggregator(on_candle_finalized=finalized.append)

    bars = [
      _bar(0, open_=100.0, high=101.0, low=99.0, close=100.5, volume=1.0),
      _bar(1, open_=100.5, high=102.0, low=100.0, close=101.5, volume=2.0),
      _bar(2, open_=101.5, high=103.0, low=101.0, close=102.0, volume=3.0),
      _bar(3, open_=102.0, high=104.0, low=101.5, close=103.5, volume=4.0),
      _bar(4, open_=103.5, high=105.0, low=103.0, close=104.0, volume=5.0),
    ]

    for bar in bars:
        agg.on_bar_update(bar)
        agg.on_candle_close("BTC-USD", bar)

    assert len(finalized) == 1
    candle = finalized[0]
    assert candle.interval == "5m"
    assert candle.timestamp == dt.datetime(2026, 3, 18, 0, 0, tzinfo=dt.timezone.utc)
    assert candle.open == 100.0
    assert candle.high == 105.0
    assert candle.low == 99.0
    assert candle.close == 104.0
    assert candle.volume == 15.0

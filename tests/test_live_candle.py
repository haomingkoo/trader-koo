from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import patch

from trader_koo.streaming import live_candle


NOW = dt.datetime(2026, 8, 26, 14, 35, tzinfo=dt.timezone.utc)


class _FixedDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def _fixed_clock():
    return SimpleNamespace(datetime=_FixedDateTime, timezone=dt.timezone)


def setup_function() -> None:
    live_candle.clear()


def teardown_function() -> None:
    live_candle.clear()


def test_aggregates_one_session_candle_without_minute_flooring() -> None:
    first_tick = NOW - dt.timedelta(hours=5, seconds=53)
    live_candle.update_tick("AAPL", price=100.0, volume=10, timestamp=first_tick)
    live_candle.update_tick(
        "AAPL",
        price=103.0,
        volume=20,
        timestamp=NOW - dt.timedelta(seconds=30),
    )
    live_candle.update_tick(
        "AAPL",
        price=99.0,
        volume=5,
        timestamp=NOW,
    )

    with patch.object(live_candle, "dt", _fixed_clock()):
        candle = live_candle.get_forming_candle("AAPL")

    assert candle == {
        "timestamp": first_tick.isoformat(),
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 99.0,
        "volume": 35,
        "tick_count": 3,
        "forming": True,
    }


def test_five_minutes_is_freshness_cutoff_not_candle_duration() -> None:
    first_tick = NOW - dt.timedelta(hours=5)
    live_candle.update_tick("AAPL", price=100.0, volume=10, timestamp=first_tick)
    live_candle.update_tick(
        "AAPL",
        price=101.0,
        volume=5,
        timestamp=NOW - dt.timedelta(minutes=5),
    )

    with patch.object(live_candle, "dt", _fixed_clock()):
        candle = live_candle.get_forming_candle("AAPL")

    assert candle is not None
    assert candle["timestamp"] == first_tick.isoformat()
    assert candle["tick_count"] == 2


def test_hides_session_candle_after_latest_tick_exceeds_five_minutes() -> None:
    live_candle.update_tick(
        "AAPL",
        price=100.0,
        volume=10,
        timestamp=NOW - dt.timedelta(minutes=5, seconds=1),
    )

    with patch.object(live_candle, "dt", _fixed_clock()):
        candle = live_candle.get_forming_candle("AAPL")

    assert candle is None

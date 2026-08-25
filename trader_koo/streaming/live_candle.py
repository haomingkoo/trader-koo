"""Session-aware live candle aggregator for equities.

Receives live ticks from Finnhub and maintains one forming session candle
per symbol for the daily chart. The candle starts at the first observed tick
and accumulates until the process or explicit test state is reset.

Usage:
    from trader_koo.streaming.live_candle import update_tick, get_forming_candle

    # Called from the Finnhub WS on_tick callback:
    update_tick("AAPL", price=182.50, volume=100, timestamp=<datetime>)

    # Called from the dashboard endpoint:
    candle = get_forming_candle("AAPL")
"""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class FormingCandle:
    """An incomplete (still-forming) candle that aggregates live ticks.

    For the daily chart, this represents "today so far" — it accumulates
    all ticks since market open and never resets within the trading day.
    """

    symbol: str
    minute_start: dt.datetime  # timestamp of first tick (or floored to minute)
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int
    last_update: dt.datetime  # timestamp of most recent tick


_lock = threading.Lock()
_candles: dict[str, FormingCandle] = {}


def update_tick(
    symbol: str,
    *,
    price: float,
    volume: int,
    timestamp: dt.datetime,
) -> None:
    """Ingest one trade tick into the symbol's forming session candle."""
    with _lock:
        existing = _candles.get(symbol)

        if existing is None:
            _candles[symbol] = FormingCandle(
                symbol=symbol,
                minute_start=timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                tick_count=1,
                last_update=timestamp,
            )
        else:
            # Update the existing forming candle
            existing.high = max(existing.high, price)
            existing.low = min(existing.low, price)
            existing.close = price
            existing.volume += volume
            existing.tick_count += 1
            existing.last_update = timestamp


def get_forming_candle(symbol: str) -> dict[str, Any] | None:
    """Return the current forming candle for *symbol* as a dict.

    Returns ``None`` if there is no live data or the candle has received no
    tick for more than five minutes.
    """
    sym = symbol.upper()
    now = dt.datetime.now(dt.timezone.utc)

    with _lock:
        candle = _candles.get(sym)
        if candle is None:
            return None

    # Discard candles if no tick received in the last 5 minutes
    # (generous window — Finnhub can have gaps between trades)
    age_since_last_tick = (now - candle.last_update).total_seconds()
    if age_since_last_tick > 300:
        return None

    return {
        "timestamp": candle.minute_start.isoformat(),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "tick_count": candle.tick_count,
        "forming": True,
    }


def clear() -> None:
    """Remove all forming candles (used in tests)."""
    with _lock:
        _candles.clear()

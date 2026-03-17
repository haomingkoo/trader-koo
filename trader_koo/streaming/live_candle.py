"""Session-aware live candle aggregator for equities.

Receives live ticks from Finnhub and maintains a "forming" 1-min candle
per symbol. When a candle closes (on minute boundary), it is finalized
and the next one starts.

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
    """An incomplete (still-forming) 1-minute candle."""

    symbol: str
    minute_start: dt.datetime  # floored to minute boundary
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int


_lock = threading.Lock()
_candles: dict[str, FormingCandle] = {}


def _floor_to_minute(ts: dt.datetime) -> dt.datetime:
    """Floor a datetime to the start of its minute."""
    return ts.replace(second=0, microsecond=0)


def update_tick(
    symbol: str,
    *,
    price: float,
    volume: int,
    timestamp: dt.datetime,
) -> None:
    """Ingest a single trade tick and update the forming candle.

    If the tick belongs to a new minute, the old candle is discarded
    (it is now "closed") and a fresh candle begins.
    """
    minute = _floor_to_minute(timestamp)

    with _lock:
        existing = _candles.get(symbol)

        if existing is None or existing.minute_start != minute:
            # New candle for this minute
            _candles[symbol] = FormingCandle(
                symbol=symbol,
                minute_start=minute,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                tick_count=1,
            )
        else:
            # Update the existing forming candle
            existing.high = max(existing.high, price)
            existing.low = min(existing.low, price)
            existing.close = price
            existing.volume += volume
            existing.tick_count += 1


def get_forming_candle(symbol: str) -> dict[str, Any] | None:
    """Return the current forming candle for *symbol* as a dict.

    Returns ``None`` if there is no live data or the candle is stale
    (more than 2 minutes old).
    """
    sym = symbol.upper()
    now = dt.datetime.now(dt.timezone.utc)

    with _lock:
        candle = _candles.get(sym)
        if candle is None:
            return None

    # Discard stale candles (more than 2 minutes old)
    age_sec = (now - candle.minute_start).total_seconds()
    if age_sec > 120:
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

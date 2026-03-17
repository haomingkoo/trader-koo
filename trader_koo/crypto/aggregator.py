"""Server-side multi-interval candle aggregator.

Receives 1m kline updates from Binance and maintains forming candles for
all supported intervals. On interval boundary, finalizes the candle and
persists to SQLite.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from trader_koo.crypto.models import CryptoBar

LOG = logging.getLogger("trader_koo.crypto.aggregator")

INTERVALS = ("1m", "5m", "15m", "30m", "1h", "4h")

INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
}


@dataclass
class FormingCandle:
    """A candle that is still accumulating ticks."""

    symbol: str
    interval: str
    bucket_start: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int = 0
    minute_bars: dict[dt.datetime, CryptoBar] = field(default_factory=dict, repr=False)

    @classmethod
    def from_bar(
        cls,
        *,
        symbol: str,
        interval: str,
        bucket_start: dt.datetime,
        bar: CryptoBar,
    ) -> "FormingCandle":
        candle = cls(
            symbol=symbol,
            interval=interval,
            bucket_start=bucket_start,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            tick_count=1,
        )
        candle.minute_bars[bar.timestamp] = bar
        return candle

    def update_from_bar(self, bar: CryptoBar) -> None:
        self.minute_bars[bar.timestamp] = bar
        ordered = sorted(self.minute_bars.values(), key=lambda item: item.timestamp)
        first = ordered[0]
        last = ordered[-1]
        self.open = first.open
        self.high = max(item.high for item in ordered)
        self.low = min(item.low for item in ordered)
        self.close = last.close
        self.volume = sum(item.volume for item in ordered)
        self.tick_count += 1

    def to_bar(self) -> CryptoBar:
        return CryptoBar(
            symbol=self.symbol,
            timestamp=self.bucket_start,
            interval=self.interval,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )

    def to_dict(self, interval_minutes: int) -> dict:
        now = dt.datetime.now(dt.timezone.utc)
        elapsed = (now - self.bucket_start).total_seconds()
        total = interval_minutes * 60
        progress_pct = min(100.0, max(0.0, (elapsed / total) * 100)) if total > 0 else 0.0
        return {
            "timestamp": self.bucket_start.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "progress_pct": round(progress_pct, 1),
        }


def _floor_timestamp(ts: dt.datetime, interval_minutes: int) -> dt.datetime:
    """Floor a timestamp to the start of its interval bucket."""
    bucket = interval_minutes * 60
    floored = int(ts.timestamp()) // bucket * bucket
    return dt.datetime.fromtimestamp(floored, tz=dt.timezone.utc)


def _is_boundary(ts: dt.datetime, interval: str) -> bool:
    """Check if a timestamp falls on an interval boundary."""
    minute = ts.minute
    hour = ts.hour
    if interval == "5m":
        return minute % 5 == 0
    if interval == "15m":
        return minute % 15 == 0
    if interval == "30m":
        return minute % 30 == 0
    if interval == "1h":
        return minute == 0
    if interval == "4h":
        return hour % 4 == 0 and minute == 0
    return False


@dataclass
class CandleAggregator:
    """Maintains forming candles for multiple symbols and intervals.

    Thread-safe: all public methods acquire ``_lock`` before accessing
    internal state.
    """

    on_candle_finalized: Callable[[CryptoBar], None] | None = None
    _forming: dict[str, dict[str, FormingCandle]] = field(default_factory=dict)
    _latest_price: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def on_bar_update(self, bar: CryptoBar) -> None:
        """Called on every 1m kline update (forming or closed) from Binance.

        The incoming ``bar`` represents the current cumulative state of the
        active 1m candle. Higher-interval forming candles are rebuilt from the
        latest 1m bucket states so volume is not double-counted across
        repeated websocket updates.
        """
        with self._lock:
            symbol = bar.symbol
            self._latest_price[symbol] = bar.close
            if symbol not in self._forming:
                self._forming[symbol] = {}

            for interval in INTERVALS:
                minutes = INTERVAL_MINUTES[interval]
                bucket = _floor_timestamp(bar.timestamp, minutes)
                forming = self._forming[symbol].get(interval)

                if forming is None or forming.bucket_start != bucket:
                    self._forming[symbol][interval] = FormingCandle.from_bar(
                        symbol=symbol,
                        interval=interval,
                        bucket_start=bucket,
                        bar=bar,
                    )
                else:
                    forming.update_from_bar(bar)

    def on_candle_close(self, symbol: str, bar: CryptoBar) -> None:
        """Called when a 1m candle closes.

        Checks if any higher-interval boundaries are crossed, finalizes
        those candles, and notifies via callback.
        """
        finalized: list[CryptoBar] = []

        with self._lock:
            # The next minute boundary
            next_ts = bar.timestamp + dt.timedelta(minutes=1)

            for interval in INTERVALS:
                if interval == "1m":
                    continue
                if not _is_boundary(next_ts, interval):
                    continue

                forming = (self._forming.get(symbol) or {}).get(interval)
                if forming is None:
                    continue

                candle = forming.to_bar()
                finalized.append(candle)
                LOG.debug(
                    "Finalized %s candle for %s at %s",
                    interval,
                    symbol,
                    candle.timestamp.isoformat(),
                )

                # Clear the forming candle so the next tick starts fresh
                if symbol in self._forming:
                    self._forming[symbol].pop(interval, None)

        # Notify outside the lock to avoid deadlocks
        if self.on_candle_finalized:
            for candle in finalized:
                try:
                    self.on_candle_finalized(candle)
                except Exception as exc:
                    LOG.warning(
                        "on_candle_finalized callback error for %s [%s]: %s",
                        candle.symbol,
                        candle.interval,
                        exc,
                    )

    def get_forming(self, symbol: str, interval: str) -> dict | None:
        """Get the current forming candle for a symbol+interval."""
        with self._lock:
            forming = (self._forming.get(symbol) or {}).get(interval)
            if forming is None:
                return None
            minutes = INTERVAL_MINUTES.get(interval, 1)
            return forming.to_dict(minutes)

    def get_snapshot(self, symbol: str) -> dict:
        """Get latest price + all forming candles for a symbol."""
        with self._lock:
            price = self._latest_price.get(symbol)
            forming_map: dict[str, dict] = {}
            for interval in INTERVALS:
                forming = (self._forming.get(symbol) or {}).get(interval)
                if forming is not None:
                    minutes = INTERVAL_MINUTES.get(interval, 1)
                    forming_map[interval] = forming.to_dict(minutes)
            return {
                "symbol": symbol,
                "price": price,
                "forming": forming_map,
            }

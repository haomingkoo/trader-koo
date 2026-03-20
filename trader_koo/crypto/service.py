"""High-level crypto service — start/stop the feed and query data.

All public functions are thread-safe (the underlying BinanceWSClient
guards its state with a ``threading.Lock``).

The ``subscribe_ticks`` / ``unsubscribe_ticks`` functions allow the
FastAPI WebSocket endpoint to receive real-time ticks via asyncio queues.

Subscription-aware fan-out: each browser WebSocket can subscribe to a
specific symbol+interval pair. The server pushes only relevant tick,
forming-candle, and candle-close events to each client.

Persistence: closed bars are periodically flushed to SQLite so history
survives restarts. On startup, recent bars are loaded from the DB to
pre-fill the in-memory deque.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from trader_koo.crypto.aggregator import CandleAggregator
from trader_koo.crypto.binance_history import fetch_recent_klines
from trader_koo.crypto.binance_ws import BinanceWSClient, SYMBOL_MAP
from trader_koo.crypto.models import CryptoBar, CryptoTick
from trader_koo.crypto.storage import (
    ensure_crypto_schema,
    load_recent_bars,
    prune_old_bars,
    save_bars,
)

LOG = logging.getLogger("trader_koo.crypto.service")

# Module-level singletons
_client: BinanceWSClient | None = None
_aggregator: CandleAggregator | None = None
_db_path_str: str | None = None
_flush_thread: threading.Thread | None = None
_flush_running = False
_backfill_lock = threading.Lock()
_warm_backfill_thread: threading.Thread | None = None
_staleness_thread: threading.Thread | None = None
_staleness_running = False

# Staleness thresholds
_STALENESS_CHECK_INTERVAL_SEC = 60
_STALENESS_THRESHOLD_SEC = 300  # 5 minutes


@dataclass
class _Subscription:
    """Tracks what a single browser WS client is subscribed to."""

    queue: asyncio.Queue[dict[str, Any]]
    symbol: str | None = None   # None = all symbols (backward compatible)
    interval: str | None = None  # None = 1m default


# Browser WebSocket subscribers: sub_id → _Subscription
_subscribers_lock = threading.Lock()
_subscribers: dict[str, _Subscription] = {}

# Maximum concurrent browser WebSocket subscribers
MAX_WS_SUBSCRIBERS = 50

# Flush interval — reduced from 300s to 60s to minimize data loss on crash
FLUSH_INTERVAL_SEC = 60
_INTERVAL_TO_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}
_NATIVE_DB_INTERVALS = {
    interval for interval in _INTERVAL_TO_MINUTES.keys() if interval != "1m"
}
_BACKFILL_TARGETS: dict[str, int] = {
    "1m": 10080,   # ~7 days
    "5m": 2016,    # ~7 days
    "15m": 2880,   # ~30 days
    "30m": 2160,   # ~45 days
    "1h": 2160,    # ~90 days
    "2h": 2160,    # ~180 days
    "4h": 1440,    # ~240 days
    "6h": 1440,    # ~360 days
    "12h": 1095,   # ~1.5 years
    "1d": 1825,    # ~5 years
    "1w": 260,     # ~5 years
}
_RECENT_NATIVE_REFRESH_BARS: dict[str, int] = {
    "5m": 576,     # ~2 days
    "15m": 384,    # ~4 days
    "30m": 336,    # ~7 days
    "1h": 336,     # ~14 days
    "2h": 240,     # ~20 days
    "4h": 180,     # ~30 days
    "6h": 180,     # ~45 days
    "12h": 120,    # ~60 days
    "1d": 90,      # ~3 months
    "1w": 52,      # ~1 year
}


def subscribe_ticks(queue: asyncio.Queue[dict[str, Any]]) -> str | None:
    """Register a browser WebSocket to receive real-time tick pushes.

    Returns a subscription ID used to unsubscribe later, or ``None``
    if the maximum subscriber limit has been reached.

    By default the subscriber receives ALL ticks (backward compatible).
    Call ``update_subscription`` to narrow to a specific symbol+interval.
    """
    sub_id = uuid.uuid4().hex[:12]
    with _subscribers_lock:
        if len(_subscribers) >= MAX_WS_SUBSCRIBERS:
            LOG.warning("Max WS subscribers reached (%d), rejecting", MAX_WS_SUBSCRIBERS)
            return None
        _subscribers[sub_id] = _Subscription(queue=queue)
    LOG.debug("Tick subscriber added: %s (total: %d)", sub_id, len(_subscribers))
    return sub_id


def unsubscribe_ticks(sub_id: str) -> None:
    """Remove a browser WebSocket subscriber."""
    with _subscribers_lock:
        _subscribers.pop(sub_id, None)
    LOG.debug("Tick subscriber removed: %s (total: %d)", sub_id, len(_subscribers))


def update_subscription(sub_id: str, symbol: str | None, interval: str | None) -> None:
    """Update the symbol+interval filter for an existing subscriber."""
    with _subscribers_lock:
        sub = _subscribers.get(sub_id)
        if sub is not None:
            sub.symbol = symbol
            sub.interval = interval
    LOG.debug(
        "Subscription updated: %s → symbol=%s interval=%s",
        sub_id,
        symbol,
        interval,
    )


def get_aggregator() -> CandleAggregator | None:
    """Return the module-level aggregator (for WS endpoint access)."""
    return _aggregator


def _broadcast_tick(tick: CryptoTick) -> None:
    """Push a tick to all connected browser WebSocket subscribers.

    Called from the Binance WS thread — puts into asyncio queues which
    are drained by the FastAPI WebSocket handlers on the event loop.
    """
    tick_payload = {
        "type": "tick",
        "symbol": tick.symbol,
        "price": tick.price,
        "volume_24h": tick.volume_24h,
        "change_pct_24h": tick.change_pct_24h,
        "timestamp": tick.timestamp.isoformat(),
    }

    with _subscribers_lock:
        dead: list[str] = []
        for sub_id, sub in _subscribers.items():
            # Filter: if subscriber has a symbol filter, skip non-matching
            if sub.symbol is not None and sub.symbol != tick.symbol:
                continue

            try:
                sub.queue.put_nowait(tick_payload)
            except asyncio.QueueFull:
                dead.append(sub_id)

            # Also push forming candle for the subscriber's interval
            if _aggregator is not None and sub.interval is not None:
                forming = _aggregator.get_forming(tick.symbol, sub.interval)
                if forming is not None:
                    forming_payload = {
                        "type": "forming",
                        "symbol": tick.symbol,
                        "interval": sub.interval,
                        **forming,
                    }
                    try:
                        sub.queue.put_nowait(forming_payload)
                    except asyncio.QueueFull:
                        if sub_id not in dead:
                            dead.append(sub_id)

        for sub_id in dead:
            _subscribers.pop(sub_id, None)
            LOG.warning("Dropped slow subscriber: %s", sub_id)


def _on_candle_finalized(candle: CryptoBar) -> None:
    """Persist a finalized higher-interval candle and broadcast to subscribers."""
    # Persist to DB
    if _db_path_str:
        try:
            conn = sqlite3.connect(_db_path_str)
            try:
                save_bars(conn, [candle])
            finally:
                conn.close()
            LOG.info(
                "Persisted finalized %s candle for %s at %s",
                candle.interval,
                candle.symbol,
                candle.timestamp.isoformat(),
            )
        except Exception as exc:
            LOG.error(
                "Failed to persist finalized candle %s [%s]: %s",
                candle.symbol,
                candle.interval,
                exc,
            )

    # Broadcast candle_close to interested subscribers
    close_payload = {
        "type": "candle_close",
        "symbol": candle.symbol,
        "interval": candle.interval,
        "bar": {
            "timestamp": candle.timestamp.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        },
    }

    with _subscribers_lock:
        dead: list[str] = []
        for sub_id, sub in _subscribers.items():
            # Only send candle_close to clients that explicitly subscribed
            if sub.interval is None:
                continue
            if sub.interval != candle.interval:
                continue
            if sub.symbol is not None and sub.symbol != candle.symbol:
                continue
            try:
                sub.queue.put_nowait(close_payload)
            except asyncio.QueueFull:
                dead.append(sub_id)
        for sub_id in dead:
            _subscribers.pop(sub_id, None)
            LOG.warning("Dropped slow subscriber: %s", sub_id)


def _on_binance_bar_update(bar: CryptoBar) -> None:
    """Feed the current 1m kline state into the multi-interval aggregator."""
    if _aggregator is not None:
        _aggregator.on_bar_update(bar)


def _flush_loop() -> None:
    """Background thread: periodically flush pending bars to SQLite."""
    global _flush_running
    while _flush_running:
        time.sleep(FLUSH_INTERVAL_SEC)
        if not _flush_running:
            break
        _flush_bars_to_db()


def _flush_bars_to_db() -> None:
    """Drain pending bars from client and write to DB."""
    if _client is None or _db_path_str is None:
        return
    bars = _client.drain_pending_bars()
    if not bars:
        return
    try:
        conn = sqlite3.connect(_db_path_str)
        try:
            save_bars(conn, bars)
            prune_old_bars(conn, retention_days=30)
        finally:
            conn.close()
        LOG.info("Flushed %d crypto bars to DB", len(bars))
    except Exception as exc:
        LOG.error("Failed to flush crypto bars to DB: %s", exc)


def _load_history_from_db() -> None:
    """Pre-fill the in-memory deque from persisted bars."""
    if _client is None or _db_path_str is None:
        return
    try:
        conn = sqlite3.connect(_db_path_str)
        try:
            for display_symbol in SYMBOL_MAP.values():
                bars = load_recent_bars(conn, display_symbol, interval="1m", limit=1440)
                if bars:
                    _client.prepend_bars(display_symbol, bars)
        finally:
            conn.close()
    except Exception as exc:
        LOG.error("Failed to load crypto history from DB: %s", exc)


def _load_recent_bars_from_db(symbol: str, limit: int, interval: str = "1m") -> list[CryptoBar]:
    if not _db_path_str or limit <= 0:
        return []
    try:
        conn = sqlite3.connect(_db_path_str)
        try:
            return load_recent_bars(conn, symbol, interval=interval, limit=limit)
        finally:
            conn.close()
    except Exception as exc:
        LOG.error("Failed to load recent crypto bars for %s [%s]: %s", symbol, interval, exc)
        return []


def _backfill_history(symbol: str, interval: str, limit: int) -> list[CryptoBar]:
    if not _db_path_str or limit <= 0:
        return []

    with _backfill_lock:
        bars = fetch_recent_klines(symbol, interval, limit)
        if not bars:
            return []
        try:
            conn = sqlite3.connect(_db_path_str)
            try:
                ensure_crypto_schema(conn)
                save_bars(conn, bars)
                prune_old_bars(conn, retention_days=30)
            finally:
                conn.close()
        except Exception as exc:
            LOG.warning(
                "Failed to persist crypto backfill for %s [%s]: %s",
                symbol,
                interval,
                exc,
            )
            return []
        LOG.info(
            "Backfilled %d crypto bars for %s [%s]",
            len(bars),
            symbol,
            interval,
        )
        return bars


def _backfill_target_limit(interval: str, requested_limit: int) -> int:
    return max(int(requested_limit), _BACKFILL_TARGETS.get(interval, int(requested_limit)))


def _latest_closed_bucket_start(interval_minutes: int, now: dt.datetime | None = None) -> dt.datetime:
    now_utc = now or dt.datetime.now(dt.timezone.utc)
    current_bucket = _floor_timestamp(now_utc, interval_minutes)
    return current_bucket - dt.timedelta(minutes=interval_minutes)


def _native_history_is_stale(bars: list[CryptoBar], interval: str) -> bool:
    """Check if bars are stale OR have gaps that need backfilling."""
    if not bars:
        return True

    interval_minutes = _INTERVAL_TO_MINUTES.get(interval)
    if interval_minutes is None:
        return False

    latest = bars[-1].timestamp
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=dt.timezone.utc)
    else:
        latest = latest.astimezone(dt.timezone.utc)

    expected_latest = _latest_closed_bucket_start(interval_minutes)
    if latest < expected_latest:
        return True

    # Check for gaps in the middle of the dataset
    # A gap = two consecutive bars more than 2x the interval apart
    gap_threshold = dt.timedelta(minutes=interval_minutes * 2)
    for i in range(1, min(len(bars), 500)):  # check last 500 bars
        t_prev = bars[i - 1].timestamp
        t_curr = bars[i].timestamp
        if t_prev.tzinfo is None:
            t_prev = t_prev.replace(tzinfo=dt.timezone.utc)
        if t_curr.tzinfo is None:
            t_curr = t_curr.replace(tzinfo=dt.timezone.utc)
        if (t_curr - t_prev) > gap_threshold:
            LOG.warning(
                "Gap detected in %s history: %s → %s (%.1fh)",
                interval,
                t_prev.isoformat(),
                t_curr.isoformat(),
                (t_curr - t_prev).total_seconds() / 3600,
            )
            return True

    return False


def _refresh_recent_native_history(
    symbol: str,
    interval: str,
    existing_bars: list[CryptoBar],
    requested_limit: int,
) -> list[CryptoBar]:
    refresh_limit = min(
        _backfill_target_limit(interval, _RECENT_NATIVE_REFRESH_BARS.get(interval, requested_limit)),
        _BACKFILL_TARGETS.get(interval, requested_limit),
    )
    refreshed = _backfill_history(symbol, interval, refresh_limit)
    if not refreshed:
        return existing_bars
    return _merge_bars(refreshed, existing_bars)


def _on_ws_gap_detected(disconnect_ts: dt.datetime, reconnect_ts: dt.datetime) -> None:
    """Called by the WS client when it detects a gap after reconnecting.

    Backfills 1m and 1h bars for the gap window from Binance REST API.
    """
    gap_minutes = int((reconnect_ts - disconnect_ts).total_seconds() / 60)
    LOG.info("Gap-fill triggered: %d minutes gap, backfilling...", gap_minutes)

    for symbol in SYMBOL_MAP.values():
        # Backfill 1m bars for the gap
        limit_1m = min(gap_minutes + 10, 1000)
        try:
            _backfill_history(symbol, "1m", limit_1m)
        except Exception as exc:
            LOG.warning("1m gap-fill failed for %s: %s", symbol, exc)

        # Also refresh 1h to cover the gap
        limit_1h = max(gap_minutes // 60 + 2, 10)
        try:
            _backfill_history(symbol, "1h", limit_1h)
        except Exception as exc:
            LOG.warning("1h gap-fill failed for %s: %s", symbol, exc)

    LOG.info("Gap-fill complete for %d-minute window", gap_minutes)


def _warm_backfill_history() -> None:
    if _db_path_str is None:
        return
    for symbol in SYMBOL_MAP.values():
        for interval, target_limit in (("1h", 2160), ("4h", 1440), ("12h", 1095), ("1d", 1825), ("1w", 260)):
            existing = _load_recent_bars_from_db(symbol, limit=target_limit, interval=interval)
            if len(existing) >= target_limit:
                continue
            try:
                _backfill_history(symbol, interval, target_limit)
            except Exception as exc:
                LOG.debug(
                    "Warm crypto backfill skipped for %s [%s]: %s",
                    symbol,
                    interval,
                    exc,
                )


def _merge_bars(primary: list[CryptoBar], secondary: list[CryptoBar]) -> list[CryptoBar]:
    merged: dict[tuple[str, dt.datetime], CryptoBar] = {}
    for bar in secondary:
        merged[(bar.interval, bar.timestamp)] = bar
    for bar in primary:
        merged[(bar.interval, bar.timestamp)] = bar
    return sorted(merged.values(), key=lambda bar: bar.timestamp)


def _floor_timestamp(ts: dt.datetime, interval_minutes: int) -> dt.datetime:
    bucket = interval_minutes * 60
    floored = int(ts.timestamp()) // bucket * bucket
    return dt.datetime.fromtimestamp(floored, tz=dt.timezone.utc)


def _aggregate_bars(bars: list[CryptoBar], interval: str) -> list[CryptoBar]:
    interval_minutes = _INTERVAL_TO_MINUTES.get(interval)
    if interval_minutes is None or interval == "1m":
        return list(bars)
    if not bars:
        return []

    grouped: list[CryptoBar] = []
    current: CryptoBar | None = None
    current_bucket: dt.datetime | None = None

    for bar in sorted(bars, key=lambda item: item.timestamp):
        bucket = _floor_timestamp(bar.timestamp, interval_minutes)
        if current is None or current_bucket != bucket:
            if current is not None:
                grouped.append(current)
            current_bucket = bucket
            current = CryptoBar(
                symbol=bar.symbol,
                timestamp=bucket,
                interval=interval,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            continue
        current.high = max(current.high, bar.high)
        current.low = min(current.low, bar.low)
        current.close = bar.close
        current.volume += bar.volume

    if current is not None:
        grouped.append(current)
    return grouped


def _on_binance_candle_close(bar: CryptoBar) -> None:
    """Called when Binance sends a closed 1m candle.

    Forwards to the aggregator so it can finalize any higher-interval
    candles whose boundaries have been crossed.
    """
    if _aggregator is not None:
        _aggregator.on_candle_close(bar.symbol, bar)


def _staleness_loop() -> None:
    """Background thread: check WS staleness every 60 seconds."""
    while _staleness_running:
        time.sleep(_STALENESS_CHECK_INTERVAL_SEC)
        if not _staleness_running or _client is None:
            break
        last_msg = _client.last_message_at
        if last_msg == 0:
            continue
        age = time.monotonic() - last_msg
        if age > _STALENESS_THRESHOLD_SEC:
            LOG.warning(
                "Crypto WS stale: no message for %.0f seconds — forcing reconnect",
                age,
            )
            try:
                ws = _client._ws
                if ws is not None:
                    ws.close()
            except Exception as exc:
                LOG.debug("Staleness reconnect close error: %s", exc)


def get_crypto_ws_health() -> dict[str, Any]:
    """Return health status for the crypto WebSocket connection."""
    if _client is None:
        return {
            "connected": False,
            "last_message_ago_sec": None,
            "symbols": 0,
        }
    last_msg = _client.last_message_at
    if last_msg == 0:
        ago: float | None = None
    else:
        ago = round(time.monotonic() - last_msg, 1)
    return {
        "connected": _client.connected,
        "last_message_ago_sec": ago,
        "symbols": _client.symbol_count,
    }


def start_crypto_feed(db_path_str: str | None = None) -> None:
    """Start the Binance WebSocket feed in a background daemon thread.

    If *db_path_str* is provided, bars are persisted to that SQLite DB
    and history is pre-loaded on startup.
    """
    global _client, _aggregator, _db_path_str, _flush_thread, _flush_running
    global _warm_backfill_thread, _staleness_thread, _staleness_running
    if _client is not None:
        LOG.warning("Crypto feed already started — ignoring duplicate call")
        return

    _db_path_str = db_path_str

    # Create the multi-interval aggregator
    _aggregator = CandleAggregator(on_candle_finalized=_on_candle_finalized)
    LOG.info("CandleAggregator initialized")

    _client = BinanceWSClient(
        on_tick=_broadcast_tick,
        on_bar_update=_on_binance_bar_update,
        on_candle_close=_on_binance_candle_close,
        on_gap_detected=_on_ws_gap_detected,
    )

    # Pre-fill from DB before starting live feed
    if _db_path_str:
        _load_history_from_db()

    _client.start()

    if _db_path_str:
        _warm_backfill_thread = threading.Thread(
            target=_warm_backfill_history,
            name="crypto-history-backfill",
            daemon=True,
        )
        _warm_backfill_thread.start()
        LOG.info("Crypto history warm backfill started")

    # Start periodic DB flush thread
    if _db_path_str:
        _flush_running = True
        _flush_thread = threading.Thread(
            target=_flush_loop,
            name="crypto-db-flush",
            daemon=True,
        )
        _flush_thread.start()
        LOG.info("Crypto DB flush thread started (interval=%ds)", FLUSH_INTERVAL_SEC)

    # Start staleness monitor
    _staleness_running = True
    _staleness_thread = threading.Thread(
        target=_staleness_loop,
        name="crypto-ws-staleness",
        daemon=True,
    )
    _staleness_thread.start()
    LOG.info("Crypto WS staleness monitor started")

    LOG.info("Crypto feed started")


def stop_crypto_feed() -> None:
    """Gracefully stop the WebSocket feed and flush remaining bars."""
    global _client, _aggregator, _flush_running, _flush_thread
    global _staleness_running, _staleness_thread
    _flush_running = False
    _staleness_running = False

    # Final flush before shutdown
    _flush_bars_to_db()

    if _client is None:
        return
    _client.stop()
    _client = None
    _aggregator = None
    _flush_thread = None
    _staleness_thread = None
    LOG.info("Crypto feed stopped")


def get_crypto_prices() -> dict[str, CryptoTick]:
    """Return the latest tick for every tracked symbol.

    Returns an empty dict if the feed has not started or no data received yet.
    """
    if _client is None:
        return {}
    return _client.get_latest_ticks()


def get_crypto_history(
    symbol: str,
    interval: str = "1m",
    limit: int = 100,
) -> list[CryptoBar]:
    """Return recent bars from the in-memory buffer.

    ``1m`` bars come from the live Binance websocket and DB cache.
    Longer intervals are served from native Binance REST backfills where
    possible, with 1m aggregation fallback for intraday views.
    """
    interval_norm = str(interval or "1m").lower()
    if interval_norm not in _INTERVAL_TO_MINUTES:
        LOG.debug("Unsupported crypto interval requested: %s", interval)
        return []

    if interval_norm in _NATIVE_DB_INTERVALS:
        native_limit = _backfill_target_limit(interval_norm, limit)
        native_bars = _load_recent_bars_from_db(symbol, limit=native_limit, interval=interval_norm)
        if len(native_bars) < native_limit:
            _backfill_history(symbol, interval_norm, native_limit)
            native_bars = _load_recent_bars_from_db(symbol, limit=native_limit, interval=interval_norm)
        elif _native_history_is_stale(native_bars, interval_norm):
            native_bars = _refresh_recent_native_history(
                symbol,
                interval_norm,
                native_bars,
                requested_limit=limit,
            )
        if interval_norm != "1w":
            minutes = _INTERVAL_TO_MINUTES[interval_norm]
            patch_limit = max(minutes * 4, 240)
            live_bars = _client.get_bars(symbol, limit=patch_limit) if _client is not None else []
            if len(live_bars) < patch_limit:
                recent_db_bars = _load_recent_bars_from_db(symbol, limit=patch_limit, interval="1m")
                recent_base_bars = _merge_bars(live_bars, recent_db_bars)
            else:
                recent_base_bars = live_bars
            if recent_base_bars:
                patched_bars = _aggregate_bars(recent_base_bars, interval_norm)
                native_bars = _merge_bars(patched_bars, native_bars)
        if native_bars:
            return native_bars[-limit:]

    minutes = _INTERVAL_TO_MINUTES[interval_norm]
    base_limit = limit if interval_norm == "1m" else max(limit * minutes + minutes, 240)

    live_bars = _client.get_bars(symbol, limit=base_limit) if _client is not None else []
    if len(live_bars) < base_limit:
        db_bars = _load_recent_bars_from_db(symbol, limit=base_limit, interval="1m")
        base_bars = _merge_bars(live_bars, db_bars)
    else:
        base_bars = live_bars

    if len(base_bars) < base_limit and interval_norm in {"1m", "5m", "15m"}:
        backfill_limit = _backfill_target_limit(interval_norm, base_limit)
        _backfill_history(symbol, "1m", backfill_limit)
        db_bars = _load_recent_bars_from_db(symbol, limit=backfill_limit, interval="1m")
        base_bars = _merge_bars(live_bars, db_bars)

    if interval_norm == "1m":
        return base_bars[-limit:]

    aggregated = _aggregate_bars(base_bars, interval_norm)
    return aggregated[-limit:]


def get_forming_candle(symbol: str, interval: str) -> CryptoBar | None:
    """Get the current forming (incomplete) candle for a higher timeframe.

    Aggregates recent 1m bars since the last interval boundary.
    E.g., for 5m at 14:23, aggregates the 1m bars from 14:20-14:23.

    Returns ``None`` for 1m (every 1m bar from the stream is already
    near-real-time) or if insufficient data is available.
    """
    interval_norm = str(interval or "1m").lower()
    interval_minutes = _INTERVAL_TO_MINUTES.get(interval_norm)
    if interval_minutes is None or interval_minutes <= 1:
        return None

    now = dt.datetime.now(dt.timezone.utc)
    bucket_start = _floor_timestamp(now, interval_minutes)

    # Fetch enough 1m bars to cover the current interval window
    bars_needed = interval_minutes + 1
    live_bars = _client.get_bars(symbol, limit=bars_needed) if _client is not None else []
    if len(live_bars) < 1:
        recent_db = _load_recent_bars_from_db(symbol, limit=bars_needed, interval="1m")
        live_bars = _merge_bars(live_bars, recent_db)

    # Filter to bars within the current bucket
    bucket_bars = [
        b for b in live_bars
        if b.timestamp >= bucket_start
    ]
    if not bucket_bars:
        return None

    # Aggregate into a single forming candle
    first = bucket_bars[0]
    candle = CryptoBar(
        symbol=symbol,
        timestamp=bucket_start,
        interval=interval_norm,
        open=first.open,
        high=first.high,
        low=first.low,
        close=first.close,
        volume=first.volume,
    )
    for bar in bucket_bars[1:]:
        candle.high = max(candle.high, bar.high)
        candle.low = min(candle.low, bar.low)
        candle.close = bar.close
        candle.volume += bar.volume

    return candle


def get_crypto_summary() -> dict[str, Any]:
    """Return a summary payload suitable for the frontend header.

    Shape::

        {
            "prices": {
                "BTC-USD": { ...CryptoTick fields... },
                "ETH-USD": { ...CryptoTick fields... },
            },
            "connected": true/false,
        }
    """
    if _client is None:
        return {"prices": {}, "connected": False}

    ticks = _client.get_latest_ticks()
    prices: dict[str, dict[str, Any]] = {}
    for symbol, tick in ticks.items():
        prices[symbol] = {
            "symbol": tick.symbol,
            "price": tick.price,
            "volume_24h": tick.volume_24h,
            "change_pct_24h": tick.change_pct_24h,
            "timestamp": tick.timestamp.isoformat(),
        }

    return {
        "prices": prices,
        "connected": _client.connected,
    }

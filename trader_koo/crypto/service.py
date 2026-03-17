"""High-level crypto service — start/stop the feed and query data.

All public functions are thread-safe (the underlying BinanceWSClient
guards its state with a ``threading.Lock``).

The ``subscribe_ticks`` / ``unsubscribe_ticks`` functions allow the
FastAPI WebSocket endpoint to receive real-time ticks via asyncio queues.

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
from typing import Any

from trader_koo.crypto.binance_ws import BinanceWSClient, SYMBOL_MAP
from trader_koo.crypto.models import CryptoBar, CryptoTick
from trader_koo.crypto.storage import (
    ensure_crypto_schema,
    load_recent_bars,
    prune_old_bars,
    save_bars,
)

LOG = logging.getLogger("trader_koo.crypto.service")

# Module-level singleton
_client: BinanceWSClient | None = None
_db_path_str: str | None = None
_flush_thread: threading.Thread | None = None
_flush_running = False

# Browser WebSocket subscribers: sub_id → asyncio.Queue
_subscribers_lock = threading.Lock()
_subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}

# Maximum concurrent browser WebSocket subscribers
MAX_WS_SUBSCRIBERS = 50

# Flush interval in seconds (5 minutes)
FLUSH_INTERVAL_SEC = 300
_INTERVAL_TO_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def subscribe_ticks(queue: asyncio.Queue[dict[str, Any]]) -> str | None:
    """Register a browser WebSocket to receive real-time tick pushes.

    Returns a subscription ID used to unsubscribe later, or ``None``
    if the maximum subscriber limit has been reached.
    """
    sub_id = uuid.uuid4().hex[:12]
    with _subscribers_lock:
        if len(_subscribers) >= MAX_WS_SUBSCRIBERS:
            LOG.warning("Max WS subscribers reached (%d), rejecting", MAX_WS_SUBSCRIBERS)
            return None
        _subscribers[sub_id] = queue
    LOG.debug("Tick subscriber added: %s (total: %d)", sub_id, len(_subscribers))
    return sub_id


def unsubscribe_ticks(sub_id: str) -> None:
    """Remove a browser WebSocket subscriber."""
    with _subscribers_lock:
        _subscribers.pop(sub_id, None)
    LOG.debug("Tick subscriber removed: %s (total: %d)", sub_id, len(_subscribers))


def _broadcast_tick(tick: CryptoTick) -> None:
    """Push a tick to all connected browser WebSocket subscribers.

    Called from the Binance WS thread — puts into asyncio queues which
    are drained by the FastAPI WebSocket handlers on the event loop.
    """
    payload = {
        "symbol": tick.symbol,
        "price": tick.price,
        "volume_24h": tick.volume_24h,
        "change_pct_24h": tick.change_pct_24h,
        "timestamp": tick.timestamp.isoformat(),
    }
    with _subscribers_lock:
        dead: list[str] = []
        for sub_id, queue in _subscribers.items():
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(sub_id)
        for sub_id in dead:
            _subscribers.pop(sub_id, None)
            LOG.warning("Dropped slow subscriber: %s", sub_id)


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
                bars = load_recent_bars(conn, display_symbol, limit=1440)
                if bars:
                    _client.prepend_bars(display_symbol, bars)
        finally:
            conn.close()
    except Exception as exc:
        LOG.error("Failed to load crypto history from DB: %s", exc)


def _load_recent_bars_from_db(symbol: str, limit: int) -> list[CryptoBar]:
    if not _db_path_str or limit <= 0:
        return []
    try:
        conn = sqlite3.connect(_db_path_str)
        try:
            return load_recent_bars(conn, symbol, limit=limit)
        finally:
            conn.close()
    except Exception as exc:
        LOG.error("Failed to load recent crypto bars for %s: %s", symbol, exc)
        return []


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


def start_crypto_feed(db_path_str: str | None = None) -> None:
    """Start the Binance WebSocket feed in a background daemon thread.

    If *db_path_str* is provided, bars are persisted to that SQLite DB
    and history is pre-loaded on startup.
    """
    global _client, _db_path_str, _flush_thread, _flush_running
    if _client is not None:
        LOG.warning("Crypto feed already started — ignoring duplicate call")
        return

    _db_path_str = db_path_str
    _client = BinanceWSClient(on_tick=_broadcast_tick)

    # Pre-fill from DB before starting live feed
    if _db_path_str:
        _load_history_from_db()

    _client.start()

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

    LOG.info("Crypto feed started")


def stop_crypto_feed() -> None:
    """Gracefully stop the WebSocket feed and flush remaining bars."""
    global _client, _flush_running, _flush_thread
    _flush_running = False

    # Final flush before shutdown
    _flush_bars_to_db()

    if _client is None:
        return
    _client.stop()
    _client = None
    _flush_thread = None
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

    Currently only ``1m`` bars are stored; the ``interval`` parameter is
    accepted for forward-compatibility but only ``"1m"`` returns data.
    """
    interval_norm = str(interval or "1m").lower()
    if interval_norm not in _INTERVAL_TO_MINUTES:
        LOG.debug("Unsupported crypto interval requested: %s", interval)
        return []

    minutes = _INTERVAL_TO_MINUTES[interval_norm]
    base_limit = limit if interval_norm == "1m" else max(limit * minutes + minutes, 240)

    live_bars = _client.get_bars(symbol, limit=base_limit) if _client is not None else []
    if len(live_bars) < base_limit:
        db_bars = _load_recent_bars_from_db(symbol, limit=base_limit)
        base_bars = _merge_bars(live_bars, db_bars)
    else:
        base_bars = live_bars

    if interval_norm == "1m":
        return base_bars[-limit:]

    aggregated = _aggregate_bars(base_bars, interval_norm)
    return aggregated[-limit:]


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

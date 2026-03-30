"""High-level equity streaming service — start/stop the feed and query data.

All public functions are thread-safe (the underlying FinnhubWSClient
guards its state with a ``threading.Lock``).

The ``subscribe_equity_ticks`` / ``unsubscribe_equity_ticks`` functions
allow the FastAPI WebSocket endpoint to receive real-time ticks via
asyncio queues (same pattern as the crypto service).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import threading
import time
import uuid
from typing import Any

from trader_koo.streaming.finnhub_ws import FinnhubWSClient, MAX_SUBSCRIPTIONS
from trader_koo.streaming.live_candle import (
    get_forming_candle as _get_forming_candle,
    update_tick as _live_candle_update,
)

LOG = logging.getLogger("trader_koo.streaming.service")

# Module-level singleton
_client: FinnhubWSClient | None = None

# Browser WebSocket subscribers: sub_id -> asyncio.Queue
_subscribers_lock = threading.Lock()
_subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}

# Maximum concurrent browser WebSocket subscribers
MAX_WS_SUBSCRIBERS = 50

# Staleness monitoring
_staleness_thread: threading.Thread | None = None
_staleness_running = False
_STALENESS_CHECK_INTERVAL_SEC = 60
_STALENESS_THRESHOLD_SEC = 300  # 5 minutes


def subscribe_equity_ticks(queue: asyncio.Queue[dict[str, Any]]) -> str | None:
    """Register a browser WebSocket to receive real-time equity tick pushes.

    Returns a subscription ID used to unsubscribe later, or ``None``
    if the maximum subscriber limit has been reached.
    """
    sub_id = uuid.uuid4().hex[:12]
    with _subscribers_lock:
        if len(_subscribers) >= MAX_WS_SUBSCRIBERS:
            LOG.warning("Max equity WS subscribers reached (%d), rejecting", MAX_WS_SUBSCRIBERS)
            return None
        _subscribers[sub_id] = queue
    LOG.debug(
        "Equity tick subscriber added: %s (total: %d)",
        sub_id,
        len(_subscribers),
    )
    return sub_id


def unsubscribe_equity_ticks(sub_id: str) -> None:
    """Remove a browser WebSocket subscriber."""
    with _subscribers_lock:
        _subscribers.pop(sub_id, None)
    LOG.debug(
        "Equity tick subscriber removed: %s (total: %d)",
        sub_id,
        len(_subscribers),
    )


def _broadcast_tick(tick: dict) -> None:
    """Push a tick to all connected browser WebSocket subscribers.

    Called from the Finnhub WS thread — puts into asyncio queues which
    are drained by the FastAPI WebSocket handlers on the event loop.
    Also feeds the live candle aggregator.
    """
    # Feed live candle aggregator
    symbol = tick.get("symbol")
    price = tick.get("price")
    if symbol and price is not None:
        try:
            ts_raw = tick.get("timestamp", "")
            ts = (
                dt.datetime.fromisoformat(ts_raw)
                if ts_raw
                else dt.datetime.now(dt.timezone.utc)
            )
            _live_candle_update(
                symbol,
                price=float(price),
                volume=int(tick.get("volume", 0)),
                timestamp=ts,
            )
        except (ValueError, TypeError):
            pass

    with _subscribers_lock:
        dead: list[str] = []
        for sub_id, queue in _subscribers.items():
            try:
                queue.put_nowait(tick)
            except asyncio.QueueFull:
                dead.append(sub_id)
        for sub_id in dead:
            _subscribers.pop(sub_id, None)
            LOG.warning("Dropped slow equity subscriber: %s", sub_id)


def _staleness_loop() -> None:
    """Background thread: check equity WS staleness every 60 seconds."""
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
                "Equity WS stale: no message for %.0f seconds — forcing reconnect",
                age,
            )
            try:
                ws = _client._ws
                if ws is not None:
                    ws.close()
            except Exception as exc:
                LOG.debug("Staleness reconnect close error: %s", exc)


def get_equity_ws_health() -> dict[str, Any]:
    """Return health status for the equity WebSocket connection."""
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
        "symbols": _client.get_subscription_count(),
    }


def start_equity_feed(api_key: str) -> None:
    """Start the Finnhub WebSocket feed in a background daemon thread."""
    global _client, _staleness_thread, _staleness_running
    if _client is not None:
        LOG.warning("Equity feed already started — ignoring duplicate call")
        return

    _client = FinnhubWSClient(
        api_key=api_key,
        always_on=["SPY", "QQQ", "DIA"],
        on_tick=_broadcast_tick,
    )
    _client.start()

    # Start staleness monitor
    _staleness_running = True
    _staleness_thread = threading.Thread(
        target=_staleness_loop,
        name="equity-ws-staleness",
        daemon=True,
    )
    _staleness_thread.start()
    LOG.info("Equity feed started (Finnhub WS for SPY/QQQ/DIA + on-demand)")


def stop_equity_feed() -> None:
    """Gracefully stop the Finnhub WebSocket feed."""
    global _client, _staleness_running, _staleness_thread
    _staleness_running = False
    if _client is None:
        return
    _client.stop()
    _client = None
    _staleness_thread = None
    LOG.info("Equity feed stopped")


def subscribe_symbol(symbol: str) -> bool:
    """Subscribe to real-time data for a symbol. Returns False if at limit."""
    if _client is None:
        LOG.warning("Equity feed not running — cannot subscribe to %s", symbol)
        return False
    return _client.subscribe(symbol)


def unsubscribe_symbol(symbol: str) -> bool:
    """Unsubscribe from a symbol. Cannot unsubscribe always-on symbols."""
    if _client is None:
        return False
    return _client.unsubscribe(symbol)


def get_equity_price(symbol: str) -> dict | None:
    """Get latest price for a symbol."""
    if _client is None:
        return None
    return _client.get_price(symbol)


def get_equity_prices() -> dict[str, dict]:
    """Get all currently streaming equity prices."""
    if _client is None:
        return {}
    return _client.get_all_prices()


def get_forming_candle(symbol: str) -> dict | None:
    """Return the current forming 1-min candle for *symbol*, or None."""
    return _get_forming_candle(symbol)


def get_subscription_info() -> dict:
    """Return subscription count, max, connected status, and symbol list."""
    if _client is None:
        return {
            "connected": False,
            "subscribed_count": 0,
            "max_symbols": MAX_SUBSCRIPTIONS,
            "symbols": [],
        }
    return {
        "connected": _client.connected,
        "subscribed_count": _client.get_subscription_count(),
        "max_symbols": MAX_SUBSCRIPTIONS,
        "symbols": _client.get_subscribed_symbols(),
    }

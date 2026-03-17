"""High-level equity streaming service — start/stop the feed and query data.

All public functions are thread-safe (the underlying FinnhubWSClient
guards its state with a ``threading.Lock``).

The ``subscribe_equity_ticks`` / ``unsubscribe_equity_ticks`` functions
allow the FastAPI WebSocket endpoint to receive real-time ticks via
asyncio queues (same pattern as the crypto service).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Any

from trader_koo.streaming.finnhub_ws import FinnhubWSClient, MAX_SUBSCRIPTIONS

LOG = logging.getLogger("trader_koo.streaming.service")

# Module-level singleton
_client: FinnhubWSClient | None = None

# Browser WebSocket subscribers: sub_id -> asyncio.Queue
_subscribers_lock = threading.Lock()
_subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}


def subscribe_equity_ticks(queue: asyncio.Queue[dict[str, Any]]) -> str:
    """Register a browser WebSocket to receive real-time equity tick pushes.

    Returns a subscription ID used to unsubscribe later.
    """
    sub_id = uuid.uuid4().hex[:12]
    with _subscribers_lock:
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
    """
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


def start_equity_feed(api_key: str) -> None:
    """Start the Finnhub WebSocket feed in a background daemon thread."""
    global _client
    if _client is not None:
        LOG.warning("Equity feed already started — ignoring duplicate call")
        return

    _client = FinnhubWSClient(
        api_key=api_key,
        always_on=["SPY", "QQQ"],
        on_tick=_broadcast_tick,
    )
    _client.start()
    LOG.info("Equity feed started (Finnhub WS for SPY/QQQ + on-demand)")


def stop_equity_feed() -> None:
    """Gracefully stop the Finnhub WebSocket feed."""
    global _client
    if _client is None:
        return
    _client.stop()
    _client = None
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

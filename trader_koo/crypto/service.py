"""High-level crypto service — start/stop the feed and query data.

All public functions are thread-safe (the underlying BinanceWSClient
guards its state with a ``threading.Lock``).

The ``subscribe_ticks`` / ``unsubscribe_ticks`` functions allow the
FastAPI WebSocket endpoint to receive real-time ticks via asyncio queues.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Any

from trader_koo.crypto.binance_ws import BinanceWSClient
from trader_koo.crypto.models import CryptoBar, CryptoTick

LOG = logging.getLogger("trader_koo.crypto.service")

# Module-level singleton
_client: BinanceWSClient | None = None

# Browser WebSocket subscribers: sub_id → asyncio.Queue
_subscribers_lock = threading.Lock()
_subscribers: dict[str, asyncio.Queue[dict[str, Any]]] = {}


def subscribe_ticks(queue: asyncio.Queue[dict[str, Any]]) -> str:
    """Register a browser WebSocket to receive real-time tick pushes.

    Returns a subscription ID used to unsubscribe later.
    """
    sub_id = uuid.uuid4().hex[:12]
    with _subscribers_lock:
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


def start_crypto_feed() -> None:
    """Start the Binance WebSocket feed in a background daemon thread."""
    global _client
    if _client is not None:
        LOG.warning("Crypto feed already started — ignoring duplicate call")
        return
    _client = BinanceWSClient(on_tick=_broadcast_tick)
    _client.start()
    LOG.info("Crypto feed started")


def stop_crypto_feed() -> None:
    """Gracefully stop the WebSocket feed."""
    global _client
    if _client is None:
        return
    _client.stop()
    _client = None
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
    if _client is None:
        return []
    if interval != "1m":
        LOG.debug(
            "Requested interval=%s but only 1m bars are buffered", interval,
        )
        return []
    return _client.get_bars(symbol, limit=limit)


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

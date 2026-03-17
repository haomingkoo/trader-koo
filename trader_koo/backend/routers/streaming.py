"""Streaming endpoints — real-time equity data from Finnhub WebSocket.

Supports SPY/QQQ always-on streaming plus on-demand symbol subscriptions.
Includes a browser-facing WebSocket at /ws/equities that pushes live ticks
directly from the Finnhub feed with sub-second latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from trader_koo.streaming.service import (
    get_equity_price,
    get_equity_prices,
    get_subscription_info,
    subscribe_equity_ticks,
    subscribe_symbol,
    unsubscribe_equity_ticks,
    unsubscribe_symbol,
)

LOG = logging.getLogger("trader_koo.routers.streaming")

router = APIRouter(tags=["streaming"])


@router.get("/api/streaming/prices")
def streaming_prices() -> dict[str, Any]:
    """All currently streaming equity prices."""
    prices = get_equity_prices()
    if not prices:
        return {
            "ok": True,
            "prices": {},
            "detail": "Equity feed not connected or no data yet",
        }
    return {"ok": True, "prices": prices}


@router.get("/api/streaming/price/{symbol}")
def streaming_price(symbol: str) -> dict[str, Any]:
    """Price for a specific symbol. Auto-subscribes if not already streaming."""
    sym = symbol.upper()
    # Auto-subscribe on first request
    subscribe_symbol(sym)
    price = get_equity_price(sym)
    if price is None:
        return {
            "ok": True,
            "symbol": sym,
            "price": None,
            "detail": "Subscribed — price will be available shortly",
        }
    return {"ok": True, "symbol": sym, **price}


@router.post("/api/streaming/subscribe/{symbol}")
def subscribe(symbol: str) -> dict[str, Any]:
    """Subscribe to real-time data for a symbol."""
    sym = symbol.upper()
    success = subscribe_symbol(sym)
    info = get_subscription_info()
    if not success:
        return {
            "ok": False,
            "symbol": sym,
            "detail": (
                f"Cannot subscribe: at {info['subscribed_count']}"
                f"/{info['max_symbols']} limit"
            ),
            **info,
        }
    return {
        "ok": True,
        "symbol": sym,
        "detail": f"Subscribed to {sym}",
        **info,
    }


@router.post("/api/streaming/unsubscribe/{symbol}")
def unsubscribe(symbol: str) -> dict[str, Any]:
    """Unsubscribe from a symbol."""
    sym = symbol.upper()
    success = unsubscribe_symbol(sym)
    info = get_subscription_info()
    if not success:
        return {
            "ok": False,
            "symbol": sym,
            "detail": f"Cannot unsubscribe {sym} (may be always-on)",
            **info,
        }
    return {
        "ok": True,
        "symbol": sym,
        "detail": f"Unsubscribed from {sym}",
        **info,
    }


@router.get("/api/streaming/status")
def streaming_status() -> dict[str, Any]:
    """Subscription count, connected status, symbol list."""
    info = get_subscription_info()
    return {"ok": True, **info}


@router.websocket("/ws/equities")
async def ws_equities(websocket: WebSocket) -> None:
    """Push live equity ticks to the browser as they arrive from Finnhub.

    Zero polling — the Finnhub WS thread puts ticks into an asyncio.Queue
    which this handler drains and forwards to the connected client.
    """
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    sub_id = subscribe_equity_ticks(queue)
    LOG.info("Equity browser WS connected (sub_id=%s)", sub_id)
    try:
        while True:
            tick_data = await queue.get()
            await websocket.send_text(json.dumps(tick_data))
    except WebSocketDisconnect:
        LOG.info("Equity browser WS disconnected (sub_id=%s)", sub_id)
    except Exception as exc:
        LOG.warning("Equity browser WS error (sub_id=%s): %s", sub_id, exc)
    finally:
        unsubscribe_equity_ticks(sub_id)

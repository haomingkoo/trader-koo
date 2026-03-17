"""Crypto endpoints — real-time BTC and ETH data from Binance WebSocket.

Includes a browser-facing WebSocket at /ws/crypto that pushes live ticks
directly from the Binance feed with sub-second latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from trader_koo.crypto.service import (
    get_crypto_history,
    get_crypto_prices,
    get_crypto_summary,
    subscribe_ticks,
    unsubscribe_ticks,
)

LOG = logging.getLogger("trader_koo.routers.crypto")

router = APIRouter(tags=["crypto"])


@router.get("/api/crypto/prices")
def crypto_prices() -> dict[str, Any]:
    """Current prices for all tracked crypto."""
    ticks = get_crypto_prices()
    if not ticks:
        return {
            "ok": True,
            "prices": {},
            "detail": "Crypto feed not connected or no data yet",
        }
    prices: dict[str, dict[str, Any]] = {}
    for symbol, tick in ticks.items():
        prices[symbol] = {
            "symbol": tick.symbol,
            "price": tick.price,
            "volume_24h": tick.volume_24h,
            "change_pct_24h": tick.change_pct_24h,
            "timestamp": tick.timestamp.isoformat(),
        }
    return {"ok": True, "prices": prices}


@router.get("/api/crypto/history/{symbol}")
def crypto_history(
    symbol: str,
    interval: str = Query("1m", description="Bar interval (currently only 1m)"),
    limit: int = Query(100, ge=1, le=1440, description="Max bars to return"),
) -> dict[str, Any]:
    """Recent OHLCV bars for a crypto symbol."""
    # Normalise symbol input: accept btc-usd, BTC-USD, btcusd, etc.
    normalised = symbol.upper().replace(" ", "")
    if normalised in ("BTCUSD", "BTC-USD", "BTC"):
        normalised = "BTC-USD"
    elif normalised in ("ETHUSD", "ETH-USD", "ETH"):
        normalised = "ETH-USD"

    bars = get_crypto_history(normalised, interval=interval, limit=limit)
    return {
        "ok": True,
        "symbol": normalised,
        "interval": interval,
        "count": len(bars),
        "bars": [
            {
                "timestamp": bar.timestamp.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ],
    }


@router.get("/api/crypto/summary")
def crypto_summary() -> dict[str, Any]:
    """Summary for header display: BTC + ETH price, 24h change."""
    return get_crypto_summary()


@router.websocket("/ws/crypto")
async def ws_crypto(websocket: WebSocket) -> None:
    """Push live crypto ticks to the browser as they arrive from Binance.

    Zero polling — the Binance WS thread puts ticks into an asyncio.Queue
    which this handler drains and forwards to the connected client.
    """
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    sub_id = subscribe_ticks(queue)
    LOG.info("Browser WS connected (sub_id=%s)", sub_id)
    try:
        while True:
            tick_data = await queue.get()
            await websocket.send_text(json.dumps(tick_data))
    except WebSocketDisconnect:
        LOG.info("Browser WS disconnected (sub_id=%s)", sub_id)
    except Exception as exc:
        LOG.warning("Browser WS error (sub_id=%s): %s", sub_id, exc)
    finally:
        unsubscribe_ticks(sub_id)

"""Crypto endpoints — real-time data from Binance WebSocket.

Supports BTC, ETH, SOL, XRP, DOGE via 1-minute kline streams.
Includes a browser-facing WebSocket at /ws/crypto that pushes live ticks
directly from the Binance feed with sub-second latency.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from trader_koo.backend.services.database import get_conn
from trader_koo.crypto.indicators import compute_all_indicators
from trader_koo.crypto.market_insights import (
    build_btc_spy_correlation,
    build_crypto_market_structure,
)
from trader_koo.crypto.structure import build_crypto_structure
from trader_koo.crypto.service import (
    get_crypto_history,
    get_crypto_prices,
    get_crypto_summary,
    subscribe_ticks,
    unsubscribe_ticks,
)

LOG = logging.getLogger("trader_koo.routers.crypto")

router = APIRouter(tags=["crypto"])

_TRACKED_SYMBOLS = ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD")

# Symbol aliases → canonical display name
_SYMBOL_ALIASES: dict[str, str] = {
    "BTC": "BTC-USD", "BTCUSD": "BTC-USD", "BTC-USD": "BTC-USD",
    "ETH": "ETH-USD", "ETHUSD": "ETH-USD", "ETH-USD": "ETH-USD",
    "SOL": "SOL-USD", "SOLUSDT": "SOL-USD", "SOL-USD": "SOL-USD", "SOLUSD": "SOL-USD",
    "XRP": "XRP-USD", "XRPUSDT": "XRP-USD", "XRP-USD": "XRP-USD", "XRPUSD": "XRP-USD",
    "DOGE": "DOGE-USD", "DOGEUSDT": "DOGE-USD", "DOGE-USD": "DOGE-USD", "DOGEUSD": "DOGE-USD",
}


def _normalise_symbol(raw: str) -> str:
    """Normalise user-provided symbol to canonical display format."""
    key = raw.upper().replace(" ", "")
    return _SYMBOL_ALIASES.get(key, key)


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
    interval: str = Query("1m", description="Bar interval: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w"),
    limit: int = Query(100, ge=1, le=5000, description="Max bars to return"),
) -> dict[str, Any]:
    """Recent OHLCV bars for a crypto symbol."""
    normalised = _normalise_symbol(symbol)

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


@router.get("/api/crypto/structure/{symbol}")
def crypto_structure(
    symbol: str,
    interval: str = Query("1m", description="Structure interval: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w"),
    limit: int = Query(240, ge=20, le=5000, description="Max bars to analyze"),
) -> dict[str, Any]:
    """Support/resistance zones, trendlines, and HMM regime context for crypto."""
    normalised = _normalise_symbol(symbol)
    bars = get_crypto_history(normalised, interval=interval, limit=limit)
    return {
        "ok": True,
        **build_crypto_structure(normalised, bars, interval=interval),
    }


@router.get("/api/crypto/correlation/{symbol}")
def crypto_correlation(
    symbol: str,
    benchmark: str = Query("SPY", description="Benchmark equity ticker from price_daily"),
    limit: int = Query(40, ge=10, le=90, description="Max daily crypto bars to align"),
) -> dict[str, Any]:
    """Cross-asset correlation and relative-strength snapshot versus a benchmark."""
    normalised = _normalise_symbol(symbol)
    bars = get_crypto_history(normalised, interval="1d", limit=limit)
    conn = get_conn()
    try:
        return {
            "ok": True,
            **build_btc_spy_correlation(
                conn,
                asset_symbol=normalised,
                benchmark_symbol=str(benchmark or "SPY").upper(),
                asset_bars=bars,
            ),
        }
    finally:
        conn.close()


@router.get("/api/crypto/market-structure")
def crypto_market_structure(
    interval: str = Query("1h", description="Structure interval: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w"),
    limit: int = Query(240, ge=20, le=5000, description="Max bars to analyze per symbol"),
) -> dict[str, Any]:
    """Broad crypto breadth, volatility, and level-context snapshot."""
    summaries = (get_crypto_summary() or {}).get("prices") or {}
    structures = []
    for symbol in _TRACKED_SYMBOLS:
        bars = get_crypto_history(symbol, interval=interval, limit=limit)
        structures.append(
            build_crypto_structure(symbol, bars, interval=interval, include_hmm=False),
        )
    payload = build_crypto_market_structure(
        interval=interval,
        summaries=summaries,
        structures=structures,
    )
    return payload


@router.get("/api/crypto/summary")
def crypto_summary() -> dict[str, Any]:
    """Summary for all tracked crypto: price and 24h change."""
    return get_crypto_summary()


@router.get("/api/crypto/indicators/{symbol}")
def crypto_indicators(symbol: str) -> dict[str, Any]:
    """Technical indicators computed from buffered 1-min bars."""
    normalised = _normalise_symbol(symbol)
    bars = get_crypto_history(normalised, interval="1m", limit=1440)
    if not bars:
        LOG.info("No bars available for indicators: %s", normalised)
        return {
            "ok": True,
            "symbol": normalised,
            "indicators": {
                "sma_20": None,
                "sma_50": None,
                "rsi_14": None,
                "macd": {"macd": None, "signal": None, "histogram": None},
                "bollinger": {
                    "upper": None,
                    "middle": None,
                    "lower": None,
                    "width": None,
                },
                "vwap": None,
            },
            "bar_count": 0,
        }
    indicators = compute_all_indicators(bars)
    return {
        "ok": True,
        "symbol": normalised,
        "indicators": indicators,
        "bar_count": len(bars),
    }


@router.websocket("/ws/crypto")
async def ws_crypto(websocket: WebSocket) -> None:
    """Push live crypto ticks to the browser as they arrive from Binance.

    Zero polling — the Binance WS thread puts ticks into an asyncio.Queue
    which this handler drains and forwards to the connected client.
    """
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    sub_id = subscribe_ticks(queue)
    if sub_id is None:
        await websocket.close(code=1013, reason="Too many connections")
        return
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

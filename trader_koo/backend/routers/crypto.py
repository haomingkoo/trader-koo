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
    get_aggregator,
    get_crypto_history,
    get_crypto_prices,
    get_crypto_summary,
    get_forming_candle,
    subscribe_ticks,
    unsubscribe_ticks,
    update_subscription,
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
    limit: int = Query(100, ge=1, le=3000, description="Max bars to return"),
) -> dict[str, Any]:
    """Recent OHLCV bars for a crypto symbol."""
    normalised = _normalise_symbol(symbol)

    bars = get_crypto_history(normalised, interval=interval, limit=limit)
    bar_dicts = [
        {
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars
    ]
    payload: dict[str, Any] = {
        "ok": True,
        "symbol": normalised,
        "interval": interval,
        "count": len(bar_dicts),
        "bars": bar_dicts,
    }

    # Candlestick pattern detection (fast, <10ms)
    if len(bars) >= 10 and interval not in ("1m", "5m"):
        try:
            import pandas as pd
            from trader_koo.features.candle_patterns import (
                CandlePatternConfig,
                detect_candlestick_patterns,
            )
            df = pd.DataFrame([
                {"date": b.timestamp.strftime("%Y-%m-%d %H:%M"), "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
                for b in bars
            ])
            candle_df = detect_candlestick_patterns(df, CandlePatternConfig(lookback_bars=min(len(bars), 180), max_rows=20))
            if not candle_df.empty:
                payload["candlestick_patterns"] = candle_df.to_dict(orient="records")
        except Exception:
            LOG.debug("Candle pattern detection failed for %s [%s]", normalised, interval, exc_info=True)

    # Append forming candle for higher timeframes (5m+)
    try:
        forming = get_forming_candle(normalised, interval=interval)
        if forming is not None:
            payload["forming_candle"] = {
                "timestamp": forming.timestamp.isoformat(),
                "open": forming.open,
                "high": forming.high,
                "low": forming.low,
                "close": forming.close,
                "volume": forming.volume,
            }
    except Exception:
        LOG.debug("Could not compute forming candle for %s [%s]", normalised, interval, exc_info=True)

    return payload


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
    """Cross-asset correlation with regime change detection."""
    normalised = _normalise_symbol(symbol)
    bars = get_crypto_history(normalised, interval="1d", limit=limit)
    conn = get_conn()
    try:
        from trader_koo.crypto.market_insights import (
            save_correlation_snapshot,
            detect_correlation_regime_change,
        )
        benchmark_norm = str(benchmark or "SPY").upper()
        corr_data = build_btc_spy_correlation(
            conn,
            asset_symbol=normalised,
            benchmark_symbol=benchmark_norm,
            asset_bars=bars,
        )

        # Save snapshot and detect regime changes
        regime_change = None
        try:
            save_correlation_snapshot(conn, asset=normalised, benchmark=benchmark_norm, correlation_data=corr_data)
            current_label = corr_data.get("relationship_label", "")
            w20 = (corr_data.get("windows") or {}).get("20d") or {}
            regime_change = detect_correlation_regime_change(
                conn, asset=normalised, benchmark=benchmark_norm,
                current_label=current_label, current_corr_20d=w20.get("correlation"),
            )
        except Exception:
            pass

        return {
            "ok": True,
            **corr_data,
            "regime_change": regime_change,
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
def crypto_indicators(
    symbol: str,
    interval: str = Query("1m", description="Interval for indicator computation"),
) -> dict[str, Any]:
    """Technical indicators computed from bars at the requested interval."""
    normalised = _normalise_symbol(symbol)
    limit = {"1m": 1440, "5m": 500, "15m": 300, "30m": 200, "1h": 200, "4h": 200, "12h": 150, "1d": 100, "1w": 60}.get(interval, 200)
    bars = get_crypto_history(normalised, interval=interval, limit=limit)
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


@router.get("/api/crypto/open-interest/{symbol}")
def crypto_open_interest(
    symbol: str,
    period: str = Query("1h", description="OI period: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d"),
    limit: int = Query(100, ge=1, le=500, description="Max OI snapshots"),
) -> dict[str, Any]:
    """Open interest history from Binance Futures."""
    from trader_koo.crypto.binance_oi import (
        fetch_current_open_interest,
        fetch_open_interest_history,
    )

    normalised = _normalise_symbol(symbol)
    snapshots = fetch_open_interest_history(normalised, period=period, limit=limit)
    current = fetch_current_open_interest(normalised)

    oi_bars = [
        {
            "timestamp": s.timestamp.isoformat(),
            "open_interest": s.sum_open_interest,
            "open_interest_value": s.sum_open_interest_value,
        }
        for s in snapshots
    ]

    # Compute 24h change if we have enough data
    oi_change_24h = None
    if len(snapshots) >= 2:
        latest_val = snapshots[-1].sum_open_interest_value
        oldest_val = snapshots[0].sum_open_interest_value
        if oldest_val > 0:
            oi_change_24h = round((latest_val - oldest_val) / oldest_val * 100, 2)

    return {
        "ok": True,
        "symbol": normalised,
        "period": period,
        "count": len(oi_bars),
        "oi_bars": oi_bars,
        "current_oi": current,
        "oi_change_24h_pct": oi_change_24h,
    }


@router.websocket("/ws/crypto")
async def ws_crypto(websocket: WebSocket) -> None:
    """Push live crypto ticks to the browser as they arrive from Binance.

    Supports subscription-based fan-out. After connecting, a client may
    send a JSON message to filter updates::

        {"action": "subscribe", "symbol": "BTC-USD", "interval": "5m"}

    If no subscribe message is sent, the client receives ALL ticks at 1m
    (backward compatible with the old protocol).

    Message types pushed to the client:

    - ``tick``  — latest price for the subscribed symbol
    - ``forming`` — current forming candle at the subscribed interval
    - ``candle_close`` — finalized candle at the subscribed interval
    """
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    sub_id = subscribe_ticks(queue)
    if sub_id is None:
        await websocket.close(code=1013, reason="Too many connections")
        return
    LOG.info("Browser WS connected (sub_id=%s)", sub_id)

    async def _reader() -> None:
        """Read subscription messages from the client."""
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if msg.get("action") == "subscribe":
                    symbol = msg.get("symbol")
                    interval = msg.get("interval")
                    if symbol:
                        symbol = _normalise_symbol(symbol)
                    update_subscription(sub_id, symbol, interval)
                    LOG.debug(
                        "WS sub_id=%s subscribed: symbol=%s interval=%s",
                        sub_id,
                        symbol,
                        interval,
                    )
                    # Send immediate snapshot if aggregator is available
                    agg = get_aggregator()
                    if agg and symbol and interval:
                        forming = agg.get_forming(symbol, interval)
                        if forming:
                            snapshot = {
                                "type": "forming",
                                "symbol": symbol,
                                "interval": interval,
                                **forming,
                            }
                            await websocket.send_text(json.dumps(snapshot))
        except (WebSocketDisconnect, Exception):
            pass  # Connection closed — _writer will also exit

    async def _writer() -> None:
        """Drain the queue and push to the client."""
        while True:
            data = await queue.get()
            try:
                await websocket.send_text(json.dumps(data))
            except WebSocketDisconnect:
                break
            except Exception:
                break

    try:
        # Run reader and writer concurrently; if either exits, cancel both
        reader_task = asyncio.create_task(_reader())
        writer_task = asyncio.create_task(_writer())
        done, pending = await asyncio.wait(
            {reader_task, writer_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        LOG.info("Browser WS disconnected (sub_id=%s)", sub_id)
    except Exception as exc:
        LOG.warning("Browser WS error (sub_id=%s): %s", sub_id, exc)
    finally:
        unsubscribe_ticks(sub_id)

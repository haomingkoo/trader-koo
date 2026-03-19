"""Binance Futures open interest data fetcher.

Uses the public Binance Futures API (no auth required) to fetch
open interest history for perpetual contracts.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger("trader_koo.crypto.binance_oi")

# Map our symbols to Binance futures symbols
_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
}

_VALID_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}

_cache_lock = threading.Lock()
_oi_cache: dict[str, Any] = {}
_CACHE_TTL_SEC = 300  # 5 minutes


@dataclass
class OpenInterestSnapshot:
    symbol: str
    timestamp: dt.datetime
    sum_open_interest: float  # in contracts
    sum_open_interest_value: float  # in USD


def _binance_symbol(symbol: str) -> str | None:
    """Convert our symbol to Binance futures symbol."""
    return _SYMBOL_MAP.get(symbol.upper())


def fetch_open_interest_history(
    symbol: str,
    *,
    period: str = "1h",
    limit: int = 100,
) -> list[OpenInterestSnapshot]:
    """Fetch OI history from Binance Futures public API.

    Returns list sorted oldest-first.
    """
    binance_sym = _binance_symbol(symbol)
    if not binance_sym:
        LOG.warning("Unknown symbol for OI: %s", symbol)
        return []

    if period not in _VALID_PERIODS:
        period = "1h"

    cache_key = f"oi_{binance_sym}_{period}_{limit}"
    with _cache_lock:
        cached = _oi_cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
            return cached["data"]

    url = (
        f"https://fapi.binance.com/futures/data/openInterestHist"
        f"?symbol={binance_sym}&period={period}&limit={limit}"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "trader-koo/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        if not isinstance(raw, list):
            LOG.warning("Unexpected OI response for %s: %s", symbol, type(raw))
            return []

        snapshots: list[OpenInterestSnapshot] = []
        for item in raw:
            try:
                ts = dt.datetime.fromtimestamp(
                    int(item["timestamp"]) / 1000, tz=dt.timezone.utc,
                )
                snapshots.append(OpenInterestSnapshot(
                    symbol=symbol,
                    timestamp=ts,
                    sum_open_interest=float(item.get("sumOpenInterest", 0)),
                    sum_open_interest_value=float(item.get("sumOpenInterestValue", 0)),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                LOG.debug("Skipping OI item: %s", exc)
                continue

        snapshots.sort(key=lambda s: s.timestamp)

        with _cache_lock:
            _oi_cache[cache_key] = {
                "data": snapshots,
                "expires_at": (
                    dt.datetime.now(dt.timezone.utc)
                    + dt.timedelta(seconds=_CACHE_TTL_SEC)
                ).timestamp(),
            }

        LOG.info("OI %s: fetched %d snapshots (%s period)", symbol, len(snapshots), period)
        return snapshots

    except Exception as exc:
        LOG.warning("OI fetch failed for %s: %s", symbol, exc)
        return []


def fetch_current_open_interest(symbol: str) -> dict[str, Any] | None:
    """Fetch the latest single OI value for a symbol."""
    binance_sym = _binance_symbol(symbol)
    if not binance_sym:
        return None

    url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={binance_sym}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "trader-koo/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return {
            "symbol": symbol,
            "open_interest": float(data.get("openInterest", 0)),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    except Exception as exc:
        LOG.warning("Current OI fetch failed for %s: %s", symbol, exc)
        return None

"""Binance REST history backfill helpers for crypto charts."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Final

import requests

from trader_koo.crypto.models import CryptoBar

LOG = logging.getLogger("trader_koo.crypto.binance_history")

BINANCE_REST_URL: Final[str] = "https://api.binance.com/api/v3/klines"
REQUEST_TIMEOUT_SEC: Final[float] = 8.0
MAX_LIMIT_PER_CALL: Final[int] = 1000
SUPPORTED_INTERVALS: Final[set[str]] = {
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
}
SYMBOL_TO_BINANCE: Final[dict[str, str]] = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
}


def fetch_recent_klines(
    symbol: str,
    interval: str,
    limit: int,
) -> list[CryptoBar]:
    """Fetch recent klines for *symbol*/*interval* from Binance REST.

    Returns bars oldest-first. Unsupported symbols/intervals and request
    failures return an empty list.
    """
    normalised_interval = str(interval or "1m").lower()
    if limit <= 0 or normalised_interval not in SUPPORTED_INTERVALS:
        return []

    binance_symbol = SYMBOL_TO_BINANCE.get(str(symbol or "").upper())
    if not binance_symbol:
        LOG.debug("Unsupported Binance history symbol: %s", symbol)
        return []

    remaining = int(limit)
    end_time_ms: int | None = None
    out: list[CryptoBar] = []

    while remaining > 0:
        batch_limit = min(remaining, MAX_LIMIT_PER_CALL)
        params: dict[str, int | str] = {
            "symbol": binance_symbol,
            "interval": normalised_interval,
            "limit": batch_limit,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        try:
            response = requests.get(
                BINANCE_REST_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            LOG.warning(
                "Binance REST history fetch failed for %s %s: %s",
                symbol,
                normalised_interval,
                exc,
            )
            break

        if not isinstance(payload, list) or not payload:
            break

        batch = _parse_klines(symbol, normalised_interval, payload)
        if not batch:
            break

        out = batch + out
        remaining -= len(batch)

        first_open_ms = int(payload[0][0])
        next_end_time = first_open_ms - 1
        if next_end_time <= 0 or len(batch) < batch_limit:
            break
        end_time_ms = next_end_time

    deduped: dict[dt.datetime, CryptoBar] = {}
    for bar in out:
        deduped[bar.timestamp] = bar
    return [deduped[ts] for ts in sorted(deduped)]


def _parse_klines(
    symbol: str,
    interval: str,
    rows: list[object],
) -> list[CryptoBar]:
    bars: list[CryptoBar] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            open_time_ms = int(row[0])
            timestamp = dt.datetime.fromtimestamp(
                open_time_ms / 1000,
                tz=dt.timezone.utc,
            )
            bars.append(
                CryptoBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    interval=interval,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        except Exception:
            continue
    return bars

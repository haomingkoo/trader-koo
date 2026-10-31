"""Shared Finnhub REST quote fetcher used across notification subsystems."""
from __future__ import annotations

import datetime as dt
import logging
from zoneinfo import ZoneInfo

import httpx

LOG = logging.getLogger("trader_koo.notifications.finnhub")

# Finnhub REST API
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_REQUEST_TIMEOUT_SEC = 10
NEW_YORK = ZoneInfo("America/New_York")


def fetch_finnhub_quote(ticker: str, api_key: str) -> float | None:
    """Fetch the current price for *ticker* via Finnhub REST ``/quote``.

    Returns the current price (``c`` field) or ``None`` on failure.
    This is a synchronous call; async callers should wrap it in
    ``loop.run_in_executor``.
    """
    detail = fetch_finnhub_quote_detail(ticker, api_key)
    return detail[0] if detail else None


def fetch_finnhub_quote_detail(ticker: str, api_key: str) -> tuple[float, str | None] | None:
    """Return ``(price, New York session date of the quote)``, or ``None``.

    Outside market hours Finnhub serves the last session's close, so callers
    need the session to tell a live price from a close they already stored.
    """
    if not api_key:
        LOG.warning(
            "FINNHUB_API_KEY not set — cannot fetch quote for %s", ticker
        )
        return None

    try:
        with httpx.Client(timeout=FINNHUB_REQUEST_TIMEOUT_SEC) as client:
            resp = client.get(
                FINNHUB_QUOTE_URL,
                params={"symbol": ticker, "token": api_key},
            )
        if resp.status_code != 200:
            LOG.warning(
                "Finnhub quote returned %d for %s", resp.status_code, ticker
            )
            return None

        data = resp.json()
        price = data.get("c")  # "c" = current price
        if price is None or price == 0:
            LOG.debug("Finnhub returned no price for %s: %s", ticker, data)
            return None
        quote_ts = data.get("t")
        session = (
            dt.datetime.fromtimestamp(int(quote_ts), NEW_YORK).date().isoformat() if quote_ts else None
        )
        return float(price), session
    except httpx.HTTPError as exc:
        LOG.warning("Finnhub HTTP error for %s: %s", ticker, exc)
        return None
    except Exception as exc:
        LOG.warning("Finnhub quote fetch failed for %s: %s", ticker, exc)
        return None

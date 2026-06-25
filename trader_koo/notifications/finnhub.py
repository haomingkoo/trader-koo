"""Shared Finnhub REST quote fetcher used across notification subsystems."""
from __future__ import annotations

import logging

import httpx

LOG = logging.getLogger("trader_koo.notifications.finnhub")

# Finnhub REST API
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_REQUEST_TIMEOUT_SEC = 10


def fetch_finnhub_quote(ticker: str, api_key: str) -> float | None:
    """Fetch the current price for *ticker* via Finnhub REST ``/quote``.

    Returns the current price (``c`` field) or ``None`` on failure.
    This is a synchronous call; async callers should wrap it in
    ``loop.run_in_executor``.
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
        return float(price)
    except httpx.HTTPError as exc:
        LOG.warning("Finnhub HTTP error for %s: %s", ticker, exc)
        return None
    except Exception as exc:
        LOG.warning("Finnhub quote fetch failed for %s: %s", ticker, exc)
        return None

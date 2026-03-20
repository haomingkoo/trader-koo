"""External macro data sources for ML feature enrichment.

Pulls data from free public APIs that aren't in our price_daily table:
- FRED (Federal Reserve Economic Data): M2 money supply, yield curve, unemployment
- Polymarket: prediction market probabilities for macro events

These are fetched on-demand and cached. They supplement the price-based
features with macro regime context.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import urllib.parse
import urllib.request
from typing import Any

LOG = logging.getLogger(__name__)

# Load .env for local development (Railway sets env vars directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd

_cache_lock = threading.Lock()
_fred_cache: dict[str, Any] = {}
_polymarket_cache: dict[str, Any] = {}
_cache_ttl_sec = 3600  # 1 hour

# In-memory store for bulk-prefetched FRED data.
# Maps series_id -> DataFrame with columns [date, value], sorted by date.
_fred_bulk_store: dict[str, pd.DataFrame] = {}


# ---------------------------------------------------------------------------
# FRED (no API key needed for CSV download)
# ---------------------------------------------------------------------------

_FRED_SERIES = {
    "M2SL": "M2 money supply (seasonally adjusted, monthly)",
    "DFF": "Federal funds effective rate (daily)",
    "T10Y2Y": "10Y-2Y treasury spread (yield curve, daily)",
    "T10Y3M": "10Y-3M treasury spread (daily)",
    "UNRATE": "Unemployment rate (monthly)",
    "CPIAUCSL": "CPI all items (monthly, seasonally adjusted)",
    "DTWEXBGS": "Trade-weighted USD index (daily)",
    "BAMLH0A0HYM2": "ICE BofA high-yield OAS (daily, credit stress)",
}


def _fred_api_key() -> str:
    return str(os.getenv("FRED_API_KEY", "")).strip()


def prefetch_fred_series_bulk(
    series_ids: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Fetch full history for multiple FRED series in bulk (one API call each).

    DFF, T10Y2Y, BAMLH0A0HYM2 are never revised, so we can safely pull the
    entire date range in a single request per series instead of making
    per-date API calls during feature extraction.

    Results are cached in the module-level ``_fred_bulk_store`` so repeated
    calls within the same process are free.

    Returns
    -------
    dict mapping series_id -> DataFrame with columns [date, value] sorted
    by date ascending.
    """
    api_key = _fred_api_key()
    if not api_key:
        LOG.warning("FRED_API_KEY not set; bulk prefetch skipped")
        return {}

    import requests as _requests

    result: dict[str, pd.DataFrame] = {}

    for series_id in series_ids:
        # Skip if already prefetched with a range that covers the request
        with _cache_lock:
            existing = _fred_bulk_store.get(series_id)
            if existing is not None and not existing.empty:
                existing_min = str(existing["date"].iloc[0])
                existing_max = str(existing["date"].iloc[-1])
                if existing_min <= start_date and existing_max >= end_date:
                    result[series_id] = existing
                    continue

        try:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            }
            resp = _requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params=params,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            rows: list[dict[str, str | float]] = []
            for obs in data.get("observations", []):
                val_str = str(obs.get("value", "")).strip()
                date_str = str(obs.get("date", "")).strip()
                if date_str and val_str and val_str != ".":
                    try:
                        rows.append({"date": date_str, "value": float(val_str)})
                    except ValueError:
                        continue

            df = pd.DataFrame(rows, columns=["date", "value"])
            df = df.sort_values("date").reset_index(drop=True)

            with _cache_lock:
                _fred_bulk_store[series_id] = df

            result[series_id] = df
            LOG.info(
                "FRED bulk prefetch %s: %d observations (%s to %s)",
                series_id, len(df), start_date, end_date,
            )

        except Exception as exc:
            LOG.warning("FRED bulk prefetch failed for %s: %s", series_id, exc)

    return result


def lookup_fred_value(series_id: str, as_of_date: str) -> float | None:
    """Look up the most recent FRED value on or before *as_of_date*.

    Uses the in-memory bulk store populated by ``prefetch_fred_series_bulk``.
    Returns ``None`` if the series has not been prefetched or no observation
    exists on or before the requested date, allowing the caller to fall back
    to the per-date ``fetch_fred_series`` API call.
    """
    with _cache_lock:
        df = _fred_bulk_store.get(series_id)

    if df is None or df.empty:
        return None

    # All dates are ISO strings — lexicographic comparison is valid
    mask = df["date"] <= as_of_date
    valid = df.loc[mask]
    if valid.empty:
        return None

    return float(valid["value"].iloc[-1])


def fetch_fred_series(
    series_id: str,
    *,
    lookback_days: int = 365,
    as_of_date: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a FRED series via the official FRED API.

    Uses FRED_API_KEY env var. Falls back to CSV scraping if no key set.
    The vintage_date parameter ensures we only see data published by
    as_of_date (prevents look-ahead bias from FRED data revisions).

    Returns list of {date: str, value: float} dicts, sorted by date.
    """
    if as_of_date:
        vintage = as_of_date
        end = dt.datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    else:
        end = dt.datetime.now(dt.timezone.utc)
        vintage = end.strftime("%Y-%m-%d")

    cache_key = f"fred_{series_id}_{lookback_days}_{vintage}"
    with _cache_lock:
        cached = _fred_cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
            return cached["data"]

    start = end - dt.timedelta(days=lookback_days)
    api_key = _fred_api_key()

    try:
        if api_key:
            # Use official FRED API via requests (urllib has connection issues)
            import requests as _requests

            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start.strftime("%Y-%m-%d"),
                "observation_end": end.strftime("%Y-%m-%d"),
                "realtime_start": vintage,
                "realtime_end": vintage,
            }
            resp = _requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            rows: list[dict[str, Any]] = []
            for obs in data.get("observations", []):
                val_str = str(obs.get("value", "")).strip()
                date_str = str(obs.get("date", "")).strip()
                if date_str and val_str and val_str != ".":
                    try:
                        rows.append({"date": date_str, "value": float(val_str)})
                    except ValueError:
                        continue
        else:
            # Fallback: CSV scraping (no API key needed but slower)
            url = (
                f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                f"?id={series_id}"
                f"&cosd={start.strftime('%Y-%m-%d')}"
                f"&coed={end.strftime('%Y-%m-%d')}"
                f"&vintage_date={vintage}"
                f"&revision_date={vintage}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "trader-koo/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")

            rows = []
            for line in raw.strip().split("\n")[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    date_str = parts[0].strip()
                    val_str = parts[1].strip()
                    if val_str and val_str != "." and val_str != "":
                        try:
                            rows.append({"date": date_str, "value": float(val_str)})
                        except ValueError:
                            continue

        with _cache_lock:
            _fred_cache[cache_key] = {
                "data": rows,
                "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_cache_ttl_sec)).timestamp(),
            }

        LOG.info("FRED %s: fetched %d observations%s", series_id, len(rows), " (API)" if api_key else " (CSV)")
        return rows

    except Exception as exc:
        LOG.warning("FRED fetch failed for %s: %s", series_id, exc)
        return []


def get_fred_latest(series_id: str) -> float | None:
    """Get the latest value of a FRED series."""
    rows = fetch_fred_series(series_id, lookback_days=90)
    return rows[-1]["value"] if rows else None


def get_yield_curve_spread() -> dict[str, float | None]:
    """Get current yield curve spreads."""
    return {
        "spread_10y_2y": get_fred_latest("T10Y2Y"),
        "spread_10y_3m": get_fred_latest("T10Y3M"),
        "fed_funds_rate": get_fred_latest("DFF"),
        "high_yield_oas": get_fred_latest("BAMLH0A0HYM2"),
    }


def get_m2_growth() -> dict[str, float | None]:
    """Get M2 money supply growth rate (YoY)."""
    rows = fetch_fred_series("M2SL", lookback_days=400)
    if len(rows) < 13:
        return {"m2_latest": None, "m2_yoy_pct": None}
    latest = rows[-1]["value"]
    year_ago = rows[-13]["value"]  # ~12 months back
    yoy = (latest - year_ago) / year_ago * 100 if year_ago > 0 else None
    return {"m2_latest": latest, "m2_yoy_pct": round(yoy, 2) if yoy else None}


# ---------------------------------------------------------------------------
# Polymarket (public CLOB API — no auth needed)
# ---------------------------------------------------------------------------

_POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"


_FINANCE_KEYWORDS = frozenset({
    "fed ", "rate cut", "rate hike", "recession", "bitcoin", "btc ",
    "inflation", "cpi ", "gdp ", "tariff", "china", "iran",
    "oil price", "crude oil", "gold price", "stock market",
    "interest rate", "central bank", "economy", "economic",
    "debt ceiling", "stimulus", "sanctions", "opec", "fomc",
    "powell", "fiscal", "monetary", "yield curve", "bond",
    "s&p 500", "nasdaq", "dow jones", "microstrategy",
    "ceasefire", "invade", "invasion",
    "crypto market", "ethereum", "defi",
})

_EXCLUDE_KEYWORDS = frozenset({
    # Team sports
    "nba", "nfl", "nhl", "mlb", "mls", "fifa", "world cup", "premier league",
    "la liga", "champions league", "serie a", "bundesliga", "euroleague",
    "cricket", "rugby", "soccer", "futbol",
    # Individual sports
    "ufc", "boxing", "tennis", "golf", "masters", "wimbledon", "mma",
    "formula 1", "f1 ", "nascar", "cycling", "marathon", "olympics",
    # Esports
    "valorant", "cs2 ", "dota", "league of legends", "esport",
    # Entertainment / pop culture
    "oscar", "grammy", "emmy", "super bowl", "mvp", "stanley cup",
    "movie", "film", "album", "tv show", "reality tv", "award show",
    "gta", "rihanna", "kardashian", "taylor swift", "celebrity",
    # Social media / meme
    "tweet", "tiktok", "youtube", "instagram", "influencer",
    "jesus", "bitboy", "airdrop", "meme coin",
    # Weather / misc
    "weather", "temperature", "earthquake", "hurricane",
})


def fetch_polymarket_events(
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Fetch curated finance-relevant Polymarket events (grouped markets).

    Uses events API for better grouping, filters for macro/finance relevance,
    sorts by total volume. Returns event-level data with embedded markets.
    """
    cache_key = f"poly_events_{limit}"
    with _cache_lock:
        cached = _polymarket_cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
            return cached["data"]

    try:
        url = f"{_POLYMARKET_GAMMA}/events?limit=200&active=true&closed=false"
        req = urllib.request.Request(url, headers={
            "User-Agent": "trader-koo/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_events = json.loads(resp.read().decode("utf-8"))

        if not isinstance(raw_events, list):
            return []

        # Filter for finance/macro relevance
        relevant: list[dict[str, Any]] = []
        for ev in raw_events:
            title = str(ev.get("title", "")).lower()
            desc = str(ev.get("description", "")).lower()
            text = f"{title} {desc}"
            # Exclude sports/entertainment first
            if any(kw in text for kw in _EXCLUDE_KEYWORDS):
                continue
            # Then require finance relevance
            if not any(kw in text for kw in _FINANCE_KEYWORDS):
                continue

            raw_markets = ev.get("markets") or []
            total_volume = sum(float(m.get("volume", 0) or 0) for m in raw_markets)

            # Parse the top market for this event (highest volume)
            top_market = None
            if raw_markets:
                sorted_mkts = sorted(raw_markets, key=lambda m: float(m.get("volume", 0) or 0), reverse=True)
                m = sorted_mkts[0]
                raw_outcomes = m.get("outcomes") or []
                # Polymarket sometimes returns outcomes as a JSON string, not a list
                if isinstance(raw_outcomes, str):
                    try:
                        import json as _json
                        raw_outcomes = _json.loads(raw_outcomes)
                    except Exception:
                        raw_outcomes = [raw_outcomes]
                outcomes = list(raw_outcomes) if isinstance(raw_outcomes, (list, tuple)) else []

                raw_prices = m.get("outcomePrices") or []
                if isinstance(raw_prices, str):
                    try:
                        import json as _json
                        raw_prices = _json.loads(raw_prices)
                    except Exception:
                        raw_prices = []
                prices_raw = list(raw_prices) if isinstance(raw_prices, (list, tuple)) else []
                prices = []
                for p in prices_raw:
                    try:
                        prices.append(round(float(p) * 100, 1))
                    except (TypeError, ValueError):
                        prices.append(None)
                top_market = {
                    "question": str(m.get("question", "")).strip(),
                    "outcomes": outcomes,
                    "prices_pct": prices,
                    "volume": round(float(m.get("volume", 0) or 0), 2),
                }

            relevant.append({
                "title": str(ev.get("title", "")).strip(),
                "slug": ev.get("slug", ""),
                "market_count": len(raw_markets),
                "total_volume": round(total_volume, 2),
                "end_date": ev.get("endDate"),
                "image": ev.get("image"),
                "url": f"https://polymarket.com/event/{ev.get('slug', '')}",
                "top_market": top_market,
            })

        # Sort by volume, return top N
        relevant.sort(key=lambda e: e["total_volume"], reverse=True)
        result = relevant[:limit]

        with _cache_lock:
            _polymarket_cache[cache_key] = {
                "data": result,
                "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_cache_ttl_sec)).timestamp(),
            }

        LOG.info("Polymarket events: %d relevant / %d total", len(relevant), len(raw_events))
        return result

    except Exception as exc:
        LOG.warning("Polymarket events fetch failed: %s", exc)
        return []


def fetch_polymarket_markets(
    *,
    limit: int = 20,
    tag: str = "",
) -> list[dict[str, Any]]:
    """Fetch active Polymarket prediction markets via Gamma API.

    Returns list of markets with question, outcome prices, volume, liquidity.
    """
    cache_key = f"poly_{tag}_{limit}"
    with _cache_lock:
        cached = _polymarket_cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
            return cached["data"]

    try:
        # Fetch a large batch and filter for finance-relevant markets
        fetch_limit = max(limit * 10, 200)  # oversample then filter
        params: dict[str, str] = {
            "limit": str(fetch_limit),
            "active": "true",
            "closed": "false",
        }
        if tag:
            params["tag"] = tag
        qs = urllib.parse.urlencode(params)
        url = f"{_POLYMARKET_GAMMA}/markets?{qs}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "trader-koo/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        markets = []
        if isinstance(data, list):
            for market in data:
                if not isinstance(market, dict):
                    continue
                outcomes = market.get("outcomes") or []
                prices_raw = market.get("outcomePrices") or []
                prices = []
                for p in prices_raw:
                    try:
                        prices.append(round(float(p) * 100, 1))
                    except (TypeError, ValueError):
                        prices.append(None)

                markets.append({
                    "question": str(market.get("question") or "").strip(),
                    "slug": market.get("slug", ""),
                    "outcomes": outcomes,
                    "prices_pct": prices,  # percentage (0-100)
                    "volume": round(float(market.get("volume") or 0), 2),
                    "liquidity": round(float(market.get("liquidity") or 0), 2),
                    "end_date": market.get("endDate"),
                    "image": market.get("image"),
                    "active": bool(market.get("active")),
                    "url": f"https://polymarket.com/event/{market.get('slug', '')}",
                })

        # Filter: exclude sports/entertainment, then require finance relevance
        relevant = []
        for m in markets:
            text = str(m.get("question", "")).lower()
            if any(kw in text for kw in _EXCLUDE_KEYWORDS):
                continue
            if not any(kw in text for kw in _FINANCE_KEYWORDS):
                continue
            relevant.append(m)

        # Sort by volume (highest first) and cap at requested limit
        relevant.sort(key=lambda m: m.get("volume", 0), reverse=True)
        result = relevant[:limit]

        with _cache_lock:
            _polymarket_cache[cache_key] = {
                "data": result,
                "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_cache_ttl_sec)).timestamp(),
            }

        LOG.info("Polymarket: %d relevant / %d total, returning %d", len(relevant), len(markets), len(result))
        return result

    except Exception as exc:
        LOG.warning("Polymarket fetch failed: %s", exc)
        return []


def get_polymarket_macro_probabilities() -> dict[str, float]:
    """Extract aggregate Polymarket probabilities for ML features.

    Returns 3 features:
    - polymarket_fed_cut_prob: highest "Yes" probability among rate-cut markets (0-100)
    - polymarket_recession_prob: highest "Yes" probability among recession markets (0-100)
    - polymarket_macro_sentiment: average "Yes" probability across all finance events
      (crude fear/greed proxy — higher = more bullish macro consensus)

    NOTE: These are LIVE market prices from Polymarket's current order book.
    They are only meaningful for live/forward scoring. During historical training,
    Polymarket odds for past dates are NOT available via API, so these will be NaN.
    The model must tolerate missing values for these columns.
    """
    try:
        events = fetch_polymarket_events(limit=50)
    except Exception as exc:
        LOG.warning("Polymarket macro prob fetch failed: %s", exc)
        return {
            "polymarket_fed_cut_prob": float("nan"),
            "polymarket_recession_prob": float("nan"),
            "polymarket_macro_sentiment": float("nan"),
        }

    if not events:
        return {
            "polymarket_fed_cut_prob": float("nan"),
            "polymarket_recession_prob": float("nan"),
            "polymarket_macro_sentiment": float("nan"),
        }

    fed_cut_probs: list[float] = []
    recession_probs: list[float] = []
    all_yes_probs: list[float] = []

    for ev in events:
        title_lower = ev.get("title", "").lower()
        top = ev.get("top_market")
        if not top:
            continue

        # Extract "Yes" probability (first outcome price)
        prices = top.get("prices_pct") or []
        outcomes = top.get("outcomes") or []
        yes_prob: float | None = None
        for outcome, price in zip(outcomes, prices):
            if str(outcome).lower() == "yes" and price is not None:
                yes_prob = float(price)
                break
        # Fallback: first price if no explicit "Yes" label
        if yes_prob is None and prices and prices[0] is not None:
            yes_prob = float(prices[0])

        if yes_prob is None:
            continue

        all_yes_probs.append(yes_prob)

        if "rate cut" in title_lower or "interest rate" in title_lower:
            fed_cut_probs.append(yes_prob)
        if "recession" in title_lower:
            recession_probs.append(yes_prob)

    return {
        # Take max prob if multiple matching markets exist (most relevant/active one)
        "polymarket_fed_cut_prob": max(fed_cut_probs) if fed_cut_probs else float("nan"),
        "polymarket_recession_prob": max(recession_probs) if recession_probs else float("nan"),
        "polymarket_macro_sentiment": (
            sum(all_yes_probs) / len(all_yes_probs) if all_yes_probs else float("nan")
        ),
    }


def get_macro_snapshot() -> dict[str, Any]:
    """Return a complete macro snapshot for display or ML feature enrichment.

    Combines FRED data + Polymarket signals into a single dict.
    """
    yield_curve = get_yield_curve_spread()
    m2 = get_m2_growth()

    return {
        "yield_curve": yield_curve,
        "m2_money_supply": m2,
        "fed_funds_rate": yield_curve.get("fed_funds_rate"),
        "spread_10y_2y": yield_curve.get("spread_10y_2y"),
        "spread_10y_3m": yield_curve.get("spread_10y_3m"),
        "high_yield_oas": yield_curve.get("high_yield_oas"),
        "m2_yoy_pct": m2.get("m2_yoy_pct"),
        "sources": ["FRED", "Polymarket"],
    }

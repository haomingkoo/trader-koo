"""Polymarket public-API archive: trading-relevant prediction-market events."""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import urllib.parse
import urllib.request
from typing import Any

LOG = logging.getLogger(__name__)

_cache_lock = threading.Lock()
_polymarket_cache: dict[str, Any] = {}
_polymarket_fetch_status: dict[str, Any] = {
    "ok": False,
    "source_fetched_at": None,
    "cache_hit": False,
    "error": "not_fetched",
}
_cache_ttl_sec = 3600  # 1 hour
POLYMARKET_TOP_N = 50  # one ranked list serves every request limit; the archival scheduler refreshes it

_POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"


_POLYMARKET_TRADING_TAGS = frozenset({
    "business",
    "commodities",
    "crypto",
    "crypto-prices",
    "economic-policy",
    "economy",
    "fed",
    "fed-rates",
    "finance",
    "stocks",
})


def _is_trading_relevant_polymarket_event(event: dict[str, Any]) -> bool:
    """Use Polymarket's own taxonomy instead of matching words in prose."""
    tag_slugs = {
        str(tag.get("slug") or "").strip().lower()
        for tag in (event.get("tags") or [])
        if isinstance(tag, dict)
    }
    return bool(tag_slugs & _POLYMARKET_TRADING_TAGS)


def _parse_single_market(m: dict[str, Any]) -> dict[str, Any]:
    """Parse a single Gamma API market object into a clean dict."""
    raw_outcomes = m.get("outcomes") or []
    if isinstance(raw_outcomes, str):
        try:
            raw_outcomes = json.loads(raw_outcomes)
        except Exception:
            raw_outcomes = [raw_outcomes]
    outcomes = list(raw_outcomes) if isinstance(raw_outcomes, (list, tuple)) else []

    raw_prices = m.get("outcomePrices") or []
    if isinstance(raw_prices, str):
        try:
            raw_prices = json.loads(raw_prices)
        except Exception:
            raw_prices = []
    prices_raw = list(raw_prices) if isinstance(raw_prices, (list, tuple)) else []
    prices: list[float | None] = []
    for p in prices_raw:
        try:
            prices.append(round(float(p) * 100, 1))
        except (TypeError, ValueError):
            prices.append(None)

    is_resolved = bool(m.get("closed") or m.get("resolved"))
    is_active = bool(m.get("active")) and not is_resolved and m.get("acceptingOrders") is not False

    return {
        "market_id": str(m.get("id") or m.get("conditionId") or ""),
        "question": str(m.get("question", "")).strip(),
        "outcomes": outcomes,
        "prices_pct": prices,
        "volume": round(float(m.get("volume", 0) or 0), 2),
        "volume_24h": round(float(m.get("volume24hr", 0) or 0), 2),
        "liquidity": round(float(m.get("liquidityNum", m.get("liquidity", 0)) or 0), 2),
        "end_date": m.get("endDate"),
        "resolved": is_resolved,
        "active": is_active,
    }


def _parse_event_markets(
    raw_markets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse and sort all sub-markets for an event.

    Sort priority: end_date ascending (earliest first), then volume
    descending as tiebreaker. Markets without an end_date go last.
    """
    parsed = [_parse_single_market(m) for m in raw_markets]

    def _sort_key(mkt: dict[str, Any]) -> tuple[int, str, float]:
        end = mkt.get("end_date")
        if end:
            return (0, str(end), -mkt["volume"])
        return (1, "", -mkt["volume"])

    parsed.sort(key=_sort_key)
    return parsed


def _classify_event_type(markets: list[dict[str, Any]]) -> str:
    """Classify an event as 'simple', 'timeline', or 'multi_outcome'.

    - simple: 1 active sub-market with YES/NO outcomes
    - timeline: active sub-markets have different end_dates
    - multi_outcome: multiple active sub-markets or >3 non-YES/NO outcomes
    """
    # Classify based on ACTIVE markets only — resolved ones don't affect display
    active = [m for m in markets if m.get("active", True)]

    if len(active) <= 1:
        if active:
            outcomes = [str(o).lower() for o in (active[0].get("outcomes") or [])]
            if set(outcomes) <= {"yes", "no"}:
                return "simple"
            return "multi_outcome"
        return "simple"

    # Multiple active sub-markets — check if end_dates differ (timeline)
    end_dates = {m.get("end_date") for m in active if m.get("end_date")}
    if len(end_dates) > 1:
        return "timeline"

    # Check if outcomes are non-YES/NO across active sub-markets
    non_yesno_count = sum(
        1 for m in active
        if not (set(str(o).lower() for o in (m.get("outcomes") or [])) <= {"yes", "no"})
    )

    if non_yesno_count > 3:
        return "multi_outcome"

    # Multiple active markets with same end_date and YES/NO outcomes
    # These are timeline-style (e.g., "by March 31?" and "by June 30?")
    if len(active) > 1:
        return "timeline"

    return "simple"


def fetch_polymarket_events(
    *,
    limit: int = 15,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Fetch curated finance-relevant Polymarket events (grouped markets).

    Uses events API for better grouping, filters for macro/finance relevance,
    sorts by total volume. Returns event-level data with embedded markets.
    """
    cache_key = "poly_events_top"
    if use_cache:
        with _cache_lock:
            cached = _polymarket_cache.get(cache_key)
            if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
                _polymarket_fetch_status.update({
                    "ok": True,
                    "source_fetched_at": cached.get("source_fetched_at"),
                    "cache_hit": True,
                    "error": None,
                })
                return cached["data"][:limit]

    try:
        raw_events: list[dict[str, Any]] = []
        page_size = 100
        for offset in (0, page_size):
            query = urllib.parse.urlencode({
                "limit": page_size,
                "offset": offset,
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
            })
            req = urllib.request.Request(
                f"{_POLYMARKET_GAMMA}/events?{query}",
                headers={"User-Agent": "trader-koo/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                page = json.loads(resp.read().decode("utf-8"))
            if not isinstance(page, list):
                raise ValueError("Polymarket events response is not a list")
            raw_events.extend(event for event in page if isinstance(event, dict))
            if len(page) < page_size:
                break

        deduped_events: dict[str, dict[str, Any]] = {}
        for event in raw_events:
            identity = str(event.get("id") or event.get("slug") or "")
            if identity:
                deduped_events[identity] = event
        raw_events = list(deduped_events.values())
        source_fetched_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

        # Filter using Polymarket's event taxonomy. Untagged events fail closed.
        relevant: list[dict[str, Any]] = []
        for ev in raw_events:
            if not _is_trading_relevant_polymarket_event(ev):
                continue

            raw_markets = ev.get("markets") or []
            total_volume = sum(float(m.get("volume", 0) or 0) for m in raw_markets)

            # Parse all sub-markets for this event
            parsed_markets = _parse_event_markets(raw_markets)
            active_markets = [market for market in parsed_markets if market.get("active")]
            if not active_markets:
                continue

            # The headline must represent a currently tradable market.
            top_market = (
                max(active_markets, key=lambda m: (m["volume_24h"], m["volume"]))
                if active_markets
                else None
            )

            # Count active vs resolved sub-markets
            active_count = sum(1 for m in parsed_markets if m.get("active"))
            resolved_count = sum(1 for m in parsed_markets if m.get("resolved"))

            # Classify event type
            event_type = _classify_event_type(parsed_markets)

            relevant.append({
                "title": str(ev.get("title", "")).strip(),
                "slug": ev.get("slug", ""),
                "market_count": len(active_markets),
                "total_volume": round(total_volume, 2),
                "active_volume_24h": round(sum(m["volume_24h"] for m in active_markets), 2),
                "active_liquidity": round(sum(m["liquidity"] for m in active_markets), 2),
                "end_date": ev.get("endDate"),
                "image": ev.get("image"),
                "url": f"https://polymarket.com/event/{ev.get('slug', '')}",
                "top_market": top_market,
                # Public consumers only need contracts that can still trade.
                # Retain aggregate resolved_count for provenance without
                # returning stale contracts as current opportunities.
                "markets": active_markets,
                "active_count": active_count,
                "resolved_count": resolved_count,
                "event_type": event_type,
                "source_fetched_at": source_fetched_at,
            })

        # Rank current activity, not lifetime volume from resolved contracts.
        relevant.sort(
            key=lambda e: (e["active_volume_24h"], e["active_liquidity"]),
            reverse=True,
        )
        with _cache_lock:
            _polymarket_cache[cache_key] = {
                "data": relevant[:POLYMARKET_TOP_N],
                "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_cache_ttl_sec)).timestamp(),
                "source_fetched_at": source_fetched_at,
            }
            _polymarket_fetch_status.update({
                "ok": True,
                "source_fetched_at": source_fetched_at,
                "cache_hit": False,
                "error": None,
            })

        LOG.info("Polymarket events: %d relevant / %d total", len(relevant), len(raw_events))
        return relevant[:min(limit, POLYMARKET_TOP_N)]

    except Exception as exc:
        LOG.warning("Polymarket events fetch failed: %s", exc)
        with _cache_lock:
            _polymarket_fetch_status.update({
                "ok": False,
                "source_fetched_at": None,
                "cache_hit": False,
                "error": type(exc).__name__,
            })
        return []


def polymarket_fetch_status() -> dict[str, Any]:
    """Return non-secret source freshness for the latest fetch attempt."""
    with _cache_lock:
        return dict(_polymarket_fetch_status)

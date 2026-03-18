"""RSS-based financial news aggregator.

Fetches headlines from free public RSS feeds (Yahoo Finance, CNBC, MarketWatch)
and scores them with a lightweight financial-sentiment lexicon.  No API keys,
no auth, no external dependencies — uses only stdlib ``xml.etree`` and
``urllib``.

Designed to supplement Finnhub's news-sentiment endpoint by providing
per-headline sentiment scores that Finnhub's company-news endpoint lacks.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import logging
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS feed registry
# ---------------------------------------------------------------------------

_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

_MARKET_FEEDS: list[dict[str, str]] = [
    {"name": "CNBC Top News", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html"},
    {"name": "CNBC Markets", "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html"},
    {"name": "MarketWatch Top Stories", "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
    {"name": "MarketWatch Markets", "url": "https://feeds.marketwatch.com/marketwatch/marketpulse/"},
]

_DEFAULT_TICKERS = ("SPY", "QQQ", "DIA", "IWM")

# ---------------------------------------------------------------------------
# Financial sentiment lexicon
# ---------------------------------------------------------------------------

_BULLISH_TERMS = frozenset({
    "rally", "rallies", "surge", "surges", "gain", "gains", "rise", "rises",
    "climb", "climbs", "jump", "jumps", "soar", "soars", "breakout", "bull",
    "bullish", "upside", "outperform", "beat", "beats", "record", "high",
    "optimism", "optimistic", "recovery", "recover", "boost", "boosts",
    "strong", "strength", "upgrade", "upgraded", "buy", "accumulate",
    "positive", "growth", "expand", "expanding", "momentum", "uptrend",
})

_BEARISH_TERMS = frozenset({
    "fall", "falls", "drop", "drops", "decline", "declines", "plunge",
    "plunges", "crash", "crashes", "sink", "sinks", "slump", "slumps",
    "sell", "selloff", "sell-off", "bear", "bearish", "downside",
    "underperform", "miss", "misses", "low", "fear", "recession",
    "warning", "risk", "weak", "weakness", "downgrade", "downgraded",
    "negative", "loss", "losses", "contraction", "downturn", "downtrend",
    "inflation", "layoff", "layoffs", "tariff", "tariffs", "deficit",
})


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())


def _score_headline(title: str, description: str = "") -> tuple[float | None, int, int]:
    """Score text on a -1.0 to +1.0 scale using the financial lexicon."""
    tokens = _tokenize(f"{title} {description}")
    if not tokens:
        return None, 0, 0
    bullish = sum(1 for t in tokens if t in _BULLISH_TERMS)
    bearish = sum(1 for t in tokens if t in _BEARISH_TERMS)
    if bullish == 0 and bearish == 0:
        return 0.0, 0, 0
    return (bullish - bearish) / (bullish + bearish), bullish, bearish


def _score_to_100(raw: float | None) -> int | None:
    if raw is None:
        return None
    return round((max(-1.0, min(1.0, raw)) + 1.0) * 50.0)


def _label_for_score(score: int | None) -> str | None:
    if score is None:
        return None
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_rfc2822(date_str: str) -> str | None:
    """Parse RFC 2822 date (RSS pubDate) to ISO UTC string."""
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return _iso_utc(parsed)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _request_timeout_sec() -> int:
    raw = str(os.getenv("TRADER_KOO_RSS_TIMEOUT_SEC", "15")).strip()
    try:
        return max(5, min(30, int(raw)))
    except ValueError:
        return 15


def _csv_env(name: str, default: tuple[str, ...]) -> list[str]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return list(default)
    return [t for t in (s.strip() for s in raw.split(",")) if t]


def _rss_enabled() -> bool:
    raw = str(os.getenv("TRADER_KOO_RSS_ENABLED", "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# RSS fetching
# ---------------------------------------------------------------------------

def _fetch_rss(url: str) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed, returning a list of item dicts."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "trader-koo/1.0 (+https://trader.kooexperience.com)",
            "Accept": "application/rss+xml, application/xml, text/xml",
        },
    )
    with urllib.request.urlopen(req, timeout=_request_timeout_sec()) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    items: list[dict[str, Any]] = []
    # Handle both standard RSS and namespaced feeds
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        description = (item.findtext("description") or "").strip()
        # Strip CDATA / HTML tags from description
        description = re.sub(r"<[^>]+>", "", description).strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()

        items.append({
            "title": title,
            "description": description[:500] if description else "",
            "url": link or None,
            "pub_date_raw": pub_date,
            "time_published": _parse_rfc2822(pub_date) if pub_date else None,
        })
    return items


def _fetch_yahoo_ticker_news(ticker: str) -> list[dict[str, Any]]:
    """Fetch Yahoo Finance RSS headlines for a single ticker."""
    url = _YAHOO_RSS.format(symbol=urllib.request.quote(ticker))
    try:
        items = _fetch_rss(url)
        for item in items:
            item["source"] = "Yahoo Finance"
            item["feed_ticker"] = ticker
        return items
    except Exception as exc:
        LOG.warning("Yahoo RSS fetch failed for %s: %s", ticker, exc)
        return []


def _fetch_market_feeds() -> list[dict[str, Any]]:
    """Fetch headlines from general market news RSS feeds."""
    all_items: list[dict[str, Any]] = []
    for feed in _MARKET_FEEDS:
        try:
            items = _fetch_rss(feed["url"])
            for item in items:
                item["source"] = feed["name"]
                item["feed_ticker"] = None
            all_items.extend(items)
        except Exception as exc:
            LOG.warning("RSS fetch failed for %s: %s", feed["name"], exc)
    return all_items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_rss_headlines(
    *,
    tickers: list[str] | None = None,
    max_headlines: int = 15,
    include_market_feeds: bool = True,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Fetch and score RSS headlines from Yahoo Finance + market news feeds.

    Returns a dict matching the structure expected by the news sentiment
    enrichment layer::

        {
            "provider": "rss_aggregator",
            "available": bool,
            "headlines": [...],       # scored headline dicts
            "score": int | None,      # aggregate 0-100
            "raw_score": float | None,
            "label": str | None,
            "article_count": int,
            "feed_breakdown": [...],  # per-source stats
            "note": str,
        }
    """
    resolved_now = now_utc or dt.datetime.now(dt.timezone.utc)
    resolved_tickers = tickers or _csv_env("TRADER_KOO_SENTIMENT_TICKERS", _DEFAULT_TICKERS)

    if not _rss_enabled():
        return {
            "provider": "rss_aggregator",
            "available": False,
            "headlines": [],
            "score": None,
            "raw_score": None,
            "label": None,
            "article_count": 0,
            "feed_breakdown": [],
            "note": "RSS news aggregation disabled (TRADER_KOO_RSS_ENABLED=0)",
        }

    all_items: list[dict[str, Any]] = []

    # Ticker-specific Yahoo RSS
    for ticker in resolved_tickers:
        all_items.extend(_fetch_yahoo_ticker_news(ticker))

    # Market-wide feeds
    if include_market_feeds:
        all_items.extend(_fetch_market_feeds())

    # Deduplicate by title (case-insensitive)
    seen_titles: set[str] = set()
    unique_items: list[dict[str, Any]] = []
    for item in all_items:
        key = item["title"].lower().strip()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique_items.append(item)

    if not unique_items:
        return {
            "provider": "rss_aggregator",
            "available": False,
            "headlines": [],
            "score": None,
            "raw_score": None,
            "label": None,
            "article_count": 0,
            "feed_breakdown": [],
            "note": "No headlines returned from RSS feeds",
        }

    # Score each headline
    weighted_total = 0.0
    weight_total = 0.0
    scored_headlines: list[dict[str, Any]] = []
    feed_stats: dict[str, dict[str, int]] = {}

    for item in unique_items:
        source = str(item.get("source") or "unknown")
        raw, bullish, bearish = _score_headline(item["title"], item.get("description", ""))
        score_100 = _score_to_100(raw)

        if raw is not None:
            weighted_total += raw
            weight_total += 1.0

        scored_headlines.append({
            "title": item["title"],
            "source": source,
            "url": item.get("url"),
            "time_published": item.get("time_published"),
            "score": score_100,
            "label": _label_for_score(score_100),
            "bullish_terms": bullish,
            "bearish_terms": bearish,
            "feed_ticker": item.get("feed_ticker"),
        })

        stats = feed_stats.setdefault(source, {"count": 0, "scored": 0})
        stats["count"] += 1
        if raw is not None and raw != 0.0:
            stats["scored"] += 1

    # Sort by recency (items with time_published first, then rest)
    scored_headlines.sort(
        key=lambda h: h.get("time_published") or "",
        reverse=True,
    )

    # Aggregate score
    agg_raw: float | None = None
    agg_score: int | None = None
    agg_label: str | None = None
    if weight_total > 0:
        agg_raw = round(weighted_total / weight_total, 4)
        agg_score = _score_to_100(agg_raw)
        agg_label = _label_for_score(agg_score)

    feed_breakdown = [
        {"source": name, "articles": s["count"], "with_sentiment": s["scored"]}
        for name, s in sorted(feed_stats.items())
    ]

    return {
        "provider": "rss_aggregator",
        "available": True,
        "headlines": scored_headlines[:max_headlines],
        "score": agg_score,
        "raw_score": agg_raw,
        "label": agg_label,
        "article_count": len(unique_items),
        "feed_breakdown": feed_breakdown,
        "note": (
            f"{len(unique_items)} headlines from {len(feed_stats)} RSS sources "
            f"({len(resolved_tickers)} ticker feeds + "
            f"{len(_MARKET_FEEDS)} market feeds)"
        ),
    }

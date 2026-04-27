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
import hashlib
import json
import logging
import os
import re
import sqlite3
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

_MACRO_FEEDS: list[dict[str, str]] = [
    {"name": "Fed Monetary Policy", "url": "https://www.federalreserve.gov/feeds/press_monetary.xml"},
    {"name": "Fed Speeches", "url": "https://www.federalreserve.gov/feeds/speeches.xml"},
    {"name": "Fed Policy Rates", "url": "https://www.federalreserve.gov/feeds/prates.xml"},
    {"name": "BLS CPI", "url": "https://www.bls.gov/feed/cpi.rss"},
    {"name": "BLS Employment Situation", "url": "https://www.bls.gov/feed/empsit.rss"},
    {"name": "BLS Latest Numbers", "url": "https://www.bls.gov/feed/bls_latest.rss"},
    {"name": "BEA Releases", "url": "https://apps.bea.gov/rss/rss.xml"},
    {"name": "Census Economic Indicators", "url": "https://www.census.gov/economic-indicators/indicator.xml"},
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

_MACRO_MARKET_TERMS = frozenset({
    "cpi", "inflation", "jobs", "payroll", "payrolls", "unemployment",
    "fomc", "federal", "fed", "rate", "rates", "gdp", "pce", "ppi",
    "retail", "sales", "housing", "claims", "yield", "treasury",
    "manufacturing", "services", "beige", "powell", "employment",
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


def _utc_now_iso() -> str:
    return _iso_utc(dt.datetime.now(dt.timezone.utc))


def _snapshot_date(value: str | None = None) -> str:
    if value:
        try:
            return dt.date.fromisoformat(str(value)[:10]).isoformat()
        except ValueError:
            pass
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _parse_rfc2822(date_str: str) -> str | None:
    """Parse RFC 2822 date (RSS pubDate) to ISO UTC string."""
    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return _iso_utc(parsed)
    except Exception:
        return None


def _headline_key(item: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(item.get("source") or ""),
            str(item.get("feed_ticker") or ""),
            str(item.get("title") or ""),
            str(item.get("url") or ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                (table,),
            ).fetchone()
            is not None
        )
    except Exception:
        return False


def ensure_rss_news_snapshot_schema(conn: sqlite3.Connection) -> None:
    """Create the RSS headline snapshot table used for point-in-time context."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rss_news_snapshots (
            snapshot_date TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
            headline_key TEXT NOT NULL,
            provider TEXT NOT NULL,
            feed_ticker TEXT,
            source TEXT,
            title TEXT NOT NULL,
            url TEXT,
            time_published TEXT,
            score REAL,
            label TEXT,
            macro_relevant INTEGER DEFAULT 0,
            raw_json TEXT,
            PRIMARY KEY (snapshot_date, headline_key)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rss_news_snapshots_date ON rss_news_snapshots(snapshot_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rss_news_snapshots_ticker ON rss_news_snapshots(feed_ticker, snapshot_date)"
    )


def persist_rss_headline_snapshot(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
    *,
    snapshot_date: str | None = None,
    snapshot_ts: str | None = None,
) -> int:
    """Persist scored RSS headlines from ``fetch_rss_headlines``."""
    headlines = payload.get("headlines") if isinstance(payload, dict) else None
    if not isinstance(headlines, list) or not headlines:
        return 0
    ensure_rss_news_snapshot_schema(conn)
    date_value = _snapshot_date(snapshot_date)
    ts_value = snapshot_ts or _utc_now_iso()
    provider = str(payload.get("provider") or "rss_aggregator")
    rows = []
    for item in headlines:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        rows.append(
            (
                date_value,
                ts_value,
                _headline_key(item),
                provider,
                str(item.get("feed_ticker") or "").upper() or None,
                item.get("source"),
                str(item.get("title") or "")[:500],
                item.get("url"),
                item.get("time_published"),
                item.get("score"),
                item.get("label"),
                1 if item.get("macro_relevant") else 0,
                json.dumps(item, sort_keys=True),
            )
        )
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO rss_news_snapshots (
            snapshot_date, snapshot_ts, headline_key, provider, feed_ticker,
            source, title, url, time_published, score, label, macro_relevant, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def load_rss_headline_snapshot(
    conn: sqlite3.Connection,
    *,
    tickers: list[str] | None = None,
    as_of_date: str | None = None,
    max_headlines: int = 30,
    lookback_days: int = 3,
) -> dict[str, Any]:
    """Load the latest stored RSS headlines at or before ``as_of_date``."""
    if not _table_exists(conn, "rss_news_snapshots"):
        return {
            "provider": "rss_snapshot",
            "available": False,
            "headlines": [],
            "score": None,
            "raw_score": None,
            "label": None,
            "article_count": 0,
            "feed_breakdown": [],
            "note": "rss_news_snapshots table missing",
        }
    date_value = _snapshot_date(as_of_date)
    try:
        asof = dt.date.fromisoformat(date_value)
    except ValueError:
        asof = dt.datetime.now(dt.timezone.utc).date()
    earliest = (asof - dt.timedelta(days=max(0, int(lookback_days)))).isoformat()
    ticker_set = {str(t or "").upper().strip() for t in (tickers or []) if str(t or "").strip()}
    rows = conn.execute(
        """
        SELECT provider, feed_ticker, source, title, url, time_published, score, label,
               macro_relevant, snapshot_date, snapshot_ts
        FROM rss_news_snapshots
        WHERE snapshot_date <= ? AND snapshot_date >= ?
          AND (
              feed_ticker IS NULL
              OR feed_ticker = ''
              OR feed_ticker IN ({placeholders})
              OR macro_relevant = 1
          )
        ORDER BY snapshot_date DESC, COALESCE(time_published, snapshot_ts) DESC
        LIMIT ?
        """.format(placeholders=",".join("?" for _ in ticker_set) or "''"),
        [date_value, earliest, *sorted(ticker_set), max(1, int(max_headlines))],
    ).fetchall()
    if not rows:
        return {
            "provider": "rss_snapshot",
            "available": False,
            "headlines": [],
            "score": None,
            "raw_score": None,
            "label": None,
            "article_count": 0,
            "feed_breakdown": [],
            "note": f"No RSS snapshots available for {date_value}",
        }

    headlines: list[dict[str, Any]] = []
    feed_stats: dict[str, int] = {}
    raw_scores: list[float] = []
    for row in rows:
        d = dict(row) if isinstance(row, sqlite3.Row) else {
            "provider": row[0],
            "feed_ticker": row[1],
            "source": row[2],
            "title": row[3],
            "url": row[4],
            "time_published": row[5],
            "score": row[6],
            "label": row[7],
            "macro_relevant": row[8],
            "snapshot_date": row[9],
            "snapshot_ts": row[10],
        }
        score = d.get("score")
        if isinstance(score, (int, float)):
            raw_scores.append((float(score) - 50.0) / 50.0)
        source = str(d.get("source") or "unknown")
        feed_stats[source] = feed_stats.get(source, 0) + 1
        headlines.append(
            {
                "title": d.get("title"),
                "source": d.get("source"),
                "url": d.get("url"),
                "time_published": d.get("time_published"),
                "score": d.get("score"),
                "label": d.get("label"),
                "feed_ticker": d.get("feed_ticker"),
                "macro_relevant": bool(d.get("macro_relevant")),
                "snapshot_date": d.get("snapshot_date"),
            }
        )
    agg_raw = round(sum(raw_scores) / len(raw_scores), 4) if raw_scores else None
    agg_score = _score_to_100(agg_raw) if agg_raw is not None else None
    return {
        "provider": "rss_snapshot",
        "available": True,
        "headlines": headlines,
        "score": agg_score,
        "raw_score": agg_raw,
        "label": _label_for_score(agg_score),
        "article_count": len(headlines),
        "feed_breakdown": [
            {"source": source, "articles": count, "with_sentiment": count}
            for source, count in sorted(feed_stats.items())
        ],
        "note": f"Loaded {len(headlines)} stored RSS headlines for {date_value}",
    }


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


def _macro_rss_enabled() -> bool:
    raw = str(os.getenv("TRADER_KOO_MACRO_RSS_ENABLED", "1")).strip().lower()
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
    feeds = [*_MARKET_FEEDS, *(_MACRO_FEEDS if _macro_rss_enabled() else [])]
    for feed in feeds:
        try:
            items = _fetch_rss(feed["url"])
            for item in items:
                item["source"] = feed["name"]
                item["feed_ticker"] = None
                item["macro_relevant"] = bool(
                    set(_tokenize(f"{item.get('title', '')} {item.get('description', '')}"))
                    & _MACRO_MARKET_TERMS
                )
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
    resolved_tickers = (
        tickers if tickers is not None
        else _csv_env("TRADER_KOO_SENTIMENT_TICKERS", _DEFAULT_TICKERS)
    )

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
            "macro_relevant": bool(item.get("macro_relevant")),
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
            f"{len(_MARKET_FEEDS)} market feeds + "
            f"{len(_MACRO_FEEDS) if _macro_rss_enabled() else 0} macro feeds)"
        ),
    }

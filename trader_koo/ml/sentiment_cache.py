"""DB-backed news sentiment cache for ML feature extraction.

Stores per-ticker per-date sentiment scores in SQLite so the ML pipeline
can look them up instantly instead of making one Finnhub API call per
ticker per date during training.

Cache is populated in two ways:
1. Nightly pipeline: batch-fetches today's sentiment for all active tickers
2. Backfill: one-time bulk fetch for historical date ranges (--backfill-sentiment)

At feature extraction time, features.py reads from the cache first and only
falls back to live API for cache misses during live scoring.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
import time
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

# Finnhub free tier: 60 calls/min.  Stay safely under with a per-call delay.
_FINNHUB_CALLS_PER_MIN = 60
_CALL_DELAY_SEC = 60.0 / _FINNHUB_CALLS_PER_MIN + 0.05  # ~1.05s


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_sentiment_cache_table(conn: sqlite3.Connection) -> None:
    """Create the news_sentiment_cache table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment_cache (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            sentiment_score REAL,
            article_count INTEGER DEFAULT 0,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentiment_cache_date "
        "ON news_sentiment_cache(date)"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def lookup_cached_sentiment(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
) -> float | None:
    """Look up a cached sentiment score for a ticker on a specific date.

    Returns the sentiment_score (float in [-1, 1]) or None on cache miss.
    """
    row = conn.execute(
        "SELECT sentiment_score FROM news_sentiment_cache "
        "WHERE ticker = ? AND date = ?",
        (ticker, date),
    ).fetchone()
    if row is None:
        return None
    return float(row[0]) if row[0] is not None else None


def lookup_cached_sentiment_batch(
    conn: sqlite3.Connection,
    tickers: list[str],
    date: str,
) -> dict[str, float]:
    """Batch lookup cached sentiment for multiple tickers on one date.

    Returns dict mapping ticker -> sentiment_score.  Tickers with no cache
    entry are omitted from the result (not returned as NaN).
    """
    if not tickers:
        return {}
    placeholders = ",".join("?" * len(tickers))
    rows = conn.execute(
        f"SELECT ticker, sentiment_score FROM news_sentiment_cache "
        f"WHERE date = ? AND ticker IN ({placeholders})",
        (date, *tickers),
    ).fetchall()
    return {
        str(r[0]): float(r[1]) if r[1] is not None else np.nan
        for r in rows
    }


# ---------------------------------------------------------------------------
# Fetch + store
# ---------------------------------------------------------------------------

def _finnhub_key() -> str:
    return str(os.getenv("FINNHUB_API_KEY", "")).strip()


def _compute_sentiment_for_ticker(
    ticker: str,
    date_from: str,
    date_to: str,
    api_key: str,
) -> tuple[float | None, int]:
    """Fetch Finnhub company-news and compute lexicon sentiment score.

    Returns (sentiment_score, article_count).  sentiment_score is None
    when no scoreable headlines are found or on API failure.
    """
    try:
        from trader_koo.news_sentiment import _finnhub_get
        from trader_koo.rss_news import _score_headline
    except ImportError:
        LOG.warning("News sentiment modules not available")
        return None, 0

    try:
        articles = _finnhub_get(
            "company-news",
            {"symbol": ticker, "from": date_from, "to": date_to},
            api_key,
        )
        if not isinstance(articles, list):
            return None, 0

        headline_scores: list[float] = []
        for item in articles[:30]:
            if not isinstance(item, dict):
                continue
            headline = str(item.get("headline") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not headline:
                continue
            raw, _bullish, _bearish = _score_headline(headline, summary)
            if raw is not None and raw != 0.0:
                headline_scores.append(raw)

        if headline_scores:
            return sum(headline_scores) / len(headline_scores), len(headline_scores)
        return None, len(articles)

    except Exception as exc:
        LOG.debug("Finnhub company-news failed for %s: %s", ticker, exc)
        return None, 0


def cache_news_sentiment_batch(
    conn: sqlite3.Connection,
    tickers: list[str],
    date: str,
    *,
    lookback_days: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch and cache news sentiment for a list of tickers on a given date.

    Rate-limited to respect Finnhub's 60 calls/min.  Skips tickers that
    already have a cache entry unless *force* is True.

    Returns summary stats: {fetched, skipped, failed, total}.
    """
    ensure_sentiment_cache_table(conn)

    api_key = _finnhub_key()
    if not api_key:
        LOG.warning("FINNHUB_API_KEY not set; cannot populate sentiment cache")
        return {"fetched": 0, "skipped": len(tickers), "failed": 0, "total": len(tickers)}

    date_to = date
    date_from = (
        dt.datetime.strptime(date, "%Y-%m-%d") - dt.timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")
    now_ts = dt.datetime.now(dt.timezone.utc).isoformat()

    # Find which tickers already have cache entries for this date
    if not force and tickers:
        placeholders = ",".join("?" * len(tickers))
        cached_rows = conn.execute(
            f"SELECT ticker FROM news_sentiment_cache "
            f"WHERE date = ? AND ticker IN ({placeholders})",
            (date, *tickers),
        ).fetchall()
        already_cached = {str(r[0]) for r in cached_rows}
    else:
        already_cached = set()

    stats = {"fetched": 0, "skipped": 0, "failed": 0, "total": len(tickers)}

    for ticker in tickers:
        if ticker in already_cached:
            stats["skipped"] += 1
            continue

        score, article_count = _compute_sentiment_for_ticker(
            ticker, date_from, date_to, api_key,
        )

        try:
            conn.execute(
                """
                INSERT INTO news_sentiment_cache (ticker, date, sentiment_score, article_count, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    sentiment_score = excluded.sentiment_score,
                    article_count = excluded.article_count,
                    fetched_at = excluded.fetched_at
                """,
                (ticker, date, score, article_count, now_ts),
            )
            conn.commit()
            stats["fetched"] += 1
        except Exception as exc:
            LOG.warning("Failed to cache sentiment for %s on %s: %s", ticker, date, exc)
            stats["failed"] += 1

        # Rate limit: sleep between API calls
        time.sleep(_CALL_DELAY_SEC)

    LOG.info(
        "Sentiment cache batch for %s: fetched=%d skipped=%d failed=%d total=%d",
        date, stats["fetched"], stats["skipped"], stats["failed"], stats["total"],
    )
    return stats


def backfill_sentiment_cache(
    conn: sqlite3.Connection,
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    sample_frequency: int = 5,
) -> dict[str, Any]:
    """Backfill the sentiment cache for a date range (one-time operation).

    Only processes trading dates that have price data in the DB.
    Respects Finnhub rate limits -- this is intentionally slow.

    Parameters
    ----------
    conn : sqlite3.Connection
    tickers : list of ticker symbols
    start_date, end_date : str (YYYY-MM-DD)
    sample_frequency : int
        Process every Nth trading day (matches trainer.py sampling).
    """
    # Get trading dates from the DB
    rows = conn.execute(
        """
        SELECT DISTINCT date FROM price_daily
        WHERE date >= ? AND date <= ? AND ticker = 'SPY'
        ORDER BY date
        """,
        (start_date, end_date),
    ).fetchall()
    trading_dates = [str(r[0]) for r in rows]
    sampled_dates = trading_dates[::sample_frequency]

    LOG.info(
        "Backfilling sentiment cache: %d dates (%d sampled), %d tickers",
        len(trading_dates), len(sampled_dates), len(tickers),
    )

    total_stats = {"fetched": 0, "skipped": 0, "failed": 0, "dates_processed": 0}

    for date in sampled_dates:
        stats = cache_news_sentiment_batch(conn, tickers, date)
        total_stats["fetched"] += stats["fetched"]
        total_stats["skipped"] += stats["skipped"]
        total_stats["failed"] += stats["failed"]
        total_stats["dates_processed"] += 1

        LOG.info(
            "Backfill progress: %d/%d dates, %d fetched so far",
            total_stats["dates_processed"], len(sampled_dates), total_stats["fetched"],
        )

    LOG.info(
        "Backfill complete: %d dates, %d fetched, %d skipped, %d failed",
        total_stats["dates_processed"],
        total_stats["fetched"],
        total_stats["skipped"],
        total_stats["failed"],
    )
    return total_stats


# ---------------------------------------------------------------------------
# CLI entry point  (python -m trader_koo.ml.sentiment_cache)
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Populate today's sentiment cache for all active tickers."""
    import argparse
    import sys
    from pathlib import Path

    # Ensure project root is on sys.path for imports
    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # Load .env for local dev
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Cache news sentiment scores")
    parser.add_argument(
        "--db-path",
        default=os.getenv("TRADER_KOO_DB_PATH", "/data/trader_koo.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--date",
        default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        help="Date to fetch sentiment for (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--backfill-start",
        default=None,
        help="Start date for backfill mode (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--backfill-end",
        default=None,
        help="End date for backfill mode (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=5,
        help="Sample every Nth trading day during backfill (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    conn = sqlite3.connect(args.db_path)
    ensure_sentiment_cache_table(conn)

    # Get all active tickers (non-index tickers with recent price data)
    ticker_rows = conn.execute(
        "SELECT DISTINCT ticker FROM price_daily "
        "WHERE date >= date(?, '-30 days') "
        "AND ticker NOT LIKE '^%%'",
        (args.date,),
    ).fetchall()
    tickers = sorted(str(r[0]) for r in ticker_rows)
    LOG.info("Found %d active tickers", len(tickers))

    if args.backfill_start:
        # Backfill mode
        end = args.backfill_end or args.date
        stats = backfill_sentiment_cache(
            conn, tickers, args.backfill_start, end,
            sample_frequency=args.sample_frequency,
        )
    else:
        # Single-day mode (nightly pipeline)
        stats = cache_news_sentiment_batch(conn, tickers, args.date)

    conn.close()
    LOG.info("Done: %s", stats)


if __name__ == "__main__":
    _cli_main()

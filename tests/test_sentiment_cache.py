"""Tests for the news sentiment DB cache (trader_koo.ml.sentiment_cache)."""
from __future__ import annotations

import datetime as dt
import sqlite3

import numpy as np
import pytest

from trader_koo.ml.sentiment_cache import (
    cache_news_sentiment_batch,
    ensure_sentiment_cache_table,
    lookup_cached_sentiment,
    lookup_cached_sentiment_batch,
)


@pytest.fixture()
def mem_db() -> sqlite3.Connection:
    """In-memory SQLite with the sentiment cache table created."""
    conn = sqlite3.connect(":memory:")
    ensure_sentiment_cache_table(conn)
    return conn


class TestEnsureSentimentCacheTable:
    def test_creates_table(self, mem_db: sqlite3.Connection) -> None:
        row = mem_db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='news_sentiment_cache' LIMIT 1"
        ).fetchone()
        assert row is not None

    def test_idempotent(self, mem_db: sqlite3.Connection) -> None:
        ensure_sentiment_cache_table(mem_db)
        ensure_sentiment_cache_table(mem_db)
        row = mem_db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='news_sentiment_cache' LIMIT 1"
        ).fetchone()
        assert row is not None


class TestLookupCachedSentiment:
    def test_returns_none_on_cache_miss(self, mem_db: sqlite3.Connection) -> None:
        result = lookup_cached_sentiment(mem_db, "AAPL", "2026-03-20")
        assert result is None

    def test_returns_score_on_hit(self, mem_db: sqlite3.Connection) -> None:
        mem_db.execute(
            "INSERT INTO news_sentiment_cache (ticker, date, sentiment_score, article_count, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("AAPL", "2026-03-20", 0.35, 5, "2026-03-20T22:00:00Z"),
        )
        mem_db.commit()

        result = lookup_cached_sentiment(mem_db, "AAPL", "2026-03-20")
        assert result == pytest.approx(0.35)

    def test_returns_none_for_null_score(self, mem_db: sqlite3.Connection) -> None:
        mem_db.execute(
            "INSERT INTO news_sentiment_cache (ticker, date, sentiment_score, article_count, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("TSLA", "2026-03-20", None, 0, "2026-03-20T22:00:00Z"),
        )
        mem_db.commit()

        result = lookup_cached_sentiment(mem_db, "TSLA", "2026-03-20")
        assert result is None


class TestLookupCachedSentimentBatch:
    def test_returns_empty_for_no_tickers(self, mem_db: sqlite3.Connection) -> None:
        result = lookup_cached_sentiment_batch(mem_db, [], "2026-03-20")
        assert result == {}

    def test_returns_cached_scores(self, mem_db: sqlite3.Connection) -> None:
        for ticker, score in [("AAPL", 0.3), ("MSFT", -0.1), ("GOOG", 0.5)]:
            mem_db.execute(
                "INSERT INTO news_sentiment_cache (ticker, date, sentiment_score, article_count, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (ticker, "2026-03-20", score, 5, "2026-03-20T22:00:00Z"),
            )
        mem_db.commit()

        result = lookup_cached_sentiment_batch(
            mem_db, ["AAPL", "MSFT", "NVDA"], "2026-03-20"
        )

        assert "AAPL" in result
        assert result["AAPL"] == pytest.approx(0.3)
        assert "MSFT" in result
        assert result["MSFT"] == pytest.approx(-0.1)
        # NVDA not in cache, so not in result
        assert "NVDA" not in result

    def test_omits_different_date(self, mem_db: sqlite3.Connection) -> None:
        mem_db.execute(
            "INSERT INTO news_sentiment_cache (ticker, date, sentiment_score, article_count, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("AAPL", "2026-03-19", 0.5, 3, "2026-03-19T22:00:00Z"),
        )
        mem_db.commit()

        result = lookup_cached_sentiment_batch(mem_db, ["AAPL"], "2026-03-20")
        assert result == {}


class TestCacheNewsSentimentBatch:
    def test_skips_when_no_api_key(
        self, mem_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)

        stats = cache_news_sentiment_batch(mem_db, ["AAPL", "MSFT"], "2026-03-20")

        assert stats["fetched"] == 0
        assert stats["skipped"] == 2
        assert stats["total"] == 2

    def test_skips_already_cached(
        self, mem_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

        # Pre-populate cache
        mem_db.execute(
            "INSERT INTO news_sentiment_cache (ticker, date, sentiment_score, article_count, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("AAPL", "2026-03-20", 0.3, 5, "2026-03-20T22:00:00Z"),
        )
        mem_db.commit()

        # Mock the API call to avoid real network
        call_count = 0

        def fake_compute(ticker: str, date_from: str, date_to: str, api_key: str) -> tuple:
            nonlocal call_count
            call_count += 1
            return 0.1, 2

        monkeypatch.setattr(
            "trader_koo.ml.sentiment_cache._compute_sentiment_for_ticker",
            fake_compute,
        )
        # Speed up test by removing sleep
        monkeypatch.setattr("trader_koo.ml.sentiment_cache._CALL_DELAY_SEC", 0.0)

        stats = cache_news_sentiment_batch(mem_db, ["AAPL", "MSFT"], "2026-03-20")

        assert stats["skipped"] == 1  # AAPL already cached
        assert stats["fetched"] == 1  # MSFT fetched
        assert call_count == 1  # Only one API call for MSFT

    def test_force_refetches_cached(
        self, mem_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

        mem_db.execute(
            "INSERT INTO news_sentiment_cache (ticker, date, sentiment_score, article_count, fetched_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("AAPL", "2026-03-20", 0.3, 5, "2026-03-20T22:00:00Z"),
        )
        mem_db.commit()

        def fake_compute(ticker: str, date_from: str, date_to: str, api_key: str) -> tuple:
            return 0.9, 10

        monkeypatch.setattr(
            "trader_koo.ml.sentiment_cache._compute_sentiment_for_ticker",
            fake_compute,
        )
        monkeypatch.setattr("trader_koo.ml.sentiment_cache._CALL_DELAY_SEC", 0.0)

        stats = cache_news_sentiment_batch(
            mem_db, ["AAPL"], "2026-03-20", force=True
        )

        assert stats["fetched"] == 1
        assert stats["skipped"] == 0

        # Verify the score was updated
        score = lookup_cached_sentiment(mem_db, "AAPL", "2026-03-20")
        assert score == pytest.approx(0.9)

    def test_stores_none_score(
        self, mem_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

        def fake_compute(ticker: str, date_from: str, date_to: str, api_key: str) -> tuple:
            return None, 0

        monkeypatch.setattr(
            "trader_koo.ml.sentiment_cache._compute_sentiment_for_ticker",
            fake_compute,
        )
        monkeypatch.setattr("trader_koo.ml.sentiment_cache._CALL_DELAY_SEC", 0.0)

        cache_news_sentiment_batch(mem_db, ["AAPL"], "2026-03-20")

        # Row exists but score is NULL
        row = mem_db.execute(
            "SELECT sentiment_score, article_count FROM news_sentiment_cache "
            "WHERE ticker = ? AND date = ?",
            ("AAPL", "2026-03-20"),
        ).fetchone()
        assert row is not None
        assert row[0] is None
        assert row[1] == 0

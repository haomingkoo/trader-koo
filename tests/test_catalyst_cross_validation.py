"""Tests for earnings marker cross-validation in catalyst_data."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pytest

from trader_koo.catalyst_data import (
    _finviz_earnings_date,
    get_ticker_earnings_markers,
    ensure_external_data_cache_table,
)


def _setup_db(conn: sqlite3.Connection, *, finviz_earnings: str | None = None) -> None:
    """Seed minimal schema for catalyst cross-validation tests."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_daily (
            ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER
        )
        """
    )
    conn.execute("INSERT INTO price_daily VALUES ('AAPL','2026-03-20',245,250,244,248,1000000)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS finviz_fundamentals (
            ticker TEXT, snapshot_ts TEXT, price REAL, discount_pct REAL, peg REAL, raw_json TEXT
        )
        """
    )
    raw_obj: dict = {}
    if finviz_earnings:
        raw_obj["Earnings Date"] = finviz_earnings
    conn.execute(
        "INSERT INTO finviz_fundamentals VALUES (?, ?, ?, ?, ?, ?)",
        ("AAPL", "2026-03-20T22:00:00Z", 248.0, 17.0, 2.37, json.dumps(raw_obj)),
    )
    ensure_external_data_cache_table(conn)
    conn.commit()


def _seed_finnhub_cache(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """Seed a Finnhub earnings cache entry."""
    today = dt.date.today()
    cache_key = f"finnhub:earnings_calendar:{today.isoformat()}:{(today + dt.timedelta(days=120)).isoformat()}"
    conn.execute(
        """
        INSERT OR REPLACE INTO external_data_cache(cache_key, provider, fetched_ts, expires_ts, payload_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            cache_key,
            "finnhub",
            "2026-03-21T00:00:00Z",
            "2026-03-22T00:00:00Z",
            json.dumps(rows),
        ),
    )
    conn.commit()


class TestFinvizEarningsDate:
    def test_returns_date_when_present(self) -> None:
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings="Apr 24 BMO")
        result = _finviz_earnings_date(conn, "AAPL", dt.date(2026, 3, 21))

        assert result == dt.date(2026, 4, 24)

    def test_returns_none_when_no_earnings(self) -> None:
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings=None)
        result = _finviz_earnings_date(conn, "AAPL", dt.date(2026, 3, 21))

        assert result is None

    def test_returns_none_for_unknown_ticker(self) -> None:
        conn = sqlite3.connect(":memory:")
        _setup_db(conn)
        result = _finviz_earnings_date(conn, "ZZZZ", dt.date(2026, 3, 21))

        assert result is None


class TestEarningsMarkerCrossValidation:
    def test_skips_marker_when_finnhub_has_no_finviz_corroboration(self) -> None:
        """Finnhub says AAPL earnings on Mar 25, but Finviz has no date → skip."""
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings=None)
        _seed_finnhub_cache(conn, [
            {"symbol": "AAPL", "date": "2026-03-25", "hour": "bmo"},
        ])

        markers = get_ticker_earnings_markers(
            conn, ticker="AAPL", market_date=dt.date(2026, 3, 21),
        )

        assert markers == []

    def test_drops_finnhub_when_contradicts_finviz(self) -> None:
        """Finnhub says Mar 25, Finviz says Apr 24 → Mar 25 dropped."""
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings="Apr 24 BMO")
        _seed_finnhub_cache(conn, [
            {"symbol": "AAPL", "date": "2026-03-25", "hour": "bmo"},
        ])

        markers = get_ticker_earnings_markers(
            conn, ticker="AAPL", market_date=dt.date(2026, 3, 21),
        )

        assert all(m["date"] != "2026-03-25" for m in markers)

    def test_skips_all_when_finviz_date_is_past(self) -> None:
        """Finviz shows past earnings (Jan 29 AMC) → no markers at all."""
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings="Jan 29 AMC")
        _seed_finnhub_cache(conn, [
            {"symbol": "AAPL", "date": "2026-03-25", "hour": "bmo"},
            {"symbol": "AAPL", "date": "2026-03-31", "hour": "bmo"},
        ])

        markers = get_ticker_earnings_markers(
            conn, ticker="AAPL", market_date=dt.date(2026, 3, 21),
        )

        assert markers == []

    def test_shows_marker_when_both_sources_agree(self) -> None:
        """Finnhub says Apr 24, Finviz says Apr 24 BMO → show marker."""
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings="Apr 24 BMO")
        _seed_finnhub_cache(conn, [
            {"symbol": "AAPL", "date": "2026-04-24", "hour": "bmo"},
        ])

        markers = get_ticker_earnings_markers(
            conn, ticker="AAPL", market_date=dt.date(2026, 3, 21),
        )

        assert len(markers) == 1
        assert markers[0]["ticker"] == "AAPL"
        assert markers[0]["date"] == "2026-04-24"
        assert markers[0]["session"] == "BMO"

    def test_shows_marker_within_tolerance(self) -> None:
        """Finnhub says Apr 23, Finviz says Apr 24 → delta=1, within tolerance."""
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings="Apr 24 BMO")
        _seed_finnhub_cache(conn, [
            {"symbol": "AAPL", "date": "2026-04-23", "hour": "bmo"},
        ])

        markers = get_ticker_earnings_markers(
            conn, ticker="AAPL", market_date=dt.date(2026, 3, 21),
        )

        assert len(markers) == 1

    def test_rejects_marker_outside_tolerance(self) -> None:
        """Finnhub says Apr 28, Finviz says Apr 24 → delta=4, outside ±3 tolerance."""
        conn = sqlite3.connect(":memory:")
        _setup_db(conn, finviz_earnings="Apr 24 BMO")
        _seed_finnhub_cache(conn, [
            {"symbol": "AAPL", "date": "2026-04-28", "hour": "bmo"},
        ])

        markers = get_ticker_earnings_markers(
            conn, ticker="AAPL", market_date=dt.date(2026, 3, 21),
        )

        finnhub_dates = [m["date"] for m in markers if m.get("source") != "fundamentals_snapshot"]
        assert "2026-04-28" not in finnhub_dates

    def test_indices_still_skipped(self) -> None:
        """Indices (^VIX etc.) skipped by prefix check."""
        conn = sqlite3.connect(":memory:")
        _setup_db(conn)

        markers = get_ticker_earnings_markers(
            conn, ticker="^VIX", market_date=dt.date(2026, 3, 21),
        )

        assert markers == []

    def test_etfs_skipped_via_cross_validation(self) -> None:
        """ETFs like SPY have no Finviz earnings date → Finnhub markers skipped."""
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS price_daily "
            "(ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)"
        )
        conn.execute("INSERT INTO price_daily VALUES ('SPY','2026-03-20',500,510,499,505,5000000)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS finviz_fundamentals "
            "(ticker TEXT, snapshot_ts TEXT, price REAL, discount_pct REAL, peg REAL, raw_json TEXT)"
        )
        # SPY has no earnings date in Finviz
        conn.execute(
            "INSERT INTO finviz_fundamentals VALUES (?, ?, ?, ?, ?, ?)",
            ("SPY", "2026-03-20T22:00:00Z", 505.0, None, None, json.dumps({})),
        )
        ensure_external_data_cache_table(conn)
        conn.commit()
        _seed_finnhub_cache(conn, [
            {"symbol": "SPY", "date": "2026-03-25", "hour": "bmo"},
        ])

        markers = get_ticker_earnings_markers(
            conn, ticker="SPY", market_date=dt.date(2026, 3, 21),
        )

        assert markers == []

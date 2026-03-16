"""Unit tests for trader_koo.backend.services.market_data."""
from __future__ import annotations

import datetime as dt
import sqlite3

import pytest

from trader_koo.backend.services.market_data import (
    days_since,
    get_data_sources,
    hours_since,
    parse_iso_utc,
)


class TestParseIsoUtc:
    def test_parses_valid_iso_string(self):
        result = parse_iso_utc("2026-03-10T12:00:00+00:00")

        assert isinstance(result, dt.datetime)
        assert result.tzinfo is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 10

    def test_parses_zulu_suffix(self):
        result = parse_iso_utc("2026-03-10T12:00:00Z")

        assert isinstance(result, dt.datetime)
        assert result.tzinfo is not None

    def test_parses_naive_iso_as_utc(self):
        result = parse_iso_utc("2026-03-10T12:00:00")

        assert isinstance(result, dt.datetime)
        assert result.tzinfo == dt.timezone.utc

    def test_returns_none_for_none(self):
        assert parse_iso_utc(None) is None

    def test_returns_none_for_empty_string(self):
        assert parse_iso_utc("") is None

    def test_returns_none_for_whitespace(self):
        assert parse_iso_utc("   ") is None

    def test_returns_none_for_malformed_input(self):
        assert parse_iso_utc("not-a-date") is None

    def test_returns_none_for_partial_date(self):
        assert parse_iso_utc("2026-03") is None


class TestHoursSince:
    def test_returns_correct_hours_elapsed(self):
        ts = "2026-03-10T10:00:00Z"
        now = dt.datetime(2026, 3, 10, 14, 0, 0, tzinfo=dt.timezone.utc)

        result = hours_since(ts, now)

        assert result is not None
        assert abs(result - 4.0) < 0.01

    def test_returns_none_for_none_timestamp(self):
        now = dt.datetime.now(dt.timezone.utc)

        assert hours_since(None, now) is None

    def test_returns_none_for_unparsable_timestamp(self):
        now = dt.datetime.now(dt.timezone.utc)

        assert hours_since("garbage", now) is None


class TestDaysSince:
    def test_returns_positive_days_for_past_date(self):
        now = dt.datetime(2026, 3, 15, 20, 0, 0, tzinfo=dt.timezone.utc)

        result = days_since("2026-03-10", now)

        assert result is not None
        assert result > 4.0

    def test_returns_none_for_none_date(self):
        now = dt.datetime.now(dt.timezone.utc)

        assert days_since(None, now) is None

    def test_returns_none_for_empty_string(self):
        now = dt.datetime.now(dt.timezone.utc)

        assert days_since("", now) is None

    def test_returns_none_for_malformed_date(self):
        now = dt.datetime.now(dt.timezone.utc)

        assert days_since("not-a-date", now) is None

    def test_returns_zero_or_positive(self):
        now = dt.datetime(2026, 3, 10, 20, 0, 0, tzinfo=dt.timezone.utc)

        result = days_since("2026-03-10", now)

        assert result is not None
        assert result >= 0.0


class TestGetDataSources:
    def test_returns_dict_with_expected_keys(self, seeded_conn: sqlite3.Connection):
        result = get_data_sources(seeded_conn, "SPY")

        assert isinstance(result, dict)
        assert "price" in result
        assert "price_timestamp" in result

    def test_returns_known_data_source_for_seeded_ticker(self, seeded_conn: sqlite3.Connection):
        result = get_data_sources(seeded_conn, "SPY")

        assert result["price"] == "yfinance"

    def test_returns_unknown_for_missing_ticker(self, seeded_conn: sqlite3.Connection):
        result = get_data_sources(seeded_conn, "ZZZZZ_FAKE")

        assert result["price"] == "unknown"
        assert result["price_timestamp"] is None

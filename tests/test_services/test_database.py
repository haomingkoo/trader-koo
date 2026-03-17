"""Unit tests for trader_koo.backend.services.database."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from trader_koo.backend.services.database import (
    DB_PATH,
    get_conn,
    get_latest_fundamentals,
    get_price_df,
    table_exists,
)


class TestDBPath:
    def test_db_path_is_path_object(self):
        assert isinstance(DB_PATH, Path)

    def test_db_path_ends_with_db_extension(self):
        assert DB_PATH.suffix == ".db"


class TestGetConn:
    def test_returns_valid_connection_when_db_exists(self, tmp_path: Path):
        db_file = tmp_path / "test.db"
        sqlite3.connect(str(db_file)).close()

        conn = get_conn(db_path=db_file)

        assert isinstance(conn, sqlite3.Connection)
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_raises_http_exception_when_db_missing(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.db"

        with pytest.raises(HTTPException) as exc_info:
            get_conn(db_path=missing)

        assert exc_info.value.status_code == 500
        assert "unavailable" in exc_info.value.detail.lower() or "not found" in exc_info.value.detail.lower()


class TestTableExists:
    def test_detects_existing_table(self, mem_conn: sqlite3.Connection):
        result = table_exists(mem_conn, "price_daily")

        assert result is True

    def test_returns_false_for_missing_table(self, mem_conn: sqlite3.Connection):
        result = table_exists(mem_conn, "nonexistent_table_xyz")

        assert result is False

    def test_returns_false_for_empty_name(self, mem_conn: sqlite3.Connection):
        result = table_exists(mem_conn, "")

        assert result is False


class TestGetPriceDf:
    def test_returns_dataframe_with_expected_columns(self, seeded_conn: sqlite3.Connection):
        df = get_price_df(seeded_conn, "SPY")

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_returns_empty_dataframe_for_unknown_ticker(self, seeded_conn: sqlite3.Connection):
        df = get_price_df(seeded_conn, "ZZZZZ_FAKE")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_results_sorted_by_date_ascending(self, seeded_conn: sqlite3.Connection):
        df = get_price_df(seeded_conn, "SPY")

        dates = df["date"].tolist()
        assert dates == sorted(dates)


class TestGetLatestFundamentals:
    def test_returns_dict_for_existing_ticker(self, seeded_conn: sqlite3.Connection):
        result = get_latest_fundamentals(seeded_conn, "SPY")

        assert isinstance(result, dict)
        assert result.get("ticker") == "SPY"
        assert "price" in result
        assert "peg" in result

    def test_returns_empty_dict_for_unknown_ticker(self, seeded_conn: sqlite3.Connection):
        result = get_latest_fundamentals(seeded_conn, "NOPE_FAKE")

        assert isinstance(result, dict)
        assert result == {}

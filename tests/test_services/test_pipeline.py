"""Unit tests for trader_koo.backend.services.pipeline."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trader_koo.backend.services.pipeline import (
    get_cached_status,
    invalidate_status_cache,
    parse_log_line_ts_utc,
    pipeline_status_snapshot,
    reconcile_stale_running_runs,
    set_cached_status,
)


class TestStatusCache:
    """set / get / invalidate cycle on the thread-safe status cache."""

    def setup_method(self):
        invalidate_status_cache()

    def teardown_method(self):
        invalidate_status_cache()

    def test_set_then_get_returns_cached_payload(self):
        now = dt.datetime(2026, 3, 10, 12, 0, 0, tzinfo=dt.timezone.utc)
        payload = {"stage": "idle", "active": False}

        set_cached_status(now, payload)
        result = get_cached_status(now + dt.timedelta(seconds=1))

        assert result is not None
        assert result["stage"] == "idle"

    def test_get_returns_none_when_cache_expired(self):
        now = dt.datetime(2026, 3, 10, 12, 0, 0, tzinfo=dt.timezone.utc)
        payload = {"stage": "idle"}

        set_cached_status(now, payload)
        far_future = now + dt.timedelta(seconds=9999)
        result = get_cached_status(far_future)

        assert result is None

    def test_invalidate_clears_cache(self):
        now = dt.datetime(2026, 3, 10, 12, 0, 0, tzinfo=dt.timezone.utc)
        set_cached_status(now, {"ok": True})

        invalidate_status_cache()

        assert get_cached_status(now) is None

    def test_get_returns_none_when_never_set(self):
        now = dt.datetime(2026, 3, 10, 12, 0, 0, tzinfo=dt.timezone.utc)

        assert get_cached_status(now) is None

    def test_cached_payload_is_copy(self):
        now = dt.datetime(2026, 3, 10, 12, 0, 0, tzinfo=dt.timezone.utc)
        payload = {"value": 1}

        set_cached_status(now, payload)
        result = get_cached_status(now)

        assert result is not None
        result["value"] = 999
        second = get_cached_status(now)
        assert second is not None
        assert second["value"] == 1


class TestParseLogLineTs:
    def test_parses_iso_8601_utc(self):
        line = "2026-03-10T14:30:00+00:00 [START] daily_update.sh"

        result = parse_log_line_ts_utc(line)

        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 10

    def test_parses_python_logging_format(self):
        line = "2026-03-10 14:30:00,123 | INFO | starting"

        result = parse_log_line_ts_utc(line)

        assert result is not None
        assert result.hour == 14

    def test_returns_none_for_none(self):
        assert parse_log_line_ts_utc(None) is None

    def test_returns_none_for_empty_string(self):
        assert parse_log_line_ts_utc("") is None

    def test_returns_none_for_garbage(self):
        assert parse_log_line_ts_utc("no timestamp here") is None


class TestPipelineStatusSnapshot:
    @patch("trader_koo.backend.services.pipeline._tail_text_file", return_value=[])
    @patch("trader_koo.backend.services.pipeline.read_latest_ingest_run", return_value=None)
    def test_returns_dict_with_required_keys(self, mock_run, mock_tail):
        result = pipeline_status_snapshot(log_lines=10)

        assert isinstance(result, dict)
        required_keys = {
            "run_log_path",
            "active",
            "stage",
            "latest_run",
            "markers",
            "stage_line",
            "stage_line_ts",
            "stage_age_sec",
            "stale_timeout_sec",
            "stale_inference",
            "running_age_sec",
            "running_stale_min",
            "running_stale",
            "last_completed_stage",
            "last_completed_status",
            "last_completed_line",
            "last_completed_ts",
            "tail",
        }
        assert required_keys.issubset(set(result.keys()))

    @patch("trader_koo.backend.services.pipeline._tail_text_file", return_value=[])
    @patch("trader_koo.backend.services.pipeline.read_latest_ingest_run", return_value=None)
    def test_inactive_when_no_logs(self, mock_run, mock_tail):
        result = pipeline_status_snapshot(log_lines=10)

        assert result["active"] is False


class TestReconcileStaleRunningRuns:
    @patch("trader_koo.backend.services.pipeline.DB_PATH", Path("/nonexistent/db.db"))
    def test_returns_empty_when_db_missing(self):
        result = reconcile_stale_running_runs()

        assert isinstance(result, dict)
        assert result["checked"] == 0
        assert result["reconciled"] == 0
        assert result["run_ids"] == []

    def test_returns_dict_structure(self, tmp_path: Path):
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        conn.execute("""
            CREATE TABLE ingest_runs (
                run_id TEXT PRIMARY KEY,
                started_ts TEXT,
                finished_ts TEXT,
                status TEXT,
                tickers_total INTEGER DEFAULT 0,
                tickers_ok INTEGER DEFAULT 0,
                tickers_failed INTEGER DEFAULT 0,
                error_message TEXT
            )
        """)
        conn.commit()
        conn.close()

        with patch("trader_koo.backend.services.pipeline.DB_PATH", db_file):
            result = reconcile_stale_running_runs()

        assert "checked" in result
        assert "reconciled" in result
        assert "run_ids" in result

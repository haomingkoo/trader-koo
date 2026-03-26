"""Unit tests for pipeline_runs DB-based stage tracking.

Tests the create/update/finish/read/resume logic added to
trader_koo.backend.services.pipeline for surviving Railway deploy kills.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trader_koo.backend.services.pipeline import (
    count_resume_attempts,
    create_pipeline_run,
    determine_resume_mode,
    ensure_pipeline_runs_schema,
    finish_pipeline_run,
    read_interrupted_pipeline_run,
    read_latest_pipeline_run,
    update_pipeline_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Path:
    """Create a DB file with the pipeline_runs table."""
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    ensure_pipeline_runs_schema(conn)
    conn.close()
    return db_file


def _read_row(db_path: Path, run_id: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# ensure_pipeline_runs_schema
# ---------------------------------------------------------------------------

class TestEnsurePipelineRunsSchema:
    def test_creates_table_when_missing(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))

        ensure_pipeline_runs_schema(conn)

        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_idempotent_on_existing_table(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))

        ensure_pipeline_runs_schema(conn)
        ensure_pipeline_runs_schema(conn)

        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
        ).fetchone()
        conn.close()
        assert row is not None


# ---------------------------------------------------------------------------
# create_pipeline_run
# ---------------------------------------------------------------------------

class TestCreatePipelineRun:
    def test_inserts_row_with_correct_fields(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)

        create_pipeline_run(
            run_id="pipe_test_001",
            mode="full",
            source="scheduler",
            db_path=db_file,
        )

        row = _read_row(db_file, "pipe_test_001")
        assert row is not None
        assert row["run_id"] == "pipe_test_001"
        assert row["mode"] == "full"
        assert row["source"] == "scheduler"
        assert row["status"] == "running"
        assert row["stage"] == "starting"
        assert row["started_ts"] is not None
        assert row["finished_ts"] is None

    def test_noop_when_db_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.db"

        create_pipeline_run(
            run_id="pipe_test_002",
            mode="full",
            source="scheduler",
            db_path=missing,
        )

        assert not missing.exists()

    def test_creates_table_if_missing(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        conn.close()

        create_pipeline_run(
            run_id="pipe_auto_schema",
            mode="yolo",
            source="admin",
            db_path=db_file,
        )

        row = _read_row(db_file, "pipe_auto_schema")
        assert row is not None
        assert row["mode"] == "yolo"


# ---------------------------------------------------------------------------
# update_pipeline_stage
# ---------------------------------------------------------------------------

class TestUpdatePipelineStage:
    def test_updates_stage_and_timestamp(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_stage_001",
            mode="full",
            source="scheduler",
            db_path=db_file,
        )

        update_pipeline_stage("pipe_stage_001", "ingest", db_path=db_file)

        row = _read_row(db_file, "pipe_stage_001")
        assert row is not None
        assert row["stage"] == "ingest"
        assert row["stage_started_ts"] is not None

    def test_multiple_stage_transitions(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_multi",
            mode="full",
            source="scheduler",
            db_path=db_file,
        )

        for stage in ("ingest", "yolo", "report"):
            update_pipeline_stage("pipe_multi", stage, db_path=db_file)

        row = _read_row(db_file, "pipe_multi")
        assert row is not None
        assert row["stage"] == "report"

    def test_noop_when_db_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.db"

        update_pipeline_stage("pipe_ghost", "ingest", db_path=missing)


# ---------------------------------------------------------------------------
# finish_pipeline_run
# ---------------------------------------------------------------------------

class TestFinishPipelineRun:
    def test_marks_run_as_ok(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_finish_ok",
            mode="full",
            source="scheduler",
            db_path=db_file,
        )

        finish_pipeline_run(
            "pipe_finish_ok",
            status="ok",
            ingest_ok=True,
            yolo_ok=True,
            report_ok=True,
            db_path=db_file,
        )

        row = _read_row(db_file, "pipe_finish_ok")
        assert row is not None
        assert row["status"] == "ok"
        assert row["stage"] == "done"
        assert row["finished_ts"] is not None
        assert row["ingest_ok"] == 1
        assert row["yolo_ok"] == 1
        assert row["report_ok"] == 1

    def test_marks_run_as_failed_with_error(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_finish_fail",
            mode="yolo",
            source="admin",
            db_path=db_file,
        )

        finish_pipeline_run(
            "pipe_finish_fail",
            status="failed",
            error_message="daily_update.sh rc=1",
            ingest_ok=False,
            yolo_ok=False,
            report_ok=False,
            db_path=db_file,
        )

        row = _read_row(db_file, "pipe_finish_fail")
        assert row is not None
        assert row["status"] == "failed"
        assert row["error_message"] == "daily_update.sh rc=1"

    def test_marks_run_as_interrupted(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_interrupted",
            mode="full",
            source="scheduler",
            db_path=db_file,
        )

        finish_pipeline_run(
            "pipe_interrupted",
            status="interrupted",
            error_message="killed_by_restart",
            ingest_ok=True,
            yolo_ok=False,
            report_ok=False,
            db_path=db_file,
        )

        row = _read_row(db_file, "pipe_interrupted")
        assert row is not None
        assert row["status"] == "interrupted"
        assert row["ingest_ok"] == 1
        assert row["yolo_ok"] == 0


# ---------------------------------------------------------------------------
# read_latest_pipeline_run
# ---------------------------------------------------------------------------

class TestReadLatestPipelineRun:
    def test_returns_most_recent_run(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        # Insert with explicit timestamps to guarantee ordering
        conn = sqlite3.connect(str(db_file))
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, started_ts, mode, source, status, stage) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("pipe_old", "2026-03-25T10:00:00Z", "full", "scheduler", "ok", "done"),
        )
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, started_ts, mode, source, status, stage) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("pipe_new", "2026-03-25T22:00:00Z", "yolo", "admin", "running", "yolo"),
        )
        conn.commit()
        conn.close()

        result = read_latest_pipeline_run(db_path=db_file)

        assert result is not None
        assert result["run_id"] == "pipe_new"

    def test_returns_none_when_no_runs(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)

        result = read_latest_pipeline_run(db_path=db_file)

        assert result is None

    def test_returns_none_when_db_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.db"

        result = read_latest_pipeline_run(db_path=missing)

        assert result is None

    def test_returns_none_when_table_missing(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))
        conn.close()

        result = read_latest_pipeline_run(db_path=db_file)

        assert result is None


# ---------------------------------------------------------------------------
# read_interrupted_pipeline_run
# ---------------------------------------------------------------------------

class TestReadInterruptedPipelineRun:
    def test_returns_running_row(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_running", mode="full", source="scheduler", db_path=db_file
        )
        update_pipeline_stage("pipe_running", "yolo", db_path=db_file)

        result = read_interrupted_pipeline_run(db_path=db_file)

        assert result is not None
        assert result["run_id"] == "pipe_running"
        assert result["status"] == "running"
        assert result["stage"] == "yolo"

    def test_returns_none_when_all_finished(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_done", mode="full", source="scheduler", db_path=db_file
        )
        finish_pipeline_run(
            "pipe_done", status="ok", ingest_ok=True, yolo_ok=True,
            report_ok=True, db_path=db_file,
        )

        result = read_interrupted_pipeline_run(db_path=db_file)

        assert result is None

    def test_returns_none_when_no_runs(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)

        result = read_interrupted_pipeline_run(db_path=db_file)

        assert result is None

    def test_returns_none_when_db_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.db"

        result = read_interrupted_pipeline_run(db_path=missing)

        assert result is None

    def test_ignores_interrupted_status(self, tmp_path: Path) -> None:
        """Only status='running' should be returned, not 'interrupted'."""
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_was_interrupted", mode="full", source="scheduler",
            db_path=db_file,
        )
        finish_pipeline_run(
            "pipe_was_interrupted", status="interrupted",
            error_message="killed_by_restart", db_path=db_file,
        )

        result = read_interrupted_pipeline_run(db_path=db_file)

        assert result is None


# ---------------------------------------------------------------------------
# count_resume_attempts
# ---------------------------------------------------------------------------

class TestCountResumeAttempts:
    def test_counts_matching_resume_runs(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_original",
            mode="full",
            source="scheduler",
            db_path=db_file,
        )
        create_pipeline_run(
            run_id="pipe_resume_1",
            mode="yolo",
            source="startup_resume:interrupted:pipe_original",
            db_path=db_file,
        )
        create_pipeline_run(
            run_id="pipe_resume_2",
            mode="yolo",
            source="startup_resume:interrupted:pipe_original",
            db_path=db_file,
        )

        count = count_resume_attempts("pipe_original", db_path=db_file)

        assert count == 2

    def test_returns_zero_when_no_resumes(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)
        create_pipeline_run(
            run_id="pipe_no_resume",
            mode="full",
            source="scheduler",
            db_path=db_file,
        )

        count = count_resume_attempts("pipe_no_resume", db_path=db_file)

        assert count == 0

    def test_returns_zero_for_empty_run_id(self, tmp_path: Path) -> None:
        db_file = _make_db(tmp_path)

        count = count_resume_attempts("", db_path=db_file)

        assert count == 0

    def test_returns_zero_when_db_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.db"

        count = count_resume_attempts("pipe_ghost", db_path=missing)

        assert count == 0


# ---------------------------------------------------------------------------
# determine_resume_mode
# ---------------------------------------------------------------------------

class TestDetermineResumeMode:
    def test_resume_yolo_when_ingest_ok(self) -> None:
        interrupted = {
            "mode": "full",
            "stage": "yolo",
            "ingest_ok": 1,
            "yolo_ok": 0,
            "report_ok": 0,
        }

        result = determine_resume_mode(interrupted)

        assert result == "yolo"

    def test_resume_report_when_yolo_ok(self) -> None:
        interrupted = {
            "mode": "full",
            "stage": "report",
            "ingest_ok": 1,
            "yolo_ok": 1,
            "report_ok": 0,
        }

        result = determine_resume_mode(interrupted)

        assert result == "report"

    def test_no_resume_when_report_ok(self) -> None:
        interrupted = {
            "mode": "full",
            "stage": "done",
            "ingest_ok": 1,
            "yolo_ok": 1,
            "report_ok": 1,
        }

        result = determine_resume_mode(interrupted)

        assert result is None

    def test_full_resume_when_ingest_interrupted(self) -> None:
        interrupted = {
            "mode": "full",
            "stage": "ingest",
            "ingest_ok": 0,
            "yolo_ok": 0,
            "report_ok": 0,
        }

        result = determine_resume_mode(interrupted)

        assert result == "full"

    def test_resume_yolo_mode_directly(self) -> None:
        interrupted = {
            "mode": "yolo",
            "stage": "yolo",
            "ingest_ok": 0,
            "yolo_ok": 0,
            "report_ok": 0,
        }

        result = determine_resume_mode(interrupted)

        assert result == "yolo"

    def test_resume_report_mode_directly(self) -> None:
        interrupted = {
            "mode": "report",
            "stage": "report",
            "ingest_ok": 0,
            "yolo_ok": 0,
            "report_ok": 0,
        }

        result = determine_resume_mode(interrupted)

        assert result == "report"

    def test_full_resume_from_starting_stage(self) -> None:
        interrupted = {
            "mode": "full",
            "stage": "starting",
            "ingest_ok": 0,
            "yolo_ok": 0,
            "report_ok": 0,
        }

        result = determine_resume_mode(interrupted)

        assert result == "full"

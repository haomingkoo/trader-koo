"""Pipeline health warnings in /api/status: the conditions the central watch alerts on."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from trader_koo.backend.routers import system
from trader_koo.backend.services import pipeline
from trader_koo.backend.services.pipeline import (
    ensure_pipeline_runs_schema,
    finish_pipeline_run,
    invalidate_status_cache,
    pipeline_status_snapshot,
    reserve_pipeline_run,
    start_pipeline_run,
)
from trader_koo.backend.services.pipeline_health import pipeline_health_warnings

UTC = dt.timezone.utc
# Saturday noon UTC: Friday's bar is the newest expected session.
SATURDAY = dt.datetime(2026, 11, 14, 12, tzinfo=UTC)


def _iso(at: dt.datetime) -> str:
    return at.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _snap(*, status=None, error=None, active_run=None) -> dict:
    return {
        "active": active_run is not None,
        "active_pipeline_run": active_run,
        "last_completed_stage": "report",
        "last_completed_status": status,
        "last_completed_error": error,
        "last_completed_ts": "2026-11-13T22:31:00Z",
    }


def _warnings(snap: dict, price_date: str | None = "2026-11-13", now: dt.datetime = SATURDAY) -> list[str]:
    return pipeline_health_warnings(snap, price_date=price_date, now=now)


# ---------------------------------------------------------------------------
# Last finished run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["failed", "partial", "interrupted", "status_nobody_expected"])
def test_unhealthy_last_run_is_a_warning(status) -> None:
    warnings = _warnings(_snap(status=status, error="daily_update.sh rc=0; failed stages: yolo"))

    assert warnings == [
        f"pipeline last run {status}: daily_update.sh rc=0; failed stages: yolo (finished 2026-11-13T22:31:00Z)"
    ]


@pytest.mark.parametrize("status", ["ok", "skipped", None])
def test_healthy_or_absent_last_run_is_quiet(status) -> None:
    assert _warnings(_snap(status=status)) == []


def test_a_running_pipeline_supersedes_the_last_result() -> None:
    running = {"run_id": "pipe_now", "stage": "yolo", "status": "running",
               "started_ts": _iso(SATURDAY - dt.timedelta(minutes=10)),
               "stage_started_ts": _iso(SATURDAY - dt.timedelta(minutes=2))}

    assert _warnings(_snap(status="failed", error="x", active_run=running)) == []


# ---------------------------------------------------------------------------
# Hung run
# ---------------------------------------------------------------------------

def _active(started_min_ago: int, stage_min_ago: int, stage: str = "yolo") -> dict:
    return {"run_id": "pipe_hung", "stage": stage, "status": "running",
            "started_ts": _iso(SATURDAY - dt.timedelta(minutes=started_min_ago)),
            "stage_started_ts": _iso(SATURDAY - dt.timedelta(minutes=stage_min_ago))}


def test_run_past_the_age_limit_is_reported_stuck() -> None:
    warnings = _warnings(_snap(active_run=_active(started_min_ago=120, stage_min_ago=100)))

    assert warnings == ["pipeline stuck: stage yolo for 100 min (run pipe_hung started 120 min ago, limit 90)"]


def test_normal_length_run_is_not_stuck() -> None:
    # A full run takes about 25 minutes.
    assert _warnings(_snap(active_run=_active(started_min_ago=30, stage_min_ago=5))) == []


def test_stuck_run_does_not_hide_the_previous_failure() -> None:
    warnings = _warnings(_snap(status="partial", error="failed stages: yolo",
                               active_run=_active(started_min_ago=300, stage_min_ago=300, stage="queued")))

    assert warnings[0].startswith("pipeline stuck: stage queued for 300 min")
    assert warnings[1].startswith("pipeline last run partial: failed stages: yolo")


# ---------------------------------------------------------------------------
# Price bar behind the expected session (cases from the retired external freshness check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("price_date", "now", "behind"),
    [
        # Sunday, Friday bar present.
        ("2026-10-23", dt.datetime(2026, 10, 25, 12, tzinfo=UTC), False),
        # Oct 22 08:00Z: Wednesday's 22:00Z ingest is overdue and only Tuesday is stored.
        # The 3-day price_daily stale rule stays quiet here.
        ("2026-10-20", dt.datetime(2026, 10, 22, 8, tzinfo=UTC), True),
        # The Oct 21-22 outage seen on Oct 23 morning.
        ("2026-10-20", dt.datetime(2026, 10, 23, 8, tzinfo=UTC), True),
        # Thanksgiving: Wednesday's bar is current all day.
        ("2026-11-25", dt.datetime(2026, 11, 26, 15, tzinfo=UTC), False),
        # 22:30Z on a trading day, while the nightly ingest runs.
        ("2026-10-21", dt.datetime(2026, 10, 22, 22, 30, tzinfo=UTC), False),
        (None, dt.datetime(2026, 10, 25, 12, tzinfo=UTC), True),
    ],
)
def test_price_behind_expected_session(price_date, now, behind) -> None:
    warnings = _warnings(_snap(), price_date=price_date, now=now)

    assert bool(warnings) is behind
    assert all(w.startswith("price_daily behind expected session") for w in warnings)


# ---------------------------------------------------------------------------
# Persisted run -> snapshot -> /api/status
# ---------------------------------------------------------------------------

@pytest.fixture
def runs_db(tmp_path: Path, monkeypatch) -> Path:
    db = tmp_path / "runs.db"
    with sqlite3.connect(db) as conn:
        ensure_pipeline_runs_schema(conn)
    monkeypatch.setattr(pipeline, "DB_PATH", db)
    monkeypatch.setattr(pipeline, "RUN_LOG_PATH", tmp_path / "cron_daily.log")
    monkeypatch.setattr(pipeline, "read_latest_ingest_run", lambda *a, **k: None)
    return db


def test_snapshot_carries_the_persisted_error(runs_db: Path) -> None:
    reserve_pipeline_run(run_id="pipe_partial", mode="full", source="scheduler", db_path=runs_db)
    start_pipeline_run(run_id="pipe_partial", mode="full", source="scheduler", db_path=runs_db)
    finish_pipeline_run("pipe_partial", status="partial",
                        error_message="daily_update.sh rc=0; failed stages: yolo",
                        ingest_ok=True, report_ok=True, db_path=runs_db)

    snap = pipeline_status_snapshot()

    assert snap["last_completed_status"] == "partial"
    assert snap["last_completed_error"] == "daily_update.sh rc=0; failed stages: yolo"


def test_status_reports_pipeline_health_as_operational(seeded_conn, tmp_path: Path) -> None:
    db_path = tmp_path / "status.db"
    db_path.touch()
    invalidate_status_cache()
    snap = _snap(status="partial", error="daily_update.sh rc=0; failed stages: yolo")

    with (
        patch.object(system, "get_conn", return_value=seeded_conn),
        patch.object(system, "DB_PATH", db_path),
        patch.object(system, "set_cached_status"),
        patch.object(system, "resolve_published_report", return_value=None),
        patch.object(system, "pipeline_status_snapshot", return_value=snap),
        patch.object(system, "post_ingest_resume_candidate", return_value=None),
    ):
        payload = system._compute_status(SATURDAY)

    expected = "pipeline last run partial: daily_update.sh rc=0; failed stages: yolo (finished 2026-11-13T22:31:00Z)"
    assert expected in payload["operational_warnings"]
    assert payload["operational_warning_count"] == len(payload["operational_warnings"])
    assert payload["ok"] is False
    assert payload["pipeline"]["last_completed_error"] == "daily_update.sh rc=0; failed stages: yolo"

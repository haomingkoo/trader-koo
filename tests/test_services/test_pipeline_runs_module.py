"""Observable contracts of the pipeline-runs module.

Callers request a run or recover after a restart; these tests drive that
interface against a real SQLite pipeline_runs table and an in-memory
scheduler, and pin the single daily_update.sh log-marker parser.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trader_koo.backend.services import pipeline, pipeline_runs
from trader_koo.backend.services.pipeline import (
    _infer_last_completed_event_from_tail,
    _infer_pipeline_from_log_tail,
    ensure_pipeline_runs_schema,
    pipeline_status_snapshot,
    read_active_pipeline_run,
    reserve_pipeline_run,
)
from trader_koo.backend.services.run_log import StageResults, run_error_message, run_status, stage_results

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 23, 8, 44, tzinfo=UTC)
RUN_ID_RE = re.compile(r"^pipe_\d{8}T\d{6}_[0-9a-f]{6}$")


class FakeScheduler:
    def __init__(self, *, fail: bool = False) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.fail = fail

    def add_job(self, func, **kwargs) -> None:
        if self.fail:
            raise RuntimeError("scheduler unavailable")
        self.jobs.append({"func": func, **kwargs})


def _db(tmp_path: Path, *, latest_price: str | None = None) -> Path:
    db = tmp_path / "runs.db"
    with sqlite3.connect(db) as conn:
        ensure_pipeline_runs_schema(conn)
        if latest_price is not None:
            conn.execute("CREATE TABLE price_daily (date TEXT)")
            conn.execute("INSERT INTO price_daily VALUES (?)", (latest_price,))
    return db


def _rows(db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM pipeline_runs ORDER BY started_ts, run_id")]


def _insert(db: Path, run_id: str, *, status: str, stage: str, mode: str = "full", **oks: int) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, started_ts, mode, source, status, stage,"
            " ingest_ok, yolo_ok, report_ok) VALUES (?, '2026-10-22T22:00:00Z', ?, 'scheduler', ?, ?, ?, ?, ?)",
            (run_id, mode, status, stage, oks.get("ingest_ok", 0), oks.get("yolo_ok", 0), oks.get("report_ok", 0)),
        )


# ---------------------------------------------------------------------------
# request_run
# ---------------------------------------------------------------------------

def test_request_run_reserves_then_queues_one_job(tmp_path: Path) -> None:
    db = _db(tmp_path)
    scheduler = FakeScheduler()

    request = pipeline_runs.request_run(scheduler, mode="report", source="admin", db_path=db, now=NOW)

    assert request.status == "queued"
    assert RUN_ID_RE.match(request.run_id)
    assert re.match(r"^manual_daily_update_20261023T084400_[0-9a-f]{6}$", request.job_id)
    [row] = _rows(db)
    assert (row["run_id"], row["status"], row["stage"], row["mode"], row["source"]) == (
        request.run_id, "queued", "queued", "report", "admin",
    )
    [job] = scheduler.jobs
    assert job["id"] == request.job_id
    assert job["trigger"] == "date" and job["run_date"] == NOW
    assert job["kwargs"] == {"mode": "report", "source": "admin", "run_id": request.run_id}


def test_request_run_job_resolves_to_the_daily_update_runner(tmp_path: Path) -> None:
    from apscheduler.schedulers.background import BackgroundScheduler

    from trader_koo.backend.services import scheduler as scheduler_module

    real = BackgroundScheduler(timezone="UTC")
    request = pipeline_runs.request_run(real, mode="report", source="admin", db_path=_db(tmp_path))

    assert real.get_job(request.job_id).func is scheduler_module._run_daily_update


def test_request_run_rejects_when_a_run_is_active(tmp_path: Path) -> None:
    db = _db(tmp_path)
    reserve_pipeline_run(run_id="pipe_active", mode="full", source="scheduler", db_path=db)
    scheduler = FakeScheduler()

    request = pipeline_runs.request_run(scheduler, mode="report", source="admin", db_path=db, now=NOW)

    assert request.status == "active"
    assert request.active_run["run_id"] == "pipe_active"
    assert scheduler.jobs == []
    assert [r["run_id"] for r in _rows(db)] == ["pipe_active"]


def test_request_run_undoes_reservation_when_scheduler_rejects(tmp_path: Path) -> None:
    db = _db(tmp_path)

    request = pipeline_runs.request_run(
        FakeScheduler(fail=True), mode="report", source="admin", db_path=db, now=NOW,
    )

    assert request.status == "rejected"
    [row] = _rows(db)
    assert (row["status"], row["stage"], row["error_message"]) == ("failed", "done", "scheduler_enqueue_failed")
    assert read_active_pipeline_run(db_path=db) is None


def test_request_run_queues_nothing_when_reservation_cannot_persist(tmp_path: Path) -> None:
    scheduler = FakeScheduler()

    request = pipeline_runs.request_run(
        scheduler, mode="full", source="admin", db_path=tmp_path / "missing.db", now=NOW,
    )

    assert request.status == "unpersisted"
    assert scheduler.jobs == []


def test_minted_run_ids_share_one_format_and_are_unique() -> None:
    ids = {pipeline_runs.mint_run_id(NOW) for _ in range(200)}

    assert len(ids) > 190
    assert all(RUN_ID_RE.match(i) and i.startswith("pipe_20261023T084400_") for i in ids)
    assert RUN_ID_RE.match(pipeline_runs.mint_run_id())


# ---------------------------------------------------------------------------
# recover_after_restart
# ---------------------------------------------------------------------------

@pytest.fixture
def recovery_env(tmp_path: Path, monkeypatch):
    """Point recovery's ingest/report/log reads at empty temp locations."""
    monkeypatch.setattr(pipeline, "DB_PATH", tmp_path / "runs.db")
    monkeypatch.setattr(pipeline, "RUN_LOG_PATH", tmp_path / "cron_daily.log")
    monkeypatch.setattr(pipeline, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(pipeline, "read_latest_ingest_run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "AUTO_RESUME_INTERRUPTED_PIPELINE", True)
    monkeypatch.setattr(pipeline, "AUTO_STARTUP_CATCHUP", True)
    monkeypatch.setattr(pipeline_runs, "_append_run_log", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_append_run_log", lambda *a, **k: None)
    return tmp_path


def test_recovery_runs_steps_in_order(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "AUTO_STARTUP_CATCHUP", True)
    monkeypatch.setattr(
        pipeline, "reconcile_stale_running_runs", lambda: calls.append("reconcile") or {"reconciled": 0},
    )
    monkeypatch.setattr(
        pipeline_runs, "queue_reserved_pipeline_run",
        lambda scheduler, **kw: calls.append("requeue") or {"scheduled": False},
    )
    monkeypatch.setattr(
        pipeline_runs, "resume_interrupted_run",
        lambda scheduler, **kw: calls.append("resume") or {"queued": False},
    )
    monkeypatch.setattr(
        pipeline_runs, "queue_post_ingest_resume",
        lambda scheduler, **kw: calls.append("legacy") or {"scheduled": False},
    )
    monkeypatch.setattr(
        pipeline_runs, "queue_startup_catchup",
        lambda scheduler, **kw: calls.append("catchup") or {"scheduled": False},
    )

    pipeline_runs.recover_after_restart(FakeScheduler())

    assert calls == ["reconcile", "requeue", "resume", "legacy", "catchup"]


def test_recovery_resumes_interrupted_run_with_a_reserved_run_id(recovery_env: Path) -> None:
    db = _db(recovery_env, latest_price="2026-10-20")
    _insert(db, "pipe_killed", status="running", stage="yolo", ingest_ok=1)
    scheduler = FakeScheduler()

    result = pipeline_runs.recover_after_restart(scheduler, db_path=db, now=NOW)

    rows = {r["run_id"]: r for r in _rows(db)}
    assert (rows["pipe_killed"]["status"], rows["pipe_killed"]["error_message"]) == ("interrupted", "killed_by_restart at stage yolo")
    [job] = scheduler.jobs  # stale prices, but the resume suppresses the catch-up
    resumed_id = job["kwargs"]["run_id"]
    assert RUN_ID_RE.match(resumed_id)
    assert re.match(r"^resume_pipeline_20261023T084400_[0-9a-f]{6}$", job["id"])
    assert job["kwargs"] == {"mode": "yolo", "source": "startup_resume:interrupted:pipe_killed", "run_id": resumed_id}
    assert (rows[resumed_id]["status"], rows[resumed_id]["mode"]) == ("queued", "yolo")
    assert result["interrupted"]["queued"] is True
    assert result["catchup"]["scheduled"] is False


def test_recovery_requeues_a_surviving_reservation_once(recovery_env: Path) -> None:
    db = _db(recovery_env, latest_price="2026-10-20")
    reserve_pipeline_run(run_id="pipe_kept", mode="report", source="admin", db_path=db)
    scheduler = FakeScheduler()

    pipeline_runs.recover_after_restart(scheduler, db_path=db, now=NOW)

    assert [(j["id"], j["kwargs"]) for j in scheduler.jobs] == [
        ("resume_reserved_pipe_kept", {"mode": "report", "source": "admin", "run_id": "pipe_kept"}),
    ]
    assert read_active_pipeline_run(db_path=db)["run_id"] == "pipe_kept"


def test_recovery_queues_only_the_outage_catchup_when_nothing_is_pending(recovery_env: Path) -> None:
    db = _db(recovery_env, latest_price="2026-10-20")
    scheduler = FakeScheduler()

    result = pipeline_runs.recover_after_restart(scheduler, db_path=db, now=NOW)

    [job] = scheduler.jobs
    assert job["id"] == "startup_catchup_20261023T084400"
    assert job["run_date"] == NOW + pipeline.STARTUP_CATCHUP_DELAY
    assert job["kwargs"] == {"mode": "full", "source": "startup_catchup"}
    assert result["catchup"]["scheduled"] is True


def test_recovery_with_auto_resume_disabled_marks_interrupted_and_lets_catchup_run(
    recovery_env: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(pipeline, "AUTO_RESUME_INTERRUPTED_PIPELINE", False)
    db = _db(recovery_env, latest_price="2026-10-20")
    _insert(db, "pipe_killed", status="running", stage="yolo", ingest_ok=1)
    scheduler = FakeScheduler()

    pipeline_runs.recover_after_restart(scheduler, db_path=db, now=NOW)

    assert [j["kwargs"] for j in scheduler.jobs] == [{"mode": "full", "source": "startup_catchup"}]
    assert {r["run_id"]: r["status"] for r in _rows(db)} == {"pipe_killed": "interrupted"}


# ---------------------------------------------------------------------------
# One daily_update.sh log-marker parser
# ---------------------------------------------------------------------------

def _log(*lines: str) -> list[str]:
    return [f"2026-10-22T22:{i:02d}:00+0000 {line}" for i, line in enumerate(lines)]


INGEST_RUNNING = _log("[START] daily_update.sh mode=full", "[INGEST] config enabled=1")
YOLO_RUNNING = INGEST_RUNNING + _log(
    "[INGEST] done rc=0 sec=900", "[YOLO]  Starting daily pattern detection (180d)...",
    "[daily] Processing AAPL",
)
REPORT_RUNNING = YOLO_RUNNING + _log(
    "[YOLO]  Daily pattern detection done. rc=0 sec=300", "[REPORT] Generating daily report...",
)
REPORT_FAILED = REPORT_RUNNING + _log(
    "[REPORT] Done. rc=1 sec=60", "[FAIL]  daily_update.sh mode=full ingest=0 yolo=0 report=1",
)
COMPLETED = REPORT_RUNNING + _log(
    "[REPORT] Done. rc=0 sec=60", "[DONE]  daily_update.sh mode=full ingest=0 yolo=0 report=0",
)


@pytest.mark.parametrize(
    ("tail", "active", "stage"),
    [
        ([], False, "unknown"),
        (INGEST_RUNNING, True, "ingest"),
        (YOLO_RUNNING, True, "yolo"),
        (REPORT_RUNNING, True, "report"),
        (COMPLETED, False, "idle"),
        (REPORT_FAILED, False, "idle"),
        (REPORT_RUNNING + _log("[REPORT] Done. rc=0 sec=60", "[WARN]  daily_update.sh mode=full ingest=2 yolo=0 report=0 (partial failures, report OK)"), False, "idle"),
        (_log("[WEEKLY] Starting weekly YOLO pass", "[weekly] Processing SPY"), True, "yolo"),
        (_log("[WEEKLY] Weekly YOLO completed OK", "[WEEKLY] Report regeneration starting"), True, "report"),
    ],
)
def test_log_tail_inference_stage(tail, active, stage) -> None:
    inferred = _infer_pipeline_from_log_tail(tail)

    assert (inferred["active"], inferred["stage"]) == (active, stage)


INGEST_FAILED = INGEST_RUNNING + _log(
    "[INGEST] done rc=2 sec=900", "[WARN]  ingest failed rc=2 (partial data) — continuing to report step",
)


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ([], None),
        (INGEST_FAILED, ("ingest", "failed")),
        (INGEST_RUNNING + _log("[INGEST] done rc=0 sec=900"), ("ingest", "ok")),
        (REPORT_RUNNING, ("yolo", "ok")),
        (REPORT_RUNNING[:-2] + _log("[YOLO]  Daily pattern detection done. rc=1 sec=30"), ("yolo", "failed")),
        (REPORT_RUNNING + _log("[REPORT] Done. rc=1 sec=60"), ("report", "failed")),
        (REPORT_RUNNING + _log("[REPORT] Done. rc=0 sec=60"), ("report", "ok")),
        (REPORT_FAILED, ("daily_update", "failed")),
        (COMPLETED, ("daily_update", "ok")),
        (REPORT_RUNNING + _log("[REPORT] Done. rc=0 sec=60", "[WARN]  daily_update.sh mode=full ingest=2 yolo=0 report=0 (partial failures, report OK)"), ("daily_update", "partial")),
        (_log("[WEEKLY] Weekly YOLO failed rc=1"), ("weekly_yolo", "failed")),
    ],
)
def test_last_completed_event_reads_the_exit_code(tail, expected) -> None:
    event = _infer_last_completed_event_from_tail(tail)

    assert (None if event is None else (event["stage"], event["status"])) == expected


@pytest.mark.parametrize(
    ("mode", "returncode", "tail", "expected"),
    [
        ("full", 0, COMPLETED, (True, True, True)),
        # Exit 0 means the report passed; ingest and YOLO still need their own rc=0.
        ("full", 0, INGEST_FAILED + _log("[YOLO]  Daily pattern detection done. rc=0 sec=300",
                                         "[REPORT] Done. rc=0 sec=60"), (False, True, True)),
        ("full", 0, [], (False, False, True)),
        ("yolo", 0, _log("[START] daily_update.sh mode=yolo",
                         "[YOLO]  Daily pattern detection done. rc=0 sec=300"), (False, True, True)),
        ("report", 0, [], (False, False, True)),
        ("yolo", 1, REPORT_FAILED, (False, False, False)),
        ("full", 1, REPORT_FAILED, (True, True, False)),
        ("full", 1, INGEST_RUNNING + _log("[INGEST] done rc=2 sec=10"), (False, False, False)),
        # Markers from an earlier run in the tail do not count for this run.
        ("full", 1, COMPLETED + _log("[START] daily_update.sh mode=full", "[INGEST] done rc=3"), (False, False, False)),
    ],
)
def test_stage_results_from_one_parser(mode, returncode, tail, expected) -> None:
    results = stage_results(mode, returncode, lambda: tail)

    assert (results.ingest_ok, results.yolo_ok, results.report_ok) == expected


@pytest.mark.parametrize(
    ("mode", "returncode", "stages", "expected"),
    [
        ("full", 0, StageResults(True, True, True), None),
        # daily_update.sh exits 0 when only YOLO failed; the error still names it.
        ("full", 0, StageResults(True, False, True), "daily_update.sh rc=0; failed stages: yolo"),
        # The six incident after the YOLO threshold: YOLO fails, the report then refuses.
        ("full", 1, StageResults(True, False, False), "daily_update.sh rc=1; failed stages: yolo, report"),
        ("yolo", 1, StageResults(False, False, False), "daily_update.sh rc=1; failed stages: yolo, report"),
        ("report", 2, StageResults(False, False, False), "daily_update.sh rc=2; failed stages: report"),
    ],
)
def test_run_error_message_names_failed_stages(mode, returncode, stages, expected) -> None:
    assert run_error_message(mode, returncode, stages) == expected


@pytest.mark.parametrize(
    ("mode", "returncode", "stages", "expected"),
    [
        ("full", 0, StageResults(True, True, True), "ok"),
        ("full", 0, StageResults(True, False, True), "partial"),
        ("full", 0, StageResults(False, True, True), "partial"),
        ("report", 0, StageResults(False, False, True), "ok"),
        ("full", 1, StageResults(True, False, False), "failed"),
    ],
)
def test_run_status_is_partial_when_a_stage_failed_but_the_report_passed(mode, returncode, stages, expected) -> None:
    assert run_status(mode, returncode, stages) == expected


# ---------------------------------------------------------------------------
# Status snapshot and import graph
# ---------------------------------------------------------------------------

def test_status_snapshot_reports_a_requested_run_as_queued(recovery_env: Path) -> None:
    db = _db(recovery_env)
    assert pipeline_status_snapshot()["stage"] == "idle"

    request = pipeline_runs.request_run(FakeScheduler(), mode="full", source="admin", db_path=db, now=NOW)
    queued = pipeline_status_snapshot()

    assert (queued["active"], queued["stage"]) == (True, "queued")
    assert queued["active_pipeline_run"]["run_id"] == request.run_id


def test_pipeline_status_does_not_import_the_scheduler() -> None:
    probe = (
        "import sys; import trader_koo.backend.services.pipeline_runs; "
        "print('trader_koo.backend.services.scheduler' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "False"

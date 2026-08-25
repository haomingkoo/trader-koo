"""Focused contracts for the authenticated manual pipeline trigger."""
from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers.admin import router as admin_router
from trader_koo.backend.services.pipeline import ensure_pipeline_runs_schema
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator

pipeline_router = importlib.import_module("trader_koo.backend.routers.admin.pipeline")


class RecordingScheduler:
    def __init__(self, before_add=None) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.before_add = before_add

    def add_job(self, func, **kwargs) -> None:
        if self.before_add is not None:
            self.before_add(kwargs)
        self.jobs.append({"func": func, **kwargs})


def _app(scheduler: RecordingScheduler) -> FastAPI:
    app = FastAPI()
    app.state.admin_authenticator = AdminAuthenticator(
        AdminAuthConfig(api_key="x" * 32)
    )
    app.state.scheduler = scheduler
    app.include_router(admin_router)
    return app


def _db(tmp_path: Path) -> Path:
    db_path = tmp_path / "trigger.db"
    conn = sqlite3.connect(db_path)
    ensure_pipeline_runs_schema(conn)
    conn.close()
    return db_path


def _configure_trigger(monkeypatch, db_path: Path) -> None:
    monkeypatch.setattr(pipeline_router, "DB_PATH", db_path)
    monkeypatch.setattr(
        pipeline_router,
        "reconcile_stale_running_runs",
        lambda: {"reconciled": 0},
    )
    monkeypatch.setattr(
        pipeline_router,
        "pipeline_status_snapshot",
        lambda **kwargs: {"active": False, "stage": "idle"},
    )


def test_trigger_authentication_is_unchanged(tmp_path: Path, monkeypatch) -> None:
    db_path = _db(tmp_path)
    scheduler = RecordingScheduler()
    _configure_trigger(monkeypatch, db_path)

    response = TestClient(_app(scheduler)).post(
        "/api/admin/trigger-update?mode=report"
    )

    assert response.status_code == 401
    assert scheduler.jobs == []
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone() == (0,)
    conn.close()


def test_trigger_persists_reservation_before_scheduling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = _db(tmp_path)

    def assert_reserved_before_add(kwargs: dict[str, Any]) -> None:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT status, stage FROM pipeline_runs WHERE run_id = ?",
            (kwargs["kwargs"]["run_id"],),
        ).fetchone()
        conn.close()
        assert row == ("queued", "queued")

    scheduler = RecordingScheduler(before_add=assert_reserved_before_add)
    _configure_trigger(monkeypatch, db_path)

    response = TestClient(_app(scheduler)).post(
        "/api/admin/trigger-update?mode=report",
        headers={"X-API-Key": "x" * 32},
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["status"] == "queued"
    assert row["stage"] == "queued"
    assert scheduler.jobs[0]["kwargs"]["run_id"] == run_id


def test_duplicate_trigger_returns_conflict_and_does_not_queue_twice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = _db(tmp_path)
    scheduler = RecordingScheduler()
    _configure_trigger(monkeypatch, db_path)
    client = TestClient(_app(scheduler))
    headers = {"X-API-Key": "x" * 32}

    first = client.post("/api/admin/trigger-update?mode=report", headers=headers)
    second = client.post("/api/admin/trigger-update?mode=full", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 409
    assert "already queued or running" in second.json()["detail"]
    assert len(scheduler.jobs) == 1
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone() == (1,)
    conn.close()


def test_scheduler_rejection_closes_the_reservation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = _db(tmp_path)

    class RejectingScheduler(RecordingScheduler):
        def add_job(self, func, **kwargs) -> None:
            raise RuntimeError("scheduler unavailable")

    _configure_trigger(monkeypatch, db_path)
    response = TestClient(_app(RejectingScheduler())).post(
        "/api/admin/trigger-update?mode=report",
        headers={"X-API-Key": "x" * 32},
    )

    assert response.status_code == 503
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT status, stage, error_message FROM pipeline_runs"
    ).fetchone()
    conn.close()
    assert row == ("failed", "done", "scheduler_enqueue_failed")

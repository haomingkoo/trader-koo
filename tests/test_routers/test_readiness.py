"""Readiness probe: scheduler, daily_update schedule, and read-only DB checks."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers import readiness
from trader_koo.backend.routers.system import router as system_router

READY_KEYS = {
    "ok",
    "scheduler_running",
    "daily_update_scheduled",
    "db_readable",
    "db_integrity_ok",
}


class FakeScheduler:
    def __init__(self, *, running: bool = True, next_run: dt.datetime | None = None, has_job: bool = True):
        self.running = running
        self._job = SimpleNamespace(next_run_time=next_run) if has_job else None

    def get_job(self, job_id: str):
        return self._job if job_id == "daily_update" else None


def _real_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, close REAL)")
    conn.commit()
    conn.close()
    return path


def _next_run() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)


@pytest.fixture(autouse=True)
def _clear_integrity_cache():
    readiness._INTEGRITY_CACHE.clear()
    yield
    readiness._INTEGRITY_CACHE.clear()


def _client(db_path: Path, scheduler=None, maintenance_mode: bool = False):
    app = FastAPI()
    app.include_router(system_router)
    app.state.maintenance_mode = maintenance_mode
    if scheduler is not None:
        app.state.scheduler = scheduler
    return app, TestClient(app, raise_server_exceptions=False)


def _get_ready(db_path: Path, scheduler=None, maintenance_mode: bool = False):
    _, client = _client(db_path, scheduler, maintenance_mode)
    with patch("trader_koo.backend.routers.readiness.DB_PATH", db_path):
        return client.get("/api/ready")


def test_ready_ok_when_scheduler_and_db_are_healthy(tmp_path):
    response = _get_ready(_real_db(tmp_path / "db.sqlite"), FakeScheduler(next_run=_next_run()))

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "scheduler_running": True,
        "daily_update_scheduled": True,
        "db_readable": True,
        "db_integrity_ok": True,
    }


def test_ready_503_when_scheduler_not_started(tmp_path):
    from trader_koo.backend.services.scheduler import create_scheduler

    response = _get_ready(_real_db(tmp_path / "db.sqlite"), create_scheduler())

    assert response.status_code == 503
    body = response.json()
    assert body["ok"] is False
    assert body["scheduler_running"] is False


def test_ready_503_when_scheduler_missing(tmp_path):
    response = _get_ready(_real_db(tmp_path / "db.sqlite"), scheduler=None)

    assert response.status_code == 503
    assert response.json()["scheduler_running"] is False


def test_ready_503_when_daily_update_paused(tmp_path):
    response = _get_ready(_real_db(tmp_path / "db.sqlite"), FakeScheduler(next_run=None))

    assert response.status_code == 503
    body = response.json()
    assert body["scheduler_running"] is True
    assert body["daily_update_scheduled"] is False


def test_ready_503_when_daily_update_job_absent(tmp_path):
    response = _get_ready(_real_db(tmp_path / "db.sqlite"), FakeScheduler(has_job=False))

    assert response.status_code == 503
    assert response.json()["daily_update_scheduled"] is False


def test_ready_503_in_maintenance_mode_even_without_the_interlock(tmp_path):
    """Defence in depth: the deployed app's interlock answers first (see below)."""
    response = _get_ready(
        _real_db(tmp_path / "db.sqlite"), FakeScheduler(next_run=_next_run()), maintenance_mode=True
    )

    assert response.status_code == 503
    assert response.json()["ok"] is False


def test_ready_503_when_db_missing_and_does_not_create_it(tmp_path):
    db_path = tmp_path / "missing.db"

    response = _get_ready(db_path, FakeScheduler(next_run=_next_run()))

    assert response.status_code == 503
    body = response.json()
    assert body["db_readable"] is False
    assert body["db_integrity_ok"] is False
    assert not db_path.exists()


def test_ready_503_when_db_is_not_sqlite(tmp_path):
    db_path = tmp_path / "garbage.db"
    db_path.write_bytes(b"not a database" * 512)

    response = _get_ready(db_path, FakeScheduler(next_run=_next_run()))

    assert response.status_code == 503
    assert response.json()["db_readable"] is False


def test_ready_503_when_quick_check_reports_corruption(tmp_path):
    db_path = tmp_path / "corrupt.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA page_size = 4096")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("CREATE INDEX t_v ON t(v)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"value-{i:06d}",) for i in range(5000)])
    conn.commit()
    conn.close()
    data = bytearray(db_path.read_bytes())
    for offset in range(4096 * 3, 4096 * 3 + 2048):  # scribble over a b-tree page
        data[offset] = 0xFF
    db_path.write_bytes(bytes(data))

    response = _get_ready(db_path, FakeScheduler(next_run=_next_run()))

    assert response.status_code == 503
    body = response.json()
    assert body["db_readable"] is True
    assert body["db_integrity_ok"] is False


def test_ready_reuses_recent_integrity_result(tmp_path):
    db_path = _real_db(tmp_path / "db.sqlite")
    scheduler = FakeScheduler(next_run=_next_run())
    calls: list[Path] = []
    real = readiness._run_quick_check

    def counting(path: Path) -> bool:
        calls.append(path)
        return real(path)

    with patch.object(readiness, "_run_quick_check", counting):
        assert _get_ready(db_path, scheduler).status_code == 200
        assert _get_ready(db_path, scheduler).status_code == 200

    assert len(calls) == 1


def test_ready_exposes_only_booleans(tmp_path):
    response = _get_ready(_real_db(tmp_path / "db.sqlite"), FakeScheduler(next_run=_next_run()))
    body = response.json()

    assert set(body) == READY_KEYS
    assert all(isinstance(value, bool) for value in body.values())
    assert str(tmp_path) not in response.text


def test_health_stays_cheap_liveness(tmp_path):
    db_path = _real_db(tmp_path / "db.sqlite")
    with patch.object(readiness, "_run_quick_check", side_effect=AssertionError("deep check")):
        _, client = _client(db_path, scheduler=None)
        with patch("trader_koo.backend.routers.system.DB_PATH", db_path):
            health = client.get("/api/health")

    assert health.status_code == 200
    assert health.json()["ok"] is True


# -- the deployed app's middleware stack --------------------------------------

@pytest.fixture()
def real_app(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADER_KOO_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRADER_KOO_LOGOS_DIR", str(tmp_path / "logos"))
    monkeypatch.setenv("TRADER_KOO_MAINTENANCE_STATE_PATH", str(tmp_path / "maintenance.sqlite3"))
    from trader_koo.backend.main import app

    db_path = _real_db(tmp_path / "db.sqlite")
    monkeypatch.setattr(readiness, "DB_PATH", db_path)
    return db_path, TestClient(app, raise_server_exceptions=False)


def test_real_app_ready_is_exempt_from_rate_limiting(real_app):
    _, client = real_app

    response = client.get("/api/ready")

    assert response.status_code == 503  # no scheduler without the lifespan
    assert set(response.json()) == READY_KEYS
    assert "X-RateLimit-Limit" not in response.headers


def test_real_app_ready_is_503_from_maintenance_interlock(real_app):
    from trader_koo.backend.services.maintenance import request_maintenance

    db_path, client = real_app
    request_maintenance(db_path, run_id="maint-test", boot_id="boot", reason="test", timeout_sec=1)

    response = client.get("/api/ready")

    assert response.status_code == 503
    assert "ok" not in response.json()
    assert client.get("/api/health").json()["maintenance"] is True

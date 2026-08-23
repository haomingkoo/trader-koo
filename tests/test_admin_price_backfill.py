from __future__ import annotations

import subprocess
import sqlite3
import threading
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from trader_koo.backend.routers.admin import data as data_router
from trader_koo.scripts.update_market_db import build_arg_parser
from trader_koo.scripts.update_market_db import require_complete_dataset


def test_seed_history_route_runs_valid_full_refresh_and_reports_completion(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output, check):
        commands.append(command)
        build_arg_parser().parse_args(command[2:])
        started.set()
        assert release.wait(timeout=2)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(data_router.subprocess, "run", fake_run)
    monkeypatch.setattr(
        data_router,
        "_verify_seed_history_ingest",
        lambda started_at, tickers: {
            "ok": True,
            "run_id": "verified-run",
            "status": "ok",
            "tickers_failed": 0,
        },
    )
    monkeypatch.setattr(data_router, "_seed_history_thread", None)
    monkeypatch.setattr(data_router, "_seed_history_state", {"status": "idle"})

    app = FastAPI()

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        request.state.admin_identity = {"username": "test-admin"}
        return await call_next(request)

    app.include_router(data_router.router)
    with TestClient(app) as client:
        response = client.post(
            "/api/admin/seed-ticker-history",
            params={"tickers": "BKNG,SNDK", "start_date": "2020-01-01"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "running"
        assert started.wait(timeout=2)
        running = client.get("/api/admin/seed-ticker-history/status").json()
        assert running["status"] == "running"

        release.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            completed = client.get("/api/admin/seed-ticker-history/status").json()
            if completed["status"] != "running":
                break
            time.sleep(0.01)

    assert completed["status"] == "succeeded"
    assert completed["returncode"] == 0
    assert "--full-price-refresh" in commands[0]
    assert "--require-full-dataset" in commands[0]
    assert "--skip-price" not in commands[0]


def test_seed_history_process_failure_is_observable(monkeypatch):
    monkeypatch.setattr(
        data_router.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2),
    )
    monkeypatch.setattr(data_router, "_seed_history_state", {"status": "running"})

    data_router._run_seed_history(["python", "update_market_db.py"])

    assert data_router._seed_history_state["status"] == "failed"
    assert data_router._seed_history_state["returncode"] == 2
    assert data_router._seed_history_state["error"] == "process_exit_2"


def test_seed_history_zero_exit_partial_run_is_never_reported_succeeded(monkeypatch):
    monkeypatch.setattr(
        data_router.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
    )
    monkeypatch.setattr(
        data_router,
        "_verify_seed_history_ingest",
        lambda started_at, tickers: {
            "ok": False,
            "run_id": "partial-run",
            "status": "partial_failed",
            "tickers_failed": 1,
            "error": "ingest_run_partial_failed",
        },
    )
    monkeypatch.setattr(
        data_router,
        "_seed_history_state",
        {"status": "running", "started_at": "2026-08-22T00:00:00+00:00"},
    )

    data_router._run_seed_history(["python", "update_market_db.py"])

    assert data_router._seed_history_state["status"] == "failed"
    assert data_router._seed_history_state["returncode"] == 0
    assert data_router._seed_history_state["error"] == "ingest_run_partial_failed"
    assert data_router._seed_history_state["ingest_run"]["status"] == "partial_failed"


def test_required_full_dataset_treats_context_ticker_failure_as_fatal():
    with pytest.raises(RuntimeError, match=r"1/2 ticker\(s\) failed"):
        require_complete_dataset(
            {"^VIX": "provider unavailable"},
            ticker_count=2,
            max_passes=2,
            required=True,
        )


def test_seed_history_verifier_reads_partial_ingest_as_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "market.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE ingest_runs (
        run_id TEXT, started_ts TEXT, finished_ts TEXT, status TEXT,
        tickers_total INTEGER, tickers_ok INTEGER, tickers_failed INTEGER,
        error_message TEXT, args_json TEXT)"""
    )
    conn.execute(
        "INSERT INTO ingest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "partial-run",
            "2026-08-22T00:00:01Z",
            "2026-08-22T00:01:00Z",
            "partial_failed",
            2,
            1,
            1,
            "one failed",
            '{"tickers":"BKNG,SNDK"}',
        ),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(data_router, "DB_PATH", db_path)

    result = data_router._verify_seed_history_ingest(
        "2026-08-22T00:00:00Z", ["BKNG", "SNDK"]
    )

    assert result["ok"] is False
    assert result["run_id"] == "partial-run"
    assert result["error"] == "ingest_run_partial_failed"

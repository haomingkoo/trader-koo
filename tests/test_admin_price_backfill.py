from __future__ import annotations

import subprocess
import threading
import time

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from trader_koo.backend.routers.admin import data as data_router
from trader_koo.scripts.update_market_db import build_arg_parser


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

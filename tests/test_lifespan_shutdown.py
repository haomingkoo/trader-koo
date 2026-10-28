"""Lifespan shutdown stops writer children before the scheduler."""
from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

import pytest

from trader_koo.backend.services import child_processes


def test_shutdown_terminates_children_before_scheduler(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    class FakeScheduler:
        def shutdown(self, wait: bool) -> None:
            calls.append(f"scheduler.shutdown(wait={wait})")

    def fake_terminate(grace_sec: float = 0.0) -> dict[str, int]:
        calls.append("terminate_running_children")
        return {"signalled": 1, "killed": 0}

    monkeypatch.setattr(child_processes, "terminate_running_children", fake_terminate)

    child_processes.shutdown_scheduler(FakeScheduler())

    assert calls == ["terminate_running_children", "scheduler.shutdown(wait=True)"]


def test_lifespan_exit_stops_writer_child_before_scheduler_shutdown(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TRADER_KOO_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRADER_KOO_LOGOS_DIR", str(tmp_path / "logos"))
    monkeypatch.delenv("TRADER_KOO_MAINTENANCE_STATE_PATH", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import trader_koo.backend.main as main
    import trader_koo.backend.services.database as database

    events: list[tuple[str, int | None]] = []
    child: dict[str, object] = {}

    class StubScheduler:
        def start(self) -> None:
            def run() -> None:
                child["result"] = child_processes.run_tracked_child(["sleep", "30"])

            child["thread"] = threading.Thread(target=run, daemon=True)
            child["thread"].start()
            deadline = time.monotonic() + 5
            while not child_processes._RUNNING_CHILDREN:
                assert time.monotonic() < deadline, "child did not register"
                time.sleep(0.02)
            child["proc"] = next(iter(child_processes._RUNNING_CHILDREN))

        def shutdown(self, wait: bool) -> None:
            events.append(("scheduler.shutdown", child["proc"].poll()))

    monkeypatch.setattr(main, "DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr(main, "create_scheduler", StubScheduler)
    monkeypatch.setattr(main, "validate_config", lambda: None)
    monkeypatch.setattr(main, "start_crypto_feed", lambda **_: None)
    monkeypatch.setattr(main, "stop_crypto_feed", lambda: None)
    monkeypatch.setattr(main, "stop_equity_feed", lambda: None)
    monkeypatch.setattr(database, "get_conn", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no db")))
    child_processes._CHILD_SHUTDOWN.clear()
    try:
        with TestClient(FastAPI(lifespan=main.lifespan)):
            assert child["proc"].poll() is None
    finally:
        child_processes.terminate_running_children(grace_sec=1.0)
        child_processes._CHILD_SHUTDOWN.clear()

    assert events == [("scheduler.shutdown", -signal.SIGTERM)]
    child["thread"].join(timeout=5)
    assert child["result"].returncode == -signal.SIGTERM


def test_child_grace_fits_inside_configured_railway_drain_window():
    import re

    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 local venvs; CI runs 3.11
        import tomli as tomllib

    repo = Path(__file__).resolve().parents[1]
    draining = tomllib.loads((repo / "railway.toml").read_text())["deploy"].get("drainingSeconds")
    if draining is None:
        pytest.skip(
            "railway.toml sets no deploy.drainingSeconds, so Railway may SIGKILL "
            "before lifespan shutdown stops writer children"
        )
    match = re.search(r"--timeout-graceful-shutdown (\d+)", (repo / "start.sh").read_text())
    assert match, "start.sh must cap uvicorn's graceful shutdown"
    assert int(match.group(1)) + child_processes.CHILD_TERMINATE_GRACE_SEC < draining

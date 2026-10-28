"""Lifespan shutdown stops writer children before the scheduler."""
from __future__ import annotations

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


def test_lifespan_uses_child_aware_shutdown(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRADER_KOO_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRADER_KOO_LOGOS_DIR", str(tmp_path / "logos"))
    import inspect

    import trader_koo.backend.main as main

    source = inspect.getsource(main.lifespan)
    assert "shutdown_scheduler(_scheduler)" in source
    assert "_scheduler.shutdown(" not in source


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

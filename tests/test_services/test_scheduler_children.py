"""Shutdown handling for scheduler-owned writer subprocesses."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from trader_koo.backend.services import child_processes as children
from trader_koo.backend.services import scheduler
from trader_koo.backend.services.pipeline import (
    ensure_pipeline_runs_schema,
    reserve_pipeline_run,
)


@pytest.fixture(autouse=True)
def _reset_child_registry():
    children._CHILD_SHUTDOWN.clear()
    yield
    children.terminate_running_children(grace_sec=1.0)
    children._CHILD_SHUTDOWN.clear()


def _start_in_thread(cmd: list[str], **kwargs) -> tuple[threading.Thread, dict]:
    out: dict = {}

    def target() -> None:
        out["result"] = children.run_tracked_child(cmd, **kwargs)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, out


def _wait_for_children(count: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with children._CHILD_LOCK:
            if len(children._RUNNING_CHILDREN) >= count:
                return
        time.sleep(0.02)
    raise AssertionError("child did not register")


def test_tracked_child_returns_exit_code_and_unregisters():
    result = children.run_tracked_child([sys.executable, "-c", "raise SystemExit(3)"])

    assert result.returncode == 3
    assert children._RUNNING_CHILDREN == set()


def test_terminate_sends_sigterm_and_waits_for_clean_exit(tmp_path: Path):
    marker = tmp_path / "terminated.txt"
    code = (
        "import signal, sys, time, pathlib\n"
        f"p = pathlib.Path({str(marker)!r})\n"
        "def handle(signum, frame):\n"
        "    p.write_text('sigterm')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, handle)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    thread, out = _start_in_thread([sys.executable, "-c", code], stdout=subprocess.DEVNULL)
    _wait_for_children(1)
    time.sleep(0.3)  # let the child install its handler

    summary = children.terminate_running_children(grace_sec=5.0)
    thread.join(timeout=5.0)

    assert summary == {"signalled": 1, "killed": 0}
    assert marker.read_text() == "sigterm"
    assert out["result"].returncode == 0


def test_terminate_kills_child_that_ignores_sigterm():
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(60)\n"
    )
    thread, out = _start_in_thread([sys.executable, "-c", code])
    _wait_for_children(1)
    time.sleep(0.3)

    started = time.monotonic()
    summary = children.terminate_running_children(grace_sec=0.5)
    elapsed = time.monotonic() - started
    thread.join(timeout=5.0)

    assert summary == {"signalled": 1, "killed": 1}
    assert elapsed < 5.0
    assert out["result"].returncode < 0


def test_terminate_reaches_grandchildren_in_the_process_group(tmp_path: Path):
    pid_file = tmp_path / "grandchild.pid"
    grandchild_py = tmp_path / "grandchild.py"
    grandchild_py.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    script = tmp_path / "parent.sh"
    script.write_text(f"#!/usr/bin/env bash\n'{sys.executable}' '{grandchild_py}'\n")
    thread, _ = _start_in_thread(["bash", str(script)])
    _wait_for_children(1)
    deadline = time.monotonic() + 5.0
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    grandchild = int(pid_file.read_text())

    children.terminate_running_children(grace_sec=5.0)
    thread.join(timeout=5.0)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(grandchild, 9)
        pytest.fail("grandchild survived scheduler shutdown")


def test_no_new_child_starts_after_shutdown_begins():
    children.terminate_running_children(grace_sec=0.1)

    with pytest.raises(children.ChildShutdownInProgress):
        children.run_tracked_child([sys.executable, "-c", "pass"])
    assert children.run_tracked_child([sys.executable, "-c", "pass"], interruptible=True) is None


def test_interruptible_child_stopped_by_shutdown_returns_none():
    code = "import time; time.sleep(60)"
    thread, out = _start_in_thread([sys.executable, "-c", code], interruptible=True)
    _wait_for_children(1)

    children.terminate_running_children(grace_sec=5.0)
    thread.join(timeout=5.0)

    assert out["result"] is None


def test_interruptible_child_failure_without_shutdown_is_returned():
    result = children.run_tracked_child([sys.executable, "-c", "raise SystemExit(4)"], interruptible=True)

    assert result.returncode == 4


def test_tracked_child_keeps_writer_lease_fd(tmp_path: Path, monkeypatch):
    lease = (tmp_path / "lease").open("w")
    monkeypatch.setenv("TRADER_KOO_WRITER_LEASE_FD", str(lease.fileno()))
    try:
        from trader_koo.backend.services.maintenance import inherited_writer_lease_fds

        result = children.run_tracked_child(
            [sys.executable, "-c", f"import os; os.fstat({lease.fileno()})"],
            pass_fds=inherited_writer_lease_fds(),
        )
    finally:
        lease.close()

    assert result.returncode == 0


def test_daily_update_interrupted_by_shutdown_stays_resumable(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "scheduler.db"
    conn = sqlite3.connect(db_path)
    ensure_pipeline_runs_schema(conn)
    conn.close()
    reserve_pipeline_run(run_id="pipe_shutdown", mode="report", source="admin", db_path=db_path)

    def interrupted_child(cmd, **kwargs):
        assert kwargs["interruptible"] is True
        return None

    monkeypatch.setattr(scheduler, "DB_PATH", db_path)
    monkeypatch.setattr(scheduler, "RUN_LOG_PATH", tmp_path / "run.log")
    monkeypatch.setattr(scheduler, "_run_tracked_child", interrupted_child)

    scheduler._run_daily_update_unlocked("report", "admin", pipeline_run_id="pipe_shutdown")

    verify = sqlite3.connect(db_path)
    row = verify.execute(
        "SELECT status FROM pipeline_runs WHERE run_id = 'pipe_shutdown'"
    ).fetchone()
    verify.close()
    # Left running so startup resume treats it as interrupted, as after a kill.
    assert row == ("running",)


def test_daily_update_failure_without_shutdown_is_recorded(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "scheduler.db"
    conn = sqlite3.connect(db_path)
    ensure_pipeline_runs_schema(conn)
    conn.close()
    reserve_pipeline_run(run_id="pipe_fail", mode="report", source="admin", db_path=db_path)

    monkeypatch.setattr(scheduler, "DB_PATH", db_path)
    monkeypatch.setattr(scheduler, "RUN_LOG_PATH", tmp_path / "run.log")
    monkeypatch.setattr(
        scheduler, "_run_tracked_child",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1),
    )

    scheduler._run_daily_update_unlocked("report", "admin", pipeline_run_id="pipe_fail")

    verify = sqlite3.connect(db_path)
    row = verify.execute("SELECT status FROM pipeline_runs WHERE run_id = 'pipe_fail'").fetchone()
    verify.close()
    assert row == ("failed",)

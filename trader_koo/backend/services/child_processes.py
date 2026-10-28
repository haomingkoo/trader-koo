"""Track scheduler writer subprocesses so shutdown can stop them in time.

Shutdown timing and its Railway dependency are in DEPLOYMENT.md.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from typing import Any

LOG = logging.getLogger("trader_koo.services.child_processes")

CHILD_TERMINATE_GRACE_SEC = 20.0
_CHILD_LOCK = threading.Lock()
_RUNNING_CHILDREN: set[subprocess.Popen] = set()
_CHILD_SHUTDOWN = threading.Event()


class ChildShutdownInProgress(RuntimeError):
    """Raised when a job tries to start a writer child after shutdown began."""


def run_tracked_child(
    cmd: list[str], *, interruptible: bool = False, **kwargs: Any
) -> subprocess.CompletedProcess | None:
    """Run *cmd* like ``subprocess.run`` but registered for shutdown.

    The child leads its own process group so shutdown signals the whole tree
    (``daily_update.sh`` runs Python grandchildren). ``pass_fds`` is forwarded
    unchanged, so the writer lease is inherited as before.

    With ``interruptible=True`` the caller gets ``None`` instead of an error or
    a failed result when shutdown refused or stopped the child.
    """
    with _CHILD_LOCK:
        if _CHILD_SHUTDOWN.is_set():
            if interruptible:
                LOG.warning("Shutdown in progress; not starting %s", cmd[:2])
                return None
            raise ChildShutdownInProgress(f"shutdown in progress; not starting {cmd[:2]}")
        proc = subprocess.Popen(cmd, start_new_session=True, **kwargs)
        _RUNNING_CHILDREN.add(proc)
    try:
        returncode = proc.wait()
    except BaseException:
        _signal_group(proc, signal.SIGKILL)
        proc.wait()
        raise
    finally:
        with _CHILD_LOCK:
            _RUNNING_CHILDREN.discard(proc)
    if interruptible and returncode != 0 and _CHILD_SHUTDOWN.is_set():
        LOG.warning("Shutdown interrupted %s (rc=%d)", cmd[:2], returncode)
        return None
    return subprocess.CompletedProcess(cmd, returncode)


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def terminate_running_children(grace_sec: float = CHILD_TERMINATE_GRACE_SEC) -> dict[str, int]:
    """Refuse new children, SIGTERM running ones, SIGKILL stragglers.

    Returns how many children were signalled and how many needed SIGKILL.
    """
    with _CHILD_LOCK:
        _CHILD_SHUTDOWN.set()
        children = list(_RUNNING_CHILDREN)
    for proc in children:
        if proc.poll() is None:
            LOG.warning("Shutdown: sending SIGTERM to writer child pid=%s", proc.pid)
            _signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + grace_sec
    killed = 0
    for proc in children:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            LOG.error(
                "Shutdown: writer child pid=%s ignored SIGTERM for %.0fs; sending SIGKILL",
                proc.pid,
                grace_sec,
            )
            _signal_group(proc, signal.SIGKILL)
            proc.wait()
            killed += 1
    return {"signalled": len(children), "killed": killed}


def shutdown_scheduler(scheduler: Any) -> None:
    """Stop writer children within the drain window, then the scheduler.

    ``shutdown(wait=True)`` alone would block on a child running without a
    timeout until Railway SIGKILLs the container mid-write.
    """
    summary = terminate_running_children()
    if summary["signalled"]:
        LOG.warning(
            "Shutdown: stopped %d writer child(ren), %d needed SIGKILL",
            summary["signalled"],
            summary["killed"],
        )
    scheduler.shutdown(wait=True)

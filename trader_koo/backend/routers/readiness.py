"""Readiness probe: GET /api/ready.

``/api/health`` stays the cheap liveness probe Railway gates deploys on. This
probe returns 503 unless the scheduler is running, ``daily_update`` has a next
run time, and the DB reads its schema and answers ``PRAGMA quick_check(1)`` over a
read-only connection. It exposes only booleans. The system router includes it.

In the deployed app the maintenance interlock answers 503 for this path before
the route runs, and the rate limiter exempts it like ``/api/health`` so a
poller sees 503 rather than 429.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from trader_koo.backend.services.database import DB_PATH
from trader_koo.security.endpoint_validator import sanitize_public_response

LOG = logging.getLogger("trader_koo.routers.readiness")

router = APIRouter()

# PRAGMA quick_check scans the whole file, so one result serves every probe
# for this long instead of letting public traffic drive repeated full scans.
INTEGRITY_CHECK_TTL_SEC = 300.0
DB_BUSY_TIMEOUT_SEC = 5.0
HTTP_READY = 200
HTTP_NOT_READY = 503
_INTEGRITY_LOCK = threading.Lock()
_INTEGRITY_CACHE: dict[str, tuple[float, bool]] = {}


def _open_db_read_only(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=DB_BUSY_TIMEOUT_SEC
    )


def _db_readable(db_path: Path) -> bool:
    try:
        conn = _open_db_read_only(db_path)
        try:
            # SELECT 1 can succeed without reading the file; the schema read cannot.
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            return True
        finally:
            conn.close()
    except sqlite3.Error as exc:
        LOG.warning("Readiness: DB is not readable: %s", exc)
        return False


def _run_quick_check(db_path: Path) -> bool:
    try:
        conn = _open_db_read_only(db_path)
        try:
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        LOG.warning("Readiness: quick_check could not run: %s", exc)
        return False
    if row != ("ok",):
        LOG.error("Readiness: quick_check failed: %s", row)
        return False
    return True


def _db_integrity_ok(db_path: Path) -> bool:
    key = str(db_path)
    with _INTEGRITY_LOCK:
        cached = _INTEGRITY_CACHE.get(key)
        if cached is not None and time.monotonic() - cached[0] < INTEGRITY_CHECK_TTL_SEC:
            return cached[1]
        result = _run_quick_check(db_path)
        _INTEGRITY_CACHE[key] = (time.monotonic(), result)
        return result


@router.get("/api/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness: scheduler running, daily_update scheduled, DB readable and intact."""
    state = request.app.state
    # Unreachable behind MaintenanceInterlockMiddleware; kept so the route
    # fails closed if it is ever mounted without the interlock.
    maintenance_mode = bool(getattr(state, "maintenance_mode", False))
    scheduler = getattr(state, "scheduler", None)
    scheduler_running = bool(getattr(scheduler, "running", False))
    daily_update_scheduled = False
    if scheduler_running:
        job = scheduler.get_job("daily_update")
        daily_update_scheduled = job is not None and job.next_run_time is not None
    db_readable = DB_PATH.is_file() and _db_readable(DB_PATH)
    db_integrity_ok = db_readable and _db_integrity_ok(DB_PATH)
    ok = (
        not maintenance_mode
        and scheduler_running
        and daily_update_scheduled
        and db_readable
        and db_integrity_ok
    )
    payload = {
        "ok": ok,
        "scheduler_running": scheduler_running,
        "daily_update_scheduled": daily_update_scheduled,
        "db_readable": db_readable,
        "db_integrity_ok": db_integrity_ok,
    }
    return JSONResponse(
        sanitize_public_response(payload), status_code=HTTP_READY if ok else HTTP_NOT_READY
    )

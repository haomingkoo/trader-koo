"""Pipeline trigger, status, force-cancel, logs, YOLO seed/status/events."""
from __future__ import annotations

import datetime as dt
import os
import secrets
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from trader_koo.backend.services.database import DB_PATH, get_conn, table_exists
from trader_koo.backend.services.market_data import get_yolo_status
from trader_koo.backend.services.pipeline import (
    pipeline_status_snapshot,
    post_ingest_resume_candidate,
    reconcile_stale_running_runs,
)
from trader_koo.backend.services.report_loader import (
    _tail_text_file,
    daily_report_response,
)
from trader_koo.backend.services.scheduler import _run_daily_update
from trader_koo.middleware.auth import require_admin_auth
from trader_koo.scripts.generate_daily_report import (
    _build_regime_context as _report_build_regime_context,
)

from trader_koo.backend.routers.admin._shared import (
    LOG,
    LOG_PATHS,
    PROJECT_DIR,
    REPORT_DIR,
    RUN_LOG_PATH,
    LOG_DIR,
    _normalize_update_mode,
    _yolo_seed_thread,
)
import trader_koo.backend.routers.admin._shared as _shared

router = APIRouter(tags=["admin", "admin-pipeline"])


@router.post("/api/admin/trigger-update")
@require_admin_auth
def trigger_update(
    request: Request,
    mode: str = Query(default="full"),
) -> dict[str, Any]:
    """Trigger daily_update.sh immediately."""
    mode_norm = _normalize_update_mode(mode)
    if mode_norm is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid mode. Use one of: full, yolo, report "
                "(aliases: yolo_report, yolo+report, report_only)."
            ),
        )
    reconcile = reconcile_stale_running_runs()
    pipeline = pipeline_status_snapshot(log_lines=120)
    if pipeline["active"]:
        stage = pipeline.get("stage", "unknown")
        return {
            "ok": False,
            "message": (
                f"daily_update already running "
                f"(stage={stage}, requested_mode={mode_norm})"
            ),
            "stage": stage,
            "requested_mode": mode_norm,
            "latest_run": pipeline.get("latest_run"),
            "run_log_path": pipeline.get("run_log_path"),
            "reconciled_stale_runs": reconcile.get("reconciled", 0),
        }
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=503,
            detail="Scheduler unavailable. Restart the app before triggering updates.",
        )
    now_utc = dt.datetime.now(dt.timezone.utc)
    manual_job_id = (
        f"manual_daily_update_{now_utc.strftime('%Y%m%dT%H%M%S')}"
        f"_{secrets.token_hex(3)}"
    )
    scheduler.add_job(
        _run_daily_update,
        trigger="date",
        run_date=now_utc,
        id=manual_job_id,
        kwargs={"mode": mode_norm, "source": "admin"},
    )
    mode_message = {
        "full": "full pipeline (ingest + yolo + report)",
        "yolo": "yolo + report (ingest skipped)",
        "report": "report only (ingest + yolo skipped)",
    }
    return {
        "ok": True,
        "message": (
            f"daily_update triggered ({mode_message.get(mode_norm, mode_norm)}) "
            "-- check /data/logs/cron_daily.log"
        ),
        "stage": "queued",
        "mode": mode_norm,
        "job_id": manual_job_id,
        "run_log_path": str(RUN_LOG_PATH),
        "reconciled_stale_runs": reconcile.get("reconciled", 0),
    }


@router.post("/api/admin/force-cancel-run")
@require_admin_auth
def force_cancel_run(request: Request) -> dict[str, Any]:
    """Force-cancel all running ingest runs by marking them as failed.

    Use when a run is stuck and the automatic stale-detection timeout
    (75 min) hasn't expired yet.
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT run_id, started_ts FROM ingest_runs WHERE status = 'running'"
        ).fetchall()
        if not rows:
            return {
                "ok": True,
                "message": "No running ingest runs found",
                "cancelled": 0,
            }
        now_iso = (
            dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        cancelled = []
        for row in rows:
            run_id = row["run_id"]
            conn.execute(
                """
                UPDATE ingest_runs
                SET status = 'failed',
                    finished_ts = ?,
                    error_message = 'force-cancelled via admin API'
                WHERE run_id = ?
                """,
                (now_iso, run_id),
            )
            cancelled.append(run_id)
        conn.commit()
        LOG.warning(
            "Force-cancelled %d running ingest run(s): %s",
            len(cancelled),
            cancelled,
        )
        return {
            "ok": True,
            "message": f"Force-cancelled {len(cancelled)} running run(s)",
            "cancelled": len(cancelled),
            "run_ids": cancelled,
        }
    finally:
        conn.close()


@router.get("/api/admin/pipeline-status")
@require_admin_auth
def pipeline_status_endpoint(
    log_lines: int = Query(default=120, ge=20, le=1000),
) -> dict[str, Any]:
    """Return current pipeline phase inferred from run logs + latest ingest run."""
    snap = pipeline_status_snapshot(log_lines=log_lines)
    resume_candidate = post_ingest_resume_candidate(
        latest_run=snap.get("latest_run"),
        pipeline_active=bool(snap.get("active")),
    )
    return {
        "ok": True,
        **snap,
        "post_ingest_resume": resume_candidate,
    }


@router.get("/api/admin/daily-report")
@require_admin_auth
def admin_daily_report(
    limit: int = Query(default=20, ge=1, le=200),
    include_markdown: bool = Query(default=False),
) -> dict[str, Any]:
    """Return latest generated daily report and recent report files."""
    return daily_report_response(
        report_dir=REPORT_DIR,
        get_conn_fn=get_conn,
        build_regime_context_fn=_report_build_regime_context,
        pipeline_status_fn=pipeline_status_snapshot,
        limit=limit,
        include_markdown=include_markdown,
        include_internal_paths=True,
        include_admin_log_hints=True,
    )


@router.get("/api/admin/logs")
@require_admin_auth
def admin_logs(
    name: str = Query(
        default="cron",
        pattern="^(cron|update_market_db|yolo|api)$",
    ),
    lines: int = Query(default=80, ge=1, le=800),
) -> dict[str, Any]:
    """Return log tail for one known service log file."""
    path = LOG_PATHS[name]
    return {
        "ok": path.exists(),
        "name": name,
        "path": str(path),
        "tail": _tail_text_file(path, lines=lines, max_bytes=256_000),
    }


@router.post("/api/admin/run-yolo-seed")
@require_admin_auth
def run_yolo_seed(timeframe: str = "both") -> dict[str, Any]:
    """Trigger full YOLO seed for all tickers in background."""
    timeframe_norm = str(timeframe or "both").strip().lower() or "both"
    if timeframe_norm not in {"daily", "weekly", "both"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid timeframe. Use one of: daily, weekly, both",
        )
    if _shared._yolo_seed_thread and _shared._yolo_seed_thread.is_alive():
        return {
            "ok": False,
            "message": "Seed already running -- wait for it to finish",
        }
    script = PROJECT_DIR / "scripts" / "run_yolo_patterns.py"
    cmd = [
        sys.executable,
        str(script),
        "--db-path",
        str(DB_PATH),
        "--timeframe",
        timeframe_norm,
        "--lookback-days",
        "180",
        "--weekly-lookback-days",
        "730",
        "--sleep",
        "0.05",
    ]

    def _run() -> None:
        subprocess.run(cmd, capture_output=False)

    _shared._yolo_seed_thread = threading.Thread(target=_run, daemon=True)
    _shared._yolo_seed_thread.start()
    return {
        "ok": True,
        "message": (
            f"YOLO seed started (timeframe={timeframe_norm}) "
            "-- tail /data/logs/yolo_patterns.log"
        ),
        "timeframe": timeframe_norm,
    }


@router.get("/api/admin/yolo-status")
@require_admin_auth
def yolo_status(
    log_lines: int = Query(default=40, ge=0, le=400),
) -> dict[str, Any]:
    """Return YOLO runner status + DB summary + recent log tail."""
    log_path = Path(
        os.getenv("TRADER_KOO_YOLO_LOG_PATH", "/data/logs/yolo_patterns.log")
    )
    conn = get_conn()
    try:
        db_status = get_yolo_status(conn)
    finally:
        conn.close()
    thread_alive = bool(
        _shared._yolo_seed_thread and _shared._yolo_seed_thread.is_alive()
    )
    return {
        "ok": True,
        "thread_running": thread_alive,
        "log_path": str(log_path),
        "log_tail": _tail_text_file(log_path, lines=log_lines),
        "db": db_status,
    }


@router.get("/api/admin/yolo-events")
@require_admin_auth
def yolo_events(
    limit: int = Query(default=200, ge=1, le=1000),
    run_id: str = Query(default=""),
    status: str = Query(default="", pattern="^(|ok|skipped|timeout|failed)$"),
) -> dict[str, Any]:
    """Return persisted per-ticker YOLO run events for diagnostics."""
    conn = get_conn()
    try:
        if not table_exists(conn, "yolo_run_events"):
            return {"ok": False, "events_table_exists": False, "rows": []}
        clauses: list[str] = []
        params: list[Any] = []
        if run_id.strip():
            clauses.append("run_id = ?")
            params.append(run_id.strip())
        if status.strip():
            clauses.append("status = ?")
            params.append(status.strip())
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT run_id, timeframe, ticker, status, reason, elapsed_sec,
                   bars, detections, as_of_date, created_ts
            FROM yolo_run_events
            {where_sql}
            ORDER BY created_ts DESC
            LIMIT ?
        """
        params.append(int(limit))
        rows = conn.execute(sql, tuple(params)).fetchall()
        latest_run_row = conn.execute(
            """
            SELECT run_id
            FROM yolo_run_events
            GROUP BY run_id
            ORDER BY MAX(created_ts) DESC
            LIMIT 1
            """
        ).fetchone()
        return {
            "ok": True,
            "events_table_exists": True,
            "latest_run_id": (
                latest_run_row["run_id"] if latest_run_row else None
            ),
            "count": len(rows),
            "rows": [dict(r) for r in rows],
        }
    finally:
        conn.close()

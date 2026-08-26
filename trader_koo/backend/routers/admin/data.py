"""Ticker seeding, database diagnostics, audit logs, storage cleanup."""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response

from trader_koo.audit import apply_retention_policy, get_audit_stats
from trader_koo.audit.api import (
    export_audit_logs,
    get_audit_summary,
    query_audit_logs,
)
from trader_koo.audit.export import export_logs_to_local
from trader_koo.backend.routers.admin._shared import (
    LOG_DIR,
    PROJECT_DIR,
    get_audit_logger,
)
from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.services.maintenance import inherited_writer_lease_fds
from trader_koo.scripts.cleanup_storage import run_cleanup

router = APIRouter(tags=["admin", "admin-data"])

_seed_history_lock = threading.Lock()
_seed_history_thread: threading.Thread | None = None
_seed_history_state: dict[str, Any] = {"status": "idle"}


def _seed_history_command(tickers: list[str], start_date: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "update_market_db.py"),
        "--tickers",
        ",".join(tickers),
        "--price-start",
        start_date,
        "--price-lookback-days",
        "0",
        "--full-price-refresh",
        "--require-full-dataset",
        "--db-path",
        str(DB_PATH),
        "--log-file",
        str(LOG_DIR / "seed_history.log"),
    ]


def _verify_seed_history_ingest(
    started_at: str | None,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Return the ingest run started by this job, failing closed if absent/partial."""
    try:
        job_start = dt.datetime.fromisoformat(str(started_at or "").replace("Z", "+00:00"))
        if job_start.tzinfo is None:
            job_start = job_start.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return {"ok": False, "error": "invalid_job_start_time"}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT run_id, started_ts, finished_ts, status,
                   tickers_total, tickers_ok, tickers_failed, error_message,
                   args_json
            FROM ingest_runs ORDER BY started_ts DESC LIMIT 20
            """
        ).fetchall()
    except Exception as exc:
        return {"ok": False, "error": f"ingest_run_not_observable:{exc}"}
    finally:
        if "conn" in locals():
            conn.close()
    for row in rows:
        try:
            run_start = dt.datetime.fromisoformat(
                str(row["started_ts"] or "").replace("Z", "+00:00")
            )
            if run_start.tzinfo is None:
                run_start = run_start.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if run_start < job_start:
            continue
        if tickers:
            try:
                args = json.loads(str(row["args_json"] or "{}"))
            except (TypeError, ValueError):
                continue
            requested = {
                item.strip().upper()
                for item in str(args.get("tickers") or "").split(",")
                if item.strip()
            }
            if requested != set(tickers):
                continue
        result = dict(row)
        result.pop("args_json", None)
        result["ok"] = (
            str(row["status"] or "") == "ok"
            and int(row["tickers_failed"] or 0) == 0
        )
        if not result["ok"]:
            result["error"] = f"ingest_run_{row['status'] or 'unknown'}"
        return result
    return {"ok": False, "error": "ingest_run_not_observed"}


def _run_seed_history(command: list[str]) -> None:
    global _seed_history_state
    try:
        lease_fds = inherited_writer_lease_fds()
        lease_kwargs = {"pass_fds": lease_fds} if lease_fds else {}
        completed = subprocess.run(
            command, capture_output=False, check=False, **lease_kwargs,
        )
        verification = (
            _verify_seed_history_ingest(
                _seed_history_state.get("started_at"),
                _seed_history_state.get("tickers"),
            )
            if completed.returncode == 0
            else None
        )
        succeeded = completed.returncode == 0 and bool(verification and verification.get("ok"))
        status = "succeeded" if succeeded else "failed"
        error = (
            None
            if succeeded
            else f"process_exit_{completed.returncode}"
            if completed.returncode != 0
            else str((verification or {}).get("error") or "ingest_run_not_verified")
        )
    except Exception as exc:
        status = "failed"
        error = str(exc)
        completed = None
    with _seed_history_lock:
        _seed_history_state = {
            **_seed_history_state,
            "status": status,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "returncode": completed.returncode if completed is not None else None,
            "error": error,
            "ingest_run": verification if completed is not None else None,
        }


@router.post("/api/admin/cleanup-storage")
def cleanup_storage(
    dry_run: bool = Query(default=True),
) -> dict[str, Any]:
    """Run storage cleanup. Use dry_run=true to preview, dry_run=false to execute."""
    results = run_cleanup(DB_PATH, dry_run=dry_run)
    return {
        "ok": True,
        "dry_run": dry_run,
        "deleted": results,
        "total_rows_affected": sum(results.values()),
    }


@router.post("/api/admin/seed-ticker-history")
def seed_ticker_history(
    request: Request,
    tickers: str = Query(
        default=(
            "GLD,USO,TLT,HYG,IEF,EEM,IWM,UUP,"
            "XLK,XLF,XLV,XLE,XLY,XLP,XLI,XLU,XLB,XLRE,XLC,IGV"
        ),
    ),
    start_date: str = Query(default="2020-01-01"),
) -> dict[str, Any]:
    """Seed historical price data for specific tickers (e.g. new commodity/sector ETFs).

    This runs a targeted ingest for just the listed tickers with full history.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return {"ok": False, "error": "No tickers specified"}

    global _seed_history_state, _seed_history_thread
    with _seed_history_lock:
        if _seed_history_thread is not None and _seed_history_thread.is_alive():
            raise HTTPException(status_code=409, detail="Price-history backfill already running")
        command = _seed_history_command(ticker_list, start_date)
        _seed_history_state = {
            "status": "running",
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished_at": None,
            "returncode": None,
            "error": None,
            "tickers": ticker_list,
            "start_date": start_date,
        }
        _seed_history_thread = threading.Thread(
            target=_run_seed_history,
            args=(command,),
            daemon=True,
            name="admin-price-history-backfill",
        )
        _seed_history_thread.start()
    return {
        "ok": True,
        "status": "running",
        "message": f"Price-history backfill accepted for {len(ticker_list)} tickers",
        "tickers": ticker_list,
        "start_date": start_date,
    }


@router.get("/api/admin/seed-ticker-history/status")
def seed_ticker_history_status(request: Request) -> dict[str, Any]:
    with _seed_history_lock:
        state = dict(_seed_history_state)
    return {"ok": state.get("status") != "failed", **state}


@router.get("/api/admin/database-stats")
def admin_database_stats() -> dict[str, Any]:
    """Return database statistics including record counts and date ranges."""
    if not DB_PATH.exists():
        return {
            "ok": False,
            "db_exists": False,
            "error": "Database file not found",
        }
    try:
        db_size_bytes = os.path.getsize(DB_PATH)
        db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        table_stats: dict[str, Any] = {}
        for tbl in tables:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM "{tbl}"')
                count = cursor.fetchone()[0]
                table_stats[tbl] = {"row_count": count}
            except Exception as exc:
                table_stats[tbl] = {"error": str(exc)}
        price_stats: dict[str, Any] = {}
        try:
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT ticker) as ticker_count,
                    COUNT(*) as total_rows,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date
                FROM price_daily
                """
            )
            row = cursor.fetchone()
            if row:
                price_stats = {
                    "ticker_count": row[0],
                    "total_rows": row[1],
                    "earliest_date": row[2],
                    "latest_date": row[3],
                }
            cursor.execute(
                """
                SELECT ticker, COUNT(*) as row_count,
                       MIN(date) as first_date, MAX(date) as last_date
                FROM price_daily
                GROUP BY ticker
                ORDER BY ticker
                """
            )
            ticker_stats = []
            for row in cursor.fetchall():
                ticker_stats.append(
                    {
                        "ticker": row[0],
                        "row_count": row[1],
                        "first_date": row[2],
                        "last_date": row[3],
                    }
                )
            price_stats["by_ticker"] = ticker_stats
        except Exception as exc:
            price_stats["error"] = str(exc)
        conn.close()
        return {
            "ok": True,
            "db_exists": True,
            "db_size_mb": db_size_mb,
            "tables": table_stats,
            "price_data": price_stats,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "db_exists": True,
            "error": str(exc),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }


# ── Audit Logging Endpoints ─────────────────────────────────────────


@router.get("/api/admin/audit-logs")
def admin_audit_logs(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    resource: str | None = Query(default=None),
    action: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Query audit logs with filtering and pagination."""
    audit_logger = get_audit_logger()
    try:
        result = query_audit_logs(
            audit_logger,
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            event_type=event_type,
            resource=resource,
            action=action,
            page=page,
            per_page=per_page,
        )
        return {"ok": True, **result}
    finally:
        audit_logger.conn.close()


@router.get("/api/admin/audit-logs/export")
def admin_audit_logs_export(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    format: str = Query(default="json", pattern="^(json|csv)$"),
) -> Any:
    """Export audit logs in JSON or CSV format."""
    audit_logger = get_audit_logger()
    try:
        result = export_audit_logs(
            audit_logger,
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            event_type=event_type,
            format=format,
        )
        if format == "csv":
            return Response(
                content=result,
                media_type="text/csv",
                headers={
                    "Content-Disposition": (
                        f"attachment; filename=audit_logs_"
                        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                },
            )
        return result
    finally:
        audit_logger.conn.close()


@router.get("/api/admin/audit-logs/summary")
def admin_audit_logs_summary(
    days: int = Query(default=7, ge=1, le=365),
) -> dict[str, Any]:
    """Get summary statistics for audit logs."""
    audit_logger = get_audit_logger()
    try:
        summary = get_audit_summary(audit_logger, days=days)
        return {"ok": True, **summary}
    finally:
        audit_logger.conn.close()


@router.get("/api/admin/audit-logs/stats")
def admin_audit_logs_stats() -> dict[str, Any]:
    """Get audit log statistics and retention info."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        stats = get_audit_stats(conn)
        return {"ok": True, **stats}
    finally:
        conn.close()


@router.post("/api/admin/audit-logs/retention")
def admin_audit_logs_retention(
    retention_days: int = Query(default=90, ge=30, le=3650),
    dry_run: bool = Query(default=True),
) -> dict[str, Any]:
    """Apply retention policy to audit logs."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if dry_run:
            cutoff_date = dt.datetime.now(
                dt.timezone.utc
            ) - dt.timedelta(days=retention_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            cursor = conn.execute(
                "SELECT COUNT(*) FROM audit_logs WHERE timestamp < ?",
                (cutoff_str,),
            )
            count = cursor.fetchone()[0]
            return {
                "ok": True,
                "dry_run": True,
                "would_delete": count,
                "retention_days": retention_days,
                "cutoff_date": cutoff_str,
            }
        else:
            deleted = apply_retention_policy(
                conn, retention_days=retention_days
            )
            return {
                "ok": True,
                "dry_run": False,
                "deleted": deleted,
                "retention_days": retention_days,
            }
    finally:
        conn.close()


@router.post("/api/admin/audit-logs/external-export")
def admin_audit_logs_external_export(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    export_format: str = Query(
        default="jsonl", pattern="^(jsonl|csv)$"
    ),
) -> dict[str, Any]:
    """Export audit logs to external storage."""
    audit_logger = get_audit_logger()
    try:
        result = export_logs_to_local(
            audit_logger,
            start_date=start_date,
            end_date=end_date,
            export_format=export_format,
        )
        return {"ok": True, **result}
    finally:
        audit_logger.conn.close()

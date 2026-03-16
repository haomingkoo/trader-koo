"""Admin endpoints: all /api/admin/* routes requiring API key auth."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import secrets
import smtplib
import sqlite3
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response

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
    latest_daily_report_json,
)
from trader_koo.backend.services.scheduler import _run_daily_update
from trader_koo.scripts.generate_daily_report import (
    _build_regime_context as _report_build_regime_context,
)

from trader_koo.audit import AuditLogger, ensure_audit_schema, apply_retention_policy, get_audit_stats
from trader_koo.audit.api import query_audit_logs, export_audit_logs, get_audit_summary
from trader_koo.audit.export import get_exporter_from_env
from trader_koo.db.sources import get_data_source_manager
from trader_koo.email_subscribers import (
    already_sent_generated_report,
    email_max_recipients,
    email_subscribe_enabled,
    parse_recipients,
    subscriber_counts,
)
from trader_koo.llm_health import (
    llm_alert_cooldown_min,
    llm_alert_enabled,
    llm_degraded_threshold,
    llm_health_summary,
    llm_token_usage_summary,
)
from trader_koo.llm_narrative import llm_status
from trader_koo.middleware.auth import (
    get_admin_endpoint_registry,
    require_admin_auth,
)
from trader_koo.paper_trades import manually_close_trade, mark_to_market
from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)

from trader_koo.backend.routers.usage import (
    _feedback_summary,
    _usage_summary,
    ensure_analytics_schema,
    ensure_feedback_schema,
)

router = APIRouter()

LOG = logging.getLogger("trader_koo.routers.admin")

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))

LOG_DIR = Path(os.getenv("TRADER_KOO_LOG_DIR", "/data/logs"))
RUN_LOG_PATH = LOG_DIR / "cron_daily.log"
LOG_PATHS: dict[str, Path] = {
    "cron": RUN_LOG_PATH,
    "update_market_db": LOG_DIR / "update_market_db.log",
    "yolo": LOG_DIR / "yolo_patterns.log",
    "api": LOG_DIR / "api.log",
}

ANALYTICS_ENABLED = str(os.getenv("TRADER_KOO_ANALYTICS_ENABLED", "1")).strip().lower() in {
    "1", "true", "yes", "on",
}

_yolo_seed_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_update_mode(mode: str | None) -> str | None:
    value = str(mode or "full").strip().lower()
    aliases = {
        "full": "full",
        "all": "full",
        "yolo": "yolo",
        "yolo_report": "yolo",
        "yolo+report": "yolo",
        "report": "report",
        "report_only": "report",
        "email": "report",
    }
    return aliases.get(value)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Failed to parse JSON file %s: %s", path.name, exc)
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _find_timeframe_summary(rows: Any, timeframe: str) -> dict[str, Any]:
    target = str(timeframe or "").strip().lower()
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("timeframe", "")).strip().lower() == target:
            return row
    return {}


def get_audit_logger() -> AuditLogger:
    conn = sqlite3.connect(str(DB_PATH))
    return AuditLogger(conn)


def _clean_optional_url(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw == "*":
        return None
    if raw.startswith(("http://", "https://")):
        return raw.rstrip("/")
    return raw


STATUS_APP_URL = _clean_optional_url(os.getenv("TRADER_KOO_APP_URL")) or _clean_optional_url(
    os.getenv("TRADER_KOO_ALLOWED_ORIGIN")
)
STATUS_BASE_URL = _clean_optional_url(os.getenv("TRADER_KOO_BASE_URL"))


# ---------------------------------------------------------------------------
# Email helpers (shared with email router but needed for admin email endpoint)
# ---------------------------------------------------------------------------

def _smtp_settings() -> dict[str, Any]:
    port_raw = os.getenv("TRADER_KOO_SMTP_PORT", "587").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 587
    timeout_raw = os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30").strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30
    security = os.getenv("TRADER_KOO_SMTP_SECURITY", "starttls").strip().lower()
    if security not in {"starttls", "ssl", "none"}:
        security = "starttls"
    return {
        "host": os.getenv("TRADER_KOO_SMTP_HOST", "").strip(),
        "port": port,
        "user": os.getenv("TRADER_KOO_SMTP_USER", "").strip(),
        "password": os.getenv("TRADER_KOO_SMTP_PASS", ""),
        "from_email": os.getenv("TRADER_KOO_SMTP_FROM", "").strip(),
        "default_to": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
        "security": security,
    }


def _resend_settings() -> dict[str, Any]:
    timeout_raw = os.getenv(
        "TRADER_KOO_RESEND_TIMEOUT_SEC", os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30")
    ).strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30
    return {
        "api_key": os.getenv("TRADER_KOO_RESEND_API_KEY", "").strip(),
        "from_email": os.getenv("TRADER_KOO_RESEND_FROM", os.getenv("TRADER_KOO_SMTP_FROM", "")).strip(),
        "default_to": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
    }


def _email_transport() -> str:
    raw = os.getenv("TRADER_KOO_EMAIL_TRANSPORT", "auto").strip().lower()
    if raw not in {"auto", "smtp", "resend"}:
        raw = "auto"
    if raw == "auto":
        resend = _resend_settings()
        return "resend" if resend.get("api_key") else "smtp"
    return raw


def _send_resend_email(
    subject: str,
    text: str,
    recipient: str,
    resend: dict[str, Any],
    *,
    html_body: str | None = None,
) -> None:
    user_agent = os.getenv("TRADER_KOO_EMAIL_USER_AGENT", "trader-koo/1.0")
    payload = {
        "from": resend["from_email"],
        "to": [recipient],
        "subject": subject,
        "text": text,
    }
    if html_body:
        payload["html"] = html_body
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {resend['api_key']}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(resend["timeout_sec"])) as resp:
            status_code = int(getattr(resp, "status", 200))
            body = resp.read().decode("utf-8", errors="replace")
        if status_code >= 300:
            raise RuntimeError(f"Resend API failed status={status_code} body={body[:500]}")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Resend API HTTP {exc.code}: {err_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Resend connect failed: {exc.reason}") from exc


def _send_smtp_email(message: EmailMessage, smtp: dict[str, Any]) -> None:
    host = smtp["host"]
    port = int(smtp["port"])
    timeout_sec = int(smtp["timeout_sec"])
    security = str(smtp["security"])
    user = str(smtp.get("user") or "")
    password = str(smtp.get("password") or "")
    if security == "ssl":
        with smtplib.SMTP_SSL(host, port, timeout=timeout_sec, context=ssl.create_default_context()) as server:
            if user:
                server.login(user, password)
            server.send_message(message)
        return
    with smtplib.SMTP(host, port, timeout=timeout_sec) as server:
        server.ehlo()
        if security == "starttls":
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        if user:
            server.login(user, password)
        server.send_message(message)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/admin/routes")
@require_admin_auth
def admin_routes() -> dict[str, Any]:
    """List all admin endpoints with their authentication status."""
    registry = get_admin_endpoint_registry()
    routes = []
    protected_count = 0
    unprotected_count = 0
    for key, info in sorted(registry.items()):
        routes.append({
            "method": info["method"],
            "path": info["path"],
            "has_auth": info["has_auth"],
            "key": key,
        })
        if info["has_auth"]:
            protected_count += 1
        else:
            unprotected_count += 1
    return {
        "total": len(routes),
        "protected": protected_count,
        "unprotected": unprotected_count,
        "all_protected": unprotected_count == 0,
        "routes": routes,
    }


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
            "message": f"daily_update already running (stage={stage}, requested_mode={mode_norm})",
            "stage": stage,
            "requested_mode": mode_norm,
            "latest_run": pipeline.get("latest_run"),
            "run_log_path": pipeline.get("run_log_path"),
            "reconciled_stale_runs": reconcile.get("reconciled", 0),
        }
    scheduler = request.app.state.scheduler
    now_utc = dt.datetime.now(dt.timezone.utc)
    manual_job_id = f"manual_daily_update_{now_utc.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(3)}"
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


@router.post("/api/admin/run-yolo-seed")
@require_admin_auth
def run_yolo_seed(timeframe: str = "both") -> dict[str, Any]:
    """Trigger full YOLO seed for all tickers in background."""
    global _yolo_seed_thread
    timeframe_norm = str(timeframe or "both").strip().lower() or "both"
    if timeframe_norm not in {"daily", "weekly", "both"}:
        raise HTTPException(status_code=400, detail="Invalid timeframe. Use one of: daily, weekly, both")
    if _yolo_seed_thread and _yolo_seed_thread.is_alive():
        return {"ok": False, "message": "Seed already running -- wait for it to finish"}
    script = PROJECT_DIR / "scripts" / "run_yolo_patterns.py"
    cmd = [
        sys.executable, str(script),
        "--db-path", str(DB_PATH),
        "--timeframe", timeframe_norm,
        "--lookback-days", "180",
        "--weekly-lookback-days", "730",
        "--sleep", "0.05",
    ]

    def _run() -> None:
        subprocess.run(cmd, capture_output=False)

    _yolo_seed_thread = threading.Thread(target=_run, daemon=True)
    _yolo_seed_thread.start()
    return {
        "ok": True,
        "message": f"YOLO seed started (timeframe={timeframe_norm}) -- tail /data/logs/yolo_patterns.log",
        "timeframe": timeframe_norm,
    }


@router.get("/api/admin/yolo-status")
@require_admin_auth
def yolo_status(log_lines: int = Query(default=40, ge=0, le=400)) -> dict[str, Any]:
    """Return YOLO runner status + DB summary + recent log tail."""
    log_path = Path(os.getenv("TRADER_KOO_YOLO_LOG_PATH", "/data/logs/yolo_patterns.log"))
    conn = get_conn()
    try:
        db_status = get_yolo_status(conn)
    finally:
        conn.close()
    thread_alive = bool(_yolo_seed_thread and _yolo_seed_thread.is_alive())
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
            "latest_run_id": latest_run_row["run_id"] if latest_run_row else None,
            "count": len(rows),
            "rows": [dict(r) for r in rows],
        }
    finally:
        conn.close()


@router.get("/api/admin/pipeline-status")
@require_admin_auth
def pipeline_status_endpoint(log_lines: int = Query(default=120, ge=20, le=1000)) -> dict[str, Any]:
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
    name: str = Query(default="cron", pattern="^(cron|update_market_db|yolo|api)$"),
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


@router.get("/api/admin/smtp-health")
@require_admin_auth
def smtp_health() -> dict[str, Any]:
    """Return email delivery config health (without secrets)."""
    smtp = _smtp_settings()
    resend = _resend_settings()
    transport = _email_transport()
    subs_enabled = email_subscribe_enabled()
    subs = subscriber_counts(DB_PATH) if subs_enabled else {"active": 0, "pending": 0, "unsubscribed": 0, "total": 0}
    auto_email = str(os.getenv("TRADER_KOO_AUTO_EMAIL", "")).strip().lower() in {"1", "true", "yes"}
    llm_alert_to = (
        str(os.getenv("TRADER_KOO_LLM_FAIL_ALERT_TO", "") or "").strip()
        or str(os.getenv("TRADER_KOO_LLM_ALERT_TO", "") or "").strip()
    )
    llm_alert_recipient_count = len(parse_recipients(llm_alert_to)) if llm_alert_to else 0
    llm_health: dict[str, Any] = {}
    try:
        llm_health = llm_health_summary(DB_PATH, recent_limit=10)
    except Exception as exc:
        LOG.warning("Failed to load LLM health summary for smtp-health: %s", exc)
        llm_health = {"error": str(exc)}
    missing: list[str] = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append("TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
        if auto_email and not resend["default_to"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
        if auto_email and not smtp["default_to"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
        if smtp["user"] and not smtp["password"]:
            missing.append("TRADER_KOO_SMTP_PASS")
    return {
        "ok": len(missing) == 0,
        "auto_email_enabled": auto_email,
        "transport": transport,
        "subscriber_registry_enabled": subs_enabled,
        "subscriber_counts": subs,
        "email_max_recipients": email_max_recipients(),
        "missing": missing,
        "smtp": {
            "host": smtp["host"],
            "port": smtp["port"],
            "security": smtp["security"],
            "timeout_sec": smtp["timeout_sec"],
            "from_email": smtp["from_email"],
            "default_to": smtp["default_to"],
            "has_user": bool(smtp["user"]),
            "has_password": bool(smtp["password"]),
        },
        "resend": {
            "has_api_key": bool(resend["api_key"]),
            "from_email": resend["from_email"],
            "default_to": resend["default_to"],
            "timeout_sec": resend["timeout_sec"],
        },
        "llm_alert": {
            "enabled": llm_alert_enabled(),
            "cooldown_min": llm_alert_cooldown_min(),
            "degraded_threshold": llm_degraded_threshold(),
            "has_override_recipients": bool(llm_alert_to),
            "override_recipient_count": llm_alert_recipient_count,
            "health": llm_health,
        },
    }


@router.get("/api/admin/llm-health")
@require_admin_auth
def admin_llm_health(
    recent_limit: int = Query(default=25, ge=1, le=200),
) -> dict[str, Any]:
    """Return LLM runtime/config health plus recent persisted failure/success events."""
    health = llm_health_summary(DB_PATH, recent_limit=recent_limit)
    status_data = llm_status()
    return {
        "ok": True,
        "status": status_data,
        "health": health,
        "alert": {
            "enabled": llm_alert_enabled(),
            "cooldown_min": llm_alert_cooldown_min(),
            "degraded_threshold": llm_degraded_threshold(),
            "has_override_recipients": bool(
                str(os.getenv("TRADER_KOO_LLM_FAIL_ALERT_TO", "") or "").strip()
                or str(os.getenv("TRADER_KOO_LLM_ALERT_TO", "") or "").strip()
            ),
        },
    }


@router.get("/api/admin/llm-usage")
@require_admin_auth
def admin_llm_usage(
    days: int = Query(default=30, ge=1, le=3650),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Return persisted LLM token usage/cost estimates."""
    summary = llm_token_usage_summary(DB_PATH, days=days, limit=limit)
    return {"ok": True, **summary}


@router.get("/api/admin/data-source-health")
@require_admin_auth
def admin_data_source_health() -> dict[str, Any]:
    """Return data source success/failure rates and metrics."""
    manager = get_data_source_manager()
    metrics = manager.get_metrics()
    alerts: list[dict[str, Any]] = []
    for source_name, source_metrics in metrics.items():
        if source_name == "yfinance" and source_metrics["failure_rate"] > 10.0:
            alerts.append({
                "source": source_name,
                "failure_rate": source_metrics["failure_rate"],
                "message": (
                    f"Primary source {source_name} failure rate "
                    f"({source_metrics['failure_rate']:.1f}%) exceeds 10% threshold"
                ),
                "severity": "warning",
            })
    return {
        "ok": True,
        "sources": metrics,
        "alerts": alerts,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


@router.get("/api/admin/database-stats")
@require_admin_auth
def admin_database_stats() -> dict[str, Any]:
    """Return database statistics including record counts and date ranges."""
    if not DB_PATH.exists():
        return {"ok": False, "db_exists": False, "error": "Database file not found"}
    try:
        db_size_bytes = os.path.getsize(DB_PATH)
        db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
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
                SELECT ticker, COUNT(*) as row_count, MIN(date) as first_date, MAX(date) as last_date
                FROM price_daily
                GROUP BY ticker
                ORDER BY ticker
                """
            )
            ticker_stats = []
            for row in cursor.fetchall():
                ticker_stats.append({
                    "ticker": row[0],
                    "row_count": row[1],
                    "first_date": row[2],
                    "last_date": row[3],
                })
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


@router.get("/api/admin/report-stability")
@require_admin_auth
def report_stability(limit: int = Query(default=60, ge=1, le=365)) -> dict[str, Any]:
    """Summarize recent report JSON files to diagnose YOLO/report stability."""
    report_dir = REPORT_DIR
    files = sorted(
        [p for p in report_dir.glob("daily_report_*.json") if p.name != "daily_report_latest.json"],
        key=lambda p: p.name,
        reverse=True,
    )
    scan_files = files[: max(1, limit)]
    rows: list[dict[str, Any]] = []

    for p in scan_files:
        modified_ts: str | None = None
        try:
            st = p.stat()
            modified_ts = (
                dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
        except OSError:
            pass

        payload = _load_json_file(p)
        if not isinstance(payload, dict):
            rows.append({
                "file": p.name,
                "generated_ts": None,
                "modified_ts": modified_ts,
                "ok": False,
                "parse_error": True,
                "yolo_present": False,
                "yolo_rows_total": 0,
                "yolo_tickers_total": 0,
            })
            continue

        yolo = payload.get("yolo") if isinstance(payload.get("yolo"), dict) else {}
        summary = yolo.get("summary") if isinstance(yolo.get("summary"), dict) else {}
        timeframes = yolo.get("timeframes") if isinstance(yolo.get("timeframes"), list) else []
        tf_daily = _find_timeframe_summary(timeframes, "daily")
        tf_weekly = _find_timeframe_summary(timeframes, "weekly")
        delta_legacy = yolo.get("delta") if isinstance(yolo.get("delta"), dict) else {}
        delta_daily = yolo.get("delta_daily") if isinstance(yolo.get("delta_daily"), dict) else delta_legacy
        delta_weekly = yolo.get("delta_weekly") if isinstance(yolo.get("delta_weekly"), dict) else {}
        counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
        freshness = payload.get("freshness") if isinstance(payload.get("freshness"), dict) else {}
        warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

        yolo_rows_total = _to_int(summary.get("rows_total"), _to_int(counts.get("yolo_rows"), 0))
        yolo_tickers_total = _to_int(summary.get("tickers_with_patterns"), 0)
        yolo_daily_tickers = _to_int(tf_daily.get("tickers_with_patterns"), 0)
        yolo_weekly_tickers = _to_int(tf_weekly.get("tickers_with_patterns"), 0)
        yolo_present = yolo_rows_total > 0
        yolo_age_hours = _to_float(freshness.get("yolo_age_hours"))
        report_kind = str(meta.get("report_kind") or "daily").strip().lower()
        if report_kind not in {"daily", "weekly"}:
            report_kind = "daily"

        rows.append({
            "file": p.name,
            "generated_ts": payload.get("generated_ts") or modified_ts,
            "modified_ts": modified_ts,
            "ok": bool(payload.get("ok", False)),
            "parse_error": False,
            "report_kind": report_kind,
            "warnings_count": len(warnings),
            "has_yolo_data_stale_warning": "yolo_data_stale" in {str(w) for w in warnings},
            "yolo_present": yolo_present,
            "yolo_rows_total": yolo_rows_total,
            "yolo_tickers_total": yolo_tickers_total,
            "yolo_daily_tickers": yolo_daily_tickers,
            "yolo_weekly_tickers": yolo_weekly_tickers,
            "yolo_latest_detected_ts": summary.get("latest_detected_ts"),
            "yolo_latest_asof_date": summary.get("latest_asof_date"),
            "yolo_age_hours": yolo_age_hours,
            "delta_daily_new": _to_int(delta_daily.get("new_count"), 0),
            "delta_daily_lost": _to_int(delta_daily.get("lost_count"), 0),
            "delta_weekly_new": _to_int(delta_weekly.get("new_count"), 0),
            "delta_weekly_lost": _to_int(delta_weekly.get("lost_count"), 0),
        })

    parsed_rows = [r for r in rows if not r.get("parse_error")]
    yolo_present_reports = sum(1 for r in parsed_rows if r.get("yolo_present"))
    yolo_missing_reports = sum(1 for r in parsed_rows if not r.get("yolo_present"))
    yolo_presence_rate_pct = (
        round((100.0 * yolo_present_reports) / len(parsed_rows), 2) if parsed_rows else None
    )
    newest_missing_streak = 0
    for r in parsed_rows:
        if r.get("yolo_present"):
            break
        newest_missing_streak += 1
    longest_missing_streak = 0
    current_streak = 0
    for r in reversed(parsed_rows):
        if r.get("yolo_present"):
            current_streak = 0
            continue
        current_streak += 1
        if current_streak > longest_missing_streak:
            longest_missing_streak = current_streak

    def _compact(values: list[float | None]) -> list[float]:
        return [float(v) for v in values if isinstance(v, (int, float))]

    newest_generated_ts = next((r.get("generated_ts") for r in parsed_rows if r.get("generated_ts")), None)
    oldest_generated_ts = next(
        (r.get("generated_ts") for r in reversed(parsed_rows) if r.get("generated_ts")), None
    )
    missing_examples: list[dict[str, Any]] = []
    for r in rows:
        if r.get("parse_error"):
            missing_examples.append({"file": r.get("file"), "generated_ts": r.get("generated_ts"), "reason": "parse_error"})
        elif not r.get("yolo_present"):
            missing_examples.append({"file": r.get("file"), "generated_ts": r.get("generated_ts"), "reason": "no_yolo_rows"})
        if len(missing_examples) >= 10:
            break

    yolo_rows_vals = [_to_float(r.get("yolo_rows_total")) for r in parsed_rows]
    yolo_tickers_vals = [_to_float(r.get("yolo_tickers_total")) for r in parsed_rows]
    yolo_daily_vals = [_to_float(r.get("yolo_daily_tickers")) for r in parsed_rows]
    yolo_weekly_vals = [_to_float(r.get("yolo_weekly_tickers")) for r in parsed_rows]
    yolo_age_vals = [_to_float(r.get("yolo_age_hours")) for r in parsed_rows if _to_float(r.get("yolo_age_hours")) is not None]
    delta_daily_new_vals = [_to_float(r.get("delta_daily_new")) for r in parsed_rows]
    delta_daily_lost_vals = [_to_float(r.get("delta_daily_lost")) for r in parsed_rows]
    delta_weekly_new_vals = [_to_float(r.get("delta_weekly_new")) for r in parsed_rows]
    delta_weekly_lost_vals = [_to_float(r.get("delta_weekly_lost")) for r in parsed_rows]

    summary_data = {
        "files_total": len(files),
        "reports_scanned": len(rows),
        "parsed_reports": len(parsed_rows),
        "parse_error_reports": len(rows) - len(parsed_rows),
        "yolo_present_reports": yolo_present_reports,
        "yolo_missing_reports": yolo_missing_reports,
        "yolo_presence_rate_pct": yolo_presence_rate_pct,
        "newest_missing_streak": newest_missing_streak,
        "longest_missing_streak": longest_missing_streak,
        "newest_generated_ts": newest_generated_ts,
        "oldest_generated_ts": oldest_generated_ts,
        "avg_yolo_rows_total": _avg(_compact(yolo_rows_vals)),
        "avg_yolo_tickers_total": _avg(_compact(yolo_tickers_vals)),
        "avg_yolo_daily_tickers": _avg(_compact(yolo_daily_vals)),
        "avg_yolo_weekly_tickers": _avg(_compact(yolo_weekly_vals)),
        "avg_yolo_age_hours": _avg(_compact(yolo_age_vals)),
        "avg_delta_daily_new": _avg(_compact(delta_daily_new_vals)),
        "avg_delta_daily_lost": _avg(_compact(delta_daily_lost_vals)),
        "avg_delta_weekly_new": _avg(_compact(delta_weekly_new_vals)),
        "avg_delta_weekly_lost": _avg(_compact(delta_weekly_lost_vals)),
    }
    return {
        "ok": True,
        "report_dir": str(report_dir),
        "sample_limit": int(limit),
        "summary": summary_data,
        "missing_examples": missing_examples,
        "rows": rows,
    }


@router.get("/api/admin/usage-summary")
@require_admin_auth
def usage_summary_endpoint(
    days: int = Query(default=7, ge=1, le=365),
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    if not ANALYTICS_ENABLED:
        return {"ok": True, "analytics_enabled": False, "detail": "Analytics collection is disabled."}
    conn = get_conn()
    try:
        summary = _usage_summary(conn, days=days, limit=limit)
    finally:
        conn.close()
    summary["analytics_enabled"] = True
    return summary


@router.get("/api/admin/feedback-summary")
@require_admin_auth
def admin_feedback_summary(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=12, ge=1, le=100),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        summary = _feedback_summary(conn, days=days)
        if not table_exists(conn, "setup_feedback"):
            return {"ok": True, "summary": summary, "top_tickers": [], "recent": []}
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
        top_rows = conn.execute(
            """
            SELECT
                ticker,
                COUNT(*) AS votes,
                SUM(CASE WHEN verdict='good' THEN 1 ELSE 0 END) AS good,
                SUM(CASE WHEN verdict='bad' THEN 1 ELSE 0 END) AS bad,
                SUM(CASE WHEN verdict='neutral' THEN 1 ELSE 0 END) AS neutral
            FROM setup_feedback
            WHERE created_ts >= ?
            GROUP BY ticker
            ORDER BY votes DESC, ticker ASC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
        recent_rows = conn.execute(
            """
            SELECT
                created_ts, ticker, verdict, source_surface, asof,
                setup_tier, setup_score
            FROM setup_feedback
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return {
        "ok": True,
        "summary": summary,
        "top_tickers": [
            {
                "ticker": str(row["ticker"] or ""),
                "votes": int(row["votes"] or 0),
                "good": int(row["good"] or 0),
                "bad": int(row["bad"] or 0),
                "neutral": int(row["neutral"] or 0),
                "good_rate_pct": round((int(row["good"] or 0) * 100.0) / int(row["votes"] or 1), 2)
                if int(row["votes"] or 0) else None,
            }
            for row in top_rows
        ],
        "recent": [dict(row) for row in recent_rows],
    }


@router.get("/api/admin/setup-eval-summary")
@require_admin_auth
def admin_setup_eval_summary(limit_families: int = Query(default=12, ge=1, le=100)) -> dict[str, Any]:
    latest_path, latest_payload = latest_daily_report_json(REPORT_DIR)
    if not isinstance(latest_payload, dict):
        return {
            "ok": True,
            "detail": "No daily report available yet.",
            "generated_ts": None,
            "summary": {},
            "top_long_families": [],
            "top_short_families": [],
        }
    signals = latest_payload.get("signals")
    setup_eval = signals.get("setup_evaluation") if isinstance(signals, dict) else {}
    if not isinstance(setup_eval, dict):
        setup_eval = {}
    families = setup_eval.get("by_family")
    if not isinstance(families, list):
        families = []
    by_validity = setup_eval.get("by_validity_days")
    if not isinstance(by_validity, list):
        by_validity = []
    improvement_actions = setup_eval.get("improvement_actions")
    if not isinstance(improvement_actions, list):
        improvement_actions = []

    def _round_stat(value: Any, digits: int = 2) -> float | None:
        num = _to_float(value)
        if num is None:
            return None
        return round(num, digits)

    def _norm_family(row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        direction = str(row.get("call_direction") or "").strip().lower()
        if direction not in {"long", "short"}:
            return None
        return {
            "setup_family": str(row.get("setup_family") or "").strip(),
            "call_direction": direction,
            "calls": int(row.get("calls") or 0),
            "hit_rate_pct": _round_stat(row.get("hit_rate_pct"), 2),
            "avg_signed_return_pct": _round_stat(row.get("avg_signed_return_pct"), 2),
            "expectancy_pct": _round_stat(row.get("expectancy_pct"), 2),
            "avg_validity_days": _round_stat(row.get("avg_validity_days"), 2),
        }

    def _norm_validity(row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        validity = int(row.get("validity_days") or 0)
        if validity <= 0:
            return None
        return {
            "validity_days": validity,
            "calls": int(row.get("calls") or 0),
            "hit_rate_pct": _round_stat(row.get("hit_rate_pct"), 2),
            "avg_signed_return_pct": _round_stat(row.get("avg_signed_return_pct"), 2),
            "expectancy_pct": _round_stat(row.get("expectancy_pct"), 2),
            "profit_factor": _round_stat(row.get("profit_factor"), 2),
        }

    normalized = [row for row in (_norm_family(item) for item in families) if isinstance(row, dict)]
    normalized.sort(
        key=lambda item: (
            int(item.get("calls") or 0),
            float(item.get("hit_rate_pct") or 0.0),
            float(item.get("avg_signed_return_pct") or 0.0),
        ),
        reverse=True,
    )
    top_by_edge = sorted(
        normalized,
        key=lambda item: (
            float(item.get("expectancy_pct") or 0.0),
            float(item.get("hit_rate_pct") or 0.0),
            int(item.get("calls") or 0),
        ),
        reverse=True,
    )[: int(limit_families)]
    weakest_by_edge = sorted(
        normalized,
        key=lambda item: (
            float(item.get("expectancy_pct") or 0.0),
            float(item.get("hit_rate_pct") or 0.0),
            -int(item.get("calls") or 0),
        ),
    )[: int(limit_families)]
    top_long = [row for row in normalized if row.get("call_direction") == "long"][: int(limit_families)]
    top_short = [row for row in normalized if row.get("call_direction") == "short"][: int(limit_families)]
    normalized_validity = [row for row in (_norm_validity(item) for item in by_validity) if isinstance(row, dict)]
    normalized_validity.sort(key=lambda item: int(item.get("validity_days") or 0))

    return {
        "ok": True,
        "report_path": str(latest_path) if latest_path else None,
        "generated_ts": latest_payload.get("generated_ts"),
        "summary": setup_eval,
        "by_validity_days": normalized_validity,
        "improvement_actions": improvement_actions,
        "top_long_families": top_long,
        "top_short_families": top_short,
        "top_families_by_edge": top_by_edge,
        "weakest_families_by_edge": weakest_by_edge,
    }


@router.get("/api/admin/setup-eval-calls")
@require_admin_auth
def admin_setup_eval_calls(
    status: str = Query(default="scored", pattern="^(open|scored|invalid|all)$"),
    ticker: str | None = Query(default=None),
    direction: str | None = Query(default=None, pattern="^(long|short|neutral)$"),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        if not table_exists(conn, "setup_call_evaluations"):
            return {"ok": True, "detail": "setup_call_evaluations table not found", "rows": [], "count": 0}
        where_parts: list[str] = []
        params: list[Any] = []
        status_norm = str(status or "").strip().lower()
        if status_norm != "all":
            where_parts.append("status = ?")
            params.append(status_norm)
        ticker_norm = str(ticker or "").strip().upper()
        if ticker_norm:
            where_parts.append("ticker = ?")
            params.append(ticker_norm)
        direction_norm = str(direction or "").strip().lower()
        if direction_norm:
            where_parts.append("call_direction = ?")
            params.append(direction_norm)
        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + " AND ".join(where_parts)
        params.append(int(limit))
        rows = conn.execute(
            f"""
            SELECT
                id, asof_date, ticker, status, call_direction, validity_days,
                setup_family, setup_tier, signal_bias, actionability, score,
                close_asof, valid_target_date, evaluated_date, close_evaluated,
                raw_return_pct, signed_return_pct, direction_hit,
                yolo_pattern, yolo_recency, generated_ts, created_ts, updated_ts
            FROM setup_call_evaluations
            {where_sql}
            ORDER BY asof_date DESC, id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return {
            "ok": True,
            "status": status_norm,
            "ticker": ticker_norm or None,
            "direction": direction_norm or None,
            "count": len(rows),
            "rows": [dict(row) for row in rows],
        }
    finally:
        conn.close()


# ── Audit Logging Endpoints ─────────────────────────────────────────

@router.get("/api/admin/audit-logs")
@require_admin_auth
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
@require_admin_auth
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
@require_admin_auth
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
@require_admin_auth
def admin_audit_logs_stats() -> dict[str, Any]:
    """Get audit log statistics and retention info."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        stats = get_audit_stats(conn)
        return {"ok": True, **stats}
    finally:
        conn.close()


@router.post("/api/admin/audit-logs/retention")
@require_admin_auth
def admin_audit_logs_retention(
    retention_days: int = Query(default=90, ge=30, le=3650),
    dry_run: bool = Query(default=True),
) -> dict[str, Any]:
    """Apply retention policy to audit logs."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if dry_run:
            cutoff_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=retention_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            cursor = conn.execute("SELECT COUNT(*) FROM audit_logs WHERE timestamp < ?", (cutoff_str,))
            count = cursor.fetchone()[0]
            return {
                "ok": True,
                "dry_run": True,
                "would_delete": count,
                "retention_days": retention_days,
                "cutoff_date": cutoff_str,
            }
        else:
            deleted = apply_retention_policy(conn, retention_days=retention_days)
            return {
                "ok": True,
                "dry_run": False,
                "deleted": deleted,
                "retention_days": retention_days,
            }
    finally:
        conn.close()


@router.post("/api/admin/audit-logs/external-export")
@require_admin_auth
def admin_audit_logs_external_export(
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
    export_format: str = Query(default="jsonl", pattern="^(jsonl|csv)$"),
) -> dict[str, Any]:
    """Export audit logs to external storage."""
    audit_logger = get_audit_logger()
    exporter = get_exporter_from_env()
    try:
        result = exporter.export_logs(
            audit_logger,
            start_date=start_date,
            end_date=end_date,
            export_format=export_format,
        )
        return {"ok": True, **result}
    finally:
        audit_logger.conn.close()


# ── Admin email ─────────────────────────────────────────

@router.post("/api/admin/email-latest-report")
@require_admin_auth
def email_latest_report(
    to: str | None = Query(default=None),
    include_markdown: bool = Query(default=True),
    attach_json: bool = Query(default=True),
) -> dict[str, Any]:
    """Send the latest daily report by email via configured transport."""
    smtp = _smtp_settings()
    resend = _resend_settings()
    transport = _email_transport()
    default_to = resend["default_to"] if transport == "resend" else smtp["default_to"]
    recipient = (to or default_to or "").strip()
    missing: list[str] = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append("TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
        if smtp["user"] and not smtp["password"]:
            missing.append("TRADER_KOO_SMTP_PASS")
    if not recipient:
        missing.append("TRADER_KOO_REPORT_EMAIL_TO (or use ?to=...)")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing email config: {', '.join(missing)}")
    report_dir = REPORT_DIR
    latest_path, latest_payload = latest_daily_report_json(report_dir)
    if latest_payload is None:
        raise HTTPException(status_code=404, detail=f"No report found in {report_dir}")
    latest_md_path = report_dir / "daily_report_latest.md"
    md_text = ""
    if latest_md_path.exists():
        try:
            md_text = latest_md_path.read_text(encoding="utf-8")
        except Exception as exc:
            LOG.warning("Failed to read markdown report: %s", exc)
            md_text = ""
    generated = str(
        latest_payload.get("generated_ts")
        or latest_payload.get("generated_at_utc")
        or latest_payload.get("snapshot_ts")
        or ""
    ).strip()
    if not generated:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    subject = build_report_email_subject(latest_payload)
    text_body, html_body = build_report_email_bodies(
        latest_payload,
        md_text if include_markdown else "",
        app_url=report_email_app_url(),
    )
    from_header = resend["from_email"] if transport == "resend" else smtp["from_email"]
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_header
    message["To"] = recipient
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    if attach_json:
        filename = latest_path.name if latest_path is not None else "daily_report_latest.json"
        json_bytes = json.dumps(latest_payload, indent=2).encode("utf-8")
        message.add_attachment(json_bytes, maintype="application", subtype="json", filename=filename)
    if include_markdown and md_text:
        message.add_attachment(
            md_text.encode("utf-8"),
            maintype="text",
            subtype="markdown",
            filename="daily_report_latest.md",
        )
    try:
        if transport == "resend":
            _send_resend_email(
                subject=subject,
                text=text_body,
                recipient=recipient,
                resend=resend,
                html_body=html_body,
            )
        else:
            _send_smtp_email(message, smtp)
    except Exception as exc:
        LOG.exception("Failed to send daily report email (transport=%s)", transport)
        raise HTTPException(status_code=500, detail=f"Email send failed: {exc}") from exc
    return {
        "ok": True,
        "transport": transport,
        "to": recipient,
        "subject": subject,
        "report_file": str(latest_path) if latest_path else None,
        "smtp_host": smtp["host"],
        "smtp_port": smtp["port"],
        "smtp_security": smtp["security"],
    }


# ── Paper trade admin ────────────────────────────────────

@router.post("/api/admin/paper-trades/close")
@require_admin_auth
def admin_close_paper_trade(
    request: Request,
    trade_id: int = Query(..., ge=1),
    exit_price: float | None = Query(default=None),
    exit_reason: str = Query(default="manual_close"),
) -> dict[str, Any]:
    """Manually close an open paper trade."""
    conn = get_conn()
    try:
        result = manually_close_trade(
            conn,
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
        return {"ok": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@router.post("/api/admin/paper-trades/mtm")
@require_admin_auth
def admin_trigger_mtm(request: Request) -> dict[str, Any]:
    """Trigger mark-to-market on all open paper trades."""
    conn = get_conn()
    try:
        result = mark_to_market(conn)
        conn.commit()
        return {"ok": True, **result}
    finally:
        conn.close()

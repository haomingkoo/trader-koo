"""System endpoints: health, config, status, VIX glossary/markers."""
from __future__ import annotations

import datetime as dt
import logging
import os
import resource
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

LOG = logging.getLogger("trader_koo.routers.system")

from trader_koo.backend.services.database import DB_PATH, get_conn, table_exists
from trader_koo.backend.services.market_data import days_since, hours_since
from trader_koo.backend.services.pipeline import (
    get_cached_status,
    pipeline_status_snapshot,
    post_ingest_resume_candidate,
    set_cached_status,
)
from trader_koo.llm_narrative import llm_status
from trader_koo.security.endpoint_validator import sanitize_public_response

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level constants (read from env once)
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.getenv("TRADER_KOO_LOG_DIR", "/data/logs"))
RUN_LOG_PATH = LOG_DIR / "cron_daily.log"

EXPOSE_STATUS_INTERNAL = str(os.getenv("TRADER_KOO_EXPOSE_STATUS_INTERNAL", "0")).strip().lower() in {
    "1", "true", "yes", "on",
}
PROCESS_START_UTC = dt.datetime.now(dt.timezone.utc)

CONTROL_CENTER_CONTRACT_VERSION = "2026-03-01"
APP_VERSION = str(os.getenv("TRADER_KOO_APP_VERSION", "0.2.0") or "0.2.0").strip() or "0.2.0"

API_KEY = os.getenv("TRADER_KOO_API_KEY", "")


from trader_koo.backend.utils import clean_optional_url as _clean_optional_url


def _first_env(names: list[str]) -> str | None:
    for name in names:
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return None


STATUS_BASE_URL = _clean_optional_url(os.getenv("TRADER_KOO_BASE_URL"))
STATUS_APP_URL = _clean_optional_url(os.getenv("TRADER_KOO_APP_URL")) or _clean_optional_url(
    os.getenv("TRADER_KOO_ALLOWED_ORIGIN")
)
STATUS_REPO_URL = _clean_optional_url(os.getenv("TRADER_KOO_REPO_URL"))
STATUS_GIT_SHA = _first_env(
    ["TRADER_KOO_GIT_SHA", "RAILWAY_GIT_COMMIT_SHA", "GIT_SHA", "COMMIT_SHA", "SOURCE_VERSION"]
)
STATUS_DEPLOYED_TS = _first_env(
    ["TRADER_KOO_DEPLOYED_TS", "TRADER_KOO_DEPLOYED_AT", "DEPLOYED_TS", "DEPLOYED_AT"]
)
STATUS_DEPLOYMENT_ID = _first_env(
    ["TRADER_KOO_DEPLOYMENT_ID", "RAILWAY_DEPLOYMENT_ID", "DEPLOYMENT_ID"]
)

CONTROL_CENTER_ACTIONS = [
    {
        "id": "run-full-update",
        "label": "Run Full Update",
        "description": "Ingest, YOLO refresh, and report generation.",
        "method": "POST",
        "path": "/api/admin/trigger-update?mode=full",
        "confirm_text": "Queue the full nightly pipeline now?",
    },
    {
        "id": "run-yolo-refresh",
        "label": "Run YOLO + Report",
        "description": "Skip ingest and rerun YOLO plus report generation.",
        "method": "POST",
        "path": "/api/admin/trigger-update?mode=yolo",
        "confirm_text": "Queue YOLO plus report now?",
    },
    {
        "id": "run-report-only",
        "label": "Rebuild Report",
        "description": "Generate the latest report without new ingest.",
        "method": "POST",
        "path": "/api/admin/trigger-update?mode=report",
    },
    {
        "id": "run-yolo-seed",
        "label": "Run YOLO Seed",
        "description": "Full backfill pattern scan across tracked tickers.",
        "method": "POST",
        "path": "/api/admin/run-yolo-seed?timeframe=both",
        "confirm_text": "Start the full YOLO seed run?",
    },
]


# ---------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------

from trader_koo.backend.utils import current_rss_mb as _current_rss_mb


def _max_rss_mb() -> float | None:
    try:
        rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            rss_kb = rss_kb / 1024.0
        return rss_kb / 1024.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/health")
def health() -> dict[str, Any]:
    """Health check endpoint (Requirement 6.5)."""
    db_exists = DB_PATH.exists()
    payload: dict[str, Any] = {"ok": db_exists, "db_exists": db_exists}
    if EXPOSE_STATUS_INTERNAL:
        payload["db_name"] = DB_PATH.name
    return sanitize_public_response(payload)


@router.get("/api/config", include_in_schema=False)
def config() -> dict[str, Any]:
    """Public client config -- never expose secrets (Requirement 6.6)."""
    response = {
        "auth": {
            "admin_api_key_required": bool(API_KEY),
            "admin_api_key_header": "X-API-Key",
        },
    }
    return sanitize_public_response(response)


@router.get("/api/vix-glossary")
def vix_glossary() -> dict[str, Any]:
    """Return glossary definitions for VIX trap/reclaim patterns."""
    from trader_koo.structure.vix_patterns import get_pattern_glossary

    return {
        "glossary": get_pattern_glossary(),
        "description": "Definitions for VIX trap and reclaim pattern terminology",
    }


@router.get("/api/vix-pattern-markers")
def vix_pattern_markers() -> dict[str, Any]:
    """Return visual marker specifications for VIX trap/reclaim patterns."""
    from trader_koo.structure.vix_patterns import get_pattern_visual_markers

    return {
        "markers": get_pattern_visual_markers(),
        "description": "Visual marker specifications for VIX trap/reclaim patterns on charts",
    }


@router.get("/api/polymarket")
def polymarket_data(limit: int = 15) -> dict[str, Any]:
    """Public endpoint: curated finance-relevant Polymarket events."""
    try:
        from trader_koo.ml.external_data import fetch_polymarket_events

        events = fetch_polymarket_events(limit=limit)
        return {"ok": True, "count": len(events), "events": events}
    except Exception as exc:
        LOG.exception("Failed to fetch Polymarket events: %s", exc)
        return {"ok": False, "error": "Unable to fetch Polymarket data", "events": []}


@router.get("/api/macro-data")
def macro_data_public() -> dict[str, Any]:
    """Public endpoint: FRED yield curve + M2 data."""
    try:
        from trader_koo.ml.external_data import get_macro_snapshot

        return {"ok": True, **get_macro_snapshot()}
    except Exception as exc:
        LOG.exception("Failed to fetch macro data: %s", exc)
        return {"ok": False, "error": "Unable to fetch macro data"}


@router.get("/api/status")
def status() -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)

    cached = get_cached_status(now)
    if cached is not None:
        return cached

    rss_now = _current_rss_mb()
    rss_max = _max_rss_mb()
    base: dict[str, Any] = {
        "service": "trader_koo-api",
        "now_utc": now.replace(microsecond=0).isoformat(),
        "db_exists": DB_PATH.exists(),
    }
    if EXPOSE_STATUS_INTERNAL:
        base["db_name"] = DB_PATH.name
        base["process"] = {
            "pid": os.getpid(),
            "rss_mb": None if rss_now is None else round(rss_now, 2),
            "rss_max_mb": None if rss_max is None else round(rss_max, 2),
            "uptime_sec": int((now - PROCESS_START_UTC).total_seconds()),
        }
    if not DB_PATH.exists():
        return {**base, "ok": False, "error": "Database file not found"}

    conn = get_conn()
    try:
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM price_daily) AS price_rows,
                (SELECT COUNT(*) FROM finviz_fundamentals) AS fundamentals_rows,
                (SELECT COUNT(*) FROM options_iv) AS options_rows,
                (SELECT COUNT(DISTINCT ticker) FROM price_daily) AS tracked_tickers,
                (SELECT MAX(date) FROM price_daily) AS latest_price_date,
                (SELECT MAX(snapshot_ts) FROM finviz_fundamentals) AS latest_fund_snapshot,
                (SELECT MAX(snapshot_ts) FROM options_iv) AS latest_opt_snapshot
            """
        ).fetchone()

        run_row = None
        ticker_status_count = None
        last_failed_run = None
        failed_runs_7d = 0
        if table_exists(conn, "ingest_runs"):
            run_row = conn.execute(
                """
                SELECT
                    run_id, started_ts, finished_ts, status,
                    tickers_total, tickers_ok, tickers_failed, error_message
                FROM ingest_runs
                ORDER BY started_ts DESC
                LIMIT 1
                """
            ).fetchone()
            ts_row = None
            if run_row and table_exists(conn, "ingest_ticker_status"):
                ts_row = conn.execute(
                    "SELECT COUNT(*) AS c FROM ingest_ticker_status WHERE run_id = ?",
                    (run_row["run_id"],),
                ).fetchone()
            ticker_status_count = int(ts_row["c"]) if ts_row else 0
            failed_cutoff = (now - dt.timedelta(days=7)).replace(microsecond=0).isoformat()
            failed_row = conn.execute(
                """
                SELECT COUNT(*) AS failed_runs_7d
                FROM ingest_runs
                WHERE status = 'failed' AND COALESCE(finished_ts, started_ts) >= ?
                """,
                (failed_cutoff,),
            ).fetchone()
            failed_runs_7d = int(failed_row["failed_runs_7d"] or 0) if failed_row is not None else 0
            last_failed_run = conn.execute(
                """
                SELECT run_id, started_ts, finished_ts, status, error_message
                FROM ingest_runs
                WHERE status = 'failed'
                ORDER BY COALESCE(finished_ts, started_ts) DESC
                LIMIT 1
                """
            ).fetchone()

        latest_price_date = counts["latest_price_date"] if counts else None
        latest_fund_snapshot = counts["latest_fund_snapshot"] if counts else None
        latest_opt_snapshot = counts["latest_opt_snapshot"] if counts else None

        price_age_days = days_since(latest_price_date, now)
        fund_age_hours = hours_since(latest_fund_snapshot, now)
        opt_age_hours = hours_since(latest_opt_snapshot, now)

        warnings: list[str] = []
        if price_age_days is None or price_age_days > 3:
            warnings.append("price_daily stale")
        if fund_age_hours is None or fund_age_hours > 48:
            warnings.append("finviz_fundamentals stale")

        latest_run = dict(run_row) if run_row is not None else None
        if latest_run and latest_run.get("status") in {"failed"}:
            warnings.append("latest ingest run failed")
        if latest_run:
            if ticker_status_count is not None:
                latest_run["tickers_processed"] = ticker_status_count
            elif latest_run.get("status") in {"ok", "failed"}:
                completed = int(latest_run.get("tickers_ok") or 0) + int(latest_run.get("tickers_failed") or 0)
                latest_run["tickers_processed"] = completed or int(latest_run.get("tickers_total") or 0)
        latest_failed = dict(last_failed_run) if last_failed_run is not None else None
        latest_error_message = None
        latest_error_ts = None
        if latest_run and latest_run.get("status") == "failed":
            latest_error_message = str(latest_run.get("error_message") or "").strip() or None
            latest_error_ts = latest_run.get("finished_ts") or latest_run.get("started_ts")
        if latest_error_message is None and latest_failed:
            latest_error_message = str(latest_failed.get("error_message") or "").strip() or None
            latest_error_ts = latest_failed.get("finished_ts") or latest_failed.get("started_ts")

        pipeline_snap = pipeline_status_snapshot(log_lines=60)
        resume_candidate = post_ingest_resume_candidate(
            latest_run=latest_run,
            pipeline_active=bool(pipeline_snap.get("active")),
            now_utc=now,
        )
        pipeline_active = bool(pipeline_snap.get("active"))
        pipeline_stage = pipeline_snap.get("stage") or "unknown"
        pipeline_stage_line = pipeline_snap.get("stage_line")
        if latest_run and latest_run.get("status") == "running" and not pipeline_snap.get("running_stale"):
            pipeline_active = True
            if pipeline_stage in {"unknown", "idle"}:
                pipeline_stage = "ingest"
        if latest_run and latest_run.get("status") == "running" and pipeline_snap.get("running_stale"):
            warnings.append("latest ingest run appears stale-running")
        if resume_candidate:
            warnings.append("post-ingest yolo/report recovery recommended")

        llm_meta = llm_status()
        llm_health_data = llm_meta.get("health") if isinstance(llm_meta.get("health"), dict) else {}
        if llm_meta.get("enabled"):
            if not llm_meta.get("ready"):
                warnings.append("llm not ready")
            if llm_meta.get("runtime_disabled"):
                warnings.append("llm in runtime cooldown")
            if llm_health_data.get("degraded"):
                warnings.append("llm degraded (rule fallback active)")

        activity = {
            "tracked_tickers": counts["tracked_tickers"] if counts else 0,
            "tickers_processed": int((latest_run or {}).get("tickers_processed") or 0),
            "tickers_total": int((latest_run or {}).get("tickers_total") or 0),
            "tickers_ok": int((latest_run or {}).get("tickers_ok") or 0),
            "tickers_failed": int((latest_run or {}).get("tickers_failed") or 0),
            "price_rows": counts["price_rows"] if counts else 0,
            "fundamentals_rows": counts["fundamentals_rows"] if counts else 0,
            "options_rows": counts["options_rows"] if counts else 0,
        }

        service_meta: dict[str, Any] = {
            "service": "trader_koo-api",
            "contract": "control-center-v1",
            "contract_version": CONTROL_CENTER_CONTRACT_VERSION,
            "version": APP_VERSION,
            "auth_header": "X-API-Key",
            "admin_auth_configured": bool(API_KEY),
            "runtime_started_ts": PROCESS_START_UTC.replace(microsecond=0).isoformat(),
            "actions": CONTROL_CENTER_ACTIONS,
        }
        if STATUS_BASE_URL:
            service_meta["base_url"] = STATUS_BASE_URL
        if STATUS_APP_URL:
            service_meta["app_url"] = STATUS_APP_URL
        if STATUS_REPO_URL:
            service_meta["repo_url"] = STATUS_REPO_URL
        if STATUS_GIT_SHA:
            service_meta["git_sha"] = STATUS_GIT_SHA
        if STATUS_DEPLOYED_TS:
            service_meta["deployed_ts"] = STATUS_DEPLOYED_TS
        if STATUS_DEPLOYMENT_ID:
            service_meta["deployment_id"] = STATUS_DEPLOYMENT_ID

        payload = {
            **base,
            "ok": len(warnings) == 0,
            "warnings": warnings,
            "warning_count": len(warnings),
            "latest_run": latest_run,
            "pipeline_active": pipeline_active,
            "pipeline_stage": pipeline_stage,
            "service_meta": service_meta,
            "errors": {
                "failed_runs_7d": failed_runs_7d,
                "latest_error_message": latest_error_message,
                "latest_error_ts": latest_error_ts,
                "latest_failed_run": latest_failed,
            },
            "pipeline": {
                "active": pipeline_active,
                "stage": pipeline_stage,
                "stage_line": pipeline_stage_line,
                "stage_line_ts": pipeline_snap.get("stage_line_ts"),
                "stage_age_sec": pipeline_snap.get("stage_age_sec"),
                "stale_timeout_sec": pipeline_snap.get("stale_timeout_sec"),
                "stale_inference": pipeline_snap.get("stale_inference"),
                "running_age_sec": pipeline_snap.get("running_age_sec"),
                "running_stale_min": pipeline_snap.get("running_stale_min"),
                "running_stale": pipeline_snap.get("running_stale"),
                "last_completed_stage": pipeline_snap.get("last_completed_stage"),
                "last_completed_status": pipeline_snap.get("last_completed_status"),
                "last_completed_line": pipeline_snap.get("last_completed_line"),
                "last_completed_ts": pipeline_snap.get("last_completed_ts"),
                "post_ingest_resume": resume_candidate,
                **({"run_log_path": str(RUN_LOG_PATH)} if EXPOSE_STATUS_INTERNAL else {}),
            },
            "freshness": {
                "price_age_days": None if price_age_days is None else round(price_age_days, 2),
                "fund_age_hours": None if fund_age_hours is None else round(fund_age_hours, 2),
                "opt_age_hours": None if opt_age_hours is None else round(opt_age_hours, 2),
            },
            "counts": {
                "tracked_tickers": counts["tracked_tickers"] if counts else 0,
                "price_rows": counts["price_rows"] if counts else 0,
                "fundamentals_rows": counts["fundamentals_rows"] if counts else 0,
                "options_rows": counts["options_rows"] if counts else 0,
            },
            "activity": activity,
            "llm": llm_meta,
            "latest_data": {
                "price_date": latest_price_date,
                "fund_snapshot": latest_fund_snapshot,
                "options_snapshot": latest_opt_snapshot,
            },
        }
        set_cached_status(now, payload)
        return sanitize_public_response(payload)
    finally:
        conn.close()

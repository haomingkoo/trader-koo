"""Pipeline status tracking and inference.

Infers the current pipeline phase (idle, ingest, yolo, report) from
cron log tails and the ``ingest_runs`` DB table.  Also provides the
status-cache machinery and stale-run reconciliation used by the
``/api/status`` and ``/api/admin/pipeline-status`` endpoints.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import secrets
import sqlite3
import threading
from pathlib import Path
from typing import Any

from trader_koo.backend.services.database import DB_PATH, table_exists
from trader_koo.backend.services.market_data import parse_iso_utc, read_latest_ingest_run
from trader_koo.backend.services.report_loader import (
    _tail_text_file,
    latest_daily_report_json,
)
from trader_koo.backend.services.scheduler import (
    RUN_LOG_PATH,
    _append_run_log,
    _run_daily_update,
)

LOG = logging.getLogger("trader_koo.services.pipeline")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))
STATUS_CACHE_TTL_SEC = max(0, int(os.getenv("TRADER_KOO_STATUS_CACHE_SEC", "20")))
PIPELINE_STALE_SEC = max(60, int(os.getenv("TRADER_KOO_PIPELINE_STALE_SEC", "1200")))
INGEST_RUNNING_STALE_MIN = max(10, int(os.getenv("TRADER_KOO_INGEST_RUNNING_STALE_MIN", "75")))
AUTO_RESUME_POST_INGEST = str(os.getenv("TRADER_KOO_AUTO_RESUME_POST_INGEST", "1")).strip().lower() in {
    "1", "true", "yes", "on",
}
AUTO_RESUME_MAX_AGE_HOURS = max(1, int(os.getenv("TRADER_KOO_AUTO_RESUME_MAX_AGE_HOURS", "18")))
AUTO_RESUME_MAX_RETRIES = max(0, int(os.getenv("TRADER_KOO_AUTO_RESUME_MAX_RETRIES", "2")))

# ---------------------------------------------------------------------------
# Status cache (thread-safe)
# ---------------------------------------------------------------------------

_STATUS_CACHE_LOCK = threading.Lock()
_STATUS_CACHE_AT: dt.datetime | None = None
_STATUS_CACHE_PAYLOAD: dict[str, Any] | None = None


def invalidate_status_cache() -> None:
    """Clear the cached ``/api/status`` payload."""
    global _STATUS_CACHE_AT, _STATUS_CACHE_PAYLOAD
    with _STATUS_CACHE_LOCK:
        _STATUS_CACHE_AT = None
        _STATUS_CACHE_PAYLOAD = None


def get_cached_status(now: dt.datetime) -> dict[str, Any] | None:
    """Return the cached status payload if still fresh, else None."""
    if STATUS_CACHE_TTL_SEC <= 0:
        return None
    with _STATUS_CACHE_LOCK:
        if (
            _STATUS_CACHE_AT is not None
            and _STATUS_CACHE_PAYLOAD is not None
            and (now - _STATUS_CACHE_AT).total_seconds() < STATUS_CACHE_TTL_SEC
        ):
            return dict(_STATUS_CACHE_PAYLOAD)
    return None


def set_cached_status(now: dt.datetime, payload: dict[str, Any]) -> None:
    """Store a fresh status payload in the cache."""
    global _STATUS_CACHE_AT, _STATUS_CACHE_PAYLOAD
    if STATUS_CACHE_TTL_SEC > 0:
        with _STATUS_CACHE_LOCK:
            _STATUS_CACHE_AT = now
            _STATUS_CACHE_PAYLOAD = payload


# ---------------------------------------------------------------------------
# Log-line timestamp parser
# ---------------------------------------------------------------------------

def parse_log_line_ts_utc(line: str | None) -> dt.datetime | None:
    """Parse a UTC timestamp from a cron log line prefix."""
    if not line:
        return None
    text = str(line).strip()
    if not text:
        return None

    m_iso = re.match(r"^(\d{4}-\d{2}-\d{2}T[0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2}))\b", text)
    if m_iso:
        ts = m_iso.group(1)
        if re.match(r".*[+-]\d{4}$", ts):
            ts = f"{ts[:-5]}{ts[-5:-2]}:{ts[-2:]}"
        return parse_iso_utc(ts)

    m_py = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,(\d{1,6}))?\b", text)
    if m_py:
        base = m_py.group(1)
        frac = (m_py.group(2) or "0")[:6]
        micro = int(frac.ljust(6, "0"))
        try:
            parsed = dt.datetime.strptime(base, "%Y-%m-%d %H:%M:%S").replace(
                microsecond=micro,
                tzinfo=dt.timezone.utc,
            )
            return parsed
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Pipeline inference from log tail
# ---------------------------------------------------------------------------

def _infer_pipeline_from_log_tail(tail: list[str]) -> dict[str, Any]:
    """Infer active/stage from log marker lines."""
    def _last_idx(lines: list[str], token: str) -> int:
        for idx in range(len(lines) - 1, -1, -1):
            if token in lines[idx]:
                return idx
        return -1

    def _last_idx_any(lines: list[str], tokens: list[str]) -> int:
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            for token in tokens:
                if token in line:
                    return idx
        return -1

    start_idx = _last_idx(tail, "[START] daily_update.sh")
    done_idx = _last_idx(tail, "[DONE]  daily_update.sh")
    report_start_idx = _last_idx(tail, "[REPORT] Generating")
    report_done_idx = _last_idx(tail, "[REPORT] Done.")
    yolo_start_idx = _last_idx(tail, "[YOLO]  Starting")
    yolo_done_idx = _last_idx(tail, "[YOLO]  Daily pattern detection done.")
    weekly_start_idx = _last_idx(tail, "[WEEKLY] Starting weekly YOLO pass")
    weekly_report_start_idx = _last_idx(tail, "[WEEKLY] Report regeneration starting")
    weekly_yolo_done_idx = _last_idx(tail, "[WEEKLY] Weekly YOLO completed OK")
    weekly_yolo_failed_idx = _last_idx(tail, "[WEEKLY] Weekly YOLO failed")
    weekly_report_done_idx = _last_idx(tail, "[WEEKLY] Report regeneration completed")
    weekly_report_failed_idx = _last_idx(tail, "[WEEKLY] Report regeneration failed")
    yolo_progress_idx = _last_idx_any(
        tail,
        [
            "[daily] Processing",
            "[weekly] Processing",
            "[daily ",
            "[weekly ",
        ],
    )
    markers = [
        ln
        for ln in tail
        if (
            "[START] daily_update.sh" in ln
            or "[YOLO]" in ln
            or "[daily] Processing" in ln
            or "[weekly] Processing" in ln
            or "[daily " in ln
            or "[weekly " in ln
            or "[REPORT]" in ln
            or "[DONE]  daily_update.sh" in ln
            or "[WEEKLY]" in ln
            or "run_id=" in ln
        )
    ]
    out: dict[str, Any] = {
        "active": False,
        "stage": "unknown" if start_idx < 0 else "idle",
        "stage_line": None,
        "start_line": tail[start_idx] if start_idx >= 0 else None,
        "done_line": tail[done_idx] if done_idx >= 0 else None,
        "marker_lines": markers[-40:],
    }
    if start_idx < 0:
        weekly_report_terminal_idx = max(weekly_report_done_idx, weekly_report_failed_idx)
        weekly_yolo_terminal_idx = max(weekly_yolo_done_idx, weekly_yolo_failed_idx, weekly_report_terminal_idx)

        if weekly_report_start_idx >= 0 and weekly_report_start_idx > weekly_report_terminal_idx:
            out["active"] = True
            out["stage"] = "report"
            out["stage_line"] = tail[weekly_report_start_idx]
            return out
        if weekly_start_idx >= 0 and weekly_start_idx > weekly_yolo_terminal_idx:
            out["active"] = True
            out["stage"] = "yolo"
            out["stage_line"] = tail[yolo_progress_idx] if yolo_progress_idx > weekly_start_idx else tail[weekly_start_idx]
            return out
        if report_start_idx >= 0 and report_start_idx > report_done_idx:
            out["active"] = True
            out["stage"] = "report"
            out["stage_line"] = tail[report_start_idx]
            return out
        if yolo_progress_idx >= 0 and yolo_progress_idx > yolo_done_idx:
            out["active"] = True
            out["stage"] = "yolo"
            out["stage_line"] = tail[yolo_progress_idx]
            return out
        return out
    if done_idx > start_idx:
        return out

    block = tail[start_idx:]
    out["active"] = True

    report_start = _last_idx(block, "[REPORT] Generating")
    report_done = _last_idx(block, "[REPORT] Done.")
    yolo_start = _last_idx(block, "[YOLO]  Starting")
    yolo_done = _last_idx(block, "[YOLO]  Daily pattern detection done.")
    yolo_progress = _last_idx_any(
        block,
        [
            "[daily] Processing",
            "[weekly] Processing",
            "[daily ",
            "[weekly ",
        ],
    )

    if report_start >= 0 and report_start > report_done:
        out["stage"] = "report"
        out["stage_line"] = block[report_start]
    elif (yolo_start >= 0 and yolo_start > yolo_done) or (yolo_progress >= 0 and yolo_progress > yolo_done):
        out["stage"] = "yolo"
        out["stage_line"] = block[yolo_progress] if yolo_progress >= 0 else block[yolo_start]
    else:
        out["stage"] = "ingest"
        out["stage_line"] = block[0] if block else None
    return out


def _infer_last_completed_event_from_tail(tail: list[str]) -> dict[str, Any] | None:
    """Find the most recent completion marker in *tail*."""
    candidates = [
        ("weekly_report", "ok", "[WEEKLY] Report regeneration completed"),
        ("weekly_report", "failed", "[WEEKLY] Report regeneration failed"),
        ("weekly_yolo", "ok", "[WEEKLY] Weekly YOLO completed OK"),
        ("weekly_yolo", "failed", "[WEEKLY] Weekly YOLO failed"),
        ("report", "ok", "[REPORT] Done."),
        ("report", "failed", "[REPORT] Failed to generate report"),
        ("daily_update", "ok", "[DONE]  daily_update.sh"),
        ("ingest", "failed", "[ERROR] ingest failed rc="),
    ]
    best_idx = -1
    best: dict[str, Any] | None = None
    for stage, status, token in candidates:
        for idx in range(len(tail) - 1, -1, -1):
            line = tail[idx]
            if token in line and idx > best_idx:
                ts = parse_log_line_ts_utc(line)
                best_idx = idx
                best = {
                    "stage": stage,
                    "status": status,
                    "line": line,
                    "ts": ts.replace(microsecond=0).isoformat() if ts else None,
                }
                break
    return best


# ---------------------------------------------------------------------------
# Pipeline snapshot
# ---------------------------------------------------------------------------

def pipeline_status_snapshot(log_lines: int = 160) -> dict[str, Any]:
    """Build the full pipeline-status dict from log tails + DB run data."""
    tail = _tail_text_file(RUN_LOG_PATH, lines=log_lines, max_bytes=256_000)
    inferred = _infer_pipeline_from_log_tail(tail)
    last_completed = _infer_last_completed_event_from_tail(tail)
    latest_run = read_latest_ingest_run()
    stage = inferred.get("stage", "unknown")
    active = bool(inferred.get("active"))
    stage_line = inferred.get("stage_line")
    stage_line_ts = parse_log_line_ts_utc(stage_line)
    stage_age_sec: float | None = None
    stale_inference = False
    now_utc = dt.datetime.now(dt.timezone.utc)
    if stage_line_ts is not None:
        stage_age_sec = max(0.0, (now_utc - stage_line_ts).total_seconds())
    running_age_sec: float | None = None
    running_stale = False
    if latest_run and latest_run.get("status") == "running":
        started = parse_iso_utc(latest_run.get("started_ts"))
        if started is not None:
            running_age_sec = max(0.0, (now_utc - started).total_seconds())
            running_stale = running_age_sec > (INGEST_RUNNING_STALE_MIN * 60)
    if latest_run and latest_run.get("status") == "running" and not running_stale:
        active = True
        stage = "ingest"
    elif latest_run and latest_run.get("status") == "running" and running_stale:
        stale_inference = True
        active = False
        stage = "stale_running"
    elif active and stage_age_sec is not None and stage_age_sec > PIPELINE_STALE_SEC:
        if not (latest_run and latest_run.get("status") == "running"):
            stale_inference = True
            active = False
            stage = "idle"
    if not active and stage != "stale_running":
        if last_completed is not None:
            stage = "idle"
        stage_line = None
        stage_line_ts = None
        stage_age_sec = None
    return {
        "run_log_path": str(RUN_LOG_PATH),
        "active": active,
        "stage": stage,
        "latest_run": latest_run,
        "markers": inferred.get("marker_lines", []),
        "stage_line": stage_line,
        "stage_line_ts": stage_line_ts.replace(microsecond=0).isoformat() if stage_line_ts else None,
        "stage_age_sec": round(stage_age_sec, 1) if stage_age_sec is not None else None,
        "stale_timeout_sec": PIPELINE_STALE_SEC,
        "stale_inference": stale_inference,
        "running_age_sec": round(running_age_sec, 1) if running_age_sec is not None else None,
        "running_stale_min": INGEST_RUNNING_STALE_MIN,
        "running_stale": running_stale,
        "last_completed_stage": last_completed.get("stage") if last_completed else None,
        "last_completed_status": last_completed.get("status") if last_completed else None,
        "last_completed_line": last_completed.get("line") if last_completed else None,
        "last_completed_ts": last_completed.get("ts") if last_completed else None,
        "tail": tail[-60:],
    }


# ---------------------------------------------------------------------------
# Stale-run reconciliation
# ---------------------------------------------------------------------------

def reconcile_stale_running_runs() -> dict[str, Any]:
    """Mark orphaned ``ingest_runs(status=running)`` as failed."""
    out: dict[str, Any] = {
        "checked": 0,
        "reconciled": 0,
        "run_ids": [],
    }
    if not DB_PATH.exists():
        return out

    now_utc = dt.datetime.now(dt.timezone.utc)
    stale_after_sec = INGEST_RUNNING_STALE_MIN * 60
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if not table_exists(conn, "ingest_runs"):
            return out

        rows = conn.execute(
            """
            SELECT run_id, started_ts, tickers_total, tickers_ok, tickers_failed
            FROM ingest_runs
            WHERE status = 'running'
            ORDER BY started_ts ASC
            """
        ).fetchall()
        out["checked"] = len(rows)
        has_ticker_status = table_exists(conn, "ingest_ticker_status")

        for row in rows:
            run_id = str(row["run_id"])
            started_ts = parse_iso_utc(row["started_ts"])
            if started_ts is None:
                run_age_sec = float(stale_after_sec + 1)
            else:
                run_age_sec = max(0.0, (now_utc - started_ts).total_seconds())
            if run_age_sec <= stale_after_sec:
                continue

            tickers_total = int(row["tickers_total"] or 0)
            tickers_ok = int(row["tickers_ok"] or 0)
            tickers_failed = int(row["tickers_failed"] or 0)
            processed = tickers_ok + tickers_failed
            if has_ticker_status:
                agg = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
                        COUNT(*) AS processed_count
                    FROM ingest_ticker_status
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if agg is not None:
                    tickers_ok = int(agg["ok_count"] or 0)
                    tickers_failed = int(agg["failed_count"] or 0)
                    processed = int(agg["processed_count"] or 0)

            unresolved = max(0, tickers_total - processed)
            tickers_failed += unresolved
            if tickers_ok + tickers_failed > tickers_total and tickers_total > 0:
                tickers_failed = max(0, tickers_total - tickers_ok)

            age_min = run_age_sec / 60.0
            finished_ts = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            reconcile_reason = (
                f"run_orphaned_after_restart: auto-failed after stale running "
                f"({age_min:.1f}m > {INGEST_RUNNING_STALE_MIN}m)"
            )
            updated = conn.execute(
                """
                UPDATE ingest_runs
                SET
                    finished_ts = ?,
                    status = 'failed',
                    tickers_ok = ?,
                    tickers_failed = ?,
                    error_message = COALESCE(NULLIF(error_message, ''), ?)
                WHERE run_id = ? AND status = 'running'
                """,
                (finished_ts, tickers_ok, tickers_failed, reconcile_reason, run_id),
            ).rowcount
            if not updated:
                continue

            out["reconciled"] += 1
            out["run_ids"].append(run_id)
            LOG.warning(
                "Reconciled stale running ingest run_id=%s age_min=%.1f total=%s processed=%s",
                run_id,
                age_min,
                tickers_total,
                processed,
            )
            _append_run_log(
                "RECOVER",
                (
                    f"run_id={run_id} stale-running auto-failed "
                    f"age_min={age_min:.1f} total={tickers_total} processed={processed}"
                ),
            )

        if out["reconciled"] > 0:
            conn.commit()
            invalidate_status_cache()
    except Exception:
        LOG.exception("Failed to reconcile stale running ingest runs")
    finally:
        conn.close()
    return out


# ---------------------------------------------------------------------------
# Post-ingest resume helpers
# ---------------------------------------------------------------------------

def _count_post_ingest_resume_attempts(
    run_id: str,
    *,
    source: str = "startup_resume",
) -> int:
    if not run_id:
        return 0
    tail = _tail_text_file(RUN_LOG_PATH, lines=5000, max_bytes=1_000_000)
    needle_source = f"source={source}"
    needle_run = f"run_id={run_id}"
    return sum(
        1
        for line in tail
        if "[RESUME]" in line and "queued mode=yolo" in line and needle_source in line and needle_run in line
    )


def post_ingest_resume_candidate(
    *,
    latest_run: dict[str, Any] | None = None,
    pipeline_active: bool | None = None,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any] | None:
    """Evaluate whether a post-ingest resume (yolo+report) should run."""
    if not AUTO_RESUME_POST_INGEST:
        return None

    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    latest_run = latest_run or read_latest_ingest_run()
    if not latest_run or str(latest_run.get("status") or "").lower() != "ok":
        return None

    run_finished_ts = parse_iso_utc(latest_run.get("finished_ts"))
    if run_finished_ts is None:
        return None
    run_age_sec = max(0.0, (now_utc - run_finished_ts).total_seconds())
    if run_age_sec > (AUTO_RESUME_MAX_AGE_HOURS * 3600):
        return None

    if pipeline_active:
        return None

    _, latest_payload = latest_daily_report_json(REPORT_DIR)
    generated_ts = parse_iso_utc((latest_payload or {}).get("generated_ts")) if latest_payload else None
    if generated_ts is not None and generated_ts >= (run_finished_ts - dt.timedelta(seconds=60)):
        return None

    reason = (
        "report_missing_after_completed_ingest"
        if generated_ts is None
        else "report_stale_after_completed_ingest"
    )
    run_id = str(latest_run.get("run_id") or "")
    resume_attempts = _count_post_ingest_resume_attempts(run_id, source="startup_resume")
    eligible = resume_attempts < AUTO_RESUME_MAX_RETRIES
    return {
        "mode": "yolo",
        "reason": reason,
        "eligible": eligible,
        "resume_attempts": resume_attempts,
        "max_retries": AUTO_RESUME_MAX_RETRIES,
        "blocked_reason": None if eligible else "max_retries_reached",
        "latest_run": latest_run,
        "run_finished_ts": run_finished_ts.replace(microsecond=0).isoformat(),
        "report_generated_ts": generated_ts.replace(microsecond=0).isoformat() if generated_ts else None,
        "run_age_sec": round(run_age_sec, 1),
        "max_age_hours": AUTO_RESUME_MAX_AGE_HOURS,
    }


def queue_post_ingest_resume(
    scheduler: Any,
    source: str = "startup",
) -> dict[str, Any]:
    """Conditionally schedule a post-ingest resume job on *scheduler*.

    Parameters
    ----------
    scheduler:
        APScheduler ``BackgroundScheduler`` instance.
    source:
        Tag for the run-log entry.
    """
    now_utc = dt.datetime.now(dt.timezone.utc)
    pipeline = pipeline_status_snapshot(log_lines=120)
    candidate = post_ingest_resume_candidate(
        latest_run=pipeline.get("latest_run"),
        pipeline_active=bool(pipeline.get("active")),
        now_utc=now_utc,
    )
    if candidate is None:
        return {"scheduled": False}
    if not candidate.get("eligible", True):
        LOG.warning(
            "Post-ingest resume not queued: mode=%s reason=%s run_id=%s retries=%s/%s",
            candidate.get("mode"),
            candidate.get("reason"),
            ((candidate.get("latest_run") or {}).get("run_id") or "-"),
            candidate.get("resume_attempts"),
            candidate.get("max_retries"),
        )
        return {"scheduled": False, **candidate}

    job_id = f"resume_daily_update_{now_utc.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(3)}"
    scheduler.add_job(
        _run_daily_update,
        trigger="date",
        run_date=now_utc,
        id=job_id,
        kwargs={"mode": str(candidate.get("mode") or "yolo"), "source": source},
    )
    _append_run_log(
        "RESUME",
        (
            f"queued mode={candidate.get('mode')} source={source} "
            f"reason={candidate.get('reason')} run_id={((candidate.get('latest_run') or {}).get('run_id') or '-')}"
        ),
    )
    LOG.warning(
        "Queued post-ingest resume mode=%s source=%s reason=%s run_id=%s",
        candidate.get("mode"),
        source,
        candidate.get("reason"),
        ((candidate.get("latest_run") or {}).get("run_id") or "-"),
    )
    return {"scheduled": True, "job_id": job_id, **candidate}

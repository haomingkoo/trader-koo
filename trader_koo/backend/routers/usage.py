"""Usage and feedback endpoints: session tracking, setup feedback."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from trader_koo.backend.services.database import DB_PATH, get_conn, table_exists

router = APIRouter()

LOG = logging.getLogger("trader_koo.routers.usage")

ANALYTICS_ENABLED = str(os.getenv("TRADER_KOO_ANALYTICS_ENABLED", "1")).strip().lower() in {
    "1", "true", "yes", "on",
}
ANALYTICS_MAX_SESSION_AGE_DAYS = max(7, int(os.getenv("TRADER_KOO_ANALYTICS_MAX_SESSION_AGE_DAYS", "180")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _clamp_int(value: Any, *, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return minimum
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _clean_session_text(value: Any, *, max_len: int = 120) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:max_len]


def _clean_feedback_text(value: Any, *, max_len: int = 400) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:max_len]


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "-"


def prune_analytics_sessions() -> None:
    if not DB_PATH.exists():
        return
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=ANALYTICS_MAX_SESSION_AGE_DAYS)).isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            "DELETE FROM ui_usage_sessions WHERE COALESCE(last_seen_ts, started_ts, created_ts) < ?",
            (cutoff,),
        )
        conn.commit()
    finally:
        conn.close()


def ensure_analytics_schema() -> None:
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_usage_sessions (
                session_id TEXT PRIMARY KEY,
                visitor_id TEXT NOT NULL,
                started_ts TEXT,
                last_seen_ts TEXT,
                active_ms INTEGER NOT NULL DEFAULT 0,
                page_views_total INTEGER NOT NULL DEFAULT 0,
                guide_views INTEGER NOT NULL DEFAULT 0,
                report_views INTEGER NOT NULL DEFAULT 0,
                earnings_views INTEGER NOT NULL DEFAULT 0,
                chart_views INTEGER NOT NULL DEFAULT 0,
                opportunities_views INTEGER NOT NULL DEFAULT 0,
                chart_loads INTEGER NOT NULL DEFAULT 0,
                last_tab TEXT,
                last_ticker TEXT,
                market TEXT,
                path TEXT,
                tz TEXT,
                created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ui_usage_sessions_visitor ON ui_usage_sessions(visitor_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ui_usage_sessions_last_seen ON ui_usage_sessions(last_seen_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ui_usage_sessions_last_ticker ON ui_usage_sessions(last_ticker)"
        )
        conn.commit()
    finally:
        conn.close()


def ensure_feedback_schema() -> None:
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS setup_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                asof TEXT,
                verdict TEXT NOT NULL CHECK (verdict IN ('good', 'bad', 'neutral')),
                source_surface TEXT,
                note TEXT,
                setup_tier TEXT,
                setup_score REAL,
                setup_family TEXT,
                signal_bias TEXT,
                actionability TEXT,
                yolo_role TEXT,
                yolo_recency TEXT,
                visitor_id TEXT,
                session_id TEXT,
                client_ip TEXT,
                user_agent TEXT,
                context_json TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_setup_feedback_created ON setup_feedback(created_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_setup_feedback_ticker ON setup_feedback(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_setup_feedback_verdict ON setup_feedback(verdict)"
        )
        conn.commit()
    finally:
        conn.close()


def _upsert_usage_session(conn: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    if not table_exists(conn, "ui_usage_sessions"):
        ensure_analytics_schema()
    session_id = _clean_session_text(payload.get("session_id"), max_len=64)
    visitor_id = _clean_session_text(payload.get("visitor_id"), max_len=64)
    if not session_id or not visitor_id:
        raise HTTPException(status_code=400, detail="session_id and visitor_id are required")
    started_ts = _clean_session_text(payload.get("started_ts"), max_len=40) or _utc_now_iso()
    last_seen_ts = _clean_session_text(payload.get("last_seen_ts"), max_len=40) or _utc_now_iso()
    active_ms = _clamp_int(payload.get("active_ms"), maximum=31_536_000_000)
    page_views_total = _clamp_int(payload.get("page_views_total"), maximum=1_000_000)
    guide_views = _clamp_int(payload.get("guide_views"), maximum=1_000_000)
    report_views = _clamp_int(payload.get("report_views"), maximum=1_000_000)
    earnings_views = _clamp_int(payload.get("earnings_views"), maximum=1_000_000)
    chart_views = _clamp_int(payload.get("chart_views"), maximum=1_000_000)
    opportunities_views = _clamp_int(payload.get("opportunities_views"), maximum=1_000_000)
    chart_loads = _clamp_int(payload.get("chart_loads"), maximum=1_000_000)
    last_tab = _clean_session_text(payload.get("last_tab"), max_len=32)
    last_ticker = _clean_session_text(payload.get("last_ticker"), max_len=24)
    market = _clean_session_text(payload.get("market"), max_len=24)
    path = _clean_session_text(payload.get("path"), max_len=200)
    tz = _clean_session_text(payload.get("tz"), max_len=64)
    conn.execute(
        """
        INSERT INTO ui_usage_sessions (
            session_id, visitor_id, started_ts, last_seen_ts,
            active_ms, page_views_total, guide_views, report_views,
            earnings_views, chart_views, opportunities_views, chart_loads,
            last_tab, last_ticker, market, path, tz,
            created_ts, updated_ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            visitor_id = excluded.visitor_id,
            started_ts = COALESCE(ui_usage_sessions.started_ts, excluded.started_ts),
            last_seen_ts = excluded.last_seen_ts,
            active_ms = MAX(ui_usage_sessions.active_ms, excluded.active_ms),
            page_views_total = MAX(ui_usage_sessions.page_views_total, excluded.page_views_total),
            guide_views = MAX(ui_usage_sessions.guide_views, excluded.guide_views),
            report_views = MAX(ui_usage_sessions.report_views, excluded.report_views),
            earnings_views = MAX(ui_usage_sessions.earnings_views, excluded.earnings_views),
            chart_views = MAX(ui_usage_sessions.chart_views, excluded.chart_views),
            opportunities_views = MAX(ui_usage_sessions.opportunities_views, excluded.opportunities_views),
            chart_loads = MAX(ui_usage_sessions.chart_loads, excluded.chart_loads),
            last_tab = COALESCE(excluded.last_tab, ui_usage_sessions.last_tab),
            last_ticker = COALESCE(excluded.last_ticker, ui_usage_sessions.last_ticker),
            market = COALESCE(excluded.market, ui_usage_sessions.market),
            path = COALESCE(excluded.path, ui_usage_sessions.path),
            tz = COALESCE(excluded.tz, ui_usage_sessions.tz),
            updated_ts = excluded.updated_ts
        """,
        (
            session_id, visitor_id, started_ts, last_seen_ts,
            active_ms, page_views_total, guide_views, report_views,
            earnings_views, chart_views, opportunities_views, chart_loads,
            last_tab, last_ticker, market, path, tz,
            _utc_now_iso(), _utc_now_iso(),
        ),
    )
    conn.commit()
    return {
        "ok": True,
        "session_id": session_id,
        "visitor_id": visitor_id,
        "active_ms": active_ms,
        "page_views_total": page_views_total,
        "chart_loads": chart_loads,
    }


def _record_setup_feedback(conn: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    if not table_exists(conn, "setup_feedback"):
        ensure_feedback_schema()
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in {"good", "bad", "neutral"}:
        raise HTTPException(status_code=400, detail="verdict must be one of: good, bad, neutral")
    ticker = str(payload.get("ticker") or "").strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    if len(ticker) > 16:
        raise HTTPException(status_code=400, detail="ticker is too long")
    context_json = payload.get("context_json")
    context_text = None
    if isinstance(context_json, (dict, list)):
        try:
            context_text = json.dumps(context_json, ensure_ascii=True)[:2000]
        except (TypeError, ValueError) as exc:
            LOG.debug("Failed to serialize context_json for feedback: %s", exc)
            context_text = None
    elif context_json is not None:
        context_text = _clean_feedback_text(context_json, max_len=2000)
    row = {
        "ticker": ticker,
        "asof": _clean_feedback_text(payload.get("asof"), max_len=24),
        "verdict": verdict,
        "source_surface": _clean_feedback_text(payload.get("source_surface"), max_len=32),
        "note": _clean_feedback_text(payload.get("note"), max_len=800),
        "setup_tier": _clean_feedback_text(payload.get("setup_tier"), max_len=4),
        "setup_score": float(payload.get("setup_score")) if payload.get("setup_score") not in (None, "") else None,
        "setup_family": _clean_feedback_text(payload.get("setup_family"), max_len=48),
        "signal_bias": _clean_feedback_text(payload.get("signal_bias"), max_len=24),
        "actionability": _clean_feedback_text(payload.get("actionability"), max_len=24),
        "yolo_role": _clean_feedback_text(payload.get("yolo_role"), max_len=32),
        "yolo_recency": _clean_feedback_text(payload.get("yolo_recency"), max_len=32),
        "visitor_id": _clean_feedback_text(payload.get("visitor_id"), max_len=64),
        "session_id": _clean_feedback_text(payload.get("session_id"), max_len=64),
        "client_ip": _clean_feedback_text(payload.get("client_ip"), max_len=96),
        "user_agent": _clean_feedback_text(payload.get("user_agent"), max_len=300),
        "context_json": context_text,
    }
    conn.execute(
        """
        INSERT INTO setup_feedback (
            ticker, asof, verdict, source_surface, note,
            setup_tier, setup_score, setup_family, signal_bias, actionability,
            yolo_role, yolo_recency, visitor_id, session_id, client_ip, user_agent, context_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["ticker"], row["asof"], row["verdict"], row["source_surface"], row["note"],
            row["setup_tier"], row["setup_score"], row["setup_family"], row["signal_bias"], row["actionability"],
            row["yolo_role"], row["yolo_recency"], row["visitor_id"], row["session_id"],
            row["client_ip"], row["user_agent"], row["context_json"],
        ),
    )
    conn.commit()
    return {"ok": True, "ticker": row["ticker"], "verdict": verdict}


def _usage_summary(conn: sqlite3.Connection, days: int = 7, limit: int = 10) -> dict[str, Any]:
    days = max(1, min(365, int(days)))
    limit = max(1, min(100, int(limit)))
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS sessions,
            COUNT(DISTINCT visitor_id) AS visitors,
            COALESCE(SUM(active_ms), 0) AS active_ms_total,
            COALESCE(AVG(active_ms), 0) AS active_ms_avg,
            COALESCE(SUM(page_views_total), 0) AS page_views_total,
            COALESCE(SUM(chart_loads), 0) AS chart_loads_total
        FROM ui_usage_sessions
        WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
        """,
        (cutoff,),
    ).fetchone()
    daily_rows = conn.execute(
        """
        SELECT
            substr(COALESCE(last_seen_ts, started_ts, created_ts), 1, 10) AS day,
            COUNT(*) AS sessions,
            COUNT(DISTINCT visitor_id) AS visitors,
            COALESCE(SUM(active_ms), 0) AS active_ms_total,
            COALESCE(SUM(page_views_total), 0) AS page_views_total,
            COALESCE(SUM(chart_loads), 0) AS chart_loads_total
        FROM ui_usage_sessions
        WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
        GROUP BY day
        ORDER BY day DESC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()
    tab_totals = []
    tab_columns = {
        "guide": "guide_views",
        "report": "report_views",
        "earnings": "earnings_views",
        "chart": "chart_views",
        "opportunities": "opportunities_views",
    }
    for tab, column in tab_columns.items():
        row = conn.execute(
            f"""
            SELECT
                COALESCE(SUM({column}), 0) AS views,
                COUNT(CASE WHEN {column} > 0 THEN 1 END) AS sessions
            FROM ui_usage_sessions
            WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
            """,
            (cutoff,),
        ).fetchone()
        tab_totals.append({
            "tab": tab,
            "views": int(row["views"] or 0),
            "sessions": int(row["sessions"] or 0),
        })
    tab_totals.sort(key=lambda row: (-row["views"], row["tab"]))
    top_tickers = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                last_ticker AS ticker,
                COUNT(*) AS sessions,
                COALESCE(SUM(chart_loads), 0) AS chart_loads,
                COALESCE(SUM(active_ms), 0) AS active_ms_total
            FROM ui_usage_sessions
            WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
              AND last_ticker IS NOT NULL AND last_ticker != ''
            GROUP BY last_ticker
            ORDER BY chart_loads DESC, sessions DESC, ticker ASC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    ]
    top_paths = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                path,
                COUNT(*) AS sessions,
                COUNT(DISTINCT visitor_id) AS visitors,
                COALESCE(SUM(page_views_total), 0) AS page_views_total,
                COALESCE(SUM(chart_loads), 0) AS chart_loads_total
            FROM ui_usage_sessions
            WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
              AND path IS NOT NULL AND path != ''
            GROUP BY path
            ORDER BY page_views_total DESC, sessions DESC, path ASC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    ]
    recent_sessions = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                session_id, visitor_id, started_ts, last_seen_ts,
                active_ms, page_views_total, chart_loads, last_tab, last_ticker
            FROM ui_usage_sessions
            WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
            ORDER BY COALESCE(last_seen_ts, started_ts, created_ts) DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    ]
    return {
        "ok": True,
        "days": days,
        "cutoff_ts": cutoff,
        "totals": {
            "sessions": int(totals["sessions"] or 0),
            "visitors": int(totals["visitors"] or 0),
            "active_hours_total": round(float(totals["active_ms_total"] or 0) / 3_600_000.0, 2),
            "avg_active_min_per_session": round(float(totals["active_ms_avg"] or 0) / 60_000.0, 2),
            "page_views_total": int(totals["page_views_total"] or 0),
            "chart_loads_total": int(totals["chart_loads_total"] or 0),
        },
        "top_tabs": tab_totals[:limit],
        "top_tickers": [
            {**row, "active_hours_total": round(float(row.get("active_ms_total") or 0) / 3_600_000.0, 2)}
            for row in top_tickers
        ],
        "top_paths": [
            {
                **row,
                "path": str(row.get("path") or ""),
                "sessions": int(row.get("sessions") or 0),
                "visitors": int(row.get("visitors") or 0),
                "page_views_total": int(row.get("page_views_total") or 0),
                "chart_loads_total": int(row.get("chart_loads_total") or 0),
            }
            for row in top_paths
        ],
        "top_pages": [
            {
                **row,
                "path": str(row.get("path") or ""),
                "sessions": int(row.get("sessions") or 0),
                "visitors": int(row.get("visitors") or 0),
                "page_views_total": int(row.get("page_views_total") or 0),
                "chart_loads_total": int(row.get("chart_loads_total") or 0),
            }
            for row in top_paths
        ],
        "daily": [
            {
                "day": row["day"],
                "sessions": int(row["sessions"] or 0),
                "visitors": int(row["visitors"] or 0),
                "active_hours_total": round(float(row["active_ms_total"] or 0) / 3_600_000.0, 2),
                "page_views_total": int(row["page_views_total"] or 0),
                "chart_loads_total": int(row["chart_loads_total"] or 0),
            }
            for row in daily_rows
        ],
        "recent_sessions": [
            {**row, "active_min": round(float(row.get("active_ms") or 0) / 60_000.0, 2)}
            for row in recent_sessions
        ],
    }


def _feedback_summary(conn: sqlite3.Connection, days: int = 30) -> dict[str, Any]:
    if not table_exists(conn, "setup_feedback"):
        return {
            "days": days,
            "total": 0,
            "good": 0,
            "bad": 0,
            "neutral": 0,
            "good_rate_pct": None,
        }
    days = max(1, min(365, int(days)))
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN verdict='good' THEN 1 ELSE 0 END) AS good,
            SUM(CASE WHEN verdict='bad' THEN 1 ELSE 0 END) AS bad,
            SUM(CASE WHEN verdict='neutral' THEN 1 ELSE 0 END) AS neutral
        FROM setup_feedback
        WHERE created_ts >= ?
        """,
        (cutoff,),
    ).fetchone()
    total = int(row["total"] or 0) if row else 0
    good = int(row["good"] or 0) if row else 0
    bad = int(row["bad"] or 0) if row else 0
    neutral = int(row["neutral"] or 0) if row else 0
    return {
        "days": days,
        "total": total,
        "good": good,
        "bad": bad,
        "neutral": neutral,
        "good_rate_pct": round((good * 100.0) / total, 2) if total else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/usage/session", include_in_schema=False)
async def usage_session(request: Request) -> dict[str, Any]:
    if not ANALYTICS_ENABLED:
        return {"ok": True, "analytics_enabled": False}
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid analytics payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Analytics payload must be an object")
    conn = get_conn()
    try:
        result = _upsert_usage_session(conn, payload)
    finally:
        conn.close()
    return {
        **result,
        "analytics_enabled": True,
    }


@router.post("/api/feedback/setup")
async def setup_feedback(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")
    if not payload.get("client_ip"):
        payload["client_ip"] = _client_ip(request)
    if not payload.get("user_agent"):
        payload["user_agent"] = request.headers.get("user-agent")
    conn = get_conn()
    try:
        result = _record_setup_feedback(conn, payload)
        summary = _feedback_summary(conn, days=30)
    finally:
        conn.close()
    return {"ok": True, "result": result, "summary": summary}

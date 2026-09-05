"""Analytics and feedback table schemas.

The HTTP endpoints that wrote these tables were removed: they were
unauthenticated writes to the production database with no caller in the
frontend bundle. main.py still calls the three schema functions here at
startup, so the tables and their existing rows are left untouched.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3

from trader_koo.backend.services.database import DB_PATH, get_conn, table_exists

LOG = logging.getLogger("trader_koo.routers.usage")

ANALYTICS_ENABLED = str(os.getenv("TRADER_KOO_ANALYTICS_ENABLED", "1")).strip().lower() in {
    "1", "true", "yes", "on",
}
ANALYTICS_MAX_SESSION_AGE_DAYS = max(7, int(os.getenv("TRADER_KOO_ANALYTICS_MAX_SESSION_AGE_DAYS", "180")))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

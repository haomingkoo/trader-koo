"""Market data fetching and processing helpers.

Functions shared by multiple endpoints for retrieving YOLO status,
ingest-run metadata, and data-source information from the database.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trader_koo.backend.services.database import DB_PATH, table_exists

LOG = logging.getLogger("trader_koo.services.market_data")

_MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")
try:
    MARKET_TZ = ZoneInfo(_MARKET_TZ_NAME)
except Exception:
    MARKET_TZ = dt.timezone.utc

MARKET_CLOSE_HOUR = min(23, max(0, int(os.getenv("TRADER_KOO_MARKET_CLOSE_HOUR", "16"))))


# ---------------------------------------------------------------------------
# Time / age helpers
# ---------------------------------------------------------------------------

def parse_iso_utc(value: str | None) -> dt.datetime | None:
    """Parse an ISO-8601 string into a UTC-aware datetime."""
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def hours_since(ts: str | None, now: dt.datetime) -> float | None:
    """Return hours elapsed since *ts*, or None if unparsable."""
    parsed = parse_iso_utc(ts)
    if parsed is None:
        return None
    return (now - parsed).total_seconds() / 3600.0


def days_since(date_str: str | None, now: dt.datetime) -> float | None:
    """Return calendar days since a market date string."""
    if not date_str:
        return None
    try:
        market_date = dt.date.fromisoformat(str(date_str).strip()[:10])
    except ValueError:
        return None
    market_close = dt.datetime.combine(
        market_date,
        dt.time(hour=MARKET_CLOSE_HOUR),
        tzinfo=MARKET_TZ,
    )
    now_market = now.astimezone(MARKET_TZ)
    age_days = (now_market - market_close).total_seconds() / 86400.0
    return max(0.0, age_days)


# ---------------------------------------------------------------------------
# YOLO status
# ---------------------------------------------------------------------------

def get_yolo_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return aggregated YOLO patterns status (table stats, latest run, events)."""
    out: dict[str, Any] = {
        "table_exists": table_exists(conn, "yolo_patterns"),
        "events_table_exists": table_exists(conn, "yolo_run_events"),
        "universe_tickers": 0,
        "summary": {},
        "timeframes": [],
        "latest_run": None,
        "latest_non_ok_events": [],
    }
    if not out["table_exists"]:
        return out

    universe_row = conn.execute("SELECT COUNT(DISTINCT ticker) AS c FROM price_daily").fetchone()
    out["universe_tickers"] = int(universe_row["c"] or 0) if universe_row else 0

    summary = conn.execute(
        """
        SELECT
            COUNT(*) AS rows_total,
            COUNT(DISTINCT ticker) AS tickers_total,
            MAX(detected_ts) AS latest_detected_ts,
            MAX(as_of_date) AS latest_asof_date
        FROM yolo_patterns
        """
    ).fetchone()
    out["summary"] = dict(summary) if summary is not None else {}

    tf_rows = conn.execute(
        """
        SELECT
            timeframe,
            COUNT(*) AS rows_total,
            COUNT(DISTINCT ticker) AS tickers_total,
            MAX(detected_ts) AS latest_detected_ts,
            MAX(as_of_date) AS latest_asof_date,
            AVG(confidence) AS avg_confidence
        FROM yolo_patterns
        GROUP BY timeframe
        ORDER BY timeframe
        """
    ).fetchall()
    out["timeframes"] = [dict(r) for r in tf_rows]

    if out["events_table_exists"]:
        latest_run = conn.execute(
            """
            SELECT run_id, MAX(created_ts) AS latest_ts
            FROM yolo_run_events
            GROUP BY run_id
            ORDER BY latest_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if latest_run is not None:
            run_id = latest_run["run_id"]
            run_stats = conn.execute(
                """
                SELECT
                    run_id,
                    MIN(created_ts) AS started_ts,
                    MAX(created_ts) AS latest_ts,
                    COUNT(*) AS events_total,
                    SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok_count,
                    SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skipped_count,
                    SUM(CASE WHEN status='timeout' THEN 1 ELSE 0 END) AS timeout_count,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count
                FROM yolo_run_events
                WHERE run_id = ?
                GROUP BY run_id
                """,
                (run_id,),
            ).fetchone()
            if run_stats is not None:
                out["latest_run"] = dict(run_stats)

            events = conn.execute(
                """
                SELECT run_id, timeframe, ticker, status, reason, elapsed_sec, bars, detections, as_of_date, created_ts
                FROM yolo_run_events
                WHERE run_id = ? AND status != 'ok'
                ORDER BY created_ts DESC
                LIMIT 100
                """,
                (run_id,),
            ).fetchall()
            out["latest_non_ok_events"] = [dict(r) for r in events]
    return out


# ---------------------------------------------------------------------------
# Data source info
# ---------------------------------------------------------------------------

def get_data_sources(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """Get data source information for a ticker.

    Returns a dict with the latest data_source and fetch_timestamp.
    """
    try:
        row = conn.execute(
            """
            SELECT data_source, fetch_timestamp
            FROM price_daily
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()

        if row:
            return {
                "price": row[0] or "unknown",
                "price_timestamp": row[1] or None,
            }
    except Exception as exc:
        LOG.warning("Failed to get data sources for %s: %s", ticker, exc)

    return {
        "price": "unknown",
        "price_timestamp": None,
    }


# ---------------------------------------------------------------------------
# Ingest run helpers
# ---------------------------------------------------------------------------

def read_latest_ingest_run(db_path: Path | None = None) -> dict[str, Any] | None:
    """Read the most recent row from ``ingest_runs``."""
    path = db_path or DB_PATH
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if not table_exists(conn, "ingest_runs"):
            return None
        row = conn.execute(
            """
            SELECT
                run_id, started_ts, finished_ts, status,
                tickers_total, tickers_ok, tickers_failed, error_message
            FROM ingest_runs
            ORDER BY started_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        ticker_status_count: int | None = None
        if table_exists(conn, "ingest_ticker_status"):
            ts_row = conn.execute(
                "SELECT COUNT(*) AS c FROM ingest_ticker_status WHERE run_id = ?",
                (out["run_id"],),
            ).fetchone()
            ticker_status_count = int(ts_row["c"] or 0) if ts_row else 0

        if ticker_status_count is not None:
            out["tickers_processed"] = ticker_status_count
        elif out.get("status") in {"ok", "failed"}:
            completed = int(out.get("tickers_ok") or 0) + int(out.get("tickers_failed") or 0)
            out["tickers_processed"] = completed or int(out.get("tickers_total") or 0)
        return out
    except Exception:
        return None
    finally:
        conn.close()

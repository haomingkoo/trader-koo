"""Database connection helpers and raw SQL query functions.

Provides the shared DB connection factory, path constants, and reusable
query helpers used across multiple endpoints (price data, fundamentals,
options, YOLO patterns, YOLO audit, fund snapshot selection).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from trader_koo.db.schema import ensure_ohlcv_schema
from trader_koo.scripts.generate_daily_report import (
    _yolo_age_factor as _report_yolo_age_factor,
    _yolo_recency_label as _report_yolo_recency_label,
)

LOG = logging.getLogger("trader_koo.services.database")

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PRIMARY = (PROJECT_DIR / "data" / "trader_koo.db").resolve()
DB_PATH = Path(os.getenv("TRADER_KOO_DB_PATH", str(DEFAULT_DB_PRIMARY)))


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a SQLite connection with Row factory enabled.

    Raises HTTPException(500) when the DB file does not exist.
    """
    path = db_path or DB_PATH
    if not path.exists():
        LOG.error("Database file not found at %s", path)
        raise HTTPException(status_code=500, detail="Database unavailable")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check whether *table_name* exists in the connected database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def get_price_df(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Load OHLCV price history for *ticker* as a DataFrame."""
    df = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume
        FROM price_daily
        WHERE ticker = ?
        ORDER BY date
        """,
        conn,
        params=(ticker,),
    )
    return ensure_ohlcv_schema(df)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def get_latest_fundamentals(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """Return the most recent Finviz fundamentals row for *ticker*."""
    row = conn.execute(
        """
        SELECT *
        FROM finviz_fundamentals
        WHERE ticker = ?
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    return dict(row) if row is not None else {}


# ---------------------------------------------------------------------------
# Options summary
# ---------------------------------------------------------------------------

def get_latest_options_summary(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """Aggregate the latest options-IV snapshot into a summary dict."""
    latest = conn.execute(
        "SELECT snapshot_ts FROM options_iv WHERE ticker = ? ORDER BY snapshot_ts DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if latest is None:
        return {}
    snap = latest["snapshot_ts"]
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS contracts,
            AVG(implied_vol) AS avg_iv,
            SUM(CASE WHEN option_type='call' THEN open_interest ELSE 0 END) AS call_oi,
            SUM(CASE WHEN option_type='put' THEN open_interest ELSE 0 END) AS put_oi
        FROM options_iv
        WHERE ticker = ? AND snapshot_ts = ?
        """,
        (ticker, snap),
    ).fetchone()
    if row is None:
        return {}

    call_oi = float(row["call_oi"] or 0.0)
    put_oi = float(row["put_oi"] or 0.0)
    put_call = (put_oi / call_oi) if call_oi > 0 else None
    return {
        "snapshot_ts": snap,
        "contracts": int(row["contracts"] or 0),
        "avg_iv": float(row["avg_iv"]) if row["avg_iv"] is not None else None,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "put_call_oi_ratio": float(put_call) if put_call is not None else None,
    }


# ---------------------------------------------------------------------------
# YOLO pattern helpers (shared across modules)
# ---------------------------------------------------------------------------

def _parse_iso_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


def _yolo_match_tolerance_days(timeframe: str) -> int:
    return 35 if str(timeframe or "").strip().lower() == "weekly" else 14


def _yolo_snapshot_matches(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Return True when two YOLO snapshots describe the same pattern instance."""
    if str(anchor.get("ticker") or "") != str(candidate.get("ticker") or ""):
        return False
    if str(anchor.get("timeframe") or "") != str(candidate.get("timeframe") or ""):
        return False
    if str(anchor.get("pattern") or "") != str(candidate.get("pattern") or ""):
        return False
    tolerance = _yolo_match_tolerance_days(str(anchor.get("timeframe") or ""))
    anchor_x0 = _parse_iso_date(anchor.get("x0_date"))
    anchor_x1 = _parse_iso_date(anchor.get("x1_date"))
    cand_x0 = _parse_iso_date(candidate.get("x0_date"))
    cand_x1 = _parse_iso_date(candidate.get("x1_date"))
    if anchor_x0 is not None and cand_x0 is not None and abs((anchor_x0 - cand_x0).days) <= tolerance:
        return True
    if anchor_x1 is not None and cand_x1 is not None and abs((anchor_x1 - cand_x1).days) <= tolerance:
        return True
    return False


def _yolo_streak_for_asofs(
    seen_asofs: set[str],
    asof_dates_desc: list[str],
    latest_asof: str | None,
) -> int:
    """Count consecutive as-of dates where a pattern was seen."""
    if not latest_asof:
        return 0
    streak = 0
    started = False
    for asof in asof_dates_desc:
        if not started:
            if asof != latest_asof:
                continue
            started = True
        if asof in seen_asofs:
            streak += 1
        else:
            break
    return streak


def _yolo_priority_score(item: dict[str, Any], *, active_now: bool) -> float:
    age_factor = float(_report_yolo_age_factor(item.get("age_days"), item.get("timeframe")) or 0.0)
    streak = min(6, int(item.get("current_streak") or 0))
    confidence = float(item.get("confidence") or 0.0)
    timeframe_bonus = 3.0 if str(item.get("timeframe") or "").strip().lower() == "daily" else 0.0
    active_bonus = 8.0 if active_now else 0.0
    return round(
        (age_factor * 100.0) + (streak * 6.0) + (confidence * 10.0) + timeframe_bonus + active_bonus,
        1,
    )


def _yolo_signal_role(item: dict[str, Any], *, active_now: bool) -> str:
    age_factor = float(_report_yolo_age_factor(item.get("age_days"), item.get("timeframe")) or 0.0)
    streak = int(item.get("current_streak") or 0)
    if not active_now:
        return "historical_context"
    if age_factor >= 0.8 and (streak >= 2 or float(item.get("confidence") or 0.0) >= 0.6):
        return "primary"
    if age_factor >= 0.5 or streak >= 2:
        return "secondary"
    if age_factor > 0.0:
        return "recent_context"
    return "stale_context"


def _yolo_role_rank(role: str) -> int:
    return {
        "primary": 4,
        "secondary": 3,
        "recent_context": 2,
        "stale_context": 1,
        "historical_context": 0,
    }.get(str(role or ""), 0)


# ---------------------------------------------------------------------------
# YOLO patterns (latest per timeframe, enriched)
# ---------------------------------------------------------------------------

def get_yolo_patterns(conn: sqlite3.Connection, ticker: str) -> list[dict[str, Any]]:
    """Return enriched YOLO patterns for *ticker* (latest per timeframe)."""
    if not table_exists(conn, "yolo_patterns"):
        return []
    ticker_key = str(ticker or "").upper().strip()
    if not ticker_key:
        return []

    history_rows = conn.execute(
        """
        SELECT ticker, timeframe, pattern, x0_date, x1_date, as_of_date
        FROM yolo_patterns
        WHERE ticker = ?
          AND as_of_date IS NOT NULL
        ORDER BY as_of_date DESC
        """,
        (ticker_key,),
    ).fetchall()
    history_payload = [
        {
            "ticker": str(row["ticker"] or ""),
            "timeframe": str(row["timeframe"] or ""),
            "pattern": str(row["pattern"] or ""),
            "x0_date": row["x0_date"],
            "x1_date": row["x1_date"],
            "as_of_date": row["as_of_date"],
        }
        for row in history_rows
    ]
    asof_dates_desc: dict[str, list[str]] = {"daily": [], "weekly": []}
    for timeframe_key in ("daily", "weekly"):
        dates = {
            str(row.get("as_of_date") or "")
            for row in history_payload
            if str(row.get("timeframe") or "").strip().lower() == timeframe_key
            and str(row.get("as_of_date") or "").strip()
        }
        asof_dates_desc[timeframe_key] = sorted(dates, reverse=True)

    rows = conn.execute(
        """
        SELECT p.timeframe, p.pattern, p.confidence, p.x0_date, p.x1_date,
               p.y0, p.y1, p.lookback_days, p.as_of_date, p.detected_ts
        FROM yolo_patterns p
        JOIN (
            SELECT timeframe, MAX(as_of_date) AS latest_asof
            FROM yolo_patterns
            WHERE ticker = ?
            GROUP BY timeframe
        ) cur
          ON p.timeframe = cur.timeframe
         AND p.as_of_date = cur.latest_asof
        WHERE p.ticker = ?
        ORDER BY p.timeframe, p.confidence DESC
        """,
        (ticker_key, ticker_key),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["ticker"] = ticker_key
        age_days = None
        as_of_date = str(item.get("as_of_date") or "").strip()
        x1_date = str(item.get("x1_date") or "").strip()
        if len(as_of_date) >= 10 and len(x1_date) >= 10:
            try:
                asof_dt = dt.date.fromisoformat(as_of_date[:10])
                x1_dt = dt.date.fromisoformat(x1_date[:10])
                age_days = max(0, (asof_dt - x1_dt).days)
            except Exception:
                age_days = None
        item["age_days"] = age_days
        seen_asofs: set[str] = set()
        timeframe_key = str(item.get("timeframe") or "").strip().lower()
        for hist in history_payload:
            if _yolo_snapshot_matches(item, hist):
                hist_asof = str(hist.get("as_of_date") or "").strip()
                if hist_asof:
                    seen_asofs.add(hist_asof)
        if seen_asofs:
            seen_sorted = sorted(seen_asofs)
            item["first_seen_asof"] = seen_sorted[0]
            item["last_seen_asof"] = seen_sorted[-1]
            item["snapshots_seen"] = len(seen_asofs)
            item["current_streak"] = _yolo_streak_for_asofs(
                seen_asofs,
                asof_dates_desc.get(timeframe_key, []),
                as_of_date or None,
            )
        else:
            item["first_seen_asof"] = as_of_date or None
            item["last_seen_asof"] = as_of_date or None
            item["snapshots_seen"] = 1 if as_of_date else 0
            item["current_streak"] = 1 if as_of_date else 0
        item["yolo_recency"] = _report_yolo_recency_label(age_days, timeframe_key)
        item["signal_role"] = _yolo_signal_role(item, active_now=True)
        item["priority_score"] = _yolo_priority_score(item, active_now=True)
        out.append(item)
    out.sort(
        key=lambda item: (
            _yolo_role_rank(str(item.get("signal_role") or "")),
            float(item.get("priority_score") or 0.0),
            int(item.get("current_streak") or 0),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return out


# ---------------------------------------------------------------------------
# YOLO audit (grouped pattern history)
# ---------------------------------------------------------------------------

def get_yolo_audit(
    conn: sqlite3.Connection,
    ticker: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return grouped YOLO audit rows for *ticker*, sorted by relevance."""
    if not table_exists(conn, "yolo_patterns"):
        return []
    ticker_key = str(ticker or "").upper().strip()
    if not ticker_key:
        return []

    rows = conn.execute(
        """
        SELECT ticker, timeframe, pattern, confidence, x0_date, x1_date, as_of_date, detected_ts
        FROM yolo_patterns
        WHERE ticker = ?
          AND as_of_date IS NOT NULL
        ORDER BY timeframe, as_of_date DESC, confidence DESC
        """,
        (ticker_key,),
    ).fetchall()
    if not rows:
        return []

    history_payload = [dict(row) for row in rows]
    asof_dates_desc: dict[str, list[str]] = {"daily": [], "weekly": []}
    latest_asof_by_timeframe: dict[str, str | None] = {"daily": None, "weekly": None}
    latest_price_date = None
    if table_exists(conn, "price_daily"):
        latest_price_row = conn.execute(
            "SELECT MAX(date) AS latest_date FROM price_daily WHERE ticker = ?",
            (ticker_key,),
        ).fetchone()
        latest_price_date = str(latest_price_row["latest_date"] or "").strip() if latest_price_row else None
        if latest_price_date == "":
            latest_price_date = None
    for timeframe_key in ("daily", "weekly"):
        dates = {
            str(row.get("as_of_date") or "")
            for row in history_payload
            if str(row.get("timeframe") or "").strip().lower() == timeframe_key
            and str(row.get("as_of_date") or "").strip()
        }
        ordered_dates = sorted(dates, reverse=True)
        asof_dates_desc[timeframe_key] = ordered_dates
        latest_asof_by_timeframe[timeframe_key] = ordered_dates[0] if ordered_dates else None

    groups: list[dict[str, Any]] = []
    for row in history_payload:
        matched = None
        for group in groups:
            if str(group.get("timeframe") or "") != str(row.get("timeframe") or ""):
                continue
            if str(group.get("pattern") or "") != str(row.get("pattern") or ""):
                continue
            if _yolo_snapshot_matches(group["anchor"], row):
                matched = group
                break
        if matched is None:
            groups.append(
                {
                    "timeframe": str(row.get("timeframe") or ""),
                    "pattern": str(row.get("pattern") or ""),
                    "anchor": row,
                    "rows": [row],
                }
            )
        else:
            matched["rows"].append(row)

    audit_rows: list[dict[str, Any]] = []
    for group in groups:
        grouped_rows = list(group.get("rows") or [])
        if not grouped_rows:
            continue
        grouped_rows.sort(
            key=lambda item: (
                str(item.get("as_of_date") or ""),
                float(item.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        latest = grouped_rows[0]
        timeframe_key = str(latest.get("timeframe") or "").strip().lower()
        seen_asofs = {
            str(item.get("as_of_date") or "").strip()
            for item in grouped_rows
            if str(item.get("as_of_date") or "").strip()
        }
        seen_sorted = sorted(seen_asofs)
        first_seen = seen_sorted[0] if seen_sorted else None
        last_seen = seen_sorted[-1] if seen_sorted else None
        age_days = None
        if last_seen and str(latest.get("x1_date") or "").strip():
            last_seen_dt = _parse_iso_date(last_seen)
            x1_dt = _parse_iso_date(latest.get("x1_date"))
            if last_seen_dt is not None and x1_dt is not None:
                age_days = max(0, (last_seen_dt - x1_dt).days)
        active_now = bool(last_seen and last_seen == latest_asof_by_timeframe.get(timeframe_key))
        streak = _yolo_streak_for_asofs(seen_asofs, asof_dates_desc.get(timeframe_key, []), last_seen)
        confidence_by_asof: dict[str, float] = {}
        for item in grouped_rows:
            asof_key = str(item.get("as_of_date") or "").strip()
            if not asof_key:
                continue
            conf = float(item.get("confidence") or 0.0)
            confidence_by_asof[asof_key] = max(conf, float(confidence_by_asof.get(asof_key) or 0.0))
        ordered_conf = sorted(confidence_by_asof.items(), key=lambda row: row[0], reverse=True)
        latest_conf = ordered_conf[0][1] if ordered_conf else float(latest.get("confidence") or 0.0)
        previous_conf = ordered_conf[1][1] if len(ordered_conf) >= 2 else None
        confidence_delta = round(float(latest_conf - previous_conf), 3) if previous_conf is not None else None
        if active_now:
            if previous_conf is None:
                confirmation_trend = "new"
            elif confidence_delta is not None and confidence_delta >= 0.05:
                confirmation_trend = "strengthening"
            elif confidence_delta is not None and confidence_delta <= -0.05:
                confirmation_trend = "weakening"
            else:
                confirmation_trend = "persisting"
        else:
            confirmation_trend = "inactive"
        includes_latest_close = bool(active_now and latest_price_date and last_seen and last_seen == latest_price_date)
        if includes_latest_close:
            lifecycle_state = "includes_latest_close"
        elif active_now:
            lifecycle_state = "active_context"
        elif latest_price_date and last_seen and str(last_seen) < str(latest_price_date):
            lifecycle_state = "no_longer_active"
        else:
            lifecycle_state = "historical_context"
        audit_item = {
            "ticker": ticker_key,
            "timeframe": latest.get("timeframe"),
            "pattern": latest.get("pattern"),
            "confidence": round(float(latest.get("confidence") or 0.0), 3),
            "x0_date": latest.get("x0_date"),
            "x1_date": latest.get("x1_date"),
            "age_days": age_days,
            "yolo_recency": _report_yolo_recency_label(age_days, timeframe_key),
            "first_seen_asof": first_seen,
            "last_seen_asof": last_seen,
            "snapshots_seen": len(seen_asofs),
            "current_streak": streak,
            "active_now": active_now,
            "latest_price_date": latest_price_date,
            "latest_close_in_pattern": includes_latest_close,
            "previous_confidence": round(float(previous_conf), 3) if previous_conf is not None else None,
            "confidence_delta": confidence_delta,
            "confirmation_trend": confirmation_trend,
            "lifecycle_state": lifecycle_state,
        }
        audit_item["signal_role"] = _yolo_signal_role(audit_item, active_now=active_now)
        audit_item["priority_score"] = _yolo_priority_score(audit_item, active_now=active_now)
        audit_rows.append(audit_item)

    audit_rows.sort(
        key=lambda item: (
            1 if bool(item.get("active_now")) else 0,
            _yolo_role_rank(str(item.get("signal_role") or "")),
            float(item.get("priority_score") or 0.0),
            str(item.get("last_seen_asof") or ""),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return audit_rows[: max(1, int(limit))]


# ---------------------------------------------------------------------------
# Fund snapshot selection
# ---------------------------------------------------------------------------

def select_fund_snapshot(
    conn: sqlite3.Connection,
    min_complete_tickers: int = 400,
) -> tuple[str | None, int]:
    """Pick the best Finviz fundamentals snapshot (latest complete or latest)."""
    latest = conn.execute(
        """
        SELECT snapshot_ts, COUNT(DISTINCT ticker) AS c
        FROM finviz_fundamentals
        GROUP BY snapshot_ts
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """
    ).fetchone()
    if latest is None:
        return None, 0

    latest_snap = latest["snapshot_ts"]
    latest_count = int(latest["c"] or 0)
    if latest_count >= min_complete_tickers:
        return latest_snap, latest_count

    latest_complete = conn.execute(
        """
        SELECT snapshot_ts, COUNT(DISTINCT ticker) AS c
        FROM finviz_fundamentals
        GROUP BY snapshot_ts
        HAVING COUNT(DISTINCT ticker) >= ?
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """,
        (min_complete_tickers,),
    ).fetchone()
    if latest_complete is not None:
        return latest_complete["snapshot_ts"], int(latest_complete["c"] or 0)

    return latest_snap, latest_count

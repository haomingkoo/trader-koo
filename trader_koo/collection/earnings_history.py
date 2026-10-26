"""Persistent earnings-event history.

``catalyst_data`` only keeps the upcoming Finnhub calendar in a TTL cache, so
past report dates, estimates and actuals disappear. This module stores every
fetched row in ``earnings_events`` and provides the one-time historical
backfill used by ``scripts/backfill_earnings_history.py``.

Missing fields are stored as NULL. A NULL or unknown (``TBD``) value never
overwrites a value already recorded for the same (ticker, report_date, source).
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import sqlite3
import time
from typing import Any, Callable, Iterable

from trader_koo.collection.config import (
    AMC_FROM,
    BMO_BEFORE,
    DATE_TEXT_LEN,
    FINNHUB_MIN_INTERVAL_SEC,
    YFINANCE_EARNINGS_LIMIT,
)

LOG = logging.getLogger(__name__)


Row = dict[str, Any]


def ensure_earnings_events_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings_events (
            ticker TEXT NOT NULL,
            report_date TEXT NOT NULL,
            session TEXT,
            eps_estimate REAL,
            eps_actual REAL,
            revenue_estimate REAL,
            revenue_actual REAL,
            surprise_pct REAL,
            source TEXT NOT NULL,
            fetched_ts TEXT NOT NULL,
            PRIMARY KEY (ticker, report_date, source)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_earnings_events_date ON earnings_events(report_date)"
    )


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _session(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if text in {"BMO", "AMC"} else None


def upsert_earnings_events(conn: sqlite3.Connection, rows: Iterable[Row]) -> int:
    """Upsert calendar-shaped rows (the ``catalyst_data`` row keys). Returns rows written."""
    ensure_earnings_events_schema(conn)
    fetched_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = []
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        report_date = str(row.get("earnings_date") or "").strip()[:DATE_TEXT_LEN]
        source = str(row.get("source") or "").strip()
        if not ticker or not report_date or not source:
            continue
        params.append(
            (
                ticker,
                report_date,
                _session(row.get("earnings_session")),
                _num(row.get("estimate_eps")),
                _num(row.get("actual_eps")),
                _num(row.get("estimate_revenue")),
                _num(row.get("actual_revenue")),
                _num(row.get("surprise_pct")),
                source,
                fetched_ts,
            )
        )
    if not params:
        return 0
    conn.executemany(
        """
        INSERT INTO earnings_events (
            ticker, report_date, session, eps_estimate, eps_actual,
            revenue_estimate, revenue_actual, surprise_pct, source, fetched_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, report_date, source) DO UPDATE SET
            session = COALESCE(excluded.session, session),
            eps_estimate = COALESCE(excluded.eps_estimate, eps_estimate),
            eps_actual = COALESCE(excluded.eps_actual, eps_actual),
            revenue_estimate = COALESCE(excluded.revenue_estimate, revenue_estimate),
            revenue_actual = COALESCE(excluded.revenue_actual, revenue_actual),
            surprise_pct = COALESCE(excluded.surprise_pct, surprise_pct),
            fetched_ts = excluded.fetched_ts
        """,
        params,
    )
    return len(params)


def session_from_timestamp(ts: Any) -> str | None:
    """BMO/AMC from a report timestamp; None when the time of day is not informative."""
    try:
        local = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    except (AttributeError, TypeError):
        return None
    t = local.time()
    if t == dt.time(0, 0):
        return None
    if t < BMO_BEFORE:
        return "BMO"
    if t >= AMC_FROM:
        return "AMC"
    return None


def yfinance_rows(ticker: str, frame: Any) -> list[Row]:
    """Map a yfinance ``get_earnings_dates`` frame to calendar-shaped rows."""
    if frame is None or getattr(frame, "empty", True):
        return []
    out: list[Row] = []
    for ts, rec in frame.iterrows():
        out.append(
            {
                "ticker": ticker,
                "earnings_date": ts.strftime("%Y-%m-%d"),
                "earnings_session": session_from_timestamp(ts),
                "estimate_eps": _num(rec.get("EPS Estimate")),
                "actual_eps": _num(rec.get("Reported EPS")),
                "surprise_pct": _num(rec.get("Surprise(%)")),
                "source": "yfinance",
            }
        )
    return out


def fetch_yfinance_history(ticker: str, *, limit: int = YFINANCE_EARNINGS_LIMIT) -> list[Row]:
    import yfinance as yf

    return yfinance_rows(ticker, yf.Ticker(ticker).get_earnings_dates(limit=limit))


def backfill_earnings_history(
    conn: sqlite3.Connection,
    tickers: Iterable[str],
    *,
    finnhub_fetch: Callable[[str], list[Row]] | None = None,
    yfinance_fetch: Callable[[str], list[Row]] | None = None,
    finnhub_min_interval_sec: float = FINNHUB_MIN_INTERVAL_SEC,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Backfill history per ticker from each provided source, recording the source."""
    summary: dict[str, Any] = {"tickers": 0, "rows": 0, "failures": {}}
    last_finnhub: float | None = None
    for raw in tickers:
        ticker = str(raw).strip().upper()
        if not ticker:
            continue
        summary["tickers"] += 1
        for name, fetch in (("finnhub", finnhub_fetch), ("yfinance", yfinance_fetch)):
            if fetch is None:
                continue
            if name == "finnhub":
                if last_finnhub is not None:
                    wait = finnhub_min_interval_sec - (time.monotonic() - last_finnhub)
                    if wait > 0:
                        sleep(wait)
                last_finnhub = time.monotonic()
            try:
                rows = fetch(ticker)
                summary["rows"] += upsert_earnings_events(conn, rows)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                LOG.warning("earnings backfill %s/%s failed: %s", ticker, name, exc)
                summary["failures"][f"{ticker}:{name}"] = str(exc)
    return summary


def persist_fetched_calendar(
    conn: sqlite3.Connection, fetched: tuple[list[Row], dict[str, Any]]
) -> tuple[list[Row], dict[str, Any]]:
    """Archive freshly fetched calendar rows; pass the (rows, meta) pair through.

    Cache hits were archived when first fetched. A storage failure is logged and
    must not break the calendar response.
    """
    rows, meta = fetched
    if rows and not meta.get("used_cache"):
        try:
            upsert_earnings_events(conn, rows)
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            LOG.warning("Failed to persist earnings events: %s", exc)
    return rows, meta

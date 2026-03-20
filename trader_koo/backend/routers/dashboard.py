"""Dashboard endpoints: ticker list, dashboard payload, YOLO patterns."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Path as FastPath, Query

from trader_koo.backend.services.chart_builder import (
    build_commentary_payload,
    build_dashboard_payload,
    build_dashboard_quick_payload,
)
from trader_koo.backend.services.database import get_conn, get_yolo_patterns

LOG = logging.getLogger("trader_koo.routers.dashboard")

router = APIRouter()

REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))


def _attach_live_candle(payload: dict[str, Any], ticker: str) -> None:
    """Attach live forming candle to *payload* in-place (best-effort)."""
    try:
        from trader_koo.streaming.service import get_forming_candle

        live_candle = get_forming_candle(ticker.upper())
        if live_candle is not None:
            payload["live_candle"] = live_candle
    except Exception:
        LOG.debug(
            "Could not attach live candle for %s", ticker, exc_info=True
        )


@router.get("/api/tickers")
def tickers(
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT ticker
            FROM price_daily
            ORDER BY ticker
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        out = [r["ticker"] for r in rows]
        return {"count": len(out), "tickers": out, "offset": offset}
    finally:
        conn.close()


@router.get("/api/dashboard/{ticker}/quick")
def dashboard_quick(
    ticker: str = FastPath(pattern=r"^[A-Z0-9._\^-]{1,20}$"),
    months: int = Query(default=3, ge=0, le=240),
) -> dict[str, Any]:
    """Fast-path dashboard: price data, levels, patterns, fundamentals.

    Skips LLM commentary and HMM regime so the chart renders immediately.
    """
    conn = get_conn()
    try:
        payload = build_dashboard_quick_payload(conn, ticker=ticker, months=months)
    finally:
        conn.close()

    _attach_live_candle(payload, ticker)
    return payload


@router.get("/api/dashboard/{ticker}/commentary")
def dashboard_commentary(
    ticker: str = FastPath(pattern=r"^[A-Z0-9._\^-]{1,20}$"),
    months: int = Query(default=3, ge=0, le=240),
    report_generated_ts: str | None = Query(default=None),
) -> dict[str, Any]:
    """Slow-path dashboard: chart commentary, debate engine, HMM regime.

    Called in the background after the quick endpoint has returned.
    """
    conn = get_conn()
    try:
        payload = build_commentary_payload(
            conn,
            ticker=ticker,
            months=months,
            report_dir=REPORT_DIR,
            report_generated_ts=report_generated_ts,
        )
    finally:
        conn.close()
    return payload


@router.get("/api/dashboard/{ticker}")
def dashboard(
    ticker: str = FastPath(pattern=r"^[A-Z0-9._\^-]{1,20}$"),
    months: int = Query(default=3, ge=0, le=240),
    report_generated_ts: str | None = Query(default=None),
) -> dict[str, Any]:
    """Full dashboard payload (backward-compatible)."""
    conn = get_conn()
    try:
        payload = build_dashboard_payload(
            conn,
            ticker=ticker,
            months=months,
            report_dir=REPORT_DIR,
            report_generated_ts=report_generated_ts,
        )
    finally:
        conn.close()

    _attach_live_candle(payload, ticker)
    return payload


@router.get("/api/yolo/{ticker}")
def yolo_ticker(ticker: str) -> dict[str, Any]:
    """Return stored YOLO pattern detections for a ticker."""
    conn = get_conn()
    try:
        t = ticker.upper().strip()
        patterns = get_yolo_patterns(conn, t)
        return {"ticker": t, "count": len(patterns), "patterns": patterns}
    finally:
        conn.close()

"""Dashboard endpoints: ticker list, dashboard payload, YOLO patterns."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Path as FastPath, Query

from trader_koo.backend.services.chart_builder import build_dashboard_payload
from trader_koo.backend.services.database import get_conn, get_yolo_patterns

LOG = logging.getLogger("trader_koo.routers.dashboard")

router = APIRouter()

REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))


@router.get("/api/tickers")
def tickers(limit: int = Query(default=200, ge=1, le=2000)) -> dict[str, Any]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT ticker
            FROM price_daily
            ORDER BY ticker
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out = [r["ticker"] for r in rows]
        return {"count": len(out), "tickers": out}
    finally:
        conn.close()


@router.get("/api/dashboard/{ticker}")
def dashboard(
    ticker: str = FastPath(pattern=r"^[A-Z0-9._\^-]{1,20}$"),
    months: int = Query(default=3, ge=0, le=240),
    report_generated_ts: str | None = Query(default=None),
) -> dict[str, Any]:
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

    # Attach live forming candle when streaming data is available
    try:
        from trader_koo.streaming.service import get_forming_candle

        live_candle = get_forming_candle(ticker.upper())
        if live_candle is not None:
            payload["live_candle"] = live_candle
    except Exception:
        LOG.debug("Could not attach live candle for %s", ticker, exc_info=True)

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

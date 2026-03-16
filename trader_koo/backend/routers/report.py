"""Report endpoints: daily report, earnings calendar, market summary."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query

from trader_koo.backend.services.database import get_conn
from trader_koo.backend.services.pipeline import pipeline_status_snapshot
from trader_koo.backend.services.report_loader import (
    daily_report_response,
    latest_daily_report_json,
)
from trader_koo.catalyst_data import build_earnings_calendar_payload
from trader_koo.scripts.generate_daily_report import (
    _build_regime_context as _report_build_regime_context,
)

router = APIRouter()

REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))
_MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")


@router.get("/api/daily-report")
def public_daily_report(
    limit: int = Query(default=20, ge=1, le=200),
    include_markdown: bool = Query(default=False),
) -> dict[str, Any]:
    """Return latest generated daily report for UI without admin auth."""
    return daily_report_response(
        report_dir=REPORT_DIR,
        get_conn_fn=get_conn,
        build_regime_context_fn=_report_build_regime_context,
        pipeline_status_fn=pipeline_status_snapshot,
        limit=limit,
        include_markdown=include_markdown,
        include_internal_paths=False,
        include_admin_log_hints=False,
    )


@router.get("/api/earnings-calendar")
def earnings_calendar(
    days: int = Query(default=21, ge=1, le=90),
    limit: int = Query(default=200, ge=1, le=1000),
    tickers: str | None = Query(default=None),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        _, latest_report = latest_daily_report_json(REPORT_DIR)
        market_date = dt.datetime.now(ZoneInfo(_MARKET_TZ_NAME)).date()
        if isinstance(latest_report, dict):
            session = latest_report.get("market_session") or {}
            raw_market_date = str(session.get("market_date") or "").strip()
            try:
                if raw_market_date:
                    market_date = dt.date.fromisoformat(raw_market_date)
            except ValueError:
                pass
        requested = {
            str(token or "").strip().upper()
            for token in str(tickers or "").split(",")
            if str(token or "").strip()
        }
        setup_lookup: dict[str, Any] = {}
        if isinstance(latest_report, dict):
            setup_lookup = ((latest_report.get("signals") or {}).get("setup_quality_lookup")) or {}
        payload = build_earnings_calendar_payload(
            conn,
            market_date=market_date,
            days=days,
            limit=limit,
            tickers=requested,
            setup_map=setup_lookup,
        )
        payload["report_generated_ts"] = (
            (latest_report or {}).get("generated_ts") if isinstance(latest_report, dict) else None
        )
        return payload
    finally:
        conn.close()


@router.get("/api/market-summary")
def market_summary(days: int = Query(90, ge=7, le=365)) -> Any:
    """Public endpoint -- SPY & QQQ price history for the portfolio chart."""
    ticker_list = ["SPY", "QQQ"]
    conn = get_conn()
    try:
        result: dict[str, Any] = {"as_of": None, "tickers": {}}
        for ticker in ticker_list:
            rows = conn.execute(
                """
                SELECT date, CAST(close AS REAL) AS close
                FROM price_daily
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (ticker, days),
            ).fetchall()
            if not rows:
                result["tickers"][ticker] = None
                continue
            history = [{"date": r[0], "close": round(float(r[1]), 2)} for r in reversed(rows)]
            latest_close = history[-1]["close"]
            prev_close = history[-2]["close"] if len(history) >= 2 else latest_close
            first_close = history[0]["close"]
            change_pct_1d = round((latest_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0
            change_pct_period = round((latest_close - first_close) / first_close * 100, 2) if first_close else 0.0
            result["tickers"][ticker] = {
                "price": latest_close,
                "change_pct_1d": change_pct_1d,
                "change_pct_period": change_pct_period,
                "history": history,
            }
            if result["as_of"] is None:
                result["as_of"] = history[-1]["date"]
        return result
    finally:
        conn.close()

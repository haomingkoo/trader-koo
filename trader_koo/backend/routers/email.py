"""Email endpoints: chart-preview (delivery lives in trader_koo.email_delivery)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response

from trader_koo.backend.services.database import get_conn
from trader_koo.email_chart_preview import (
    build_email_chart_preview_png,
    verify_chart_preview_signature,
)

router = APIRouter()


@router.get("/api/email/chart-preview")
def email_chart_preview(
    ticker: str = Query(..., min_length=1, max_length=16),
    timeframe: str = Query(default="daily"),
    report_ts: str | None = Query(default=None),
    exp: int = Query(...),
    sig: str = Query(..., min_length=32, max_length=128),
) -> Response:
    if not verify_chart_preview_signature(
        ticker=ticker,
        timeframe=timeframe,
        report_ts=report_ts,
        exp=exp,
        sig=sig,
    ):
        raise HTTPException(status_code=403, detail="Invalid chart preview signature")
    conn = get_conn()
    try:
        png = build_email_chart_preview_png(
            conn,
            ticker=ticker,
            timeframe=timeframe,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        conn.close()
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )

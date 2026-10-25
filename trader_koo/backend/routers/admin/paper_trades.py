"""Manual paper-trade admin endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from trader_koo.backend.services.database import get_conn
from trader_koo.paper_trades import (
    PaperTradeWritesDisabled,
    manually_close_trade,
    mark_to_market,
)

router = APIRouter(tags=["admin", "admin-paper-trades"])


@router.post("/api/admin/paper-trades/close")
def admin_close_paper_trade(
    request: Request,
    trade_id: int = Query(..., ge=1),
    exit_price: float | None = Query(default=None),
    exit_reason: str = Query(default="manual_close"),
) -> dict[str, Any]:
    """Manually close an open paper trade."""
    conn = get_conn()
    try:
        result = manually_close_trade(
            conn,
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
        return {"ok": True, **result}
    except PaperTradeWritesDisabled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@router.post("/api/admin/paper-trades/mtm")
def admin_trigger_mtm(request: Request) -> dict[str, Any]:
    """Trigger mark-to-market on all open paper trades."""
    conn = get_conn()
    try:
        try:
            result = mark_to_market(conn)
        except PaperTradeWritesDisabled as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        conn.commit()
        return {"ok": True, **result}
    finally:
        conn.close()

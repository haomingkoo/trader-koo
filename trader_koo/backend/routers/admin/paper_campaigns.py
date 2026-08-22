"""Authenticated paper-campaign lifecycle transitions."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from trader_koo.backend.services.database import get_conn
from trader_koo.middleware.auth import require_admin_auth
from trader_koo.paper_trade.campaign import transition_campaign
from trader_koo.paper_trade.schema import ensure_paper_trade_schema

router = APIRouter(tags=["admin", "admin-paper-campaigns"])


class CampaignTransition(BaseModel):
    action: Literal["activate", "rollback"]
    reason: str = Field(min_length=3, max_length=500)
    idempotency_key: str = Field(min_length=8, max_length=120)


@router.post("/api/admin/paper-campaigns/{campaign_id}/transition")
@require_admin_auth
def admin_transition_paper_campaign(
    request: Request,
    campaign_id: str,
    body: CampaignTransition,
) -> dict[str, Any]:
    identity = getattr(request.state, "admin_identity", {}) or {}
    actor = str(identity.get("username") or identity.get("user_id") or "authenticated-admin")
    conn = get_conn()
    try:
        ensure_paper_trade_schema(conn)
        try:
            result = transition_campaign(
                conn,
                campaign_id=campaign_id,
                action=body.action,
                actor=actor,
                reason=body.reason,
                idempotency_key=body.idempotency_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "transition": result}
    finally:
        conn.close()

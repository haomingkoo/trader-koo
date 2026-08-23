"""Authenticated, redacted LLM and agent observability endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from trader_koo.backend.routers.admin._shared import DB_PATH
from trader_koo.llm.observability import observability_summary, observability_trace

router = APIRouter()


@router.get("/api/admin/agent-observability")
def agent_observability(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Return sanitized traces and reconciled aggregate health."""
    return {"ok": True, **observability_summary(DB_PATH, limit=limit)}


@router.get("/api/admin/agent-observability/{trace_id}")
def agent_observability_trace(trace_id: str) -> dict[str, Any]:
    detail = observability_trace(DB_PATH, trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="LLM trace not found")
    return {"ok": True, **detail}

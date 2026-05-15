"""Options premium proxy endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from trader_koo.backend.services.database import get_conn
from trader_koo.config import get_options_config
from trader_koo.options_research import build_options_premium_proxy

router = APIRouter(tags=["options"])
_OPTIONS_PREMIUM = get_options_config().premium


@router.get("/api/options/premium")
def options_premium(
    limit: int = Query(
        default=_OPTIONS_PREMIUM.default_limit,
        ge=_OPTIONS_PREMIUM.min_limit,
        le=_OPTIONS_PREMIUM.max_limit,
    ),
    sort_by: str = Query(
        default=_OPTIONS_PREMIUM.default_sort_by,
        pattern=_OPTIONS_PREMIUM.sort_pattern,
    ),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        return build_options_premium_proxy(conn, limit=limit, sort_by=sort_by)
    finally:
        conn.close()

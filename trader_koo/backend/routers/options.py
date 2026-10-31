"""Options premium proxy endpoint."""
from __future__ import annotations

import sqlite3
import threading
from typing import Any

from fastapi import APIRouter, Query

from trader_koo.backend.services.database import get_conn
from trader_koo.config import get_options_config
from trader_koo.options_research import build_options_premium_proxy

router = APIRouter(tags=["options"])
_OPTIONS_PREMIUM = get_options_config().premium

# The proxy scans all options_iv history, so results are reused until an ingest
# adds rows. Only the current data version is kept.
_PREMIUM_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_PREMIUM_CACHE_LOCK = threading.Lock()


def _options_data_version(conn: sqlite3.Connection) -> tuple[Any, ...] | None:
    """Latest snapshot and newest row id; both are index lookups, and any insert or replace changes them."""
    try:
        row = conn.execute("SELECT MAX(snapshot_ts), MAX(rowid) FROM options_iv").fetchone()
    except sqlite3.OperationalError:  # table missing: let the builder report it, uncached
        return None
    return tuple(row)


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
        version = _options_data_version(conn)
        key = (version, limit, sort_by)
        with _PREMIUM_CACHE_LOCK:
            cached = _PREMIUM_CACHE.get(key) if version is not None else None
        if cached is not None:
            return cached
        payload = build_options_premium_proxy(conn, limit=limit, sort_by=sort_by)
        if version is not None:
            with _PREMIUM_CACHE_LOCK:
                if any(existing[0] != version for existing in _PREMIUM_CACHE):
                    _PREMIUM_CACHE.clear()
                _PREMIUM_CACHE[key] = payload
        return payload
    finally:
        conn.close()

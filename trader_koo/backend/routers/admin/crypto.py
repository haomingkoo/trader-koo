"""Crypto admin endpoints — manual backfill trigger and data diagnostics."""
from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any

from fastapi import APIRouter, Query, Request

from trader_koo.backend.services.database import DB_PATH
from trader_koo.crypto.storage import get_crypto_data_status
from trader_koo.middleware.auth import require_admin_auth

LOG = logging.getLogger("trader_koo.admin.crypto")

router = APIRouter(tags=["admin", "admin-crypto"])

_backfill_thread: threading.Thread | None = None

_BACKFILL_TARGETS: dict[str, int] = {
    "1h": 2160,
    "4h": 1440,
    "12h": 1095,
    "1d": 1825,
    "1w": 260,
}


@router.post("/api/admin/crypto/backfill")
@require_admin_auth
def admin_crypto_backfill(
    request: Request,
    symbol: str = Query(default=""),
    interval: str = Query(default=""),
) -> dict[str, Any]:
    """Trigger manual crypto backfill in background thread."""
    global _backfill_thread
    if _backfill_thread and _backfill_thread.is_alive():
        return {"ok": False, "message": "Backfill already running"}

    from trader_koo.crypto.binance_ws import SYMBOL_MAP
    from trader_koo.crypto.service import _backfill_history

    symbols = [symbol.upper()] if symbol.strip() else list(SYMBOL_MAP.values())
    intervals = [(interval, _BACKFILL_TARGETS.get(interval, 500))] if interval.strip() else list(_BACKFILL_TARGETS.items())
    targets = [(s, i, lim) for s in symbols for i, lim in intervals]

    def _run() -> None:
        for sym, ivl, limit in targets:
            try:
                _backfill_history(sym, ivl, limit)
                LOG.info("Manual backfill OK: %s [%s] limit=%d", sym, ivl, limit)
            except Exception as exc:
                LOG.warning("Manual backfill failed: %s [%s]: %s", sym, ivl, exc)

    _backfill_thread = threading.Thread(target=_run, daemon=True, name="admin-crypto-backfill")
    _backfill_thread.start()
    return {
        "ok": True,
        "message": f"Backfill started for {len(targets)} target(s)",
        "targets": [{"symbol": s, "interval": i, "limit": l} for s, i, l in targets],
    }


@router.get("/api/admin/crypto/data-status")
@require_admin_auth
def admin_crypto_data_status(request: Request) -> dict[str, Any]:
    """Row counts and freshness per symbol/interval in crypto_bars."""
    from trader_koo.crypto.service import get_crypto_ws_health

    conn = sqlite3.connect(str(DB_PATH))
    try:
        status = get_crypto_data_status(conn)
    finally:
        conn.close()
    return {"ok": True, "ws_health": get_crypto_ws_health(), "data": status}

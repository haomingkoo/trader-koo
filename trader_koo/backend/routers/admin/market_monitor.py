"""Admin endpoints for prediction market + crypto spike monitoring."""
from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Query

from trader_koo.middleware.auth import require_admin_auth

from trader_koo.backend.routers.admin._shared import DB_PATH, LOG

router = APIRouter(tags=["admin", "admin-market-monitor"])


@router.get("/api/admin/market-spikes")
@require_admin_auth
def market_spikes(
    hours: int = Query(default=24, ge=1, le=168),
) -> dict[str, Any]:
    """Return recent prediction market + crypto spikes detected in the last N hours."""
    try:
        from trader_koo.notifications.market_monitor import get_recent_spikes

        return get_recent_spikes(DB_PATH, hours=hours)
    except Exception as exc:
        LOG.exception("market-spikes endpoint failed: %s", exc)
        return {
            "ok": False,
            "error": "Unable to fetch market spikes",
            "polymarket_spikes": [],
            "crypto_spikes": [],
        }


@router.post("/api/admin/market-snapshot")
@require_admin_auth
def trigger_market_snapshot() -> dict[str, Any]:
    """Manually trigger a Polymarket snapshot."""
    try:
        from trader_koo.notifications.market_monitor import snapshot_polymarket

        count = snapshot_polymarket(DB_PATH)
        return {
            "ok": True,
            "snapshots_saved": count,
            "triggered_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
    except Exception as exc:
        LOG.exception("manual market snapshot failed: %s", exc)
        return {"ok": False, "error": f"Snapshot failed: {exc}"}


@router.post("/api/admin/test-spike-alerts")
@require_admin_auth
def test_spike_alerts() -> dict[str, Any]:
    """Manually run spike detection and send alerts via Telegram."""
    try:
        from trader_koo.notifications.market_monitor import send_spike_alerts
        from trader_koo.backend.routers.admin._shared import REPORT_DIR

        alerts_sent = send_spike_alerts(DB_PATH, REPORT_DIR)
        return {
            "ok": True,
            "alerts_sent": alerts_sent,
            "triggered_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
    except Exception as exc:
        LOG.exception("test spike alerts failed: %s", exc)
        return {"ok": False, "error": f"Alert test failed: {exc}"}

"""Telegram alert admin endpoints — list recent alerts and send test messages."""
from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Query

from trader_koo.middleware.auth import require_admin_auth
from trader_koo.notifications.telegram import send_telegram_message

from trader_koo.backend.routers.admin._shared import DB_PATH, LOG

router = APIRouter(tags=["admin", "admin-telegram"])


@router.get("/api/admin/telegram-alerts")
@require_admin_auth
def list_telegram_alerts(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Return the most recent Telegram price alerts."""
    from trader_koo.notifications.alert_engine import AlertEngine

    engine = AlertEngine(
        db_path=DB_PATH,
        report_dir=DB_PATH.parent / "reports",
    )
    alerts = engine.get_recent_alerts(limit=limit)
    watchlist = engine.get_watchlist_summary()
    return {
        "ok": True,
        "count": len(alerts),
        "watchlist": watchlist,
        "alerts": alerts,
    }


@router.post("/api/admin/telegram-test")
@require_admin_auth
def send_test_message() -> dict[str, Any]:
    """Send a test message to verify the Telegram connection."""
    text = (
        "\U0001f3af *Trader Koo — Test Alert*\n"
        "\n"
        "Telegram integration is working correctly.\n"
        f"Timestamp: {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    sent = send_telegram_message(text)
    if sent:
        LOG.info("Telegram test message sent successfully")
    else:
        LOG.warning("Telegram test message failed")
    return {
        "ok": sent,
        "detail": "Test message sent" if sent else "Failed to send — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
    }

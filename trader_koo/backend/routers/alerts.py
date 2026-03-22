"""Public alerts endpoint — merges telegram price alerts and market spikes."""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Query

from trader_koo.backend.services.database import DB_PATH

LOG = logging.getLogger("trader_koo.routers.alerts")

router = APIRouter(tags=["alerts"])


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Ensure required tables exist (no-op if already present)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telegram_alerts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker     TEXT    NOT NULL,
            level      REAL    NOT NULL,
            price      REAL    NOT NULL,
            alert_type TEXT    NOT NULL,
            setup_tier TEXT,
            bias       TEXT,
            sent_at    TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS polymarket_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_slug TEXT NOT NULL,
            event_title TEXT NOT NULL,
            market_question TEXT NOT NULL,
            probability REAL NOT NULL,
            volume REAL,
            snapshot_ts TEXT NOT NULL
        )
    """)
    conn.commit()


def _severity_from_change(change_pts: float) -> str:
    """Determine severity based on absolute change magnitude."""
    abs_change = abs(change_pts)
    if abs_change >= 10:
        return "high"
    if abs_change >= 5:
        return "medium"
    return "low"


def _price_alert_severity(alert_type: str) -> str:
    """Determine severity for price alerts based on alert type."""
    if "breakout" in alert_type or "breakdown" in alert_type:
        return "high"
    return "medium"


def _format_alert_type(alert_type: str) -> str:
    """Convert raw alert_type to human-readable display type."""
    if "crypto" in alert_type.lower():
        return "crypto_spike"
    return "price_alert"


def _time_ago(ts_str: str) -> str:
    """Convert ISO timestamp to human-readable time-ago string."""
    try:
        ts = dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        now = dt.datetime.now(dt.timezone.utc)
        delta = now - ts
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except Exception:
        return ""


def _fetch_price_alerts(
    conn: sqlite3.Connection,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch recent telegram price alerts."""
    try:
        rows = conn.execute(
            """
            SELECT id, ticker, level, price, alert_type,
                   setup_tier, bias, sent_at
            FROM telegram_alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        alerts: list[dict[str, Any]] = []
        for row in rows:
            alert_type = row["alert_type"]
            ticker = row["ticker"]
            price = row["price"]
            level = row["level"]

            display_type = _format_alert_type(alert_type)
            readable_type = alert_type.replace("_", " ").title()

            alerts.append({
                "id": f"price-{row['id']}",
                "type": display_type,
                "title": f"{ticker} {readable_type}",
                "message": f"${price:.2f} near ${level:.2f} level ({row['setup_tier'] or '-'} tier)",
                "severity": _price_alert_severity(alert_type),
                "timestamp": row["sent_at"],
                "time_ago": _time_ago(row["sent_at"]),
            })
        return alerts
    except Exception as exc:
        LOG.warning("Failed to fetch price alerts: %s", exc)
        return []


def _fetch_market_spikes(
    conn: sqlite3.Connection,
    limit: int,
) -> list[dict[str, Any]]:
    """Detect and return recent polymarket spikes as alert items."""
    try:
        from trader_koo.notifications.market_monitor import detect_polymarket_spikes

        spikes = detect_polymarket_spikes(DB_PATH, lookback_hours=6, threshold_pct=5.0)

        alerts: list[dict[str, Any]] = []
        for i, spike in enumerate(spikes[:limit]):
            change = spike["change_pct"]
            sign = "+" if change > 0 else ""
            severity = _severity_from_change(change)

            # Use snapshot_ts from DB if available, else now
            now_iso = dt.datetime.now(dt.timezone.utc).replace(
                microsecond=0,
            ).isoformat()

            event_slug = spike.get("event_slug", "")
            alerts.append({
                "id": f"poly-{i}",
                "type": "market_spike",
                "title": spike["event_title"],
                "message": (
                    f"{spike['old_prob']:.0f}% -> {spike['new_prob']:.0f}% "
                    f"({sign}{change:.1f} pts in {spike['lookback_hours']}h)"
                ),
                "severity": severity,
                "timestamp": now_iso,
                "time_ago": "recent",
                "external_url": f"https://polymarket.com/event/{event_slug}" if event_slug else None,
                "internal_path": "/pred-markets",
            })
        return alerts
    except Exception as exc:
        LOG.warning("Failed to fetch market spikes: %s", exc)
        return []


def _fetch_crypto_spikes(
    conn: sqlite3.Connection,
    limit: int,
) -> list[dict[str, Any]]:
    """Detect and return recent crypto spikes as alert items."""
    try:
        from trader_koo.notifications.market_monitor import detect_crypto_spikes

        spikes = detect_crypto_spikes(DB_PATH, lookback_hours=4)

        alerts: list[dict[str, Any]] = []
        for i, spike in enumerate(spikes[:limit]):
            symbol = spike["symbol"]
            change = spike["price_change_pct"]
            sign = "+" if change > 0 else ""
            severity = "high" if abs(change) >= 10 else "medium"

            now_iso = dt.datetime.now(dt.timezone.utc).replace(
                microsecond=0,
            ).isoformat()

            alerts.append({
                "id": f"crypto-{i}",
                "type": "crypto_spike",
                "title": f"{symbol} Price Spike",
                "message": (
                    f"${spike['old_price']:,.2f} -> ${spike['new_price']:,.2f} "
                    f"({sign}{change:.1f}% in {spike['lookback_hours']}h)"
                ),
                "severity": severity,
                "timestamp": now_iso,
                "time_ago": "recent",
                "internal_path": "/crypto",
            })
        return alerts
    except Exception as exc:
        LOG.warning("Failed to fetch crypto spikes: %s", exc)
        return []


@router.get("/api/alerts")
def get_alerts(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Return recent alerts from multiple sources, merged and sorted."""
    if not DB_PATH.exists():
        return {"alerts": [], "unread_count": 0}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        _ensure_tables(conn)

        price_alerts = _fetch_price_alerts(conn, limit)
        market_spikes = _fetch_market_spikes(conn, limit)
        crypto_spikes = _fetch_crypto_spikes(conn, limit)

        all_alerts = price_alerts + market_spikes + crypto_spikes

        # Sort by timestamp descending
        all_alerts.sort(
            key=lambda a: a.get("timestamp", ""),
            reverse=True,
        )

        # Trim to limit
        all_alerts = all_alerts[:limit]

        # Add read=false to all (frontend tracks read state via localStorage)
        for alert in all_alerts:
            alert["read"] = False

        return {
            "alerts": all_alerts,
            "unread_count": len(all_alerts),
        }
    except Exception as exc:
        LOG.exception("Failed to fetch alerts: %s", exc)
        return {"alerts": [], "unread_count": 0}
    finally:
        conn.close()


@router.post("/api/alerts/mark-read")
def mark_alerts_read() -> dict[str, Any]:
    """Acknowledge that alerts have been read.

    Read state is tracked client-side via localStorage.
    This endpoint exists to provide a consistent API surface.
    """
    return {
        "ok": True,
        "marked_at": dt.datetime.now(dt.timezone.utc).replace(
            microsecond=0,
        ).isoformat(),
    }

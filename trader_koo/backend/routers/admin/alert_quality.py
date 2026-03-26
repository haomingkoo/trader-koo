"""Alert quality scoring -- scores past alerts by subsequent price action.

For each alert fired in the last 30 days, checks the price 1, 3, and 5
trading days after the alert.  Assigns a quality grade (good / neutral / bad)
based on whether the price moved in the direction the alert implied.

- "approaching_support" / "breakdown_below_support":
    good  = price recovered above the level (bounce)
    bad   = price fell further below the level (breakdown confirmed)

- "approaching_resistance" / "breakout_above_resistance":
    good  = price held above the level (breakout confirmed)
    bad   = price fell back below the level (rejection)
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Query

from trader_koo.backend.services.database import DB_PATH, table_exists
from trader_koo.middleware.auth import require_admin_auth

LOG = logging.getLogger("trader_koo.routers.admin.alert_quality")

router = APIRouter(tags=["admin", "admin-alert-quality"])

# Alert types that imply "price should go up" (support bounce / breakout)
_BULLISH_ALERT_TYPES = {"approaching_support", "breakout_above_resistance"}
# Alert types that imply "price should go down" (resistance rejection / breakdown)
_BEARISH_ALERT_TYPES = {"approaching_resistance", "breakdown_below_support"}


def _score_alert(
    alert: dict[str, Any],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Score a single alert by checking price action 1/3/5 days after.

    Returns the alert dict enriched with quality fields.
    """
    ticker = alert["ticker"]
    sent_at = alert["sent_at"]
    level = float(alert["level"])
    alert_type = alert["alert_type"]

    # Parse alert date
    try:
        alert_dt = dt.datetime.fromisoformat(
            sent_at.replace("Z", "+00:00")
        )
        alert_date = alert_dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        alert["quality"] = "unknown"
        alert["quality_detail"] = "unparseable alert timestamp"
        alert["follow_up"] = {}
        return alert

    # Fetch daily close prices after alert date
    rows = conn.execute(
        """
        SELECT date, close FROM price_daily
        WHERE ticker = ? AND date > ? AND close IS NOT NULL
        ORDER BY date ASC
        LIMIT 10
        """,
        (ticker, alert_date),
    ).fetchall()

    if not rows:
        alert["quality"] = "pending"
        alert["quality_detail"] = "no subsequent price data"
        alert["follow_up"] = {}
        return alert

    follow_up: dict[str, dict[str, Any]] = {}
    for offset_label, offset_idx in [("1d", 0), ("3d", 2), ("5d", 4)]:
        if offset_idx < len(rows):
            close = float(rows[offset_idx]["close"])
            pct_from_level = round(((close - level) / level) * 100, 2)
            pct_from_alert_price = round(
                ((close - float(alert["price"])) / float(alert["price"])) * 100, 2
            )
            follow_up[offset_label] = {
                "date": rows[offset_idx]["date"],
                "close": round(close, 2),
                "pct_from_level": pct_from_level,
                "pct_from_alert_price": pct_from_alert_price,
            }

    alert["follow_up"] = follow_up

    # Use 5d if available, else 3d, else 1d for quality grade
    eval_key = "5d" if "5d" in follow_up else ("3d" if "3d" in follow_up else "1d")
    if eval_key not in follow_up:
        alert["quality"] = "pending"
        alert["quality_detail"] = "insufficient follow-up data"
        return alert

    eval_close = follow_up[eval_key]["close"]

    if alert_type in _BULLISH_ALERT_TYPES:
        # For support/breakout alerts, price going up is good
        if eval_close > level * 1.005:
            quality = "good"
            detail = f"price held above level by {eval_key}"
        elif eval_close < level * 0.995:
            quality = "bad"
            detail = f"price broke below level by {eval_key}"
        else:
            quality = "neutral"
            detail = f"price near level at {eval_key}"
    elif alert_type in _BEARISH_ALERT_TYPES:
        # For resistance/breakdown alerts, price going down is expected
        if eval_close < level * 0.995:
            quality = "good"
            detail = f"price stayed below level by {eval_key}"
        elif eval_close > level * 1.005:
            quality = "bad"
            detail = f"price recovered above level by {eval_key}"
        else:
            quality = "neutral"
            detail = f"price near level at {eval_key}"
    else:
        quality = "unknown"
        detail = f"unrecognized alert type: {alert_type}"

    alert["quality"] = quality
    alert["quality_detail"] = detail
    return alert


@router.get("/api/admin/alert-quality")
@require_admin_auth
def get_alert_quality(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """Score past alerts by price action 1/3/5 days after firing.

    Returns each alert enriched with quality (good/neutral/bad/pending)
    and follow-up price data, plus aggregate stats.
    """
    if not DB_PATH.exists():
        return {"alerts": [], "stats": {}, "error": "database not found"}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if not table_exists(conn, "telegram_alerts"):
            return {"alerts": [], "stats": {}, "error": "no alerts table"}

        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        ).isoformat()

        rows = conn.execute(
            """
            SELECT id, ticker, level, price, alert_type,
                   setup_tier, bias, sent_at
            FROM telegram_alerts
            WHERE sent_at >= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()

        scored: list[dict[str, Any]] = []
        for row in rows:
            alert = dict(row)
            scored.append(_score_alert(alert, conn))

        # Aggregate stats
        total = len(scored)
        quality_counts: dict[str, int] = {}
        for a in scored:
            q = a.get("quality", "unknown")
            quality_counts[q] = quality_counts.get(q, 0) + 1

        scorable = quality_counts.get("good", 0) + quality_counts.get("bad", 0) + quality_counts.get("neutral", 0)
        good = quality_counts.get("good", 0)
        accuracy_pct = round((good / scorable) * 100, 1) if scorable > 0 else None

        stats = {
            "total_alerts": total,
            "days": days,
            "quality_counts": quality_counts,
            "scorable": scorable,
            "accuracy_pct": accuracy_pct,
        }

        return {"alerts": scored, "stats": stats}
    except Exception as exc:
        LOG.exception("Alert quality scoring failed: %s", exc)
        return {"alerts": [], "stats": {}, "error": str(exc)}
    finally:
        conn.close()

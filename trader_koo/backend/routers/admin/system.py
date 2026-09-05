"""System health, routes, LLM health/usage, data-source health, report stability,
usage/feedback summaries, setup evaluation, WebSocket health."""
from __future__ import annotations

import datetime as dt
import os
from typing import Any

from fastapi import APIRouter, Query, Request

from trader_koo.backend.services.database import get_conn, table_exists
from trader_koo.backend.services.report_loader import latest_daily_report_json
from trader_koo.crypto.service import get_crypto_ws_health
from trader_koo.llm_health import (
    llm_alert_cooldown_min,
    llm_alert_enabled,
    llm_degraded_threshold,
    llm_health_summary,
    llm_token_usage_summary,
)
from trader_koo.llm_narrative import llm_status
from trader_koo.middleware.auth import admin_route_inventory
from trader_koo.streaming.service import get_equity_ws_health

from trader_koo.backend.routers.admin._shared import (
    DB_PATH,
    LOG,
    REPORT_DIR,
    _to_float,
)

router = APIRouter(tags=["admin", "admin-system"])


@router.get("/api/admin/routes")
def admin_routes(request: Request) -> dict[str, Any]:
    """List the resolved runtime admin surface and its native dependency state."""
    routes = admin_route_inventory(request.app)
    protected_count = sum(bool(row["has_auth"]) for row in routes)
    unprotected_count = len(routes) - protected_count
    return {
        "total": len(routes),
        "protected": protected_count,
        "unprotected": unprotected_count,
        "all_protected": unprotected_count == 0,
        "routes": routes,
    }


@router.get("/api/admin/llm-health")
def admin_llm_health(
    recent_limit: int = Query(default=25, ge=1, le=200),
) -> dict[str, Any]:
    """Return LLM runtime/config health plus recent persisted failure/success events."""
    health = llm_health_summary(DB_PATH, recent_limit=recent_limit)
    status_data = llm_status()
    return {
        "ok": True,
        "status": status_data,
        "health": health,
        "alert": {
            "enabled": llm_alert_enabled(),
            "cooldown_min": llm_alert_cooldown_min(),
            "degraded_threshold": llm_degraded_threshold(),
            "has_override_recipients": bool(
                str(
                    os.getenv("TRADER_KOO_LLM_FAIL_ALERT_TO", "") or ""
                ).strip()
                or str(
                    os.getenv("TRADER_KOO_LLM_ALERT_TO", "") or ""
                ).strip()
            ),
        },
    }


@router.get("/api/admin/llm-usage")
def admin_llm_usage(
    days: int = Query(default=30, ge=1, le=3650),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Return persisted LLM token usage/cost estimates."""
    summary = llm_token_usage_summary(DB_PATH, days=days, limit=limit)
    return {"ok": True, **summary}


@router.get("/api/admin/setup-eval-summary")
def admin_setup_eval_summary(
    limit_families: int = Query(default=12, ge=1, le=100),
) -> dict[str, Any]:
    latest_path, latest_payload = latest_daily_report_json(REPORT_DIR)
    if not isinstance(latest_payload, dict):
        return {
            "ok": True,
            "detail": "No daily report available yet.",
            "generated_ts": None,
            "summary": {},
            "top_long_families": [],
            "top_short_families": [],
        }
    signals = latest_payload.get("signals")
    setup_eval = (
        signals.get("setup_evaluation") if isinstance(signals, dict) else {}
    )
    if not isinstance(setup_eval, dict):
        setup_eval = {}
    families = setup_eval.get("by_family")
    if not isinstance(families, list):
        families = []
    by_validity = setup_eval.get("by_validity_days")
    if not isinstance(by_validity, list):
        by_validity = []
    improvement_actions = setup_eval.get("improvement_actions")
    if not isinstance(improvement_actions, list):
        improvement_actions = []

    def _round_stat(value: Any, digits: int = 2) -> float | None:
        num = _to_float(value)
        if num is None:
            return None
        return round(num, digits)

    def _norm_family(row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        direction = str(
            row.get("call_direction") or ""
        ).strip().lower()
        if direction not in {"long", "short"}:
            return None
        return {
            "setup_family": str(
                row.get("setup_family") or ""
            ).strip(),
            "call_direction": direction,
            "calls": int(row.get("calls") or 0),
            "hit_rate_pct": _round_stat(row.get("hit_rate_pct"), 2),
            "avg_signed_return_pct": _round_stat(
                row.get("avg_signed_return_pct"), 2
            ),
            "expectancy_pct": _round_stat(row.get("expectancy_pct"), 2),
            "avg_validity_days": _round_stat(
                row.get("avg_validity_days"), 2
            ),
        }

    def _norm_validity(row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        validity = int(row.get("validity_days") or 0)
        if validity <= 0:
            return None
        return {
            "validity_days": validity,
            "calls": int(row.get("calls") or 0),
            "hit_rate_pct": _round_stat(row.get("hit_rate_pct"), 2),
            "avg_signed_return_pct": _round_stat(
                row.get("avg_signed_return_pct"), 2
            ),
            "expectancy_pct": _round_stat(row.get("expectancy_pct"), 2),
            "profit_factor": _round_stat(row.get("profit_factor"), 2),
        }

    normalized = [
        row
        for row in (_norm_family(item) for item in families)
        if isinstance(row, dict)
    ]
    normalized.sort(
        key=lambda item: (
            int(item.get("calls") or 0),
            float(item.get("hit_rate_pct") or 0.0),
            float(item.get("avg_signed_return_pct") or 0.0),
        ),
        reverse=True,
    )
    top_by_edge = sorted(
        normalized,
        key=lambda item: (
            float(item.get("expectancy_pct") or 0.0),
            float(item.get("hit_rate_pct") or 0.0),
            int(item.get("calls") or 0),
        ),
        reverse=True,
    )[: int(limit_families)]
    weakest_by_edge = sorted(
        normalized,
        key=lambda item: (
            float(item.get("expectancy_pct") or 0.0),
            float(item.get("hit_rate_pct") or 0.0),
            -int(item.get("calls") or 0),
        ),
    )[: int(limit_families)]
    top_long = [
        row
        for row in normalized
        if row.get("call_direction") == "long"
    ][: int(limit_families)]
    top_short = [
        row
        for row in normalized
        if row.get("call_direction") == "short"
    ][: int(limit_families)]
    normalized_validity = [
        row
        for row in (_norm_validity(item) for item in by_validity)
        if isinstance(row, dict)
    ]
    normalized_validity.sort(
        key=lambda item: int(item.get("validity_days") or 0)
    )

    return {
        "ok": True,
        "report_path": str(latest_path) if latest_path else None,
        "generated_ts": latest_payload.get("generated_ts"),
        "summary": setup_eval,
        "by_validity_days": normalized_validity,
        "improvement_actions": improvement_actions,
        "top_long_families": top_long,
        "top_short_families": top_short,
        "top_families_by_edge": top_by_edge,
        "weakest_families_by_edge": weakest_by_edge,
    }


@router.get("/api/admin/setup-eval-calls")
def admin_setup_eval_calls(
    status: str = Query(
        default="scored", pattern="^(open|scored|invalid|all)$"
    ),
    ticker: str | None = Query(default=None),
    direction: str | None = Query(
        default=None, pattern="^(long|short|neutral)$"
    ),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        if not table_exists(conn, "setup_call_evaluations"):
            return {
                "ok": True,
                "detail": "setup_call_evaluations table not found",
                "rows": [],
                "count": 0,
            }
        where_parts: list[str] = []
        params: list[Any] = []
        status_norm = str(status or "").strip().lower()
        if status_norm != "all":
            where_parts.append("status = ?")
            params.append(status_norm)
        ticker_norm = str(ticker or "").strip().upper()
        if ticker_norm:
            where_parts.append("ticker = ?")
            params.append(ticker_norm)
        direction_norm = str(direction or "").strip().lower()
        if direction_norm:
            where_parts.append("call_direction = ?")
            params.append(direction_norm)
        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + " AND ".join(where_parts)
        params.append(int(limit))
        rows = conn.execute(
            f"""
            SELECT
                id, asof_date, ticker, status, call_direction,
                validity_days, setup_family, setup_tier, signal_bias,
                actionability, score, close_asof, valid_target_date,
                evaluated_date, close_evaluated, raw_return_pct,
                signed_return_pct, direction_hit, yolo_pattern,
                yolo_recency, generated_ts, created_ts, updated_ts
            FROM setup_call_evaluations
            {where_sql}
            ORDER BY asof_date DESC, id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return {
            "ok": True,
            "status": status_norm,
            "ticker": ticker_norm or None,
            "direction": direction_norm or None,
            "count": len(rows),
            "rows": [dict(row) for row in rows],
        }
    finally:
        conn.close()


@router.get("/api/admin/ws-health")
def admin_ws_health() -> dict[str, Any]:
    """Return health status for all WebSocket feed connections."""
    return {
        "crypto_ws": get_crypto_ws_health(),
        "equity_ws": get_equity_ws_health(),
    }


@router.post("/api/admin/calibration/run-pulse")
def admin_run_calibration_pulse() -> dict[str, Any]:
    """Manually trigger the calibration pulse.

    Normally runs Mon/Wed/Fri 23:15 UTC. Use this to force an immediate
    recompute — e.g. after a large batch of paper trades closes, or when
    investigating why a family is/isn't being demoted.
    """
    from trader_koo.report.calibration_pulse import run_calibration_pulse, ensure_calibration_schema

    conn = get_conn()
    try:
        ensure_calibration_schema(conn)
        summary = run_calibration_pulse(conn, trigger="manual")
        LOG.info(
            "Admin manual calibration pulse: families=%d changes=%d",
            summary.get("families_updated", 0),
            len(summary.get("changes") or []),
        )
        return summary
    finally:
        conn.close()


@router.get("/api/admin/calibration/state")
def admin_calibration_state() -> dict[str, Any]:
    """Return the current calibration_state table — all family score adjustments and blocks."""
    conn = get_conn()
    try:
        if not table_exists(conn, "calibration_state"):
            return {"ok": True, "rows": [], "detail": "calibration_state table not yet created"}
        rows = conn.execute(
            """
            SELECT family, direction, score_adjustment, block_new_entries,
                   hit_rate_pct, expectancy_pct, combined_sample_count,
                   eval_sample_count, paper_sample_count, last_updated, notes
            FROM calibration_state
            ORDER BY expectancy_pct ASC NULLS LAST
            """
        ).fetchall()
        return {
            "ok": True,
            "count": len(rows),
            "rows": [dict(r) for r in rows],
        }
    finally:
        conn.close()

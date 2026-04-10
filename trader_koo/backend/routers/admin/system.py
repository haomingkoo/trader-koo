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
from trader_koo.db.sources import get_data_source_manager
from trader_koo.llm_health import (
    llm_alert_cooldown_min,
    llm_alert_enabled,
    llm_degraded_threshold,
    llm_health_summary,
    llm_token_usage_summary,
)
from trader_koo.llm_narrative import llm_status
from trader_koo.middleware.auth import (
    get_admin_endpoint_registry,
    require_admin_auth,
)
from trader_koo.streaming.service import get_equity_ws_health

from trader_koo.backend.routers.admin._shared import (
    ANALYTICS_ENABLED,
    DB_PATH,
    LOG,
    REPORT_DIR,
    _avg,
    _find_timeframe_summary,
    _load_json_file,
    _to_float,
    _to_int,
)

from trader_koo.backend.routers.usage import (
    _feedback_summary,
    _usage_summary,
)

router = APIRouter(tags=["admin", "admin-system"])


@router.get("/api/admin/routes")
@require_admin_auth
def admin_routes() -> dict[str, Any]:
    """List all admin endpoints with their authentication status."""
    registry = get_admin_endpoint_registry()
    routes = []
    protected_count = 0
    unprotected_count = 0
    for key, info in sorted(registry.items()):
        routes.append(
            {
                "method": info["method"],
                "path": info["path"],
                "has_auth": info["has_auth"],
                "key": key,
            }
        )
        if info["has_auth"]:
            protected_count += 1
        else:
            unprotected_count += 1
    return {
        "total": len(routes),
        "protected": protected_count,
        "unprotected": unprotected_count,
        "all_protected": unprotected_count == 0,
        "routes": routes,
    }


@router.get("/api/admin/llm-health")
@require_admin_auth
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
@require_admin_auth
def admin_llm_usage(
    days: int = Query(default=30, ge=1, le=3650),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Return persisted LLM token usage/cost estimates."""
    summary = llm_token_usage_summary(DB_PATH, days=days, limit=limit)
    return {"ok": True, **summary}


@router.get("/api/admin/data-source-health")
@require_admin_auth
def admin_data_source_health() -> dict[str, Any]:
    """Return data source success/failure rates and metrics."""
    manager = get_data_source_manager()
    metrics = manager.get_metrics()
    alerts: list[dict[str, Any]] = []
    for source_name, source_metrics in metrics.items():
        if (
            source_name == "yfinance"
            and source_metrics["failure_rate"] > 10.0
        ):
            alerts.append(
                {
                    "source": source_name,
                    "failure_rate": source_metrics["failure_rate"],
                    "message": (
                        f"Primary source {source_name} failure rate "
                        f"({source_metrics['failure_rate']:.1f}%) "
                        "exceeds 10% threshold"
                    ),
                    "severity": "warning",
                }
            )
    return {
        "ok": True,
        "sources": metrics,
        "alerts": alerts,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


@router.get("/api/admin/report-stability")
@require_admin_auth
def report_stability(
    limit: int = Query(default=60, ge=1, le=365),
) -> dict[str, Any]:
    """Summarize recent report JSON files to diagnose YOLO/report stability."""
    report_dir = REPORT_DIR
    files = sorted(
        [
            p
            for p in report_dir.glob("daily_report_*.json")
            if p.name != "daily_report_latest.json"
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    scan_files = files[: max(1, limit)]
    rows: list[dict[str, Any]] = []

    for p in scan_files:
        modified_ts: str | None = None
        try:
            st = p.stat()
            modified_ts = (
                dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
        except OSError:
            pass

        payload = _load_json_file(p)
        if not isinstance(payload, dict):
            rows.append(
                {
                    "file": p.name,
                    "generated_ts": None,
                    "modified_ts": modified_ts,
                    "ok": False,
                    "parse_error": True,
                    "yolo_present": False,
                    "yolo_rows_total": 0,
                    "yolo_tickers_total": 0,
                }
            )
            continue

        yolo = (
            payload.get("yolo")
            if isinstance(payload.get("yolo"), dict)
            else {}
        )
        summary = (
            yolo.get("summary")
            if isinstance(yolo.get("summary"), dict)
            else {}
        )
        timeframes = (
            yolo.get("timeframes")
            if isinstance(yolo.get("timeframes"), list)
            else []
        )
        tf_daily = _find_timeframe_summary(timeframes, "daily")
        tf_weekly = _find_timeframe_summary(timeframes, "weekly")
        delta_legacy = (
            yolo.get("delta")
            if isinstance(yolo.get("delta"), dict)
            else {}
        )
        delta_daily = (
            yolo.get("delta_daily")
            if isinstance(yolo.get("delta_daily"), dict)
            else delta_legacy
        )
        delta_weekly = (
            yolo.get("delta_weekly")
            if isinstance(yolo.get("delta_weekly"), dict)
            else {}
        )
        counts = (
            payload.get("counts")
            if isinstance(payload.get("counts"), dict)
            else {}
        )
        freshness = (
            payload.get("freshness")
            if isinstance(payload.get("freshness"), dict)
            else {}
        )
        warnings = (
            payload.get("warnings")
            if isinstance(payload.get("warnings"), list)
            else []
        )
        meta = (
            payload.get("meta")
            if isinstance(payload.get("meta"), dict)
            else {}
        )

        yolo_rows_total = _to_int(
            summary.get("rows_total"),
            _to_int(counts.get("yolo_rows"), 0),
        )
        yolo_tickers_total = _to_int(
            summary.get("tickers_with_patterns"), 0
        )
        yolo_daily_tickers = _to_int(
            tf_daily.get("tickers_with_patterns"), 0
        )
        yolo_weekly_tickers = _to_int(
            tf_weekly.get("tickers_with_patterns"), 0
        )
        yolo_present = yolo_rows_total > 0
        yolo_age_hours = _to_float(freshness.get("yolo_age_hours"))
        report_kind = str(
            meta.get("report_kind") or "daily"
        ).strip().lower()
        if report_kind not in {"daily", "weekly"}:
            report_kind = "daily"

        rows.append(
            {
                "file": p.name,
                "generated_ts": payload.get("generated_ts") or modified_ts,
                "modified_ts": modified_ts,
                "ok": bool(payload.get("ok", False)),
                "parse_error": False,
                "report_kind": report_kind,
                "warnings_count": len(warnings),
                "has_yolo_data_stale_warning": "yolo_data_stale"
                in {str(w) for w in warnings},
                "yolo_present": yolo_present,
                "yolo_rows_total": yolo_rows_total,
                "yolo_tickers_total": yolo_tickers_total,
                "yolo_daily_tickers": yolo_daily_tickers,
                "yolo_weekly_tickers": yolo_weekly_tickers,
                "yolo_latest_detected_ts": summary.get(
                    "latest_detected_ts"
                ),
                "yolo_latest_asof_date": summary.get("latest_asof_date"),
                "yolo_age_hours": yolo_age_hours,
                "delta_daily_new": _to_int(
                    delta_daily.get("new_count"), 0
                ),
                "delta_daily_lost": _to_int(
                    delta_daily.get("lost_count"), 0
                ),
                "delta_weekly_new": _to_int(
                    delta_weekly.get("new_count"), 0
                ),
                "delta_weekly_lost": _to_int(
                    delta_weekly.get("lost_count"), 0
                ),
            }
        )

    parsed_rows = [r for r in rows if not r.get("parse_error")]
    yolo_present_reports = sum(
        1 for r in parsed_rows if r.get("yolo_present")
    )
    yolo_missing_reports = sum(
        1 for r in parsed_rows if not r.get("yolo_present")
    )
    yolo_presence_rate_pct = (
        round((100.0 * yolo_present_reports) / len(parsed_rows), 2)
        if parsed_rows
        else None
    )
    newest_missing_streak = 0
    for r in parsed_rows:
        if r.get("yolo_present"):
            break
        newest_missing_streak += 1
    longest_missing_streak = 0
    current_streak = 0
    for r in reversed(parsed_rows):
        if r.get("yolo_present"):
            current_streak = 0
            continue
        current_streak += 1
        if current_streak > longest_missing_streak:
            longest_missing_streak = current_streak

    def _compact(values: list[float | None]) -> list[float]:
        return [float(v) for v in values if isinstance(v, (int, float))]

    newest_generated_ts = next(
        (
            r.get("generated_ts")
            for r in parsed_rows
            if r.get("generated_ts")
        ),
        None,
    )
    oldest_generated_ts = next(
        (
            r.get("generated_ts")
            for r in reversed(parsed_rows)
            if r.get("generated_ts")
        ),
        None,
    )
    missing_examples: list[dict[str, Any]] = []
    for r in rows:
        if r.get("parse_error"):
            missing_examples.append(
                {
                    "file": r.get("file"),
                    "generated_ts": r.get("generated_ts"),
                    "reason": "parse_error",
                }
            )
        elif not r.get("yolo_present"):
            missing_examples.append(
                {
                    "file": r.get("file"),
                    "generated_ts": r.get("generated_ts"),
                    "reason": "no_yolo_rows",
                }
            )
        if len(missing_examples) >= 10:
            break

    yolo_rows_vals = [
        _to_float(r.get("yolo_rows_total")) for r in parsed_rows
    ]
    yolo_tickers_vals = [
        _to_float(r.get("yolo_tickers_total")) for r in parsed_rows
    ]
    yolo_daily_vals = [
        _to_float(r.get("yolo_daily_tickers")) for r in parsed_rows
    ]
    yolo_weekly_vals = [
        _to_float(r.get("yolo_weekly_tickers")) for r in parsed_rows
    ]
    yolo_age_vals = [
        _to_float(r.get("yolo_age_hours"))
        for r in parsed_rows
        if _to_float(r.get("yolo_age_hours")) is not None
    ]
    delta_daily_new_vals = [
        _to_float(r.get("delta_daily_new")) for r in parsed_rows
    ]
    delta_daily_lost_vals = [
        _to_float(r.get("delta_daily_lost")) for r in parsed_rows
    ]
    delta_weekly_new_vals = [
        _to_float(r.get("delta_weekly_new")) for r in parsed_rows
    ]
    delta_weekly_lost_vals = [
        _to_float(r.get("delta_weekly_lost")) for r in parsed_rows
    ]

    summary_data = {
        "files_total": len(files),
        "reports_scanned": len(rows),
        "parsed_reports": len(parsed_rows),
        "parse_error_reports": len(rows) - len(parsed_rows),
        "yolo_present_reports": yolo_present_reports,
        "yolo_missing_reports": yolo_missing_reports,
        "yolo_presence_rate_pct": yolo_presence_rate_pct,
        "newest_missing_streak": newest_missing_streak,
        "longest_missing_streak": longest_missing_streak,
        "newest_generated_ts": newest_generated_ts,
        "oldest_generated_ts": oldest_generated_ts,
        "avg_yolo_rows_total": _avg(_compact(yolo_rows_vals)),
        "avg_yolo_tickers_total": _avg(_compact(yolo_tickers_vals)),
        "avg_yolo_daily_tickers": _avg(_compact(yolo_daily_vals)),
        "avg_yolo_weekly_tickers": _avg(_compact(yolo_weekly_vals)),
        "avg_yolo_age_hours": _avg(_compact(yolo_age_vals)),
        "avg_delta_daily_new": _avg(_compact(delta_daily_new_vals)),
        "avg_delta_daily_lost": _avg(_compact(delta_daily_lost_vals)),
        "avg_delta_weekly_new": _avg(_compact(delta_weekly_new_vals)),
        "avg_delta_weekly_lost": _avg(_compact(delta_weekly_lost_vals)),
    }
    return {
        "ok": True,
        "report_dir": str(report_dir),
        "sample_limit": int(limit),
        "summary": summary_data,
        "missing_examples": missing_examples,
        "rows": rows,
    }


@router.get("/api/admin/usage-summary")
@require_admin_auth
def usage_summary_endpoint(
    days: int = Query(default=7, ge=1, le=365),
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    if not ANALYTICS_ENABLED:
        return {
            "ok": True,
            "analytics_enabled": False,
            "detail": "Analytics collection is disabled.",
        }
    conn = get_conn()
    try:
        summary = _usage_summary(conn, days=days, limit=limit)
    finally:
        conn.close()
    summary["analytics_enabled"] = True
    return summary


@router.get("/api/admin/feedback-summary")
@require_admin_auth
def admin_feedback_summary(
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=12, ge=1, le=100),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        summary = _feedback_summary(conn, days=days)
        if not table_exists(conn, "setup_feedback"):
            return {
                "ok": True,
                "summary": summary,
                "top_tickers": [],
                "recent": [],
            }
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        ).isoformat()
        top_rows = conn.execute(
            """
            SELECT
                ticker,
                COUNT(*) AS votes,
                SUM(CASE WHEN verdict='good' THEN 1 ELSE 0 END) AS good,
                SUM(CASE WHEN verdict='bad' THEN 1 ELSE 0 END) AS bad,
                SUM(CASE WHEN verdict='neutral' THEN 1 ELSE 0 END) AS neutral
            FROM setup_feedback
            WHERE created_ts >= ?
            GROUP BY ticker
            ORDER BY votes DESC, ticker ASC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
        recent_rows = conn.execute(
            """
            SELECT
                created_ts, ticker, verdict, source_surface, asof,
                setup_tier, setup_score
            FROM setup_feedback
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return {
        "ok": True,
        "summary": summary,
        "top_tickers": [
            {
                "ticker": str(row["ticker"] or ""),
                "votes": int(row["votes"] or 0),
                "good": int(row["good"] or 0),
                "bad": int(row["bad"] or 0),
                "neutral": int(row["neutral"] or 0),
                "good_rate_pct": round(
                    (int(row["good"] or 0) * 100.0)
                    / int(row["votes"] or 1),
                    2,
                )
                if int(row["votes"] or 0)
                else None,
            }
            for row in top_rows
        ],
        "recent": [dict(row) for row in recent_rows],
    }


@router.get("/api/admin/setup-eval-summary")
@require_admin_auth
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
@require_admin_auth
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
@require_admin_auth
def admin_ws_health() -> dict[str, Any]:
    """Return health status for all WebSocket feed connections."""
    return {
        "crypto_ws": get_crypto_ws_health(),
        "equity_ws": get_equity_ws_health(),
    }


@router.post("/api/admin/calibration/run-pulse")
@require_admin_auth
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
@require_admin_auth
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


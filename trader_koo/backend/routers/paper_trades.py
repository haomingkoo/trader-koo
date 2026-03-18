"""Paper trade endpoints: list, summary, detail."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from trader_koo.backend.services.database import get_conn
from trader_koo.paper_trades import ensure_paper_trade_schema, list_paper_trades, paper_trade_summary

router = APIRouter()


@router.get("/api/paper-trades")
def api_paper_trades(
    status: str = Query(default="all", pattern="^(all|open|closed|stopped_out|target_hit|expired)$"),
    ticker: str | None = Query(default=None),
    direction: str | None = Query(default=None, pattern="^(long|short)$"),
    family: str | None = Query(default=None),
    from_date: str | None = Query(default=None),
    to_date: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List paper trades with optional filters."""
    conn = get_conn()
    try:
        trades = list_paper_trades(
            conn,
            status=status,
            ticker=ticker,
            direction=direction,
            family=family,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
        )
        return {"ok": True, "count": len(trades), "trades": trades}
    finally:
        conn.close()


@router.get("/api/paper-trades/summary")
def api_paper_trade_summary(
    window_days: int = Query(default=180, ge=7, le=730),
) -> dict[str, Any]:
    """Paper trading performance summary with metrics and equity curve."""
    conn = get_conn()
    try:
        summary = paper_trade_summary(conn, window_days=window_days)
        return {"ok": True, **summary}
    finally:
        conn.close()


@router.get("/api/paper-trades/{trade_id}")
def api_paper_trade_detail(trade_id: int) -> dict[str, Any]:
    """Get a single paper trade by ID."""
    conn = get_conn()
    try:
        ensure_paper_trade_schema(conn)
        row = conn.execute(
            """
            SELECT id, report_date, ticker, direction, entry_price, entry_date,
                   target_price, stop_loss, atr_at_entry, exit_price, exit_date,
                   exit_reason, status, current_price, unrealized_pnl_pct,
                   pnl_pct, r_multiple, high_water_mark, low_water_mark,
                   setup_family, setup_tier, score, signal_bias, actionability,
                   observation, action_text, risk_note, yolo_pattern, yolo_recency,
                   debate_agreement_score, last_mtm_date, created_ts, updated_ts,
                   decision_version, decision_state, analyst_stage, debate_stage,
                   risk_stage, portfolio_decision, decision_summary,
                   decision_reasons, risk_flags,
                   position_size_pct, risk_budget_pct, stop_distance_pct,
                   expected_reward_pct, expected_r_multiple,
                   entry_plan, exit_plan, sizing_summary,
                   review_status, review_summary
            FROM paper_trades WHERE id = ?
            """,
            (trade_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Paper trade {trade_id} not found")
        keys = [
            "id", "report_date", "ticker", "direction", "entry_price", "entry_date",
            "target_price", "stop_loss", "atr_at_entry", "exit_price", "exit_date",
            "exit_reason", "status", "current_price", "unrealized_pnl_pct",
            "pnl_pct", "r_multiple", "high_water_mark", "low_water_mark",
            "setup_family", "setup_tier", "score", "signal_bias", "actionability",
            "observation", "action_text", "risk_note", "yolo_pattern", "yolo_recency",
            "debate_agreement_score", "last_mtm_date", "created_ts", "updated_ts",
            "decision_version", "decision_state", "analyst_stage", "debate_stage",
            "risk_stage", "portfolio_decision", "decision_summary",
            "decision_reasons", "risk_flags",
            "position_size_pct", "risk_budget_pct", "stop_distance_pct",
            "expected_reward_pct", "expected_r_multiple",
            "entry_plan", "exit_plan", "sizing_summary",
            "review_status", "review_summary",
        ]
        trade = dict(zip(keys, row))
        for key in ("decision_reasons", "risk_flags"):
            raw = trade.get(key)
            if raw is None:
                trade[key] = []
                continue
            try:
                payload = json.loads(str(raw))
            except Exception:
                payload = []
            trade[key] = payload if isinstance(payload, list) else []
        return {"ok": True, "trade": trade}
    finally:
        conn.close()

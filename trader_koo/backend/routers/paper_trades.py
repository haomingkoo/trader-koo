"""Paper trade endpoints: list, summary, detail, notes."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from trader_koo.backend.services.database import get_conn
from trader_koo.paper_trades import (
    ensure_paper_trade_schema,
    list_paper_trades,
    paper_trade_summary,
    require_paper_trade_writes,
)
from trader_koo.research.strategy_evidence import evidence_snapshot_by_hash
from trader_koo.research.next_open_baseline import artifact_state
from trader_koo.research.experiment_results import (
    experiment_catalogue,
    analyze_experiment,
    experiment_download,
    experiment_result,
)
from trader_koo.paper_trade.campaign import EvidenceIntegrityError, verify_decision_set


class NotesUpdate(BaseModel):
    notes: str = ""


class ExperimentAnalysisRequest(BaseModel):
    question: str

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
    campaign_id: str = Query(default="paper-v2", min_length=1, max_length=80),
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
            campaign_id=campaign_id,
        )
        return {"ok": True, "count": len(trades), "trades": trades}
    finally:
        conn.close()


@router.get("/api/paper-trades/summary")
def api_paper_trade_summary(
    window_days: int = Query(default=180, ge=7, le=730),
    campaign_id: str = Query(default="paper-v2", min_length=1, max_length=80),
) -> dict[str, Any]:
    """Paper trading performance summary with metrics and equity curve."""
    conn = get_conn()
    try:
        summary = paper_trade_summary(conn, window_days=window_days, campaign_id=campaign_id)
        return {"ok": True, **summary}
    finally:
        conn.close()


@router.get("/api/research/strategy-evidence/{artifact_hash}/inputs/{input_hash}")
def api_strategy_evidence_provenance(artifact_hash: str, input_hash: str) -> dict[str, Any]:
    """Resolve the exact audited evidence manifest identified by both hashes."""
    state = evidence_snapshot_by_hash(artifact_hash, input_hash)
    if state is None:
        raise HTTPException(status_code=404, detail="Strategy evidence snapshot not found")
    return {"ok": True, "strategy_evidence": state}


@router.get("/api/research/next-open-baseline")
def api_next_open_baseline() -> dict[str, Any]:
    """Return the latest hash-verified baseline, or an ineligible unavailable state."""
    return {"ok": True, "baseline": artifact_state()}


@router.get("/api/research/experiments")
def api_experiment_results() -> dict[str, Any]:
    """List hash-verified experiments, including failed and invalid runs."""
    experiments = experiment_catalogue()
    return {"ok": True, "count": len(experiments), "experiments": experiments}


@router.get("/api/research/experiments/{experiment_id}")
def api_experiment_result(experiment_id: str) -> dict[str, Any]:
    result = experiment_result(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment result not found")
    return {"ok": True, "experiment": result}


@router.get("/api/research/experiments/{experiment_id}/download/{component}")
def api_experiment_download(experiment_id: str, component: str) -> dict[str, Any]:
    if component not in {"manifest", "ledger"}:
        raise HTTPException(status_code=404, detail="Experiment artifact not found")
    artifact = experiment_download(experiment_id, component)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Experiment artifact not found")
    return artifact


@router.post("/api/research/experiments/{experiment_id}/analysis")
def api_experiment_analysis(
    experiment_id: str, body: ExperimentAnalysisRequest
) -> dict[str, Any]:
    question = " ".join(body.question.split())
    if not 3 <= len(question) <= 500:
        raise HTTPException(status_code=422, detail="Question must be 3 to 500 characters")
    analysis = analyze_experiment(experiment_id, question)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Experiment result not found")
    return {"ok": True, "experiment_id": experiment_id, "analysis": analysis}


@router.get("/api/paper-trades/decisions")
def api_paper_candidate_decisions(
    campaign_id: str = Query(default="paper-v2", min_length=1, max_length=80),
    report_run_id: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    """Expose sealed rank annotations and exact policy dispositions."""
    conn = get_conn()
    try:
        ensure_paper_trade_schema(conn)
        clauses = ["d.campaign_id=?"]
        params: list[Any] = [campaign_id]
        if report_run_id:
            clauses.append("d.report_run_id=?")
            params.append(report_run_id)
        params.append(limit)
        rows = conn.execute(
            f"""SELECT d.report_run_id,d.report_date,d.generated_ts,d.candidate_rank,
                       d.ticker,d.eligibility_passed,d.final_gate,d.reason_code,
                       d.disposition,
                       CASE WHEN po.status='filled' THEN 'actionable'
                            WHEN po.status='pending' THEN 'pending_next_open'
                            ELSE d.tradeability END,
                       po.status,d.expected_r_multiple,d.inputs_hash,
                       s.candidates_hash,s.report_complete,s.is_canonical
                FROM paper_candidate_decisions d JOIN paper_decision_sets s
                  ON s.report_run_id=d.report_run_id AND s.campaign_id=d.campaign_id
                LEFT JOIN paper_pending_orders po
                  ON po.report_run_id=d.report_run_id AND po.campaign_id=d.campaign_id
                 AND po.candidate_rank=d.candidate_rank
                WHERE {' AND '.join(clauses)}
                ORDER BY d.report_date DESC,d.generated_ts DESC,d.candidate_rank
                LIMIT ?""",
            params,
        ).fetchall()
        keys = [
            "report_run_id", "report_date", "generated_ts", "candidate_rank", "ticker",
            "eligibility_passed", "final_gate", "reason_code", "disposition",
            "tradeability", "execution_status", "expected_r_multiple", "inputs_hash",
            "candidates_hash", "report_complete", "is_canonical",
        ]
        try:
            for sealed_run_id in {str(row[0]) for row in rows}:
                verify_decision_set(
                    conn, report_run_id=sealed_run_id, campaign_id=campaign_id
                )
        except EvidenceIntegrityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        decisions = [dict(zip(keys, row)) for row in rows]
        for item in decisions:
            item["eligibility_passed"] = bool(item["eligibility_passed"])
            item["report_complete"] = bool(item["report_complete"])
            item["is_canonical"] = bool(item["is_canonical"])
        return {"ok": True, "campaign_id": campaign_id, "count": len(decisions), "decisions": decisions}
    finally:
        conn.close()


@router.get("/api/paper-trades/{trade_id}")
def api_paper_trade_detail(trade_id: int) -> dict[str, Any]:
    """Get a single paper trade by ID."""
    conn = get_conn()
    try:
        ensure_paper_trade_schema(conn)
        row = conn.execute(
            "SELECT * FROM paper_trades WHERE id = ?",
            (trade_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Paper trade {trade_id} not found")
        trade = dict(row)
        annotation = conn.execute(
            "SELECT notes FROM paper_trade_annotations WHERE trade_id=?", (trade_id,)
        ).fetchone()
        if annotation:
            trade["notes"] = annotation[0]
        for key in ("decision_reasons", "risk_flags", "entry_evidence", "entry_risks"):
            raw = trade.get(key)
            if raw is None:
                trade[key] = []
                continue
            try:
                payload = json.loads(str(raw))
            except Exception:
                payload = []
            trade[key] = payload if isinstance(payload, list) else []
        trade_events = []
        for event in conn.execute(
            """SELECT event_type,event_date,payload_json,payload_hash,created_ts
               FROM paper_trade_events WHERE trade_id=? ORDER BY event_date,id""",
            (trade_id,),
        ).fetchall():
            try:
                payload = json.loads(str(event[2]))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {"unavailable": True}
            trade_events.append({
                "source": "trade",
                "event_type": event[0],
                "event_date": event[1],
                "payload": payload,
                "payload_hash": event[3],
                "created_ts": event[4],
            })
        order_events = []
        if trade.get("report_run_id") and trade.get("ticker"):
            rows = conn.execute(
                """SELECT e.event_type,e.event_date,e.payload_json,e.payload_hash,e.created_ts
                   FROM paper_pending_orders o
                   JOIN paper_order_events e ON e.order_id=o.order_id
                   WHERE o.report_run_id=? AND o.ticker=? AND o.direction=?
                   ORDER BY e.event_date,e.id""",
                (trade["report_run_id"], trade["ticker"], trade["direction"]),
            ).fetchall()
            for event in rows:
                try:
                    payload = json.loads(str(event[2]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    payload = {"unavailable": True}
                order_events.append({
                    "source": "order",
                    "event_type": event[0],
                    "event_date": event[1],
                    "payload": payload,
                    "payload_hash": event[3],
                    "created_ts": event[4],
                })
        timeline = sorted(
            [*order_events, *trade_events],
            key=lambda item: (str(item["event_date"]), str(item["created_ts"]), item["source"]),
        )
        event_types = {item["event_type"] for item in trade_events}
        trade["event_trace_status"] = (
            "unreconciled_legacy" if "fill" not in event_types
            else "complete" if trade.get("status") != "open" and "close" in event_types
            else "active"
        )
        trade["timeline"] = timeline
        return {"ok": True, "trade": trade}
    finally:
        conn.close()


@router.patch("/api/paper-trades/{trade_id}/notes")
def api_update_trade_notes(trade_id: int, body: NotesUpdate) -> dict[str, Any]:
    """Update the notes field on a paper trade."""
    try:
        require_paper_trade_writes()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    conn = get_conn()
    try:
        ensure_paper_trade_schema(conn)
        row = conn.execute(
            "SELECT id FROM paper_trades WHERE id = ?",
            (trade_id,),
        ).fetchone()
        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Paper trade {trade_id} not found",
            )
        conn.execute(
            """INSERT INTO paper_trade_annotations (trade_id,notes,actor,updated_ts)
               VALUES (?,?,'user',CURRENT_TIMESTAMP)
               ON CONFLICT(trade_id) DO UPDATE SET notes=excluded.notes,actor=excluded.actor,updated_ts=CURRENT_TIMESTAMP""",
            (trade_id, body.notes),
        )
        conn.commit()
        return {"ok": True, "trade_id": trade_id, "notes": body.notes}
    finally:
        conn.close()

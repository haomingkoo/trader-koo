"""Governed paper campaigns, pure policy decisions, and sealed evidence sets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig, config_snapshot


class DivergentDecisionSetError(RuntimeError):
    """A retry reused report identity with different decision evidence."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _base_decision(*, row: dict[str, Any], rank: int, evaluation: dict[str, Any], config: PaperTradeConfig, context: dict[str, Any]) -> dict[str, Any]:
    rank_inputs = {key: row.get(key) for key in (
        "score", "setup_tier", "confirmation_count", "contradiction_count",
        "pct_change", "actionability", "signal_bias", "setup_family",
    )}
    effective_context = {key: context.get(key) for key in (
        "entry_price", "vix_level", "avg_daily_volume", "portfolio_block",
        "critic_outcome", "campaign_active", "duplicate", "market_context",
        "portfolio_context", "source_context",
    )}
    evidence = {"candidate": row, "candidate_rank": rank, "policy": config_snapshot(config), "context": effective_context}
    return {
        "ticker": str(row.get("ticker") or "").upper().strip(),
        "candidate_rank": rank,
        "rank_inputs": rank_inputs,
        "eligibility_passed": bool(evaluation.get("approved")),
        "evaluation": evaluation,
        "inputs_hash": canonical_hash(evidence),
        "inputs": evidence,
        "policy_hash": canonical_hash(evidence["policy"]),
        "context_hash": canonical_hash(effective_context),
        "stop_loss": None,
        "target_price": None,
        "expected_r_multiple": None,
        "critic_outcome": context.get("critic_outcome") or {},
        "sizing": {},
    }


def _finish(decision: dict[str, Any], gate: str, code: str, reasons: list[str], disposition: str = "rejected") -> dict[str, Any]:
    return {
        **decision,
        "final_gate": gate,
        "reason_code": code,
        "reasons": [str(reason) for reason in reasons] or [code],
        "disposition": disposition,
    }


def decide_candidate(*, row: Any, rank: int, config: PaperTradeConfig, context: dict[str, Any]) -> dict[str, Any]:
    """Pure live/replay policy interface returning one complete decision."""
    from trader_koo.paper_trade.decision import compute_position_plan, compute_stop_and_target, evaluate_setup_for_paper_trade

    if not isinstance(row, dict):
        malformed = {"ticker": f"__MALFORMED_{rank}", "raw_type": type(row).__name__}
        evaluation = {"approved": False, "gate_failures": [{"gate": "candidate_shape", "reason_code": "candidate_not_object"}]}
        return _finish(_base_decision(row=malformed, rank=rank, evaluation=evaluation, config=config, context=context), "candidate_shape", "candidate_not_object", ["Candidate must be an object."])

    normalized = dict(row)
    ticker = str(normalized.get("ticker") or "").upper().strip()
    if not ticker:
        normalized["ticker"] = f"__MISSING_{rank}"
        evaluation = {"approved": False, "gate_failures": [{"gate": "candidate_identity", "reason_code": "missing_ticker"}]}
        return _finish(_base_decision(row=normalized, rank=rank, evaluation=evaluation, config=config, context=context), "candidate_identity", "missing_ticker", ["Candidate ticker is missing."])
    normalized["ticker"] = ticker
    evaluation = evaluate_setup_for_paper_trade(normalized, config=config)
    decision = _base_decision(row=normalized, rank=rank, evaluation=evaluation, config=config, context=context)
    if not evaluation["approved"]:
        failures = list(evaluation.get("gate_failures") or [])
        first = failures[0] if failures else {"gate": "eligibility", "reason_code": "eligibility_rejected"}
        return _finish(decision, str(first["gate"]), str(first["reason_code"]), list(evaluation.get("decision_reasons") or ["Candidate failed eligibility policy."]))

    block = context.get("portfolio_block")
    if isinstance(block, dict):
        return _finish(decision, str(block["gate"]), str(block["reason_code"]), [str(block["detail"])])

    entry_price = context.get("entry_price")
    try:
        entry = float(entry_price) if entry_price is not None else float(normalized["close"])
        levels = compute_stop_and_target(normalized, str(evaluation["direction"]), config=config, entry_price=entry)
        plan = compute_position_plan(normalized, evaluation, levels, config=config, vix_level=context.get("vix_level"), entry_price=entry)
    except (KeyError, TypeError, ValueError) as exc:
        return _finish(decision, "trade_plan", "invalid_stop_target_or_fill", [f"Trade plan could not be computed: {type(exc).__name__}."])

    decision.update(
        stop_loss=levels.get("stop_loss"), target_price=levels.get("target_price"),
        expected_r_multiple=plan.get("expected_r_multiple"),
        sizing={key: plan.get(key) for key in (
            "position_size_pct", "risk_budget_pct", "stop_distance_pct", "expected_reward_pct",
            "expected_r_multiple", "entry_plan", "exit_plan", "sizing_summary", "review_status", "review_summary",
        )},
        plan=plan, levels=levels,
    )
    avg_volume = context.get("avg_daily_volume")
    if isinstance(avg_volume, (int, float)) and avg_volume > 0 and entry > 0:
        position_dollars = config.starting_capital * float(plan.get("position_size_pct") or 0) / 100
        adv_pct = (position_dollars / entry) / float(avg_volume) * 100
        decision["adv_pct"] = round(adv_pct, 6)
        if adv_pct > config.max_adv_pct:
            return _finish(decision, "liquidity", "position_exceeds_adv_limit", [f"Planned position is {adv_pct:.1f}% of ADV; policy maximum is {config.max_adv_pct:.1f}%."])

    expected_r = plan.get("expected_r_multiple")
    if not isinstance(expected_r, (int, float)) or expected_r < config.min_reward_r_multiple:
        return _finish(decision, "reward_risk", "minimum_reward_r_not_met", [f"Expected {float(expected_r or 0):.2f}R is below policy minimum {config.min_reward_r_multiple:.2f}R."])

    critic = context.get("critic_outcome") or {}
    if not critic.get("approved"):
        if critic.get("error"):
            return _finish(
                decision, "critic", "critic_infrastructure_error",
                ["Critic infrastructure failed; policy rejects by default."],
            )
        failed_check = str(critic.get("failed_check") or "infrastructure")
        return _finish(decision, f"critic.{failed_check}", f"critic_{failed_check}_rejected", list(critic.get("rejections") or ["Critic rejected candidate."]))
    if context.get("campaign_active") is not True:
        return _finish(decision, "campaign_lifecycle", "campaign_not_active", ["Campaign is not active; decision recorded in shadow mode."])
    if context.get("duplicate"):
        return _finish(decision, "admission", "duplicate_candidate", ["This campaign already contains the same report-date ticker and direction."], "duplicate")
    return _finish(decision, "admission", "admitted", ["Candidate passed the versioned paper campaign policy."], "admitted")


def persist_decision_set(conn: sqlite3.Connection, *, report_run_id: str, report_date: str, generated_ts: str, campaign_id: str, policy_version: str, request_hash: str, policy_hash: str, context_hash: str, decisions: list[dict[str, Any]], report_complete: bool, is_canonical: bool = True) -> bool:
    """Seal one ranked set; exact retries no-op and divergent retries fail."""
    payload = [{key: decision.get(key) for key in (
        "ticker", "candidate_rank", "rank_inputs", "eligibility_passed", "final_gate",
        "reason_code", "reasons", "inputs_hash", "policy_hash", "context_hash",
        "disposition", "stop_loss", "target_price", "expected_r_multiple", "critic_outcome", "sizing",
    )} for decision in decisions]
    candidates_hash = canonical_hash(payload)
    identity = (candidates_hash, policy_hash, context_hash, len(payload), int(report_complete), int(is_canonical))
    existing = conn.execute(
        "SELECT request_hash,candidates_hash,policy_hash,context_hash,candidate_count,report_complete,is_canonical FROM paper_decision_sets WHERE report_run_id=? AND campaign_id=?",
        (report_run_id, campaign_id),
    ).fetchone()
    if existing:
        if str(existing[0]) == request_hash and tuple(existing[1:]) == identity:
            return False
        raise DivergentDecisionSetError(f"divergent retry for report_run_id={report_run_id} campaign_id={campaign_id}")
    conn.execute(
        """INSERT INTO paper_decision_sets
           (report_run_id,campaign_id,report_date,generated_ts,policy_version,candidate_count,request_hash,candidates_hash,policy_hash,context_hash,report_complete,is_canonical,status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'sealed')""",
        (report_run_id,campaign_id,report_date,generated_ts,policy_version,len(payload),request_hash,candidates_hash,policy_hash,context_hash,int(report_complete),int(is_canonical)),
    )
    conn.executemany(
        """INSERT INTO paper_candidate_decisions
           (report_run_id,report_date,generated_ts,campaign_id,policy_version,ticker,candidate_rank,rank_inputs_json,eligibility_passed,final_gate,reason_code,reasons_json,inputs_hash,policy_hash,context_hash,disposition,stop_loss,target_price,expected_r_multiple,critic_outcome_json,sizing_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(
            report_run_id,report_date,generated_ts,campaign_id,policy_version,item["ticker"],item["candidate_rank"],canonical_json(item["rank_inputs"]),
            int(bool(item["eligibility_passed"])),item["final_gate"],item["reason_code"],canonical_json(item["reasons"]),item["inputs_hash"],item["policy_hash"],
            item["context_hash"],item["disposition"],item["stop_loss"],item["target_price"],item["expected_r_multiple"],canonical_json(item["critic_outcome"]),canonical_json(item["sizing"]),
        ) for item in payload],
    )
    return True


def transition_campaign(conn: sqlite3.Connection, *, campaign_id: str, action: str, actor: str, reason: str, idempotency_key: str) -> dict[str, Any]:
    """Atomically activate or roll back a campaign and append an audit fact."""
    if action not in {"activate", "rollback"}:
        raise ValueError("action must be activate or rollback")
    if not actor.strip() or not reason.strip() or not idempotency_key.strip():
        raise ValueError("actor, reason, and idempotency_key are required")
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Recheck only after acquiring the write lock so concurrent retries
        # resolve to the same canonical audit fact instead of racing its insert.
        prior = conn.execute(
            "SELECT campaign_id,action,from_status,to_status FROM paper_campaign_audit WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        if prior:
            if str(prior[0]) != campaign_id or str(prior[1]) != action:
                raise ValueError("idempotency key was already used for a different transition")
            conn.commit()
            return {
                "campaign_id": prior[0], "action": prior[1],
                "from_status": prior[2], "to_status": prior[3], "idempotent": True,
            }
        row = conn.execute("SELECT status FROM paper_campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
        if not row:
            raise ValueError(f"unknown campaign {campaign_id}")
        before = str(row[0])
        if action == "activate":
            if before == "frozen":
                raise ValueError("frozen campaigns cannot be reactivated")
            conn.execute("UPDATE paper_campaigns SET status='draft' WHERE status='active' AND campaign_id!=?", (campaign_id,))
            after = "active"
        else:
            if before != "active":
                raise ValueError("only the active campaign can be rolled back")
            after = "draft"
        conn.execute("UPDATE paper_campaigns SET status=?,updated_ts=CURRENT_TIMESTAMP WHERE campaign_id=?", (after,campaign_id))
        conn.execute(
            "INSERT INTO paper_campaign_audit (campaign_id,action,actor,reason,idempotency_key,from_status,to_status) VALUES (?,?,?,?,?,?,?)",
            (campaign_id,action,actor,reason,idempotency_key,before,after),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"campaign_id": campaign_id, "action": action, "from_status": before, "to_status": after, "idempotent": False}


def campaign_health(conn: sqlite3.Connection, *, campaign_id: str) -> dict[str, Any]:
    campaign = conn.execute(
        "SELECT campaign_id,label,policy_version,status,starting_capital,zero_admission_streak_limit,replay_live_parity FROM paper_campaigns WHERE campaign_id=?",
        (campaign_id,),
    ).fetchone()
    if not campaign:
        return {"available": False, "campaign_id": campaign_id}
    campaigns = [{
        "campaign_id": row[0], "label": row[1], "policy_version": row[2], "status": row[3],
        "starting_capital": float(row[4]), "trade_count": int(row[5] or 0),
    } for row in conn.execute(
        """SELECT c.campaign_id,c.label,c.policy_version,c.status,c.starting_capital,COUNT(t.id)
           FROM paper_campaigns c LEFT JOIN paper_trades t ON t.campaign_id=c.campaign_id
           GROUP BY c.campaign_id ORDER BY c.created_ts,c.campaign_id"""
    ).fetchall()]
    rows = conn.execute(
        """SELECT s.report_run_id,s.report_date,s.generated_ts,s.candidate_count,
                  COALESCE(SUM(d.eligibility_passed),0),COALESCE(SUM(d.disposition='rejected'),0),
                  COALESCE(SUM(d.disposition='admitted'),0),
                  COALESCE(SUM(CASE WHEN d.disposition='admitted' THEN json_extract(d.sizing_json,'$.position_size_pct') ELSE 0 END),0)
           FROM paper_decision_sets s LEFT JOIN paper_candidate_decisions d
             ON d.report_run_id=s.report_run_id AND d.campaign_id=s.campaign_id
           WHERE s.campaign_id=? AND s.status='sealed' AND s.report_complete=1 AND s.is_canonical=1
             AND NOT EXISTS (SELECT 1 FROM paper_decision_sets newer
                 WHERE newer.campaign_id=s.campaign_id AND newer.report_date=s.report_date
                   AND newer.status='sealed' AND newer.report_complete=1 AND newer.is_canonical=1
                   AND (newer.generated_ts>s.generated_ts OR (newer.generated_ts=s.generated_ts AND newer.report_run_id>s.report_run_id)))
           GROUP BY s.report_run_id,s.report_date,s.generated_ts,s.candidate_count
           ORDER BY s.report_date,s.generated_ts,s.report_run_id""",
        (campaign_id,),
    ).fetchall()
    reports = [{
        "report_run_id": row[0], "report_date": row[1], "generated_ts": row[2], "ranked": int(row[3]),
        "eligible": int(row[4]), "rejected": int(row[5]), "admitted": int(row[6]), "exposure_pct": round(float(row[7]),2),
        "conversion_rate_pct": round(float(row[6])/float(row[4])*100,1) if row[4] else 0.0,
    } for row in rows]
    streak = 0
    for report in reversed(reports):
        if report["eligible"] > 0 and report["admitted"] == 0:
            streak += 1
        else:
            break
    latest = reports[-1] if reports else None
    if latest:
        latest["rejections_by_gate"] = [{"gate": row[0], "reason_code": row[1], "count": int(row[2])} for row in conn.execute(
            """SELECT final_gate,reason_code,COUNT(*) FROM paper_candidate_decisions
               WHERE campaign_id=? AND report_run_id=? AND disposition='rejected'
               GROUP BY final_gate,reason_code ORDER BY COUNT(*) DESC,final_gate,reason_code""",
            (campaign_id,latest["report_run_id"]),
        ).fetchall()]
        latest["candidates"] = [{
            "rank": row[0], "ticker": row[1], "eligibility_passed": bool(row[2]), "final_gate": row[3],
            "reason_code": row[4], "disposition": row[5], "expected_r_multiple": row[6],
        } for row in conn.execute(
            """SELECT candidate_rank,ticker,eligibility_passed,final_gate,reason_code,disposition,expected_r_multiple
               FROM paper_candidate_decisions WHERE campaign_id=? AND report_run_id=? ORDER BY candidate_rank""",
            (campaign_id,latest["report_run_id"]),
        ).fetchall()]
    limit = int(campaign[5])
    parity = str(campaign[6])
    reasons: list[str] = []
    if streak >= limit:
        reasons.append("eligible_candidate_zero_admission_streak")
    if parity == "diverged":
        reasons.append("replay_live_divergence")
    elif parity != "matched":
        reasons.append("replay_live_parity_not_measured")
    if str(campaign[3]) != "active":
        reasons.append("campaign_not_active")
    return {
        "available": True, "campaign_id": campaign[0], "campaigns": campaigns, "label": campaign[1],
        "policy_version": campaign[2], "status": campaign[3], "starting_capital": float(campaign[4]),
        "reports_observed": len(reports), "latest_report": latest,
        "consecutive_eligible_zero_admission_reports": streak, "zero_admission_streak_limit": limit,
        "replay_live_parity": parity, "healthy": not reasons, "health_reasons": reasons,
    }

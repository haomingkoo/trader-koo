"""Governed paper campaigns, pure policy decisions, and sealed evidence sets."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig, config_snapshot


class DivergentDecisionSetError(RuntimeError):
    """A retry reused report identity with different decision evidence."""


class EvidenceIntegrityError(RuntimeError):
    """Persisted campaign evidence does not match its sealed manifest."""


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
        "portfolio_context", "source_context", "execution_ready",
        "execution_pending_reason",
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
    if context.get("execution_ready") is False:
        pending_reason = str(
            context.get("execution_pending_reason") or "scheduled_ticker_open_missing"
        )
        return _finish(
            decision,
            "execution.next_open",
            pending_reason,
            ["Order is pending until the exact scheduled-session observation is backfilled."],
            "pending",
        )
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
    # Children are written first.  The manifest insert is the atomic seal and a
    # database trigger prevents every later child INSERT as well as mutations.
    conn.executemany(
        """INSERT INTO paper_candidate_decisions
           (report_run_id,report_date,generated_ts,campaign_id,policy_version,ticker,candidate_rank,rank_inputs_json,eligibility_passed,final_gate,reason_code,reasons_json,inputs_hash,policy_hash,context_hash,disposition,tradeability,inputs_json,stop_loss,target_price,expected_r_multiple,critic_outcome_json,sizing_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(
            report_run_id,report_date,generated_ts,campaign_id,policy_version,item["ticker"],item["candidate_rank"],canonical_json(item["rank_inputs"]),
            int(bool(item["eligibility_passed"])),item["final_gate"],item["reason_code"],canonical_json(item["reasons"]),item["inputs_hash"],item["policy_hash"],
            item["context_hash"],item["disposition"],
            ("actionable" if item["disposition"] == "admitted" else "pending_next_open" if item["disposition"] == "pending" else "not_actionable"),
            canonical_json(decisions[index].get("inputs") or {}),
            item["stop_loss"],item["target_price"],item["expected_r_multiple"],canonical_json(item["critic_outcome"]),canonical_json(item["sizing"]),
        ) for index, item in enumerate(payload)],
    )
    conn.execute(
        """INSERT INTO paper_decision_sets
           (report_run_id,campaign_id,report_date,generated_ts,policy_version,candidate_count,request_hash,candidates_hash,policy_hash,context_hash,report_complete,is_canonical,status)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'sealed')""",
        (report_run_id,campaign_id,report_date,generated_ts,policy_version,len(payload),request_hash,candidates_hash,policy_hash,context_hash,int(report_complete),int(is_canonical)),
    )
    return True


def verify_decision_set(conn: sqlite3.Connection, *, report_run_id: str, campaign_id: str) -> None:
    """Fail closed if API-visible children differ from their sealed manifest."""
    manifest = conn.execute(
        "SELECT candidate_count,candidates_hash FROM paper_decision_sets WHERE report_run_id=? AND campaign_id=? AND status='sealed'",
        (report_run_id, campaign_id),
    ).fetchone()
    if not manifest:
        raise EvidenceIntegrityError("decision set is not sealed")
    rows = conn.execute(
        """SELECT ticker,candidate_rank,rank_inputs_json,eligibility_passed,final_gate,
                  reason_code,reasons_json,inputs_hash,policy_hash,context_hash,
                  disposition,stop_loss,target_price,expected_r_multiple,
                  critic_outcome_json,sizing_json
           FROM paper_candidate_decisions
           WHERE report_run_id=? AND campaign_id=? ORDER BY candidate_rank""",
        (report_run_id, campaign_id),
    ).fetchall()
    payload = []
    for row in rows:
        payload.append({
            "ticker": row[0], "candidate_rank": row[1],
            "rank_inputs": json.loads(row[2]), "eligibility_passed": bool(row[3]),
            "final_gate": row[4], "reason_code": row[5], "reasons": json.loads(row[6]),
            "inputs_hash": row[7], "policy_hash": row[8], "context_hash": row[9],
            "disposition": row[10], "stop_loss": row[11], "target_price": row[12],
            "expected_r_multiple": row[13], "critic_outcome": json.loads(row[14] or "{}"),
            "sizing": json.loads(row[15] or "{}"),
        })
    if len(payload) != int(manifest[0]) or canonical_hash(payload) != str(manifest[1]):
        raise EvidenceIntegrityError("sealed decision set candidate count/hash mismatch")


def record_experiment_preregistration(
    conn: sqlite3.Connection,
    *,
    preregistration_id: str,
    campaign_id: str,
    policy_version: str,
    policy_hash: str,
    dataset_hash: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Seal gates and exact replay cohort before any result is recorded."""
    for label, value in (("policy_hash", policy_hash), ("dataset_hash", dataset_hash)):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError(f"{label} must be a SHA-256 hex digest")
    campaign = conn.execute(
        "SELECT policy_version,policy_hash FROM paper_campaigns WHERE campaign_id=?",
        (campaign_id,),
    ).fetchone()
    if not campaign or str(campaign[0]) != policy_version:
        raise ValueError("experiment policy version does not match campaign")
    if str(campaign[1] or "") and str(campaign[1]) != policy_hash:
        raise ValueError("experiment policy hash does not match sealed campaign policy")
    if not str(campaign[1] or ""):
        conn.execute(
            "UPDATE paper_campaigns SET policy_hash=? WHERE campaign_id=?",
            (policy_hash, campaign_id),
        )
    risk = gates.get("risk_gates") or {}
    active = gates.get("active_return_gate") or {}
    if not risk or not active:
        raise ValueError("preregistered risk and active-return gates are required")
    if "max_drawdown_pct" not in risk or "minimum_pct" not in active:
        raise ValueError("promotion gate thresholds are incomplete")
    artifact = {
        "preregistration_id": preregistration_id, "campaign_id": campaign_id,
        "policy_version": policy_version, "policy_hash": policy_hash,
        "dataset_hash": dataset_hash, "gates": gates,
    }
    artifact_hash = canonical_hash(artifact)
    conn.execute(
        """INSERT INTO paper_campaign_preregistrations
           (preregistration_id,campaign_id,policy_version,policy_hash,dataset_hash,
            gates_json,artifact_hash) VALUES (?,?,?,?,?,?,?)""",
        (preregistration_id,campaign_id,policy_version,policy_hash,dataset_hash,
         canonical_json(gates),artifact_hash),
    )
    return {**artifact, "artifact_hash": artifact_hash}


def record_promotion_experiment(
    conn: sqlite3.Connection,
    *,
    experiment_id: str,
    preregistration_id: str,
    campaign_id: str,
    policy_version: str,
    policy_hash: str,
    dataset_hash: str,
    metrics: dict[str, Any],
    parity_status: str,
) -> dict[str, Any]:
    """Seal replay results against an earlier immutable preregistration."""
    if parity_status not in {"matched", "diverged"}:
        raise ValueError("parity_status must be measured")
    prereg = conn.execute(
        """SELECT campaign_id,policy_version,policy_hash,dataset_hash,gates_json
           FROM paper_campaign_preregistrations WHERE preregistration_id=?""",
        (preregistration_id,),
    ).fetchone()
    if not prereg or tuple(map(str, prereg[:4])) != (
        campaign_id, policy_version, policy_hash, dataset_hash
    ):
        raise ValueError("replay evidence does not match immutable preregistration")
    gates = json.loads(str(prereg[4]))
    risk = gates["risk_gates"]
    active = gates["active_return_gate"]
    if "max_drawdown_pct" not in metrics or "matched_spy_active_return_pct" not in metrics:
        raise ValueError("promotion metrics are incomplete")
    required_metrics = {
        "closed_trades", "conversion_rate_pct", "average_exposure_pct",
        "turnover_pct", "portfolio_return_pct", "matched_spy_return_pct",
        "profit_factor", "mean_trade_return_pct_ci95", "walk_forward",
        "held_out", "engine_version",
    }
    if not required_metrics.issubset(metrics):
        raise ValueError("promotion requires complete replay, walk-forward, and held-out metrics")
    if metrics.get("engine_version") != "portfolio-execution-v1.0":
        raise ValueError("promotion replay engine version is not eligible")
    walk_forward = metrics.get("walk_forward") or {}
    held_out = metrics.get("held_out") or {}
    if not walk_forward.get("folds") or not isinstance(held_out.get("metrics"), dict):
        raise ValueError("promotion requires non-empty walk-forward folds and held-out evidence")
    max_dd = float(metrics["max_drawdown_pct"])
    active_return = float(metrics["matched_spy_active_return_pct"])
    if not math.isfinite(max_dd) or not math.isfinite(active_return):
        raise ValueError("promotion metrics must be finite")
    risk_pass = max_dd <= float(risk["max_drawdown_pct"])
    active_pass = active_return >= float(active["minimum_pct"])
    eligible = parity_status == "matched" and risk_pass and active_pass
    evidence = {
        "experiment_id": experiment_id, "preregistration_id": preregistration_id,
        "campaign_id": campaign_id,
        "policy_version": policy_version, "policy_hash": policy_hash,
        "dataset_hash": dataset_hash, "gates": gates,
        "metrics": metrics, "parity_status": parity_status,
        "risk_gate_passed": risk_pass,
        "active_return_gate_passed": active_pass, "eligible": eligible,
    }
    evidence_hash = canonical_hash(evidence)
    conn.execute(
        """INSERT INTO paper_campaign_experiments
           (experiment_id,preregistration_id,campaign_id,policy_version,policy_hash,dataset_hash,
            preregistration_json,metrics_json,parity_status,risk_gate_passed,
            active_return_gate_passed,eligible,evidence_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (experiment_id,preregistration_id,campaign_id,policy_version,policy_hash,dataset_hash,
         canonical_json(gates),canonical_json(metrics),parity_status,
         int(risk_pass),int(active_pass),int(eligible),evidence_hash),
    )
    conn.execute(
        "UPDATE paper_campaigns SET replay_live_parity=?,updated_ts=CURRENT_TIMESTAMP WHERE campaign_id=?",
        (parity_status, campaign_id),
    )
    return {**evidence, "evidence_hash": evidence_hash}


def record_human_approval(
    conn: sqlite3.Connection,
    *,
    approval_id: str,
    experiment_id: str,
    campaign_id: str,
    actor: str,
    reason: str,
    experiment_evidence_hash: str,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Append an explicit immutable human approval artifact."""
    if not actor.strip() or not reason.strip() or artifact.get("approved") is not True:
        raise ValueError("actor, reason, and an explicit approved=true artifact are required")
    experiment = conn.execute(
        """SELECT evidence_hash,eligible FROM paper_campaign_experiments
           WHERE experiment_id=? AND campaign_id=?""",
        (experiment_id, campaign_id),
    ).fetchone()
    if (
        experiment is None
        or int(experiment[1]) != 1
        or str(experiment[0]) != str(experiment_evidence_hash)
    ):
        raise ValueError("approval must bind the exact eligible experiment evidence hash")
    body = {
        "approval_id": approval_id, "experiment_id": experiment_id,
        "campaign_id": campaign_id, "actor": actor, "reason": reason,
        "experiment_evidence_hash": experiment_evidence_hash,
        "artifact": artifact,
    }
    artifact_hash = canonical_hash(body)
    conn.execute(
        """INSERT INTO paper_campaign_approvals
           (approval_id,experiment_id,campaign_id,actor,reason,
            experiment_evidence_hash,artifact_json,artifact_hash)
           VALUES (?,?,?,?,?,?,?,?)""",
        (approval_id,experiment_id,campaign_id,actor,reason,
         experiment_evidence_hash,canonical_json(artifact),artifact_hash),
    )
    return {**body, "artifact_hash": artifact_hash}


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
        request_hash = canonical_hash({
            "campaign_id": campaign_id, "action": action, "actor": actor,
            "reason": reason, "idempotency_key": idempotency_key,
        })
        prior = conn.execute(
            "SELECT campaign_id,action,from_status,to_status,request_hash FROM paper_campaign_audit WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        if prior:
            if str(prior[4]) != request_hash:
                raise ValueError("idempotency key was already used for a different request payload")
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
            from trader_koo.paper_trade.schema import require_contracted_paper_schema

            require_contracted_paper_schema(conn)
            if before == "frozen":
                raise ValueError("frozen campaigns cannot be reactivated")
            promotion = conn.execute(
                """SELECT e.experiment_id,e.policy_version,e.policy_hash,
                          e.eligible,e.parity_status,e.risk_gate_passed,
                          e.active_return_gate_passed,
                          EXISTS (
                              SELECT 1 FROM paper_campaign_approvals a
                              WHERE a.experiment_id=e.experiment_id
                                AND a.campaign_id=e.campaign_id
                                AND a.experiment_evidence_hash=e.evidence_hash
                          )
                   FROM paper_campaign_experiments e
                   WHERE e.campaign_id=?
                   ORDER BY e.rowid DESC LIMIT 1""",
                (campaign_id,),
            ).fetchone()
            campaign_policy = conn.execute(
                "SELECT policy_version,replay_live_parity,policy_hash FROM paper_campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
            decision_set = conn.execute(
                """SELECT policy_version,policy_hash,report_complete,is_canonical,status
                   FROM paper_decision_sets
                   WHERE campaign_id=?
                   ORDER BY rowid DESC LIMIT 1""",
                (campaign_id,),
            ).fetchone()
            promotion_ready = bool(
                promotion and campaign_policy
                and int(promotion[3]) == 1 and str(promotion[4]) == "matched"
                and int(promotion[5]) == 1 and int(promotion[6]) == 1
                and int(promotion[7]) == 1 and str(campaign_policy[1]) == "matched"
            )
            observation_ready = bool(
                decision_set and campaign_policy
                and int(decision_set[2]) == 1
                and int(decision_set[3]) == 1
                and str(decision_set[4]) == "sealed"
            )
            if not promotion_ready and not observation_ready:
                raise ValueError(
                    "activation requires a canonical sealed report decision or eligible promotion evidence"
                )
            evidence = promotion if promotion_ready else decision_set
            evidence_label = "promotion evidence" if promotion_ready else "report decision"
            if str(evidence[1 if promotion_ready else 0]) != str(campaign_policy[0]):
                raise ValueError(f"{evidence_label} policy version does not match campaign")
            if str(evidence[2 if promotion_ready else 1]) != str(campaign_policy[2]):
                raise ValueError(f"{evidence_label} policy hash does not match campaign")
            conn.execute("UPDATE paper_campaigns SET status='draft' WHERE status='active' AND campaign_id!=?", (campaign_id,))
            after = "active"
        else:
            if before != "active":
                raise ValueError("only the active campaign can be rolled back")
            after = "draft"
        conn.execute("UPDATE paper_campaigns SET status=?,updated_ts=CURRENT_TIMESTAMP WHERE campaign_id=?", (after,campaign_id))
        conn.execute(
            "INSERT INTO paper_campaign_audit (campaign_id,action,actor,reason,idempotency_key,request_hash,from_status,to_status) VALUES (?,?,?,?,?,?,?,?)",
            (campaign_id,action,actor,reason,idempotency_key,request_hash,before,after),
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
                  COALESCE(SUM(d.eligibility_passed),0),
                  COALESCE(SUM(d.disposition='rejected' OR (d.disposition='pending' AND po.status='rejected')),0),
                  COALESCE(SUM(d.disposition='admitted' OR (d.disposition='pending' AND po.status='filled')),0),
                  COALESCE(SUM(CASE WHEN d.disposition='admitted' OR (d.disposition='pending' AND po.status='filled') THEN json_extract(d.sizing_json,'$.position_size_pct') ELSE 0 END),0)
           FROM paper_decision_sets s LEFT JOIN paper_candidate_decisions d
             ON d.report_run_id=s.report_run_id AND d.campaign_id=s.campaign_id
           LEFT JOIN paper_pending_orders po
             ON po.report_run_id=d.report_run_id AND po.campaign_id=d.campaign_id
            AND po.candidate_rank=d.candidate_rank
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
            "tradeability": row[7], "execution_status": row[8],
        } for row in conn.execute(
            """SELECT d.candidate_rank,d.ticker,d.eligibility_passed,d.final_gate,
                      d.reason_code,d.disposition,d.expected_r_multiple,
                      CASE WHEN po.status='filled' THEN 'actionable'
                           WHEN po.status='pending' THEN 'pending_next_open'
                           ELSE d.tradeability END,
                      po.status
               FROM paper_candidate_decisions d
               LEFT JOIN paper_pending_orders po
                 ON po.report_run_id=d.report_run_id AND po.campaign_id=d.campaign_id
                AND po.candidate_rank=d.candidate_rank
               WHERE d.campaign_id=? AND d.report_run_id=? ORDER BY d.candidate_rank""",
            (campaign_id,latest["report_run_id"]),
        ).fetchall()]
    limit = int(campaign[5])
    parity = str(campaign[6])
    reasons: list[str] = []
    if streak >= limit:
        reasons.append("eligible_candidate_zero_admission_streak")
    if parity == "diverged":
        reasons.append("replay_live_divergence")
    if str(campaign[3]) != "active":
        reasons.append("campaign_not_active")
    latest_experiment = conn.execute(
        """SELECT experiment_id,eligible,risk_gate_passed,
                  active_return_gate_passed,parity_status
           FROM paper_campaign_experiments WHERE campaign_id=?
           ORDER BY rowid DESC LIMIT 1""",
        (campaign_id,),
    ).fetchone()
    promotion = None
    if latest_experiment:
        promotion = {
            "experiment_id": latest_experiment[0],
            "eligible": bool(latest_experiment[1]),
            "risk_gate_passed": bool(latest_experiment[2]),
            "active_return_gate_passed": bool(latest_experiment[3]),
            "parity_status": latest_experiment[4],
        }
        if not promotion["risk_gate_passed"]:
            reasons.append("promotion_risk_gate_failed")
        if not promotion["active_return_gate_passed"]:
            reasons.append("promotion_active_return_gate_failed")
    return {
        "available": True, "campaign_id": campaign[0], "campaigns": campaigns, "label": campaign[1],
        "policy_version": campaign[2], "status": campaign[3], "starting_capital": float(campaign[4]),
        "reports_observed": len(reports), "latest_report": latest,
        "consecutive_eligible_zero_admission_reports": streak, "zero_admission_streak_limit": limit,
        "replay_live_parity": parity, "healthy": not reasons, "health_reasons": reasons,
        "latest_promotion_experiment": promotion,
    }

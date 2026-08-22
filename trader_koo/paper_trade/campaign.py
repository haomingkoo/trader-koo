"""Versioned paper-campaign decision ledger and funnel health."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig, config_snapshot


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_candidate_decision(
    *,
    row: dict[str, Any],
    rank: int,
    evaluation: dict[str, Any],
    levels: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    critic: dict[str, Any] | None,
    final_gate: str,
    reason_code: str,
    reasons: list[str],
    disposition: str,
    config: PaperTradeConfig,
) -> dict[str, Any]:
    """Return the canonical live/replay decision record without touching storage."""
    levels = levels or {}
    plan = plan or {}
    critic = critic or {}
    rank_inputs = {
        key: row.get(key)
        for key in (
            "score",
            "setup_tier",
            "confirmation_count",
            "contradiction_count",
            "pct_change",
            "actionability",
            "signal_bias",
            "setup_family",
        )
    }
    policy = config_snapshot(config)
    return {
        "ticker": str(row.get("ticker") or "").upper().strip(),
        "candidate_rank": rank,
        "rank_inputs": rank_inputs,
        "eligibility_passed": bool(evaluation.get("approved")),
        "final_gate": final_gate,
        "reason_code": reason_code,
        "reasons": reasons,
        "inputs_hash": canonical_hash({"candidate": row, "policy": policy}),
        "disposition": disposition,
        "stop_loss": levels.get("stop_loss"),
        "target_price": levels.get("target_price"),
        "expected_r_multiple": plan.get("expected_r_multiple"),
        "critic_outcome": critic,
        "sizing": {
            key: plan.get(key)
            for key in ("position_size_pct", "risk_budget_pct", "sizing_summary")
        },
    }


def persist_candidate_decision(
    conn: sqlite3.Connection,
    *,
    report_run_id: str,
    report_date: str,
    generated_ts: str,
    campaign_id: str,
    policy_version: str,
    decision: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO paper_candidate_decisions (
            report_run_id, report_date, generated_ts, campaign_id, policy_version,
            ticker, candidate_rank, rank_inputs_json, eligibility_passed,
            final_gate, reason_code, reasons_json, inputs_hash, disposition,
            stop_loss, target_price, expected_r_multiple, critic_outcome_json,
            sizing_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(report_run_id, campaign_id, ticker) DO NOTHING
        """,
        (
            report_run_id,
            report_date,
            generated_ts,
            campaign_id,
            policy_version,
            decision["ticker"],
            decision["candidate_rank"],
            json.dumps(decision["rank_inputs"], sort_keys=True),
            int(decision["eligibility_passed"]),
            decision["final_gate"],
            decision["reason_code"],
            json.dumps(decision["reasons"]),
            decision["inputs_hash"],
            decision["disposition"],
            decision["stop_loss"],
            decision["target_price"],
            decision["expected_r_multiple"],
            json.dumps(decision["critic_outcome"], sort_keys=True),
            json.dumps(decision["sizing"], sort_keys=True),
        ),
    )


def campaign_health(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
) -> dict[str, Any]:
    campaign = conn.execute(
        """SELECT campaign_id, label, policy_version, status, starting_capital,
                  zero_admission_streak_limit, replay_live_parity
           FROM paper_campaigns WHERE campaign_id = ?""",
        (campaign_id,),
    ).fetchone()
    if not campaign:
        return {"available": False, "campaign_id": campaign_id}
    campaigns = [
        {
            "campaign_id": row[0],
            "label": row[1],
            "policy_version": row[2],
            "status": row[3],
            "starting_capital": float(row[4]),
            "trade_count": int(row[5] or 0),
        }
        for row in conn.execute(
            """SELECT c.campaign_id, c.label, c.policy_version, c.status,
                      c.starting_capital, COUNT(t.id)
               FROM paper_campaigns c
               LEFT JOIN paper_trades t ON t.campaign_id = c.campaign_id
               GROUP BY c.campaign_id
               ORDER BY c.created_ts, c.campaign_id"""
        ).fetchall()
    ]

    rows = conn.execute(
        """
        SELECT report_run_id, report_date, MAX(generated_ts), COUNT(*),
               SUM(eligibility_passed),
               SUM(CASE WHEN disposition = 'rejected' THEN 1 ELSE 0 END),
               SUM(CASE WHEN disposition = 'admitted' THEN 1 ELSE 0 END),
               SUM(CASE WHEN disposition = 'admitted'
                        THEN COALESCE(json_extract(sizing_json, '$.position_size_pct'), 0)
                        ELSE 0 END)
        FROM paper_candidate_decisions
        WHERE campaign_id = ?
        GROUP BY report_run_id, report_date
        ORDER BY report_date, MAX(generated_ts), report_run_id
        """,
        (campaign_id,),
    ).fetchall()
    reports = [
        {
            "report_run_id": row[0],
            "report_date": row[1],
            "generated_ts": row[2],
            "ranked": int(row[3] or 0),
            "eligible": int(row[4] or 0),
            "rejected": int(row[5] or 0),
            "admitted": int(row[6] or 0),
            "exposure_pct": round(float(row[7] or 0), 2),
            "conversion_rate_pct": (
                round(float(row[6] or 0) / float(row[4]) * 100, 1)
                if row[4]
                else 0.0
            ),
        }
        for row in rows
    ]
    streak = 0
    for report in reversed(reports):
        if report["eligible"] > 0 and report["admitted"] == 0:
            streak += 1
        else:
            break
    streak_limit = int(campaign[5])
    parity = str(campaign[6])
    latest = reports[-1] if reports else None
    if latest:
        latest["rejections_by_gate"] = [
            {"gate": row[0], "reason_code": row[1], "count": int(row[2])}
            for row in conn.execute(
                """SELECT final_gate, reason_code, COUNT(*)
                   FROM paper_candidate_decisions
                   WHERE campaign_id = ? AND report_run_id = ?
                     AND disposition = 'rejected'
                   GROUP BY final_gate, reason_code
                   ORDER BY COUNT(*) DESC, final_gate, reason_code""",
                (campaign_id, latest["report_run_id"]),
            ).fetchall()
        ]
    return {
        "available": True,
        "campaign_id": campaign[0],
        "campaigns": campaigns,
        "label": campaign[1],
        "policy_version": campaign[2],
        "status": campaign[3],
        "starting_capital": float(campaign[4]),
        "reports_observed": len(reports),
        "latest_report": latest,
        "consecutive_eligible_zero_admission_reports": streak,
        "zero_admission_streak_limit": streak_limit,
        "replay_live_parity": parity,
        "healthy": streak < streak_limit and parity != "diverged",
        "health_reasons": [
            reason
            for condition, reason in (
                (streak >= streak_limit, "eligible_candidate_zero_admission_streak"),
                (parity == "diverged", "replay_live_divergence"),
            )
            if condition
        ],
    }

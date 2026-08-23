"""Frozen P0/P1 prospective breadth shadow with no order-writing interface."""
from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import replace
from typing import Any

from trader_koo.db.price_contract import research_price_contract
from trader_koo.paper_trade.campaign import canonical_hash, canonical_json
from trader_koo.paper_trade.config import PaperTradeConfig, config_snapshot
from trader_koo.paper_trade.decision import (
    direction_from_row,
    evaluate_setup_for_paper_trade,
)
from trader_koo.research.next_open_baseline import (
    BaselineConfig,
    ExecutionDecision,
    SessionPrice,
    simulate_portfolio,
)

SHADOW_START_TS = "2026-08-23T00:00:00Z"
SHADOW_START_DATE = "2026-08-23"
SHADOW_POLICIES = {
    "P0": {"label": "current_a_b_control", "tier_change": None},
    "P1": {"label": "tier_c_breadth_challenger", "tier_change": "include_C"},
}


def ensure_shadow_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_shadow_policies (
            policy_id TEXT PRIMARY KEY,
            start_ts TEXT NOT NULL,
            specification_json TEXT NOT NULL,
            specification_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paper_shadow_decisions (
            decision_id TEXT PRIMARY KEY,
            report_run_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            generated_ts TEXT NOT NULL,
            policy_id TEXT NOT NULL REFERENCES paper_shadow_policies(policy_id),
            policy_version TEXT NOT NULL,
            candidate_rank INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            disposition TEXT NOT NULL CHECK (disposition IN ('accepted','rejected')),
            gate TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            feature_snapshot_json TEXT NOT NULL,
            feature_snapshot_hash TEXT NOT NULL,
            source_timestamps_json TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_run_id, policy_id, candidate_rank)
        );
        CREATE TABLE IF NOT EXISTS paper_shadow_decision_sets (
            report_run_id TEXT NOT NULL,
            policy_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            generated_ts TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            accepted_count INTEGER NOT NULL,
            decisions_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status='sealed'),
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (report_run_id, policy_id)
        );
        CREATE TABLE IF NOT EXISTS paper_shadow_outcomes (
            outcome_id TEXT PRIMARY KEY,
            decision_id TEXT NOT NULL UNIQUE REFERENCES paper_shadow_decisions(decision_id),
            intended_entry_date TEXT NOT NULL,
            entry_date TEXT,
            exit_date TEXT,
            status TEXT NOT NULL CHECK (status IN ('pending','resolved','invalid')),
            result_json TEXT NOT NULL,
            result_hash TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TRIGGER IF NOT EXISTS paper_shadow_policies_no_update
        BEFORE UPDATE ON paper_shadow_policies
        BEGIN SELECT RAISE(ABORT,'shadow policies are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_policies_no_delete
        BEFORE DELETE ON paper_shadow_policies
        BEGIN SELECT RAISE(ABORT,'shadow policies are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_decisions_no_update
        BEFORE UPDATE ON paper_shadow_decisions
        BEGIN SELECT RAISE(ABORT,'shadow decisions are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_decisions_no_delete
        BEFORE DELETE ON paper_shadow_decisions
        BEGIN SELECT RAISE(ABORT,'shadow decisions are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_sets_no_update
        BEFORE UPDATE ON paper_shadow_decision_sets
        BEGIN SELECT RAISE(ABORT,'shadow decision sets are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_sets_no_delete
        BEFORE DELETE ON paper_shadow_decision_sets
        BEGIN SELECT RAISE(ABORT,'shadow decision sets are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_decisions_no_insert_after_seal
        BEFORE INSERT ON paper_shadow_decisions
        WHEN EXISTS (
            SELECT 1 FROM paper_shadow_decision_sets
            WHERE report_run_id=NEW.report_run_id AND policy_id=NEW.policy_id
        )
        BEGIN SELECT RAISE(ABORT,'sealed shadow decision set is not appendable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_outcomes_no_update
        BEFORE UPDATE ON paper_shadow_outcomes
        BEGIN SELECT RAISE(ABORT,'shadow outcomes are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS paper_shadow_outcomes_no_delete
        BEFORE DELETE ON paper_shadow_outcomes
        BEGIN SELECT RAISE(ABORT,'shadow outcomes are immutable'); END;
    """)


def _policy_config(base: PaperTradeConfig, policy_id: str) -> PaperTradeConfig:
    if policy_id == "P0":
        return base
    if policy_id == "P1":
        return replace(
            base,
            min_tier="C",
            qualifying_tiers=frozenset({*base.qualifying_tiers, "C"}),
        )
    raise ValueError(f"unknown shadow policy {policy_id}")


def _specification(base: PaperTradeConfig, policy_id: str) -> dict[str, Any]:
    config = _policy_config(base, policy_id)
    return {
        "schema_version": "paper-breadth-shadow-v1",
        "policy_id": policy_id,
        "label": SHADOW_POLICIES[policy_id]["label"],
        "prospective_start_ts": SHADOW_START_TS,
        "primary_endpoint": "next_session_open_to_tenth_session_close",
        "maximum_rank": 20,
        "promotion": "separate_human_approved_issue_required",
        "can_create_orders": False,
        "policy": config_snapshot(config),
    }


def _register_policies(conn: sqlite3.Connection, base: PaperTradeConfig) -> None:
    for policy_id in SHADOW_POLICIES:
        spec = _specification(base, policy_id)
        spec_hash = canonical_hash(spec)
        prior = conn.execute(
            "SELECT start_ts,specification_hash FROM paper_shadow_policies WHERE policy_id=?",
            (policy_id,),
        ).fetchone()
        if prior:
            if tuple(map(str, prior)) != (SHADOW_START_TS, spec_hash):
                raise ValueError(f"frozen shadow policy {policy_id} changed")
            continue
        conn.execute(
            """INSERT INTO paper_shadow_policies
               (policy_id,start_ts,specification_json,specification_hash)
               VALUES (?,?,?,?)""",
            (policy_id, SHADOW_START_TS, canonical_json(spec), spec_hash),
        )


def _decision(
    row: Any,
    rank: int,
    policy_id: str,
    config: PaperTradeConfig,
    source_timestamps: dict[str, Any],
) -> dict[str, Any]:
    candidate = dict(row) if isinstance(row, dict) else {
        "ticker": f"__MALFORMED_{rank}", "raw_type": type(row).__name__,
    }
    ticker = str(candidate.get("ticker") or f"__MISSING_{rank}").upper().strip()
    evaluation = evaluate_setup_for_paper_trade(candidate, config=config)
    failures = list(evaluation.get("gate_failures") or [])
    if rank > 20:
        failures.insert(0, {
            "gate": "ranking", "reason_code": "rank_exceeds_shadow_maximum",
        })
    accepted = bool(evaluation.get("approved")) and rank <= 20
    first = failures[0] if failures else {
        "gate": "shadow_eligibility", "reason_code": "accepted",
    }
    reasons = list(evaluation.get("decision_reasons") or [])
    if rank > 20:
        reasons.insert(0, "Candidate rank exceeds the frozen maximum of 20.")
    feature_snapshot = candidate
    candidate_sources = {
        key: value for key, value in candidate.items()
        if key in {"price_date", "source_timestamps", "data_sources"}
        or key.endswith(("_date", "_timestamp", "_ts"))
    }
    effective_source_timestamps = {
        **source_timestamps,
        "candidate_sources": candidate_sources,
    }
    return {
        "policy_id": policy_id,
        "policy_version": config.decision_version,
        "candidate_rank": rank,
        "ticker": ticker,
        "disposition": "accepted" if accepted else "rejected",
        "gate": str(first["gate"]),
        "reason_code": str(first["reason_code"]),
        "reasons": reasons or [str(first["reason_code"])],
        "feature_snapshot": feature_snapshot,
        "feature_snapshot_hash": canonical_hash(feature_snapshot),
        "source_timestamps": effective_source_timestamps,
    }


def record_breadth_shadow(
    conn: sqlite3.Connection,
    *,
    report_run_id: str,
    report_date: str,
    generated_ts: str,
    setup_rows: list[Any],
    base_config: PaperTradeConfig,
) -> bool:
    """Seal P0/P1 decisions. This interface cannot create orders or trades."""
    if report_date < SHADOW_START_DATE:
        return False
    _register_policies(conn, base_config)
    report_sources: dict[str, Any] = {}
    report_row = conn.execute(
        "SELECT source_timestamps_json FROM report_runs WHERE run_id=?",
        (report_run_id,),
    ).fetchone()
    if report_row and report_row[0]:
        try:
            parsed_sources = json.loads(str(report_row[0]))
            if isinstance(parsed_sources, dict):
                report_sources = parsed_sources
        except json.JSONDecodeError:
            report_sources = {"status": "invalid_report_source_timestamps"}
    source_timestamps = {
        "report_date": report_date,
        "report_generated_ts": generated_ts,
        "report_sources": report_sources,
    }
    inserted = False
    for policy_id in SHADOW_POLICIES:
        config = _policy_config(base_config, policy_id)
        decisions = [
            _decision(row, rank, policy_id, config, source_timestamps)
            for rank, row in enumerate(setup_rows, start=1)
        ]
        decisions_hash = canonical_hash(decisions)
        prior = conn.execute(
            """SELECT candidate_count,accepted_count,decisions_hash
               FROM paper_shadow_decision_sets
               WHERE report_run_id=? AND policy_id=?""",
            (report_run_id, policy_id),
        ).fetchone()
        identity = (
            len(decisions),
            sum(row["disposition"] == "accepted" for row in decisions),
            decisions_hash,
        )
        if prior:
            if tuple(prior) != identity:
                raise ValueError(f"divergent shadow retry for {report_run_id}:{policy_id}")
            continue
        conn.executemany(
            """INSERT INTO paper_shadow_decisions
               (decision_id,report_run_id,report_date,generated_ts,policy_id,
                policy_version,candidate_rank,ticker,disposition,gate,reason_code,
                reasons_json,feature_snapshot_json,feature_snapshot_hash,
                source_timestamps_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(
                canonical_hash({
                    "report_run_id": report_run_id,
                    "policy_id": policy_id,
                    "candidate_rank": row["candidate_rank"],
                }),
                report_run_id, report_date, generated_ts, policy_id,
                row["policy_version"], row["candidate_rank"], row["ticker"],
                row["disposition"], row["gate"], row["reason_code"],
                canonical_json(row["reasons"]),
                canonical_json(row["feature_snapshot"]),
                row["feature_snapshot_hash"],
                canonical_json(row["source_timestamps"]),
            ) for row in decisions],
        )
        conn.execute(
            """INSERT INTO paper_shadow_decision_sets
               (report_run_id,policy_id,report_date,generated_ts,candidate_count,
                accepted_count,decisions_hash,status)
               VALUES (?,?,?,?,?,?,?,'sealed')""",
            (report_run_id, policy_id, report_date, generated_ts, *identity),
        )
        inserted = True
    return inserted


def _execution_result(
    *,
    decision_id: str,
    ticker: str,
    direction: str,
    report_date: str,
    entry_date: str,
    exit_date: str,
    sessions: list[str],
    prices: list[SessionPrice],
    capacity_notional: float,
    config: PaperTradeConfig,
) -> dict[str, Any]:
    execution = simulate_portfolio(
        [ExecutionDecision(
            decision_id=decision_id,
            ticker=ticker,
            direction=direction,
            signal_date=report_date,
            entry_date=entry_date,
            exit_date=exit_date,
            score=0,
            capacity_notional=capacity_notional,
            locked_notional=10_000,
        )],
        prices,
        sessions,
        BaselineConfig(
            initial_capital=100_000,
            max_positions=1,
            position_pct=10,
            max_name_pct=100,
            max_adv_pct=config.max_adv_pct,
            entry_slippage_bps=config.entry_slippage_bps,
            exit_slippage_bps=config.exit_slippage_bps,
            commission_bps_per_side=0,
            minimum_commission_per_side=config.commission_per_trade,
            short_borrow_bps_annual=config.short_borrow_annual_pct * 100,
            cash_rate_bps_annual=0,
            holding_sessions=10,
        ),
    )
    trade = dict(execution.trades[0]) if execution.trades else None
    return {
        "trade": trade,
        "rejections": [dict(row) for row in execution.exclusions],
        "ledger_sha256": execution.ledger["provenance"]["ledger_sha256"],
    }


def resolve_breadth_shadow_outcomes(
    conn: sqlite3.Connection,
    *,
    through_date: str,
    base_config: PaperTradeConfig,
) -> dict[str, int]:
    """Append matured P0/P1 outcomes; missing or unverified evidence fails closed."""
    rows = conn.execute(
        """SELECT d.decision_id,d.report_run_id,d.report_date,d.policy_id,d.ticker,
                  d.feature_snapshot_json
           FROM paper_shadow_decisions d
           LEFT JOIN paper_shadow_outcomes o ON o.decision_id=d.decision_id
           WHERE d.disposition='accepted' AND o.decision_id IS NULL
           ORDER BY d.report_date,d.report_run_id,d.policy_id,d.candidate_rank"""
    ).fetchall()
    if not rows:
        return {"resolved": 0, "invalid": 0, "pending": 0}
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='price_daily'"
    ).fetchone():
        return {"resolved": 0, "invalid": 0, "pending": len(rows)}
    sessions = [
        str(row[0]) for row in conn.execute(
            "SELECT date FROM price_daily WHERE ticker='SPY' AND date<=? ORDER BY date",
            (through_date,),
        )
    ]
    resolved = invalid = pending = 0
    for decision_id, _run_id, report_date, policy_id, ticker, feature_json in rows:
        future = [date for date in sessions if date > str(report_date)]
        if len(future) < 10:
            pending += 1
            continue
        entry_date, exit_date = future[0], future[9]
        contract = research_price_contract(conn, [str(ticker), "SPY"])
        if (
            not contract.get("eligible")
            or contract.get("basis") not in {
                "split_adjusted_price_only", "total_return",
            }
        ):
            pending += 1
            continue
        price_rows = conn.execute(
            """SELECT ticker,date,open,close,volume FROM price_daily
               WHERE ticker IN (?,?) AND date>=? AND date<=?
               ORDER BY ticker,date""",
            (str(ticker), "SPY", entry_date, exit_date),
        ).fetchall()
        prices = [SessionPrice(
            str(row[0]), str(row[1]), float(row[2]) if row[2] is not None else None,
            float(row[3]) if row[3] is not None else None,
            volume=float(row[4]) if row[4] is not None else None,
        ) for row in price_rows]
        prior_dollar_volume = [
            float(row[0]) * float(row[1])
            for row in conn.execute(
                """SELECT close,volume FROM price_daily
                   WHERE ticker=? AND date<=? AND close>0 AND volume>0
                   ORDER BY date DESC LIMIT 20""",
                (str(ticker), str(report_date)),
            )
        ]
        capacity = (
            statistics.median(prior_dollar_volume) * base_config.max_adv_pct / 100
            if prior_dollar_volume else 0.0
        )
        feature = json.loads(str(feature_json))
        direction = direction_from_row(feature)
        strategy = _execution_result(
            decision_id=str(decision_id), ticker=str(ticker), direction=direction,
            report_date=str(report_date), entry_date=entry_date, exit_date=exit_date,
            sessions=future[:10], prices=prices, capacity_notional=capacity,
            config=base_config,
        )
        control = _execution_result(
            decision_id=f"{decision_id}:spy", ticker="SPY", direction=direction,
            report_date=str(report_date), entry_date=entry_date, exit_date=exit_date,
            sessions=future[:10], prices=prices, capacity_notional=1e18,
            config=base_config,
        )
        stress_config = replace(
            base_config,
            entry_slippage_bps=base_config.entry_slippage_bps * 2,
            exit_slippage_bps=base_config.exit_slippage_bps * 2,
            commission_per_trade=base_config.commission_per_trade * 2,
            short_borrow_annual_pct=base_config.short_borrow_annual_pct * 2,
        )
        stress = _execution_result(
            decision_id=f"{decision_id}:stress", ticker=str(ticker),
            direction=direction, report_date=str(report_date),
            entry_date=entry_date, exit_date=exit_date, sessions=future[:10],
            prices=prices, capacity_notional=capacity, config=stress_config,
        )
        strategy_trade = strategy["trade"]
        control_trade = control["trade"]
        status = "resolved" if strategy_trade and control_trade else "invalid"
        result = {
            "schema_version": "paper-breadth-shadow-outcome-v1",
            "decision_id": decision_id,
            "policy_id": policy_id,
            "ticker": ticker,
            "direction": direction,
            "primary_endpoint": "next_session_open_to_tenth_session_close",
            "return_basis": contract["basis"],
            "adjustment_version": contract["version"],
            "price_revision": contract["revision"],
            "strategy": strategy,
            "matched_spy": control,
            "cost_stress_2x": stress,
            "net_return_pct": (
                float(strategy_trade["net_pnl"]) / float(strategy_trade["entry_notional"]) * 100
                if strategy_trade else None
            ),
            "matched_spy_return_pct": (
                float(control_trade["net_pnl"]) / float(control_trade["entry_notional"]) * 100
                if control_trade else None
            ),
            "causal_valid": status == "resolved",
        }
        result_hash = canonical_hash(result)
        conn.execute(
            """INSERT INTO paper_shadow_outcomes
               (outcome_id,decision_id,intended_entry_date,entry_date,exit_date,
                status,result_json,result_hash) VALUES (?,?,?,?,?,?,?,?)""",
            (
                canonical_hash({"decision_id": decision_id, "price_revision": contract["revision"]}),
                decision_id, entry_date,
                entry_date if strategy_trade else None,
                exit_date if strategy_trade else None,
                status, canonical_json(result), result_hash,
            ),
        )
        if status == "resolved":
            resolved += 1
        else:
            invalid += 1
    return {"resolved": resolved, "invalid": invalid, "pending": pending}


def breadth_shadow_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return bounded prospective evidence for the frozen P0/P1 comparison."""
    policies = {
        str(row[0]): {
            "start_ts": row[1], "specification_hash": row[2],
        }
        for row in conn.execute(
            "SELECT policy_id,start_ts,specification_hash FROM paper_shadow_policies"
        )
    }
    decision_rows = conn.execute(
        """SELECT decision_id,report_run_id,policy_id,candidate_rank,ticker,
                  disposition,feature_snapshot_json,report_date
           FROM paper_shadow_decisions
           ORDER BY report_date,report_run_id,policy_id,candidate_rank"""
    ).fetchall()
    outcomes = {
        str(row[0]): json.loads(str(row[1]))
        for row in conn.execute(
            "SELECT decision_id,result_json FROM paper_shadow_outcomes WHERE status='resolved'"
        )
    }
    counts = {
        policy_id: {
            "candidate_count": sum(str(row[2]) == policy_id for row in decision_rows),
            "accepted_count": sum(
                str(row[2]) == policy_id and str(row[5]) == "accepted"
                for row in decision_rows
            ),
        }
        for policy_id in SHADOW_POLICIES
    }
    by_key = {
        (str(row[1]), int(row[3]), str(row[2])): row for row in decision_rows
    }
    incremental = [
        row for row in decision_rows
        if str(row[2]) == "P1" and str(row[5]) == "accepted"
        and str(by_key.get((str(row[1]), int(row[3]), "P0"), [None] * 6)[5])
        != "accepted"
    ]
    incremental_results = [
        outcomes[str(row[0])] for row in incremental if str(row[0]) in outcomes
    ]

    def mean(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    incremental_returns = [
        float(row["net_return_pct"])
        for row in incremental_results if row.get("net_return_pct") is not None
    ]
    incremental_active = [
        float(row["net_return_pct"]) - float(row["matched_spy_return_pct"])
        for row in incremental_results
        if row.get("net_return_pct") is not None
        and row.get("matched_spy_return_pct") is not None
    ]
    stress_returns = []
    for row in incremental_results:
        trade = (row.get("cost_stress_2x") or {}).get("trade")
        if trade and trade.get("entry_notional"):
            stress_returns.append(
                float(trade["net_pnl"]) / float(trade["entry_notional"]) * 100
            )
    accepted_p0 = counts["P0"]["accepted_count"]
    accepted_p1 = counts["P1"]["accepted_count"]
    breadth_increase_pct = (
        (accepted_p1 / accepted_p0 - 1) * 100 if accepted_p0 else None
    )
    exit_dates = sorted({
        str(row.get("strategy", {}).get("trade", {}).get("exit_date"))
        for row in outcomes.values()
        if row.get("strategy", {}).get("trade", {}).get("exit_date")
    })
    sessions = [str(row[0]) for row in conn.execute(
        "SELECT date FROM price_daily WHERE ticker='SPY' ORDER BY date"
    )] if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='price_daily'"
    ).fetchone() else []
    session_index = {date: index for index, date in enumerate(sessions)}
    blocks: list[str] = []
    for date in exit_dates:
        if date not in session_index:
            continue
        if not blocks or session_index[date] - session_index[blocks[-1]] >= 10:
            blocks.append(date)
    accepted_total = sum(row["accepted_count"] for row in counts.values())
    matured_accepted = sum(
        str(row[5]) == "accepted"
        and sum(date > str(row[7]) for date in sessions) >= 10
        for row in decision_rows
    )
    unresolved_matured = matured_accepted - len(outcomes)
    immature_accepted = accepted_total - matured_accepted
    p1_accepted = [
        row for row in decision_rows
        if str(row[2]) == "P1" and str(row[5]) == "accepted"
    ]
    ticker_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for row in p1_accepted:
        ticker_counts[str(row[4])] = ticker_counts.get(str(row[4]), 0) + 1
        feature = json.loads(str(row[6]))
        family = str(feature.get("setup_family") or "Unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
    gates = {
        "breadth_increase_at_least_50_pct": (
            breadth_increase_pct is not None and breadth_increase_pct >= 50
        ),
        "positive_incremental_active_return": (
            mean(incremental_active) is not None and mean(incremental_active) > 0
        ),
        "double_cost_non_negative": (
            mean(stress_returns) is not None and mean(stress_returns) >= 0
        ),
        "minimum_12_non_overlapping_blocks": len(blocks) >= 12,
        "complete_matured_outcomes": (
            unresolved_matured == 0 and matured_accepted > 0
        ),
    }
    return {
        "schema_version": "paper-breadth-shadow-summary-v1",
        "policies": policies,
        "primary_endpoint": "next_session_open_to_tenth_session_close",
        "policy_counts": counts,
        "incremental_cohort": {
            "accepted_count": len(incremental),
            "resolved_count": len(incremental_results),
            "mean_net_return_pct": mean(incremental_returns),
            "mean_matched_active_return_pct": mean(incremental_active),
            "mean_2x_cost_return_pct": mean(stress_returns),
        },
        "coverage": {
            "report_count": conn.execute(
                "SELECT COUNT(DISTINCT report_run_id) FROM paper_shadow_decision_sets"
            ).fetchone()[0],
            "resolved_outcome_count": len(outcomes),
            "unresolved_matured_count": unresolved_matured,
            "immature_accepted_count": immature_accepted,
            "effective_non_overlapping_block_count": len(blocks),
        },
        "concentration": {
            "largest_ticker_pct": (
                max(ticker_counts.values(), default=0) / accepted_p1 * 100
                if accepted_p1 else None
            ),
            "largest_family_pct": (
                max(family_counts.values(), default=0) / accepted_p1 * 100
                if accepted_p1 else None
            ),
            "by_ticker": dict(sorted(ticker_counts.items())),
            "by_family": dict(sorted(family_counts.items())),
        },
        "breadth_increase_pct": breadth_increase_pct,
        "gates": gates,
        "human_promotion_review_eligible": all(gates.values()),
        "causal_state": (
            "promotion_review" if all(gates.values()) else "prospectively_accumulating"
        ),
        "automatic_promotion": False,
        "can_create_orders": False,
    }

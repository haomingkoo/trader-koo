"""Versioned paper-campaign migration, decision-ledger, and health tests."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trade.campaign import (
    canonical_hash,
    DivergentDecisionSetError,
    EvidenceIntegrityError,
    decide_candidate,
    record_experiment_preregistration,
    record_human_approval,
    record_promotion_experiment,
    transition_campaign,
    verify_decision_set,
)
from trader_koo.paper_trade.config import config_snapshot
from trader_koo.paper_trades import _build_config
from trader_koo.paper_trades import create_paper_trades_from_report as _create_paper_trades_from_report
from trader_koo.paper_trades import fill_pending_paper_orders
from trader_koo.paper_trades import paper_trade_summary
from trader_koo.paper_trade.replay import replay_campaign


def _candidate(
    ticker: str,
    *,
    tier: str = "A",
    resistance: float = 165.0,
) -> dict:
    return {
        "ticker": ticker,
        "setup_tier": tier,
        "score": 80.0,
        "actionability": "higher-probability",
        "signal_bias": "bullish",
        "close": 150.0,
        "setup_family": "Bullish Breakout",
        "atr_pct_14": 2.5,
        "support_level": 140.0,
        "resistance_level": resistance,
        "debate_agreement_score": 80.0,
        "risk_note": "Standard risk controls.",
    }


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    conn.execute(
        """CREATE TABLE price_daily (
               ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
               close REAL, volume REAL, UNIQUE(ticker, date)
           )"""
    )
    conn.execute(
        """CREATE TABLE report_runs (
               run_id TEXT PRIMARY KEY, status TEXT NOT NULL,
               is_generation_canonical INTEGER NOT NULL
           )"""
    )
    return conn


def _publish(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO report_runs VALUES (?,'published',1)", (run_id,)
    )


def create_paper_trades_from_report(conn: sqlite3.Connection, **kwargs):
    _publish(conn, str(kwargs["report_run_id"]))
    return _create_paper_trades_from_report(conn, **kwargs)


def _promotion_metrics(*, drawdown: float = 1.0, active_return: float = 1.0) -> dict:
    return {
        "engine_version": "paper-replay-v2.0", "closed_trades": 2,
        "conversion_rate_pct": 50.0, "average_exposure_pct": 10.0,
        "turnover_pct": 20.0, "portfolio_return_pct": 2.0,
        "matched_spy_return_pct": 1.0,
        "matched_spy_active_return_pct": active_return,
        "max_drawdown_pct": drawdown, "profit_factor": 1.5,
        "mean_trade_return_pct_ci95": [-1.0, 2.0],
        "walk_forward": {"folds": [{"metrics": {"closed_trades": 1}}]},
        "held_out": {"metrics": {"closed_trades": 1}, "trades": []},
    }


def _seed_promotion(conn: sqlite3.Connection) -> None:
    config = _build_config()
    experiment_id = f"experiment-{id(conn)}"
    preregistration_id = f"prereg-{id(conn)}"
    record_experiment_preregistration(
        conn, preregistration_id=preregistration_id, campaign_id="paper-v2",
        policy_version=config.decision_version,
        policy_hash=canonical_hash(config_snapshot(config)), dataset_hash="d" * 64,
        gates={
            "risk_gates": {"max_drawdown_pct": 10.0},
            "active_return_gate": {"minimum_pct": 0.0},
        },
    )
    record_promotion_experiment(
        conn, experiment_id=experiment_id,
        preregistration_id=preregistration_id, campaign_id="paper-v2",
        policy_version=config.decision_version,
        policy_hash=canonical_hash(config_snapshot(config)), dataset_hash="d" * 64,
        metrics=_promotion_metrics(),
        parity_status="matched",
    )
    record_human_approval(
        conn, approval_id=f"approval-{id(conn)}", experiment_id=experiment_id,
        campaign_id="paper-v2", actor="human-reviewer", reason="approved test evidence",
        artifact={"approved": True, "signed": True},
    )
    conn.commit()


def _activate(conn: sqlite3.Connection) -> None:
    _seed_promotion(conn)
    transition_campaign(
        conn, campaign_id="paper-v2", action="activate", actor="test-admin",
        reason="test activation", idempotency_key=f"activate-{id(conn)}",
    )


def test_legacy_trades_are_backfilled_to_immutable_v1_once():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE paper_trades (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               report_date TEXT NOT NULL,
               generated_ts TEXT,
               ticker TEXT NOT NULL,
               direction TEXT NOT NULL,
               entry_price REAL NOT NULL,
               entry_date TEXT NOT NULL,
               status TEXT NOT NULL,
               setup_family TEXT,
               decision_version TEXT,
               UNIQUE(report_date, ticker, direction)
           )"""
    )
    for index in range(42):
        conn.execute(
            """INSERT INTO paper_trades (
                   report_date, generated_ts, ticker, direction,
                   entry_price, entry_date, status, decision_version
               ) VALUES ('2026-03-18', ?, ?, 'long', 100, '2026-03-19',
                         'closed', 'paper-trade-eval-v1')""",
            (f"run-{index}", f"T{index:02d}"),
        )
    conn.execute(
        """CREATE TABLE paper_portfolio_snapshots (
               snapshot_date TEXT PRIMARY KEY, open_trades INTEGER DEFAULT 0,
               total_unrealized_pnl_pct REAL DEFAULT 0.0, snapshot_ts TEXT
           )"""
    )
    conn.execute(
        "INSERT INTO paper_portfolio_snapshots (snapshot_date,open_trades) VALUES ('2026-03-18',0)"
    )

    ensure_paper_trade_schema(conn)

    assert conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE campaign_id = 'paper-v1'"
    ).fetchone()[0] == 42
    assert conn.execute(
        "SELECT status FROM paper_campaigns WHERE campaign_id = 'paper-v1'"
    ).fetchone()[0] == "frozen"
    with pytest.raises(sqlite3.IntegrityError, match="v1 is immutable"):
        conn.execute("UPDATE paper_trades SET ticker = 'CHANGED' WHERE id = 1")
    # The v1 ledger stays frozen, while explicit user annotation remains legal.
    conn.execute(
        "INSERT INTO paper_trade_annotations (trade_id,notes,actor) VALUES (1,'reviewed','tester')"
    )
    assert conn.execute(
        "SELECT notes FROM paper_trade_annotations WHERE trade_id=1"
    ).fetchone()[0] == "reviewed"
    # Legacy global unique keys are rebuilt as campaign-scoped keys.
    conn.execute(
        """INSERT INTO paper_trades
           (campaign_id,report_date,ticker,direction,entry_price,entry_date,status)
           VALUES ('paper-v2','2026-03-18','T00','long',100,'2026-03-19','closed')"""
    )
    conn.execute(
        """INSERT INTO paper_portfolio_snapshots
           (campaign_id,snapshot_date,open_trades)
           VALUES ('paper-v2','2026-03-18',0)"""
    )
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 43
    assert conn.execute("SELECT COUNT(*) FROM paper_portfolio_snapshots").fetchone()[0] == 2


def test_live_path_persists_every_ranked_candidate_and_exact_disposition():
    conn = _db()
    _activate(conn)
    conn.execute(
        "INSERT INTO price_daily VALUES ('PASS','2026-08-22',150,155,149,154,1000000)"
    )

    inserted = create_paper_trades_from_report(
        conn,
        setup_rows=[_candidate("PASS"), _candidate("FAIL", tier="F")],
        report_date="2026-08-21",
        generated_ts="report-run-1",
        report_run_id="immutable-run-id-1",
    )

    assert inserted == 1
    decisions = conn.execute(
        """SELECT ticker, candidate_rank, eligibility_passed, final_gate,
                  reason_code, disposition, inputs_hash
           FROM paper_candidate_decisions ORDER BY candidate_rank"""
    ).fetchall()
    assert decisions[0][:6] == ("PASS", 1, 1, "admission", "admitted", "admitted")
    assert decisions[1][:6] == (
        "FAIL", 2, 0, "eligibility.tier", "tier_below_minimum", "rejected"
    )
    assert all(len(row[6]) == 64 for row in decisions)
    trade = conn.execute(
        "SELECT campaign_id, report_run_id, policy_version FROM paper_trades"
    ).fetchone()
    assert trade == ("paper-v2", "immutable-run-id-1", "paper-campaign-v2.0")


def test_three_eligible_zero_admission_reports_make_campaign_unhealthy():
    conn = _db()
    _activate(conn)
    for index in range(3):
        create_paper_trades_from_report(
            conn,
            setup_rows=[_candidate(f"LOWR{index}", resistance=154.0)],
            report_date=f"2026-08-{19 + index:02d}",
            generated_ts=f"report-run-{index}",
            report_run_id=f"report-run-{index}",
        )

    health = paper_trade_summary(conn)["campaign_health"]

    assert health["latest_report"]["ranked"] == 1
    assert health["latest_report"]["eligible"] == 1
    assert health["latest_report"]["admitted"] == 0
    assert health["latest_report"]["rejections_by_gate"] == [
        {
            "gate": "reward_risk",
            "reason_code": "minimum_reward_r_not_met",
            "count": 1,
        }
    ]
    assert health["consecutive_eligible_zero_admission_reports"] == 3
    assert health["healthy"] is False
    assert health["health_reasons"] == ["eligible_candidate_zero_admission_streak"]


def test_candidate_decision_rows_are_immutable():
    conn = _db()
    _activate(conn)
    create_paper_trades_from_report(
        conn,
        setup_rows=[_candidate("FAIL", tier="F")],
        report_date="2026-08-21",
        generated_ts="report-run-1",
        report_run_id="report-run-1",
    )

    with pytest.raises(sqlite3.IntegrityError, match="decisions are immutable"):
        conn.execute("UPDATE paper_candidate_decisions SET reason_code = 'changed'")


def test_inactive_campaign_records_sealed_shadow_set_without_trading():
    conn = _db()
    inserted = create_paper_trades_from_report(
        conn, setup_rows=[_candidate("SHADOW")], report_date="2026-08-21",
        generated_ts="shadow-ts", report_run_id="shadow-run",
    )
    assert inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    assert conn.execute(
        "SELECT status,candidate_count,report_complete FROM paper_decision_sets"
    ).fetchone() == ("sealed", 1, 1)
    assert conn.execute(
        "SELECT final_gate,reason_code FROM paper_candidate_decisions"
    ).fetchone() == ("campaign_lifecycle", "campaign_not_active")


def test_lifecycle_is_idempotent_audited_atomic_and_reversible():
    conn = _db()
    _seed_promotion(conn)
    first = transition_campaign(
        conn, campaign_id="paper-v2", action="activate", actor="alice",
        reason="paper validation approved", idempotency_key="activate-paper-v2-001",
    )
    retry = transition_campaign(
        conn, campaign_id="paper-v2", action="activate", actor="alice",
        reason="paper validation approved", idempotency_key="activate-paper-v2-001",
    )
    assert first["to_status"] == "active" and retry["idempotent"] is True
    rollback = transition_campaign(
        conn, campaign_id="paper-v2", action="rollback", actor="alice",
        reason="rollback drill", idempotency_key="rollback-paper-v2-001",
    )
    assert rollback["to_status"] == "draft"
    assert conn.execute("SELECT COUNT(*) FROM paper_campaign_audit").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError, match="audit is immutable"):
        conn.execute("DELETE FROM paper_campaign_audit")
    with pytest.raises(sqlite3.IntegrityError, match="v1 metadata is immutable"):
        conn.execute("UPDATE paper_campaigns SET label='changed' WHERE campaign_id='paper-v1'")


def test_decision_sets_preserve_duplicate_ranks_empty_partial_and_retry_identity():
    conn = _db()
    rows = [_candidate("DUP", tier="F"), _candidate("DUP", tier="F"), "bad-row"]
    create_paper_trades_from_report(
        conn, setup_rows=rows, report_date="2026-08-21", generated_ts="set-ts",
        report_run_id="set-run",
    )
    assert conn.execute(
        "SELECT candidate_count,report_complete FROM paper_decision_sets WHERE report_run_id='set-run'"
    ).fetchone() == (3, 1)
    assert conn.execute(
        "SELECT candidate_rank,ticker FROM paper_candidate_decisions ORDER BY candidate_rank"
    ).fetchall() == [(1, "DUP"), (2, "DUP"), (3, "__MALFORMED_3")]
    assert create_paper_trades_from_report(
        conn, setup_rows=rows, report_date="2026-08-21", generated_ts="set-ts",
        report_run_id="set-run",
    ) == 0
    with pytest.raises(DivergentDecisionSetError):
        create_paper_trades_from_report(
                conn, setup_rows=[_candidate("CHANGED", tier="F")], report_date="2026-08-21",
                generated_ts="set-ts", report_run_id="set-run",
        )
    create_paper_trades_from_report(
        conn, setup_rows=[], report_date="2026-08-22", generated_ts="empty-ts",
        report_run_id="empty-run",
    )
    empty_manifest = conn.execute(
        "SELECT candidate_count,policy_hash,context_hash FROM paper_decision_sets WHERE report_run_id='empty-run'"
    ).fetchone()
    assert empty_manifest[:2] == (
        0, canonical_hash(config_snapshot(_build_config()))
    )
    assert len(empty_manifest[2]) == 64
    create_paper_trades_from_report(
        conn, setup_rows=[_candidate("")], report_date="2026-08-23",
        generated_ts="missing-ts", report_run_id="missing-run",
    )
    assert conn.execute(
        "SELECT ticker,final_gate,reason_code FROM paper_candidate_decisions WHERE report_run_id='missing-run'"
    ).fetchone() == ("__MISSING_1", "candidate_identity", "missing_ticker")


def test_report_admission_rolls_back_trade_if_seal_crashes(monkeypatch):
    conn = _db()
    _activate(conn)

    def crash_before_seal(*args, **kwargs):
        raise RuntimeError("simulated seal crash")

    monkeypatch.setattr(
        "trader_koo.paper_trade.trading.persist_decision_set", crash_before_seal
    )
    with pytest.raises(RuntimeError, match="simulated seal crash"):
        create_paper_trades_from_report(
            conn, setup_rows=[_candidate("ATOMIC")], report_date="2026-08-21",
            generated_ts="atomic-ts", report_run_id="atomic-run",
        )
    # Even a caller that catches and commits cannot publish an unsealed trade.
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_decision_sets").fetchone()[0] == 0


def test_live_and_replay_share_policy_and_hash_every_effective_knob():
    config = _build_config()
    context = {
        "entry_price": 150.15, "vix_level": 18.0, "avg_daily_volume": 1_000_000,
        "portfolio_block": None, "critic_outcome": {"approved": True},
        "campaign_active": True, "duplicate": False,
        "market_context": {"regime_state_at_entry": "bull_normal"},
        "portfolio_context": {"open_count": 0}, "source_context": {"price_date": "2026-08-21"},
    }
    live = decide_candidate(row=_candidate("PARITY"), rank=1, config=config, context={**context, "mode": "live"})
    replay = decide_candidate(row=_candidate("PARITY"), rank=1, config=config, context={**context, "mode": "replay"})
    assert live == replay
    changed = decide_candidate(
        row=_candidate("PARITY"), rank=1,
        config=replace(config, risk_per_trade_pct=3.0, entry_slippage_bps=99.0, max_adv_pct=2.0),
        context={**context, "mode": "replay"},
    )
    assert changed["inputs_hash"] != live["inputs_hash"]
    assert changed["policy_hash"] != live["policy_hash"]


def test_missing_next_open_creates_pending_order_then_fills_actual_later_open():
    conn = _db()
    _activate(conn)
    inserted = create_paper_trades_from_report(
        conn, setup_rows=[_candidate("WAIT")], report_date="2026-08-21",
        generated_ts="wait-ts", report_run_id="wait-run",
    )
    assert inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    assert conn.execute(
        "SELECT disposition,tradeability FROM paper_candidate_decisions"
    ).fetchone() == ("pending", "pending_next_open")
    assert conn.execute(
        "SELECT status FROM paper_pending_orders"
    ).fetchone()[0] == "pending"

    conn.execute(
        "INSERT INTO price_daily VALUES ('WAIT','2026-08-24',152,160,151,158,1000000)"
    )
    result = fill_pending_paper_orders(conn, through_date="2026-08-24")
    assert result == {"filled": 1, "rejected": 0, "still_pending": 0}
    trade = conn.execute(
        "SELECT entry_price,entry_date,report_run_id FROM paper_trades"
    ).fetchone()
    assert trade == (152.152, "2026-08-24", "wait-run")
    assert conn.execute(
        "SELECT event_type FROM paper_order_events ORDER BY id"
    ).fetchall() == [("created",), ("filled",)]
    health = paper_trade_summary(conn)["campaign_health"]
    assert health["latest_report"]["admitted"] == 1
    assert health["latest_report"]["candidates"][0]["tradeability"] == "actionable"


def test_seal_rejects_later_insert_and_verifier_detects_forged_child():
    conn = _db()
    create_paper_trades_from_report(
        conn, setup_rows=[_candidate("ONE", tier="F")], report_date="2026-08-21",
        generated_ts="sealed-ts", report_run_id="sealed-run",
    )
    with pytest.raises(sqlite3.IntegrityError, match="not appendable"):
        conn.execute(
            """INSERT INTO paper_candidate_decisions
               (report_run_id,report_date,generated_ts,campaign_id,policy_version,
                ticker,candidate_rank,rank_inputs_json,eligibility_passed,final_gate,
                reason_code,reasons_json,inputs_hash,policy_hash,context_hash,disposition)
               SELECT report_run_id,report_date,generated_ts,campaign_id,policy_version,
                      'FORGED',2,rank_inputs_json,eligibility_passed,final_gate,
                      reason_code,reasons_json,inputs_hash,policy_hash,context_hash,disposition
               FROM paper_candidate_decisions WHERE candidate_rank=1"""
        )
    verify_decision_set(conn, report_run_id="sealed-run", campaign_id="paper-v2")
    conn.execute("DROP TRIGGER paper_candidate_decisions_no_insert_after_seal")
    conn.execute(
        """INSERT INTO paper_candidate_decisions
           (report_run_id,report_date,generated_ts,campaign_id,policy_version,
            ticker,candidate_rank,rank_inputs_json,eligibility_passed,final_gate,
            reason_code,reasons_json,inputs_hash,policy_hash,context_hash,disposition)
           SELECT report_run_id,report_date,generated_ts,campaign_id,policy_version,
                  'FORGED',2,rank_inputs_json,eligibility_passed,final_gate,
                  reason_code,reasons_json,inputs_hash,policy_hash,context_hash,disposition
           FROM paper_candidate_decisions WHERE candidate_rank=1"""
    )
    with pytest.raises(EvidenceIntegrityError, match="count/hash mismatch"):
        verify_decision_set(conn, report_run_id="sealed-run", campaign_id="paper-v2")


def test_lineage_and_activation_are_fail_closed_and_idempotency_binds_payload():
    conn = _db()
    conn.execute("INSERT INTO report_runs VALUES ('started-run','started',1)")
    with pytest.raises(ValueError, match="published canonical"):
        _create_paper_trades_from_report(
            conn, setup_rows=[], report_date="2026-08-21", generated_ts="x",
            report_run_id="started-run",
        )
    conn.commit()
    with pytest.raises(ValueError, match="promotion evidence"):
        transition_campaign(
            conn, campaign_id="paper-v2", action="activate", actor="alice",
            reason="attempt without evidence", idempotency_key="no-evidence-001",
        )
    _seed_promotion(conn)
    transition_campaign(
        conn, campaign_id="paper-v2", action="activate", actor="alice",
        reason="approved evidence", idempotency_key="payload-bound-001",
    )
    with pytest.raises(ValueError, match="different request payload"):
        transition_campaign(
            conn, campaign_id="paper-v2", action="activate", actor="mallory",
            reason="changed reason", idempotency_key="payload-bound-001",
        )


def test_chronological_replay_models_costs_overlap_exits_and_parity():
    config = replace(_build_config(), max_open=2, expiry_days=2)
    runs = [
        {"report_run_id": "r1", "report_date": "2026-08-20", "candidates": [
            {**_candidate("AAA"), "critic_outcome": {"approved": True}},
            {**_candidate("BBB"), "critic_outcome": {"approved": True}},
        ]},
        {"report_run_id": "r2", "report_date": "2026-08-21", "candidates": [
            {**_candidate("CCC"), "critic_outcome": {"approved": True}},
        ]},
    ]
    prices = [
        {"ticker": "AAA", "date": "2026-08-21", "open": 150, "high": 151, "low": 149, "close": 150, "volume": 1_000_000},
        {"ticker": "BBB", "date": "2026-08-21", "open": 150, "high": 151, "low": 149, "close": 150, "volume": 1_000_000},
        {"ticker": "AAA", "date": "2026-08-24", "open": 150, "high": 166, "low": 149, "close": 165, "volume": 1_000_000},
        {"ticker": "BBB", "date": "2026-08-24", "open": 150, "high": 151, "low": 140, "close": 141, "volume": 1_000_000},
        {"ticker": "CCC", "date": "2026-08-24", "open": 150, "high": 151, "low": 149, "close": 150, "volume": 1_000_000},
        {"ticker": "CCC", "date": "2026-08-25", "open": 150, "high": 151, "low": 149, "close": 150, "volume": 1_000_000},
    ]
    spy = [
        {"date": "2026-08-21", "open": 100},
        {"date": "2026-08-25", "close": 101},
    ]
    first = replay_campaign(candidate_runs=runs, price_rows=prices, spy_rows=spy, config=config)
    expected = {
        item["execution_key"]: {
            "disposition": item["disposition"], "inputs_hash": item["inputs_hash"]
        }
        for item in first["decisions"]
    }
    second = replay_campaign(
        candidate_runs=runs, price_rows=prices, spy_rows=spy, config=config,
        expected_execution=expected,
    )
    assert second["replay_live_parity"] == "matched"
    assert second["metrics"]["admitted_count"] == 3
    assert second["metrics"]["turnover_pct"] > 0
    assert second["metrics"]["matched_spy_active_return_pct"] is not None
    assert {trade["exit_reason"] for trade in second["trades"]} >= {"target_hit", "stopped_out"}
    assert second["walk_forward"]["training_dates"]
    assert "held_out" in second


def test_activation_never_falls_back_to_older_eligible_experiment():
    conn = _db()
    _seed_promotion(conn)
    with pytest.raises(sqlite3.IntegrityError, match="preregistrations are immutable"):
        conn.execute(
            "UPDATE paper_campaign_preregistrations SET gates_json='{}'"
        )
    conn.rollback()
    config = _build_config()
    record_experiment_preregistration(
        conn, preregistration_id="newer-failed-prereg", campaign_id="paper-v2",
        policy_version=config.decision_version,
        policy_hash=canonical_hash(config_snapshot(config)), dataset_hash="e" * 64,
        gates={
            "risk_gates": {"max_drawdown_pct": 5.0},
            "active_return_gate": {"minimum_pct": 1.0},
        },
    )
    record_promotion_experiment(
        conn, experiment_id="newer-failed",
        preregistration_id="newer-failed-prereg", campaign_id="paper-v2",
        policy_version=config.decision_version,
        policy_hash=canonical_hash(config_snapshot(config)), dataset_hash="e" * 64,
        metrics=_promotion_metrics(drawdown=12.0, active_return=-2.0),
        parity_status="matched",
    )
    conn.commit()
    with pytest.raises(ValueError, match="promotion evidence"):
        transition_campaign(
            conn, campaign_id="paper-v2", action="activate", actor="alice",
            reason="must not use stale evidence", idempotency_key="stale-evidence-001",
        )

"""Versioned paper-campaign migration, decision-ledger, and health tests."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trade.campaign import (
    canonical_hash,
    DivergentDecisionSetError,
    decide_candidate,
    transition_campaign,
)
from trader_koo.paper_trade.config import config_snapshot
from trader_koo.paper_trades import _build_config
from trader_koo.paper_trades import create_paper_trades_from_report, paper_trade_summary


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
    return conn


def _activate(conn: sqlite3.Connection) -> None:
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
    assert health["health_reasons"] == [
        "eligible_candidate_zero_admission_streak",
        "replay_live_parity_not_measured",
    ]


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
        report_run_id="set-run", report_complete=False,
    )
    assert conn.execute(
        "SELECT candidate_count,report_complete FROM paper_decision_sets WHERE report_run_id='set-run'"
    ).fetchone() == (3, 0)
    assert conn.execute(
        "SELECT candidate_rank,ticker FROM paper_candidate_decisions ORDER BY candidate_rank"
    ).fetchall() == [(1, "DUP"), (2, "DUP"), (3, "__MALFORMED_3")]
    assert create_paper_trades_from_report(
        conn, setup_rows=rows, report_date="2026-08-21", generated_ts="set-ts",
        report_run_id="set-run", report_complete=False,
    ) == 0
    with pytest.raises(DivergentDecisionSetError):
        create_paper_trades_from_report(
            conn, setup_rows=[_candidate("CHANGED", tier="F")], report_date="2026-08-21",
            generated_ts="set-ts", report_run_id="set-run", report_complete=False,
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

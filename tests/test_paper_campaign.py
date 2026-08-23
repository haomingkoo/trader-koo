"""Versioned paper-campaign migration, decision-ledger, and health tests."""

from __future__ import annotations

import sqlite3
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trade.chronology import (
    next_scheduled_session_after,
    publication_precedes_session_open,
)
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
from trader_koo.paper_trade.trading import (
    ReportLineageError,
    _require_published_canonical_report,
)
from trader_koo.research.next_open_baseline import (
    BaselineConfig,
    ExecutionDecision,
    SessionPrice,
    canonical_json_bytes,
    simulate_portfolio,
)
from trader_koo.report.runs import (
    admit_published_report,
    complete_report_run,
    publish_report_run,
    sha256_file,
    start_report_run,
)
from trader_koo.report.serializer import write_reports
from trader_koo.db.price_contract import (
    ensure_price_series_revision_schema,
    record_price_series_revision,
)


@pytest.fixture(autouse=True)
def _fixed_report_publication_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep next-open chronology deterministic as wall-clock time advances."""
    monkeypatch.setattr(
        "trader_koo.report.runs._utc_now", lambda: "2026-08-21T12:02:00Z"
    )


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


def test_publication_cutoff_is_strict_and_dst_aware() -> None:
    assert publication_precedes_session_open(
        "2026-08-24T13:29:59Z", "2026-08-24"
    ) is True
    assert publication_precedes_session_open(
        "2026-08-24T13:30:00Z", "2026-08-24"
    ) is False


def test_scheduled_session_uses_versioned_historical_exchange_calendar() -> None:
    assert next_scheduled_session_after("2021-06-17") == "2021-06-18"
    assert next_scheduled_session_after("2018-12-04") == "2018-12-06"
    assert publication_precedes_session_open(
        "2026-01-05T14:29:59Z", "2026-01-05"
    ) is True


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    conn.execute(
        """CREATE TABLE price_daily (
               ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
               close REAL, volume REAL, data_source TEXT DEFAULT 'fixture',
               fetch_timestamp TEXT DEFAULT '2026-08-21T00:00:00Z',
               adjustment_basis TEXT DEFAULT 'split_adjusted_price_only',
               adjustment_version TEXT DEFAULT 'fixture-v1',
               basis_status TEXT DEFAULT 'verified', unresolved_reason TEXT,
               UNIQUE(ticker, date)
           )"""
    )
    ensure_price_series_revision_schema(conn)
    return conn


def _publish(
    conn: sqlite3.Connection,
    run_id: str,
    setup_rows: list[dict[str, object]],
) -> None:
    if conn.execute("SELECT 1 FROM report_runs WHERE run_id=?", (run_id,)).fetchone():
        return
    unique_rows: dict[str, dict[str, object]] = {}
    for row in setup_rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker not in unique_rows:
            unique_rows[ticker] = row
    decisions = [
        {
            "ticker": str(row["ticker"]).upper(),
            "selected_rank": rank,
            "decision": "accepted",
            "reason_codes": ["selected_report_cohort"],
            "inputs": dict(row),
        }
        for rank, row in enumerate(unique_rows.values(), start=1)
    ]
    report = {
        "generated_ts": "2026-08-21T12:00:00Z",
        "meta": {"report_kind": "daily"},
        "latest_data": {"price_date": "2026-08-21"},
        "signals": {
            "report_decisions": decisions,
            "scanned_universe": [item["ticker"] for item in decisions],
        },
        "counts": {},
        "risk_filters": {},
        "warnings": [],
        "ok": True,
    }
    config_json = "{}"
    conn.execute(
        """INSERT INTO report_runs
           (run_id,report_kind,status,started_ts,config_json,config_hash,code_version)
           VALUES (?,'daily','started','2026-08-21T11:59:00Z',?,?,?)""",
        (run_id, config_json, hashlib.sha256(config_json.encode()).hexdigest(), "a" * 40),
    )
    conn.commit()
    report_dir = Path(tempfile.mkdtemp(prefix="paper-campaign-report-"))
    paths = write_reports(report, report_dir, run_id=run_id, publish_latest=False)
    artifact = Path(paths["json_path"])
    complete_report_run(
        conn,
        run_id=run_id,
        report=report,
        artifact_path=artifact,
        markdown_path=Path(paths["md_path"]),
        content_hash=sha256_file(artifact),
        completed_ts="2026-08-21T12:01:00Z",
    )
    publish_report_run(conn, run_id=run_id, report_dir=report_dir)


def create_paper_trades_from_report(conn: sqlite3.Connection, **kwargs):
    setup_rows = kwargs.get("setup_rows")
    publish_rows = list(setup_rows) if isinstance(setup_rows, list) else []
    _publish(conn, str(kwargs["report_run_id"]), publish_rows)
    for row in publish_rows:
        ticker = str(row.get("ticker") or "").upper() if isinstance(row, dict) else ""
        if ticker and conn.execute(
            "SELECT 1 FROM price_daily WHERE ticker=?", (ticker,)
        ).fetchone():
            record_price_series_revision(
                conn, ticker,
                evidence={"provider": "fixture", "vendor_action_ledger_checked": True,
                          "vendor_action_ledger": []},
                fetch_timestamp="2026-08-21T00:00:00Z",
            )
    return _create_paper_trades_from_report(conn, **kwargs)


def _promotion_metrics(*, drawdown: float = 1.0, active_return: float = 1.0) -> dict:
    return {
        "engine_version": "portfolio-execution-v1.0", "closed_trades": 2,
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
    experiment = record_promotion_experiment(
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
        experiment_evidence_hash=experiment["evidence_hash"],
        artifact={"approved": True, "signed": True},
    )
    conn.commit()


def _activate(conn: sqlite3.Connection) -> None:
    # Business-path fixtures exercise a historically active campaign. The v4
    # production transition itself is unconditionally interlocked.
    conn.execute(
        "UPDATE paper_campaigns SET status='active',replay_live_parity='matched' "
        "WHERE campaign_id='paper-v2'"
    )
    conn.commit()


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
    conn.commit()

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
    # Expand compatibility keeps the old global key for image rollback while
    # adding the campaign-aware key for the new image.
    unique_indexes = [
        row[1] for row in conn.execute("PRAGMA index_list(paper_trades)")
        if row[2]
    ]
    assert any(
        [item[2] for item in conn.execute(f"PRAGMA index_info({index})")]
        == ["campaign_id", "report_date", "ticker", "direction"]
        for index in unique_indexes
    )
    assert any(
        [item[2] for item in conn.execute(f"PRAGMA index_info({index})")]
        == ["report_date", "ticker", "direction"]
        for index in unique_indexes
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO paper_portfolio_snapshots
               (campaign_id,snapshot_date,open_trades)
               VALUES ('paper-v2','2026-03-18',0)"""
        )
    conn.execute(
        """INSERT INTO paper_portfolio_snapshots
           (campaign_id,snapshot_date,open_trades)
           VALUES ('paper-v2','2026-03-19',0)"""
    )
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 42
    assert conn.execute("SELECT COUNT(*) FROM paper_portfolio_snapshots").fetchone()[0] == 2


def test_live_path_persists_every_ranked_candidate_and_exact_disposition():
    conn = _db()
    _activate(conn)
    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('PASS','2026-08-24',150,155,149,154,1000000),
                  ('SPY','2026-08-24',650,651,649,650,1000000)"""
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


def test_lifecycle_is_idempotent_audited_atomic_and_reversible(monkeypatch):
    conn = _db()
    _seed_promotion(conn)
    monkeypatch.setattr(
        "trader_koo.paper_trade.schema.require_contracted_paper_schema",
        lambda conn: None,
    )
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
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('WAIT','2026-08-24',152,160,151,158,1000000),
                  ('SPY','2026-08-24',650,651,649,650,1000000)"""
    )
    record_price_series_revision(
        conn, "WAIT",
        evidence={"provider": "fixture", "vendor_action_ledger_checked": True,
                  "vendor_action_ledger": []},
        fetch_timestamp="2026-08-24T00:00:00Z",
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


def test_pending_order_never_skips_a_missing_immediate_session_open():
    conn = _db()
    _activate(conn)
    create_paper_trades_from_report(
        conn, setup_rows=[_candidate("LATE")], report_date="2026-08-21",
        generated_ts="late-ts", report_run_id="late-run",
    )
    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('SPY','2026-08-24',650,651,649,650,1000000),
                  ('SPY','2026-08-25',651,652,650,651,1000000),
                  ('LATE','2026-08-25',152,160,151,158,1000000)"""
    )

    assert fill_pending_paper_orders(conn, through_date="2026-08-25") == {
        "filled": 0, "rejected": 0, "still_pending": 1,
    }
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0

    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('LATE','2026-08-24',151,152,150,151,1000000)"""
    )
    record_price_series_revision(
        conn, "LATE",
        evidence={"provider": "fixture", "vendor_action_ledger_checked": True,
                  "vendor_action_ledger": []},
        fetch_timestamp="2026-08-25T00:00:00Z",
    )
    result = fill_pending_paper_orders(conn, through_date="2026-08-25")
    assert result == {"filled": 1, "rejected": 0, "still_pending": 0}
    assert conn.execute("SELECT entry_date FROM paper_trades").fetchone()[0] == "2026-08-24"


def test_immediate_admission_does_not_skip_a_missing_ticker_open() -> None:
    conn = _db()
    _activate(conn)
    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('SPY','2026-08-24',650,651,649,650,1000000),
                  ('SPY','2026-08-25',651,652,650,651,1000000),
                  ('SKIP','2026-08-25',152,160,151,158,1000000)"""
    )

    inserted = create_paper_trades_from_report(
        conn, setup_rows=[_candidate("SKIP")], report_date="2026-08-21",
        generated_ts="skip-ts", report_run_id="skip-run",
    )

    assert inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM paper_pending_orders").fetchone()[0] == "pending"


def test_missing_immediate_spy_observation_never_rolls_to_a_later_session() -> None:
    conn = _db()
    _activate(conn)
    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('SPY','2026-08-25',651,652,650,651,1000000),
                  ('NOSPY','2026-08-25',152,160,151,158,1000000)"""
    )

    inserted = create_paper_trades_from_report(
        conn, setup_rows=[_candidate("NOSPY")], report_date="2026-08-21",
        generated_ts="no-spy-ts", report_run_id="no-spy-run",
    )
    resolved = fill_pending_paper_orders(conn, through_date="2026-08-25")

    assert inserted == 0
    assert resolved == {"filled": 0, "rejected": 0, "still_pending": 1}
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    assert conn.execute(
        "SELECT reason_code FROM paper_candidate_decisions "
        "WHERE report_run_id='no-spy-run'"
    ).fetchone() == ("scheduled_spy_open_missing",)


def test_report_published_after_intended_open_cannot_backdate_a_fill(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "trader_koo.report.runs._utc_now", lambda: "2026-08-24T14:02:00Z"
    )
    conn = _db()
    _activate(conn)
    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('STALE','2026-08-24',150,151,149,150,1000000),
                  ('SPY','2026-08-24',650,651,649,650,1000000)"""
    )

    assert create_paper_trades_from_report(
        conn, setup_rows=[_candidate("STALE")], report_date="2026-08-21",
        generated_ts="stale-ts", report_run_id="stale-run",
    ) == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    decision_row = conn.execute(
        """SELECT disposition,reason_code,inputs_json,inputs_hash
           FROM paper_candidate_decisions WHERE report_run_id='stale-run'"""
    ).fetchone()
    assert decision_row[:2] == (
        "rejected", "report_published_after_intended_open",
    )
    assert conn.execute("SELECT COUNT(*) FROM paper_pending_orders").fetchone()[0] == 0
    sealed_inputs = json.loads(decision_row[2])
    replay = replay_campaign(
        candidate_runs=[{
            "report_run_id": "stale-run",
            "report_date": "2026-08-21",
            "published_ts": "2026-08-24T14:02:00Z",
            "candidates": [{
                "__sealed_candidate": sealed_inputs["candidate"],
                "__sealed_context": sealed_inputs["context"],
            }],
        }],
        price_rows=[
            {"ticker": "STALE", "date": "2026-08-24", "open": 150,
             "high": 151, "low": 149, "close": 150, "volume": 1_000_000},
            {"ticker": "SPY", "date": "2026-08-24", "open": 650,
             "high": 651, "low": 649, "close": 650, "volume": 1_000_000},
        ],
        spy_rows=[], config=_build_config(), _include_splits=False,
        expected_execution={
            "stale-run:1": {
                "disposition": decision_row[0],
                "inputs_hash": decision_row[3],
            }
        },
    )
    assert replay["decisions"][0]["inputs"] == sealed_inputs
    assert replay["replay_live_parity"] == "matched", replay
    assert replay["decisions"][0]["inputs_hash"] == decision_row[3]


def test_late_publication_precedes_missing_spy_and_ticker_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trader_koo.report.runs._utc_now", lambda: "2026-08-24T14:02:00Z"
    )
    conn = _db()
    _activate(conn)

    assert create_paper_trades_from_report(
        conn, setup_rows=[_candidate("LATEGAP")], report_date="2026-08-21",
        generated_ts="late-gap-ts", report_run_id="late-gap-run",
    ) == 0

    assert conn.execute(
        "SELECT disposition,reason_code FROM paper_candidate_decisions "
        "WHERE report_run_id='late-gap-run'"
    ).fetchone() == ("rejected", "report_published_after_intended_open")
    assert conn.execute("SELECT COUNT(*) FROM paper_pending_orders").fetchone()[0] == 0


def test_pending_order_payload_is_immutable_and_hash_verified_before_fill():
    conn = _db()
    _activate(conn)
    create_paper_trades_from_report(
        conn, setup_rows=[_candidate("SEALED")], report_date="2026-08-21",
        generated_ts="sealed-order-ts", report_run_id="sealed-order-run",
    )
    with pytest.raises(sqlite3.IntegrityError, match="payload is immutable"):
        conn.execute(
            "UPDATE paper_pending_orders SET candidate_json='{}' WHERE ticker='SEALED'"
        )
    conn.execute("DROP TRIGGER paper_pending_orders_immutable_payload")
    conn.execute(
        "UPDATE paper_pending_orders SET candidate_json='{}' WHERE ticker='SEALED'"
    )
    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('SEALED','2026-08-24',152,160,151,158,1000000)"""
    )
    record_price_series_revision(
        conn, "SEALED",
        evidence={"provider": "fixture", "vendor_action_ledger_checked": True,
                  "vendor_action_ledger": []},
        fetch_timestamp="2026-08-24T00:00:00Z",
    )
    with pytest.raises(ValueError, match="immutable hash verification"):
        fill_pending_paper_orders(conn, through_date="2026-08-24")


def test_human_approval_requires_exact_eligible_experiment_even_without_foreign_keys():
    conn = _db()
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    _seed_promotion(conn)
    experiment_id, evidence_hash = conn.execute(
        "SELECT experiment_id,evidence_hash FROM paper_campaign_experiments"
    ).fetchone()
    with pytest.raises(ValueError, match="exact eligible experiment evidence hash"):
        record_human_approval(
            conn, approval_id="wrong-hash-approval", experiment_id=experiment_id,
            campaign_id="paper-v2", actor="reviewer", reason="wrong evidence",
            experiment_evidence_hash="0" * 64,
            artifact={"approved": True},
        )
    with pytest.raises(ValueError, match="exact eligible experiment evidence hash"):
        record_human_approval(
            conn, approval_id="missing-experiment-approval", experiment_id="missing",
            campaign_id="paper-v2", actor="reviewer", reason="missing evidence",
            experiment_evidence_hash=evidence_hash,
            artifact={"approved": True},
        )
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


def test_lineage_and_activation_are_fail_closed_and_idempotency_binds_payload(monkeypatch):
    conn = _db()
    started_run = start_report_run(
        conn,
        report_kind="daily",
        configuration={},
        code_version="a" * 40,
    )
    with pytest.raises(ReportLineageError, match="verified report artifact") as lineage_error:
        _create_paper_trades_from_report(
            conn, setup_rows=[], report_date="2026-08-21", generated_ts="x",
            report_run_id=started_run,
        )
    assert lineage_error.value.code == "report_not_verified_published"
    assert conn.execute("SELECT COUNT(*) FROM paper_decision_sets").fetchone()[0] == 0
    conn.commit()
    with pytest.raises(ValueError, match="activation interlock"):
        transition_campaign(
            conn, campaign_id="paper-v2", action="activate", actor="alice",
            reason="attempt without evidence", idempotency_key="no-evidence-001",
        )
    _seed_promotion(conn)
    monkeypatch.setattr(
        "trader_koo.paper_trade.schema.require_contracted_paper_schema",
        lambda conn: None,
    )
    transition_campaign(
        conn, campaign_id="paper-v2", action="activate", actor="alice",
        reason="approved evidence", idempotency_key="payload-bound-001",
    )
    with pytest.raises(ValueError, match="different request payload"):
        transition_campaign(
            conn, campaign_id="paper-v2", action="activate", actor="mallory",
            reason="changed reason", idempotency_key="payload-bound-001",
        )


def test_malformed_report_registry_is_a_structural_lineage_error() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE report_runs(run_id TEXT PRIMARY KEY, artifact_path TEXT)")
    conn.execute("INSERT INTO report_runs VALUES ('broken-run','/tmp/broken.json')")

    with pytest.raises(ReportLineageError) as error:
        _require_published_canonical_report(conn, "broken-run")

    assert error.value.code == "report_publication_lineage_invalid"


def test_outer_admission_persists_lineage_failure_without_candidate_writes(
    tmp_path: Path,
) -> None:
    conn = _db()
    run_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version="a" * 40
    )

    with pytest.raises(ReportLineageError) as error:
        admit_published_report(conn, run_id=run_id, report_dir=tmp_path)

    assert error.value.code == "report_not_verified_published"
    assert conn.execute(
        "SELECT status,error_code FROM report_admission_attempts WHERE run_id=?",
        (run_id,),
    ).fetchone() == ("failed", "report_not_verified_published")
    assert conn.execute("SELECT COUNT(*) FROM paper_decision_sets").fetchone()[0] == 0
    with pytest.raises(sqlite3.IntegrityError, match="admission attempts are immutable"):
        conn.execute("DELETE FROM report_admission_attempts")


def test_chronological_replay_models_costs_overlap_exits_and_parity():
    config = replace(_build_config(), max_open=2, expiry_days=2)
    runs = [
        {"report_run_id": "r1", "report_date": "2026-08-20",
         "published_ts": "2026-08-20T12:00:00Z", "candidates": [
            {**_candidate("AAA"), "critic_outcome": {"approved": True}},
            {**_candidate("BBB"), "critic_outcome": {"approved": True}},
        ]},
        {"report_run_id": "r2", "report_date": "2026-08-21",
         "published_ts": "2026-08-21T12:00:00Z", "candidates": [
            {**_candidate("CCC"), "critic_outcome": {"approved": True}},
        ]},
    ]
    prices = [
        {"ticker": "SPY", "date": "2026-08-20", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "SPY", "date": "2026-08-21", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "SPY", "date": "2026-08-24", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "SPY", "date": "2026-08-25", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "AAA", "date": "2026-08-20", "open": 149, "high": 151, "low": 148, "close": 150, "volume": 900_000},
        {"ticker": "BBB", "date": "2026-08-20", "open": 149, "high": 151, "low": 148, "close": 150, "volume": 900_000},
        {"ticker": "AAA", "date": "2026-08-21", "open": 150, "high": 151, "low": 149, "close": 150, "volume": 1_000_000},
        {"ticker": "BBB", "date": "2026-08-21", "open": 150, "high": 151, "low": 149, "close": 150, "volume": 1_000_000},
        {"ticker": "CCC", "date": "2026-08-21", "open": 149, "high": 151, "low": 148, "close": 150, "volume": 900_000},
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


def test_baseline_and_campaign_replay_seal_identical_complete_ledgers():
    config = replace(_build_config(), max_open=2, expiry_days=2)
    runs = [{
        "report_run_id": "ledger-run", "report_date": "2026-08-20",
        "published_ts": "2026-08-20T12:00:00Z",
        "candidates": [{**_candidate("AAA"), "critic_outcome": {"approved": True}}],
    }]
    prices = [
        {"ticker": "SPY", "date": "2026-08-20", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "SPY", "date": "2026-08-21", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "SPY", "date": "2026-08-24", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "AAA", "date": "2026-08-20", "open": 149, "high": 151,
         "low": 148, "close": 150, "volume": 900_000},
        {"ticker": "AAA", "date": "2026-08-21", "open": 150, "high": 151,
         "low": 149, "close": 150, "volume": 1_000_000},
        {"ticker": "AAA", "date": "2026-08-24", "open": 150, "high": 166,
         "low": 149, "close": 165, "volume": 1_000_000},
    ]
    campaign = replay_campaign(
        candidate_runs=runs, price_rows=prices, spy_rows=[], config=config,
        _include_splits=False,
    )
    fixture_decisions = []
    for raw in campaign["execution_ledger"]["decisions"]:
        payload = dict(raw)
        payload["metadata"] = tuple(tuple(item) for item in payload["metadata"])
        fixture_decisions.append(ExecutionDecision(**payload))
    fixture_prices = [SessionPrice(**row) for row in campaign[
        "execution_ledger"
    ]["market_data"]["prices"]]
    baseline = simulate_portfolio(
        fixture_decisions, fixture_prices,
        campaign["execution_ledger"]["market_data"]["sessions"],
        BaselineConfig(**campaign["execution_ledger"]["config"]),
    )

    assert canonical_json_bytes(baseline.ledger) == canonical_json_bytes(
        campaign["execution_ledger"]
    )
    assert baseline.ledger["provenance"]["ledger_sha256"] == campaign[
        "execution_ledger_hash"
    ]


def test_replay_rejects_missing_immediate_open_and_uses_only_causal_volume():
    config = replace(_build_config(), expiry_days=2)
    runs = [{
        "report_run_id": "causal-run", "report_date": "2026-08-20",
        "published_ts": "2026-08-20T12:00:00Z",
        "candidates": [{**_candidate("LATE"), "critic_outcome": {"approved": True}}],
    }]
    prices = [
        {"ticker": "LATE", "date": "2026-08-20", "open": 150, "high": 151,
         "low": 149, "close": 150, "volume": 100},
        {"ticker": "SPY", "date": "2026-08-21", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "LATE", "date": "2026-08-24", "open": 150, "high": 151,
         "low": 149, "close": 150, "volume": 100_000_000},
    ]

    replay = replay_campaign(
        candidate_runs=runs, price_rows=prices, spy_rows=[], config=config,
        _include_splits=False,
    )

    decision = replay["decisions"][0]
    assert decision["disposition"] == "rejected"
    assert decision["inputs"]["context"]["execution_ready"] is False
    assert decision["inputs"]["context"]["source_context"]["intended_session"] == "2026-08-21"
    assert decision["inputs"]["context"]["avg_daily_volume"] == 100
    assert replay["trades"] == []


def test_replay_parity_uses_the_exact_sealed_live_decision_inputs():
    conn = _db()
    _activate(conn)
    candidate = {**_candidate("PARITY"), "critic_outcome": {"approved": True}}
    conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('PARITY','2026-08-24',150,151,149,150,1000000),
                  ('SPY','2026-08-24',650,651,649,650,1000000)"""
    )
    create_paper_trades_from_report(
        conn, setup_rows=[candidate], report_date="2026-08-21",
        generated_ts="parity-live-ts", report_run_id="parity-live-run",
    )
    disposition, inputs_hash, inputs_json = conn.execute(
        """SELECT disposition,inputs_hash,inputs_json
           FROM paper_candidate_decisions
           WHERE report_run_id='parity-live-run'"""
    ).fetchone()
    evidence = json.loads(inputs_json)
    replay = replay_campaign(
        candidate_runs=[{
            "report_run_id": "parity-live-run", "report_date": "2026-08-21",
            "published_ts": "2026-08-21T12:02:00Z",
            "candidates": [{
                "__sealed_candidate": evidence["candidate"],
                "__sealed_context": evidence["context"],
            }],
        }],
        price_rows=[{
                "ticker": "PARITY", "date": "2026-08-24", "open": 150,
            "high": 151, "low": 149, "close": 150, "volume": 1_000_000,
        }, {
            "ticker": "SPY", "date": "2026-08-24", "open": 650,
            "high": 651, "low": 649, "close": 650, "volume": 1_000_000,
        }],
        spy_rows=[], config=_build_config(),
        expected_execution={
            "parity-live-run:1": {
                "disposition": disposition, "inputs_hash": inputs_hash,
            }
        },
    )
    assert replay["decisions"][0]["inputs"] == evidence
    assert replay["replay_live_parity"] == "matched"
    assert replay["decisions"][0]["inputs_hash"] == inputs_hash


def test_replay_rejects_the_same_late_publication_as_live_execution():
    replay = replay_campaign(
        candidate_runs=[{
            "report_run_id": "late-publication-run",
            "report_date": "2026-08-21",
            "published_ts": "2026-08-24T14:00:00Z",
            "candidates": [{
                **_candidate("LATEPUB"),
                "critic_outcome": {"approved": True},
            }],
        }],
        price_rows=[{
            "ticker": "LATEPUB", "date": "2026-08-24", "open": 150,
            "high": 151, "low": 149, "close": 150, "volume": 1_000_000,
        }, {
            "ticker": "SPY", "date": "2026-08-24", "open": 650,
            "high": 651, "low": 649, "close": 650, "volume": 1_000_000,
        }],
        spy_rows=[], config=_build_config(), _include_splits=False,
    )

    decision = replay["decisions"][0]
    assert decision["disposition"] == "rejected"
    assert decision["inputs"]["context"]["portfolio_block"]["reason_code"] == (
        "report_published_after_intended_open"
    )
    assert replay["trades"] == []


def test_replay_fails_closed_without_an_observed_spy_calendar():
    replay = replay_campaign(
        candidate_runs=[{
            "report_run_id": "no-calendar-run", "report_date": "2026-08-21",
            "published_ts": "2026-08-21T12:00:00Z",
            "candidates": [{**_candidate("NOCAL"), "critic_outcome": {"approved": True}}],
        }],
        price_rows=[{
            "ticker": "NOCAL", "date": "2026-08-24", "open": 150,
            "high": 151, "low": 149, "close": 150, "volume": 1_000_000,
        }],
        spy_rows=[], config=_build_config(), _include_splits=False,
    )

    assert replay["decisions"][0]["disposition"] == "pending"
    assert replay["decisions"][0]["reason_code"] == "scheduled_spy_open_missing"
    assert replay["decisions"][0]["inputs"]["context"]["execution_ready"] is False
    assert replay["trades"] == []


def test_activation_never_falls_back_to_older_eligible_experiment(monkeypatch):
    conn = _db()
    _seed_promotion(conn)
    monkeypatch.setattr(
        "trader_koo.paper_trade.schema.require_contracted_paper_schema",
        lambda conn: None,
    )
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

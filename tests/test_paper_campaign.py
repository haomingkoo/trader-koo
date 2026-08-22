"""Versioned paper-campaign migration, decision-ledger, and health tests."""

from __future__ import annotations

import sqlite3

import pytest

from trader_koo.paper_trade.schema import ensure_paper_trade_schema
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

    ensure_paper_trade_schema(conn)

    assert conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE campaign_id = 'paper-v1'"
    ).fetchone()[0] == 42
    assert conn.execute(
        "SELECT status FROM paper_campaigns WHERE campaign_id = 'paper-v1'"
    ).fetchone()[0] == "frozen"
    with pytest.raises(sqlite3.IntegrityError, match="v1 is immutable"):
        conn.execute("UPDATE paper_trades SET ticker = 'CHANGED' WHERE id = 1")


def test_live_path_persists_every_ranked_candidate_and_exact_disposition():
    conn = _db()

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
        "FAIL", 2, 0, "eligibility", "initial_eligibility_rejected", "rejected"
    )
    assert all(len(row[6]) == 64 for row in decisions)
    trade = conn.execute(
        "SELECT campaign_id, report_run_id, policy_version FROM paper_trades"
    ).fetchone()
    assert trade == ("paper-v2", "immutable-run-id-1", "paper-campaign-v2.0")


def test_three_eligible_zero_admission_reports_make_campaign_unhealthy():
    conn = _db()
    for index in range(3):
        create_paper_trades_from_report(
            conn,
            setup_rows=[_candidate(f"LOWR{index}", resistance=154.0)],
            report_date=f"2026-08-{19 + index:02d}",
            generated_ts=f"report-run-{index}",
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
        "eligible_candidate_zero_admission_streak"
    ]


def test_candidate_decision_rows_are_immutable():
    conn = _db()
    create_paper_trades_from_report(
        conn,
        setup_rows=[_candidate("FAIL", tier="F")],
        report_date="2026-08-21",
        generated_ts="report-run-1",
    )

    with pytest.raises(sqlite3.IntegrityError, match="decisions are immutable"):
        conn.execute("UPDATE paper_candidate_decisions SET reason_code = 'changed'")

from __future__ import annotations

import datetime as dt
import inspect
import json
import sqlite3

import pytest

from trader_koo.paper_trade import shadow
from trader_koo.db.price_contract import record_price_series_revision
from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trade.shadow import (
    breadth_shadow_summary,
    record_breadth_shadow,
    resolve_breadth_shadow_outcomes,
)
from trader_koo.paper_trades import _build_config


def _candidate(ticker: str, tier: str, score: float = 80) -> dict[str, object]:
    return {
        "ticker": ticker,
        "setup_tier": tier,
        "score": score,
        "actionability": "higher-probability",
        "signal_bias": "bullish",
        "setup_family": "Bullish Breakout",
        "close": 100.0,
    }


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    return conn


def test_p1_changes_only_tier_admission_and_cannot_create_orders() -> None:
    conn = _db()
    rows = [
        _candidate("AAA", "A"),
        _candidate("BBB", "B"),
        _candidate("CCC", "C"),
        _candidate("LOW", "A", 59),
    ]

    assert record_breadth_shadow(
        conn,
        report_run_id="run-1",
        report_date="2026-08-23",
        generated_ts="2026-08-23T01:00:00Z",
        setup_rows=rows,
        base_config=_build_config(),
    ) is True

    assert conn.execute(
        "SELECT policy_id,accepted_count FROM paper_shadow_decision_sets ORDER BY policy_id"
    ).fetchall() == [("P0", 2), ("P1", 3)]
    assert conn.execute(
        """SELECT policy_id,ticker,disposition,reason_code
           FROM paper_shadow_decisions WHERE ticker IN ('CCC','LOW')
           ORDER BY policy_id,ticker"""
    ).fetchall() == [
        ("P0", "CCC", "rejected", "tier_below_minimum"),
        ("P0", "LOW", "rejected", "score_below_minimum"),
        ("P1", "CCC", "accepted", "accepted"),
        ("P1", "LOW", "rejected", "score_below_minimum"),
    ]
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_pending_orders").fetchone()[0] == 0
    source = inspect.getsource(shadow)
    assert "paper_pending_orders" not in source
    assert "INSERT INTO paper_trades" not in source

    specs = {
        row[0]: json.loads(row[1])
        for row in conn.execute(
            "SELECT policy_id,specification_json FROM paper_shadow_policies"
        )
    }
    p0, p1 = specs["P0"]["policy"], specs["P1"]["policy"]
    for field in ("min_tier", "qualifying_tiers"):
        p0.pop(field)
        p1.pop(field)
    assert p0 == p1


def test_shadow_sets_are_idempotent_sealed_and_prospective_only() -> None:
    conn = _db()
    kwargs = {
        "report_run_id": "run-1",
        "report_date": "2026-08-23",
        "generated_ts": "2026-08-23T01:00:00Z",
        "setup_rows": [_candidate("AAA", "A")],
        "base_config": _build_config(),
    }
    assert record_breadth_shadow(conn, **kwargs) is True
    assert record_breadth_shadow(conn, **kwargs) is False
    with pytest.raises(ValueError, match="divergent shadow retry"):
        record_breadth_shadow(
            conn, **{**kwargs, "setup_rows": [_candidate("CHANGED", "A")]}
        )
    with pytest.raises(sqlite3.IntegrityError, match="shadow decisions are immutable"):
        conn.execute("UPDATE paper_shadow_decisions SET ticker='CHANGED'")

    before = conn.execute("SELECT COUNT(*) FROM paper_shadow_decision_sets").fetchone()[0]
    assert record_breadth_shadow(
        conn,
        report_run_id="historical-run",
        report_date="2026-08-22",
        generated_ts="2026-08-22T01:00:00Z",
        setup_rows=[_candidate("OLD", "A")],
        base_config=_build_config(),
    ) is False
    assert conn.execute("SELECT COUNT(*) FROM paper_shadow_decision_sets").fetchone()[0] == before


def test_mature_shadow_outcomes_use_exact_next_open_and_tenth_close() -> None:
    conn = _db()
    conn.execute("""
        CREATE TABLE price_daily (
            ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
            volume REAL, adjustment_basis TEXT, adjustment_version TEXT,
            basis_status TEXT, unresolved_reason TEXT,
            UNIQUE(ticker,date)
        )
    """)
    dates: list[str] = []
    day = dt.date(2026, 7, 27)
    while len(dates) < 30:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += dt.timedelta(days=1)
    for ticker, offset in (("AAA", 0.0), ("CCC", 20.0), ("SPY", 50.0)):
        conn.executemany(
            "INSERT INTO price_daily VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
            [(
                ticker, date, 100 + offset + index, 102 + offset + index,
                99 + offset + index, 101 + offset + index, 1_000_000,
                "total_return", "fixture-total-return-v1", "verified",
            ) for index, date in enumerate(dates)],
        )
        record_price_series_revision(
            conn,
            ticker,
            evidence={"vendor_action_ledger_checked": True, "vendor_action_ledger": []},
            fetch_timestamp="2026-09-04T00:00:00Z",
        )
    assert record_breadth_shadow(
        conn,
        report_run_id="run-outcome",
        report_date="2026-08-23",
        generated_ts="2026-08-23T01:00:00Z",
        setup_rows=[_candidate("AAA", "A"), _candidate("CCC", "C")],
        base_config=_build_config(),
    ) is True

    first = resolve_breadth_shadow_outcomes(
        conn, through_date=dates[-1], base_config=_build_config()
    )
    retry = resolve_breadth_shadow_outcomes(
        conn, through_date=dates[-1], base_config=_build_config()
    )

    assert first == {"resolved": 3, "invalid": 0, "pending": 0}
    assert retry == {"resolved": 0, "invalid": 0, "pending": 0}
    outcomes = conn.execute(
        "SELECT intended_entry_date,exit_date,status,result_json FROM paper_shadow_outcomes"
    ).fetchall()
    assert {row[0] for row in outcomes} == {"2026-08-24"}
    assert {row[1] for row in outcomes} == {"2026-09-04"}
    assert {row[2] for row in outcomes} == {"resolved"}
    assert all(json.loads(row[3])["return_basis"] == "total_return" for row in outcomes)
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM paper_pending_orders").fetchone()[0] == 0
    summary = breadth_shadow_summary(conn)
    assert summary["policy_counts"] == {
        "P0": {"candidate_count": 2, "accepted_count": 1},
        "P1": {"candidate_count": 2, "accepted_count": 2},
    }
    assert summary["incremental_cohort"]["accepted_count"] == 1
    assert summary["incremental_cohort"]["resolved_count"] == 1
    assert summary["breadth_increase_pct"] == 100.0
    assert summary["coverage"]["effective_non_overlapping_block_count"] == 1
    assert summary["human_promotion_review_eligible"] is False
    assert summary["automatic_promotion"] is False

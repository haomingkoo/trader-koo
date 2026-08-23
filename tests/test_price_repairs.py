from __future__ import annotations

import json
import sqlite3

import pytest

from trader_koo.db.price_repairs import apply_price_repair, plan_price_repair


KNOWN_TICKERS = ("BKNG", "KLAC", "CVNA", "CRWD", "DD", "MNST")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """CREATE TABLE price_daily (
               ticker TEXT NOT NULL,date TEXT NOT NULL,open REAL,high REAL,low REAL,
               close REAL,volume REAL,data_source TEXT,fetch_timestamp TEXT,
               adjustment_basis TEXT,adjustment_version TEXT,basis_status TEXT,
               unresolved_reason TEXT,PRIMARY KEY(ticker,date));
           CREATE TABLE price_corporate_actions (
               ticker TEXT NOT NULL,action_date TEXT NOT NULL,action_type TEXT NOT NULL,
               provider TEXT NOT NULL,value REAL NOT NULL,applied_to_prices INTEGER NOT NULL,
               adjustment_version TEXT NOT NULL,fetch_timestamp TEXT,evidence_json TEXT,
               UNIQUE(ticker,action_date,action_type,provider));
           CREATE TABLE paper_trades (ticker TEXT,entry_date TEXT);
           CREATE TABLE setup_calls (ticker TEXT,report_date TEXT);"""
    )
    for ticker in KNOWN_TICKERS:
        conn.execute(
            """INSERT INTO price_daily VALUES
               (?, '2026-01-02',1000,1010,990,1000,1000000,'fixture','ts',
                'split_adjusted_price_only','contaminated-v1','unresolved','ratio break')""",
            (ticker,),
        )
        conn.execute(
            """INSERT INTO price_daily VALUES
               (?, '2026-01-03',100,101,99,100,1000000,'fixture','ts',
                'split_adjusted_price_only','contaminated-v1','unresolved','ratio break')""",
            (ticker,),
        )
    return conn


def _proposals() -> list[dict[str, object]]:
    return [{
        "ticker": ticker,
        "date": "2026-01-02",
        "open": 100,
        "high": 101,
        "low": 99,
        "close": 100,
        "volume": 1_000_000,
        "action": {
            "action_type": "split",
            "action_date": "2026-01-03",
            "factor": 10.0,
        },
    } for ticker in KNOWN_TICKERS]


def _evidence() -> dict[str, object]:
    return {
        "provider": "fixture_vendor",
        "vendor_action_ledger_checked": True,
        "full_history_verified": True,
        "vendor_action_ledger": [{"ticker": ticker} for ticker in KNOWN_TICKERS],
    }


def test_repair_dry_run_is_exact_and_does_not_mutate_prices() -> None:
    conn = _db()

    plan = plan_price_repair(
        conn,
        _proposals(),
        adjustment_version="restated-v2",
        reason="known provider rebase contamination",
        provider_evidence=_evidence(),
    )

    assert plan["apply_eligible"] is True
    assert len(plan["changes"]) == 6
    assert plan["unresolved"] == []
    assert len(plan["plan_sha256"]) == 64
    assert conn.execute(
        "SELECT close FROM price_daily WHERE ticker='BKNG' AND date='2026-01-02'"
    ).fetchone()[0] == 1000


def test_apply_preserves_originals_is_idempotent_and_removes_ratio_breaks() -> None:
    conn = _db()
    plan = plan_price_repair(
        conn,
        _proposals(),
        adjustment_version="restated-v2",
        reason="known provider rebase contamination",
        provider_evidence=_evidence(),
    )

    first = apply_price_repair(conn, plan)
    second = apply_price_repair(conn, plan)

    assert first["changed_rows"] == 6
    assert second == {
        **first,
        "changed_rows": 0,
        "already_applied_rows": 6,
        "tickers": [],
    }
    assert conn.execute("SELECT COUNT(*) FROM price_corrections").fetchone()[0] == 6
    original = json.loads(conn.execute(
        "SELECT original_json FROM price_corrections WHERE ticker='BKNG'"
    ).fetchone()[0])
    assert original["close"] == 1000
    for ticker in KNOWN_TICKERS:
        closes = [row[0] for row in conn.execute(
            "SELECT close FROM price_daily WHERE ticker=? ORDER BY date", (ticker,)
        )]
        assert closes == [100, 100]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE price_corrections SET reason='forged'")


def test_apply_rejects_source_drift_and_unresolved_plan() -> None:
    conn = _db()
    plan = plan_price_repair(
        conn,
        _proposals(),
        adjustment_version="restated-v2",
        reason="known provider rebase contamination",
        provider_evidence=_evidence(),
    )
    conn.execute(
        "UPDATE price_daily SET close=999 WHERE ticker='BKNG' AND date='2026-01-02'"
    )
    with pytest.raises(ValueError, match="source row drifted"):
        apply_price_repair(conn, plan)

    unresolved = plan_price_repair(
        conn,
        [{**_proposals()[0], "ticker": "MISSING"}],
        adjustment_version="restated-v2",
        reason="missing row",
        provider_evidence=_evidence(),
    )
    with pytest.raises(ValueError, match="not apply eligible"):
        apply_price_repair(conn, unresolved)

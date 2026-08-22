from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from trader_koo.notifications.macro_monitor import _get_prev_close
from trader_koo.report.setup_scoring import (
    _persist_setup_call_candidates,
    _score_open_setup_call_outcomes,
    ensure_setup_call_eval_schema,
)
from trader_koo.scripts.update_market_db import (
    ensure_schema,
    reconcile_vendor_action_ledger,
    write_price_daily,
)


def _seed_verified_prices(conn: sqlite3.Connection, ticker: str = "AAA") -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-08-18", "open": 99, "high": 101, "low": 98, "close": 100, "volume": 1000},
            {"date": "2026-08-19", "open": 100, "high": 103, "low": 99, "close": 102, "volume": 1100},
            {"date": "2026-08-20", "open": 102, "high": 105, "low": 101, "close": 104, "volume": 1200},
        ]
    )
    frame.attrs.update(
        adjustment_basis="split_adjusted_price_only",
        adjustment_version="fixture-v1",
        basis_status="verified",
    )
    reconcile_vendor_action_ledger(frame, [])
    write_price_daily(conn, ticker, frame, fetch_timestamp="2026-08-20T22:00:00Z")
    conn.commit()


def test_setup_call_mutations_pause_when_price_revision_is_unresolved() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    ensure_setup_call_eval_schema(conn)
    _seed_verified_prices(conn)
    conn.execute("UPDATE price_daily SET basis_status='unresolved' WHERE ticker='AAA'")
    conn.execute(
        """INSERT INTO setup_call_evaluations (
               asof_date,ticker,report_kind,call_direction,validity_days,close_asof,status
           ) VALUES ('2026-08-18','AAA','daily','long',2,100,'open')"""
    )
    conn.commit()

    inserted = _persist_setup_call_candidates(
        conn,
        generated_ts="2026-08-20T22:00:00Z",
        report_kind="weekly",
        asof_date="2026-08-20",
        setup_rows=[{"ticker": "AAA", "signal_bias": "bullish", "close": 104}],
    )
    scored = _score_open_setup_call_outcomes(conn)

    assert inserted == 0
    assert scored == 0
    assert conn.execute(
        "SELECT status FROM setup_call_evaluations WHERE ticker='AAA'"
    ).fetchone()[0] == "open"


def test_macro_reference_close_fails_closed_after_revision_break(tmp_path: Path) -> None:
    db_path = tmp_path / "prices.db"
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    _seed_verified_prices(conn, "SPY")
    conn.close()
    assert _get_prev_close(db_path, "SPY") == 104.0

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE price_daily SET close=5, basis_status='unresolved' WHERE ticker='SPY'")
    conn.commit()
    conn.close()
    assert _get_prev_close(db_path, "SPY") is None

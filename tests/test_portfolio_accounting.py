from __future__ import annotations

import sqlite3

import pytest

from trader_koo.paper_trade.portfolio_accounting import reconcile_portfolio


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE paper_trades (
               id INTEGER PRIMARY KEY,
               campaign_id TEXT NOT NULL,
               ticker TEXT NOT NULL,
               direction TEXT NOT NULL,
               status TEXT NOT NULL,
               entry_price REAL,
               current_price REAL,
               exit_price REAL,
               quantity REAL,
               entry_notional REAL,
               entry_commission REAL,
               exit_commission REAL,
               borrow_cost REAL,
               realized_pnl_usd REAL,
               entry_date TEXT,
               exit_date TEXT,
               last_mtm_date TEXT,
               accounting_status TEXT NOT NULL
           )"""
    )
    return conn


def test_cash_positions_and_pnl_reconcile_for_open_position() -> None:
    conn = _db()
    conn.execute(
        """INSERT INTO paper_trades VALUES
           (1,'paper-v2','AAA','long','open',100,105,NULL,100,10000,1,NULL,NULL,NULL,
            '2026-08-21',NULL,'2026-08-22','reconciled')"""
    )

    account = reconcile_portfolio(
        conn, campaign_id="paper-v2", starting_capital=100_000,
    )

    assert account["cash"] == pytest.approx(89_999)
    assert account["equity"] == pytest.approx(100_499)
    assert account["unrealized_pnl_usd"] == pytest.approx(499)
    assert account["accounting_breaks"] == []


def test_closed_position_reconciles_and_legacy_rows_are_excluded() -> None:
    conn = _db()
    conn.execute(
        """INSERT INTO paper_trades VALUES
           (1,'paper-v2','AAA','long','target_hit',100,110,110,100,10000,1,1,0,998,
            '2026-08-21','2026-08-25','2026-08-25','reconciled')"""
    )
    conn.execute(
        """INSERT INTO paper_trades VALUES
           (2,'paper-v2','OLD','long','open',50,55,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
            '2026-01-01',NULL,NULL,'legacy_unreconciled')"""
    )

    account = reconcile_portfolio(
        conn, campaign_id="paper-v2", starting_capital=100_000,
    )

    assert account["cash"] == pytest.approx(100_998)
    assert account["equity"] == pytest.approx(100_998)
    assert account["realized_pnl_usd"] == pytest.approx(998)
    assert account["legacy_unreconciled_count"] == 1
    assert account["accounting_breaks"] == []

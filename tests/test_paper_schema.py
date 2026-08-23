from __future__ import annotations

import sqlite3

from trader_koo.paper_trade.schema import _rebuild_unique_key


def test_parent_unique_key_rebuild_preserves_child_foreign_keys() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            campaign_id TEXT,
            report_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            UNIQUE(report_date, ticker, direction)
        );
        CREATE TABLE paper_trade_reflections (
            trade_id INTEGER PRIMARY KEY REFERENCES paper_trades(id)
        );
        INSERT INTO paper_trades VALUES (1,'paper-v1','2026-01-01','AAA','long');
        INSERT INTO paper_trade_reflections VALUES (1);
        """
    )

    _rebuild_unique_key(
        conn,
        "paper_trades",
        "UNIQUE(report_date, ticker, direction)",
        "UNIQUE(campaign_id, report_date, ticker, direction)",
    )

    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_list(paper_trade_reflections)").fetchone()[2] == "paper_trades"
    assert conn.execute("SELECT trade_id FROM paper_trade_reflections").fetchall() == [(1,)]

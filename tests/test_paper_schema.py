from __future__ import annotations

import sqlite3

import pytest

from trader_koo.paper_trade.schema import _rebuild_unique_key, ensure_paper_trade_schema


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


def test_unique_key_rebuild_does_not_commit_a_caller_transaction() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE item(id INTEGER PRIMARY KEY, value TEXT, UNIQUE(value))")
    conn.commit()
    conn.execute("INSERT INTO item(value) VALUES ('caller-owned')")

    with pytest.raises(RuntimeError, match="transaction boundary"):
        _rebuild_unique_key(conn, "item", "UNIQUE(value)", "UNIQUE(id, value)")

    assert conn.in_transaction is True
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM item").fetchone()[0] == 0


def test_public_schema_initializer_rejects_caller_owned_work() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE caller_work(value TEXT)")
    conn.execute("INSERT INTO caller_work VALUES ('uncommitted')")

    with pytest.raises(RuntimeError, match="clean transaction boundary"):
        ensure_paper_trade_schema(conn)

    assert conn.in_transaction is True
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM caller_work").fetchone()[0] == 0


def test_unique_key_rebuild_rolls_back_before_committing_fk_violations() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        PRAGMA foreign_keys=OFF;
        CREATE TABLE parent(id INTEGER PRIMARY KEY, value TEXT, UNIQUE(value));
        CREATE TABLE child(parent_id INTEGER REFERENCES parent(id));
        INSERT INTO child VALUES (99);
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="foreign-key violations"):
        _rebuild_unique_key(
            conn, "parent", "UNIQUE(value)", "UNIQUE(id, value)"
        )

    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='parent'"
    ).fetchone()[0]
    assert "UNIQUE(value)" in sql
    assert conn.execute("SELECT COUNT(*) FROM parent").fetchone()[0] == 0

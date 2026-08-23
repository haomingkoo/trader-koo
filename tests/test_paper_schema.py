from __future__ import annotations

import sqlite3

import pytest

from trader_koo.paper_trade.schema import (
    PAPER_TRADE_SCHEMA_VERSION,
    _rebuild_unique_key,
    ensure_paper_trade_schema,
)
from trader_koo.scripts.release_evidence import _schema_contract


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


def test_v3_database_runs_v4_expand_contract_instead_of_short_circuiting() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    conn.execute("DROP INDEX idx_paper_trades_legacy_compat")
    conn.execute("UPDATE paper_trade_schema_meta SET schema_version=3 WHERE id=1")
    conn.commit()

    ensure_paper_trade_schema(conn)
    contract = _schema_contract(conn)

    assert PAPER_TRADE_SCHEMA_VERSION == 4
    assert conn.execute(
        "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (4,)
    assert contract["passed"] is True
    assert contract["malformed_indexes"] == []


def test_v4_expand_contract_accepts_legacy_trade_table_without_lineage_fk() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    table_sql = str(conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='paper_trades'"
    ).fetchone()[0])
    legacy_sql = table_sql.replace(
        "CREATE TABLE paper_trades", "CREATE TABLE paper_trades_legacy", 1
    ).replace(
        "report_run_id TEXT REFERENCES report_runs(run_id)", "report_run_id TEXT", 1
    )
    assert legacy_sql != table_sql
    columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(paper_trades)")]
    column_list = ",".join(f'"{column}"' for column in columns)

    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(legacy_sql)
    conn.execute(
        f"INSERT INTO paper_trades_legacy ({column_list}) "
        f"SELECT {column_list} FROM paper_trades"
    )
    conn.execute("DROP TABLE paper_trades")
    conn.execute("ALTER TABLE paper_trades_legacy RENAME TO paper_trades")
    conn.execute("UPDATE paper_trade_schema_meta SET schema_version=3 WHERE id=1")
    conn.commit()

    ensure_paper_trade_schema(conn)
    contract = _schema_contract(conn)

    assert contract["passed"] is True
    assert not any(
        str(row[2]) == "report_runs" and str(row[3]) == "report_run_id"
        for row in conn.execute("PRAGMA foreign_key_list(paper_trades)")
    )


def test_v4_expand_contract_rejects_malformed_optional_lineage_fk() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE wrong_runs(id TEXT PRIMARY KEY);
        CREATE TABLE paper_trades(
            id INTEGER PRIMARY KEY,
            report_run_id TEXT REFERENCES wrong_runs(id)
        );
        """
    )

    contract = _schema_contract(conn)

    assert contract["malformed_optional_foreign_keys"] == [{
        "table": "paper_trades",
        "column": "report_run_id",
        "constraints": [{
            "id": 0,
            "mappings": [["report_run_id", "wrong_runs", "id"]],
        }],
    }]
    assert contract["passed"] is False


def test_v4_expand_contract_rejects_composite_optional_lineage_fk() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE report_runs(
            run_id TEXT,
            report_kind TEXT,
            UNIQUE(run_id, report_kind)
        );
        CREATE TABLE paper_trades(
            id INTEGER PRIMARY KEY,
            report_run_id TEXT,
            report_kind TEXT,
            FOREIGN KEY(report_run_id, report_kind)
                REFERENCES report_runs(run_id, report_kind)
        );
        """
    )

    contract = _schema_contract(conn)

    malformed = contract["malformed_optional_foreign_keys"]
    assert len(malformed) == 1
    assert malformed[0]["constraints"][0]["mappings"] == [
        ["report_run_id", "report_runs", "run_id"],
        ["report_kind", "report_runs", "report_kind"],
    ]
    assert contract["passed"] is False


def test_schema_contract_rejects_correctly_named_malformed_objects() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    conn.execute("DROP INDEX idx_paper_trades_legacy_compat")
    conn.execute(
        "CREATE INDEX idx_paper_trades_legacy_compat ON paper_trades(ticker)"
    )
    conn.execute("DROP TRIGGER paper_v1_trades_no_delete")
    conn.execute(
        "CREATE TRIGGER paper_v1_trades_no_delete BEFORE DELETE ON paper_trades "
        "BEGIN SELECT 1; END"
    )

    contract = _schema_contract(conn)

    assert contract["passed"] is False
    assert contract["malformed_indexes"] == ["idx_paper_trades_legacy_compat"]
    assert contract["malformed_triggers"] == ["paper_v1_trades_no_delete"]


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

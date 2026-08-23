from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from trader_koo.paper_trade.schema import (
    PAPER_TRADE_SCHEMA_VERSION,
    _rebuild_unique_key,
    ensure_paper_trade_schema,
)
from trader_koo.scripts.release_evidence import _schema_contract, migrate_copy


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


def test_release_copy_rescans_current_v4_admission_ledger(tmp_path: Path) -> None:
    source = tmp_path / "current-v4.db"
    with sqlite3.connect(source) as conn:
        ensure_paper_trade_schema(conn)
        run_id = conn.execute(
            """INSERT INTO report_runs
               (run_id,report_kind,status,started_ts,config_json,config_hash,code_version)
               VALUES ('legacy-run','daily','started','2026-08-21T00:00:00Z',
                       '{}',?,?) RETURNING run_id""",
            ("a" * 64, "b" * 40),
        ).fetchone()[0]
        conn.execute("DROP TABLE report_admission_attempts")
        conn.execute(
            """CREATE TABLE report_admission_attempts (
                   attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   run_id TEXT,
                   status TEXT,
                   error_code TEXT,
                   error_message TEXT,
                   attempted_ts TEXT
               )"""
        )
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed','admission_finalize_failed',?,
                       '2026-08-22T00:00:00Z')""",
            (run_id, "\t"),
        )
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed',NULL,'ValueError','2026-08-22T00:00:01Z')""",
            (run_id,),
        )
        conn.execute(
            "DELETE FROM report_schema_migrations "
            "WHERE migration='admission-ledger-contract-v3'"
        )
        conn.execute(
            "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) "
            "VALUES ('admission-ledger-contract-v2','2026-08-21T00:00:00Z')"
        )
        conn.commit()

    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    with pytest.raises(RuntimeError, match="report-admission contract failed"):
        migrate_copy(source, output_dir)
    failure = json.loads(
        (output_dir / "database-migration-manifest.json").read_text()
    )
    assert failure["passed"] is False
    assert failure["report_admission_contract"] == {
        "passed": False,
        "violation": "legacy_rows_invalid",
        "invalid_row_count": 1,
        "affected_attempts": [{
            "attempt_id": 2,
            "violations": ["failure_error_metadata_invalid"],
        }],
    }
    assert failure["migrated_copy_sha256"] == hashlib.sha256(
        (output_dir / "database-copy.db").read_bytes()
    ).hexdigest()


def test_release_copy_records_verified_admission_contract(tmp_path: Path) -> None:
    source = tmp_path / "clean.db"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE release_seed(id INTEGER PRIMARY KEY)")
    output_dir = tmp_path / "clean-evidence"
    output_dir.mkdir()

    manifest = migrate_copy(source, output_dir)

    assert manifest["passed"] is True
    assert manifest["report_admission_contract"] == {
        "passed": True,
        "migration": "admission-ledger-contract-v3",
    }


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
            "mappings": [[
                "report_run_id", "wrong_runs", "id",
                "NO ACTION", "NO ACTION", "NONE",
            ]],
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
        ["report_run_id", "report_runs", "run_id", "NO ACTION", "NO ACTION", "NONE"],
        ["report_kind", "report_runs", "report_kind", "NO ACTION", "NO ACTION", "NONE"],
    ]
    assert contract["passed"] is False


def test_v4_expand_contract_rejects_cascading_optional_lineage_fk() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE report_runs(run_id TEXT PRIMARY KEY);
        CREATE TABLE paper_trades(
            id INTEGER PRIMARY KEY,
            report_run_id TEXT REFERENCES report_runs(run_id) ON DELETE CASCADE
        );
        """
    )

    contract = _schema_contract(conn)

    mappings = contract["malformed_optional_foreign_keys"][0]["constraints"][0]["mappings"]
    assert mappings == [[
        "report_run_id", "report_runs", "run_id",
        "NO ACTION", "CASCADE", "NONE",
    ]]
    assert contract["passed"] is False


def test_v4_expand_contract_rejects_duplicate_optional_lineage_fks() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE report_runs(run_id TEXT PRIMARY KEY);
        CREATE TABLE paper_trades(
            id INTEGER PRIMARY KEY,
            report_run_id TEXT,
            FOREIGN KEY(report_run_id) REFERENCES report_runs(run_id),
            FOREIGN KEY(report_run_id) REFERENCES report_runs(run_id)
        );
        """
    )

    contract = _schema_contract(conn)

    assert len(
        contract["malformed_optional_foreign_keys"][0]["constraints"]
    ) == 2
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

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from trader_koo.paper_trade.schema import (
    PAPER_TRADE_SCHEMA_VERSION,
    PaperSchemaInitializationError,
    _verified_v5_paths,
    ensure_paper_trade_schema,
    require_contracted_paper_schema,
)
from trader_koo.paper_trade.schema_v5_migration import (
    _logical_database_hash,
    migrate_paper_schema_v4_to_v5,
)
from trader_koo.paper_trade.schema_v5_verifier import (
    PaperSchemaV5VerificationError,
)
from trader_koo.scripts.release_evidence import migrate_copy


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def _fixture(name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript((FIXTURES / name).read_text(encoding="utf-8"))
    return conn


def _file_fixture(path: Path, name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript((FIXTURES / name).read_text(encoding="utf-8"))
    return conn


def test_fresh_database_stays_expand_compatible_v4_and_is_idempotent() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")

    ensure_paper_trade_schema(conn)
    first = _logical_database_hash(conn)
    ensure_paper_trade_schema(conn)

    assert PAPER_TRADE_SCHEMA_VERSION == 4
    assert conn.execute(
        "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (4,)
    assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert {
        "idx_paper_trades_legacy_compat",
        "idx_paper_portfolio_legacy_compat",
    } <= {
        str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
    }
    conn.execute(
        "SELECT report_date,ticker,direction,status FROM paper_trades LIMIT 1"
    ).fetchall()
    conn.execute(
        "SELECT snapshot_date,open_trades,equity_index "
        "FROM paper_portfolio_snapshots LIMIT 1"
    ).fetchall()
    assert _logical_database_hash(conn) == first


def test_production_like_v4_is_a_read_only_noop() -> None:
    conn = _fixture("paper_schema_v4_legacy_production_like.sql")
    before = (_logical_database_hash(conn), conn.total_changes)

    ensure_paper_trade_schema(conn)

    assert (_logical_database_hash(conn), conn.total_changes) == before
    assert conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE campaign_id='paper-v1'"
    ).fetchone() == (42,)


def test_exact_v5_is_verified_read_only_without_calling_v4_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _fixture("paper_schema_v5_target.sql")
    before = (_logical_database_hash(conn), conn.total_changes)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("v5 must not call v4 schema helpers")

    monkeypatch.setattr(
        "trader_koo.report.runs.ensure_report_run_schema", forbidden,
    )
    monkeypatch.setattr(
        "trader_koo.paper_trade.shadow.ensure_shadow_schema", forbidden,
    )
    ensure_paper_trade_schema(conn)
    ensure_paper_trade_schema(conn)

    assert (_logical_database_hash(conn), conn.total_changes) == before


@pytest.mark.parametrize(
    "index",
    ["idx_paper_trades_legacy_compat", "idx_paper_portfolio_legacy_compat"],
)
def test_v4_rejects_same_name_nonunique_rollback_index(index: str) -> None:
    conn = _fixture("paper_schema_v4_fresh.sql")
    table, columns = {
        "idx_paper_trades_legacy_compat": (
            "paper_trades", "report_date,ticker,direction",
        ),
        "idx_paper_portfolio_legacy_compat": (
            "paper_portfolio_snapshots", "snapshot_date",
        ),
    }[index]
    conn.execute(f'DROP INDEX "{index}"')
    conn.execute(f'CREATE INDEX "{index}" ON "{table}"({columns})')
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes)

    with pytest.raises(PaperSchemaInitializationError) as raised:
        ensure_paper_trade_schema(conn)

    assert raised.value.diagnostics == ({
        "code": "v4_rollback_interlock_missing", "index": index,
    },)
    assert (_logical_database_hash(conn), conn.total_changes) == before


def test_v4_expand_rejects_malformed_existing_index_before_mutation() -> None:
    conn = _fixture("paper_schema_v4_fresh.sql")
    conn.execute("UPDATE paper_trade_schema_meta SET schema_version=3 WHERE id=1")
    conn.execute("DROP INDEX idx_paper_trades_legacy_compat")
    conn.execute(
        "CREATE INDEX idx_paper_trades_legacy_compat "
        "ON paper_trades(report_date,ticker,direction)"
    )
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes)

    with pytest.raises(PaperSchemaInitializationError) as raised:
        ensure_paper_trade_schema(conn)

    assert raised.value.diagnostics == ({
        "code": "v4_rollback_interlock_missing",
        "index": "idx_paper_trades_legacy_compat",
    },)
    assert (_logical_database_hash(conn), conn.total_changes) == before
    assert conn.execute(
        "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (3,)


@pytest.mark.parametrize(
    ("occurrence", "table"),
    [(0, "paper_portfolio_snapshots"), (1, "paper_trades")],
)
def test_v4_rejects_nullable_campaign_columns_with_valid_defaults(
    occurrence: int,
    table: str,
) -> None:
    sql = (FIXTURES / "paper_schema_v4_fresh.sql").read_text(encoding="utf-8")
    old = "campaign_id TEXT NOT NULL DEFAULT 'paper-v2'"
    positions = [index for index in range(len(sql)) if sql.startswith(old, index)]
    position = positions[occurrence]
    malformed = sql[:position] + old.replace(" NOT NULL", "") + sql[position + len(old):]
    conn = sqlite3.connect(":memory:")
    conn.executescript(malformed)

    with pytest.raises(PaperSchemaInitializationError) as raised:
        ensure_paper_trade_schema(conn)

    assert {
        "code": "v4_rollback_interlock_missing",
        "table": table,
        "column": "campaign_id",
    } in raised.value.diagnostics


@pytest.mark.parametrize("version", [6, "not-a-version"])
def test_unknown_v4_identity_versions_fail_without_mutation(version: object) -> None:
    conn = _fixture("paper_schema_v4_fresh.sql")
    conn.execute(
        "UPDATE paper_trade_schema_meta SET schema_version=? WHERE id=1", (version,),
    )
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes, conn.in_transaction)

    with pytest.raises(PaperSchemaInitializationError):
        ensure_paper_trade_schema(conn)

    assert (
        _logical_database_hash(conn), conn.total_changes, conn.in_transaction,
    ) == before


def test_multiple_v4_identity_rows_fail_without_mutation() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE paper_trades(campaign_id TEXT DEFAULT 'paper-v1');
        CREATE TABLE paper_portfolio_snapshots(
            campaign_id TEXT DEFAULT 'paper-v1'
        );
        CREATE TABLE paper_trade_schema_meta(id INTEGER,schema_version INTEGER);
        INSERT INTO paper_trade_schema_meta VALUES (1,4),(2,4);
    """)
    before = (_logical_database_hash(conn), conn.total_changes)

    with pytest.raises(PaperSchemaInitializationError) as raised:
        ensure_paper_trade_schema(conn)

    assert raised.value.diagnostics == ({
        "code": "ambiguous_schema_phase", "reason": "invalid_v4_identity",
    },)
    assert (_logical_database_hash(conn), conn.total_changes) == before


def test_partial_legacy_schema_without_identity_fails_before_ddl() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE paper_trades(id INTEGER PRIMARY KEY)")
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes)

    with pytest.raises(PaperSchemaInitializationError) as raised:
        ensure_paper_trade_schema(conn)

    assert raised.value.diagnostics == ({
        "code": "ambiguous_schema_phase", "reason": "partial_legacy_schema",
    },)
    assert (_logical_database_hash(conn), conn.total_changes) == before


def test_partial_v5_marker_never_falls_back_to_v4_repair() -> None:
    conn = _fixture("paper_schema_v4_fresh.sql")
    conn.execute(
        "INSERT INTO schema_migrations(migration_id) "
        "VALUES ('paper_schema_contract_v5_20260826')"
    )
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes)

    with pytest.raises(PaperSchemaV5VerificationError):
        ensure_paper_trade_schema(conn)

    assert (_logical_database_hash(conn), conn.total_changes) == before
    assert conn.execute(
        "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (4,)


def test_on_disk_partial_v5_identity_reaches_stable_verifier_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial-v5.db"
    conn = _file_fixture(path, "paper_schema_v5_target.sql")
    conn.execute("DROP TABLE paper_trade_schema_meta")
    conn.execute(
        """CREATE TABLE paper_trade_schema_meta(
             id INTEGER PRIMARY KEY,
             schema_version INTEGER NOT NULL,
             contract_id TEXT NOT NULL
           )"""
    )
    conn.execute(
        "INSERT INTO paper_trade_schema_meta VALUES "
        "(1,5,'paper-schema-contract-v5')"
    )
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes, conn.in_transaction)

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        ensure_paper_trade_schema(conn)

    assert "missing_v5_identity" in {
        item["code"] for item in raised.value.diagnostics
    }
    assert (
        _logical_database_hash(conn), conn.total_changes, conn.in_transaction,
    ) == before


def test_v5_objects_without_identity_or_target_marker_do_not_fall_back() -> None:
    conn = _fixture("paper_schema_v5_target.sql")
    conn.execute("DROP TABLE paper_trade_schema_meta")
    conn.execute(
        "DELETE FROM schema_migrations "
        "WHERE migration_id='paper_schema_contract_v5_20260826'"
    )
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes)

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        ensure_paper_trade_schema(conn)

    assert "missing_v5_identity" in {
        item["code"] for item in raised.value.diagnostics
    }
    assert (_logical_database_hash(conn), conn.total_changes) == before


def test_v5_schema_rewritten_to_v4_identity_is_rejected_before_mutation() -> None:
    conn = _fixture("paper_schema_v5_target.sql")
    conn.execute("DROP TABLE paper_trade_schema_meta")
    conn.execute(
        "CREATE TABLE paper_trade_schema_meta(id INTEGER PRIMARY KEY,schema_version INTEGER)"
    )
    conn.execute("INSERT INTO paper_trade_schema_meta VALUES (1,4)")
    conn.commit()
    before = (_logical_database_hash(conn), conn.total_changes)

    with pytest.raises(PaperSchemaV5VerificationError):
        ensure_paper_trade_schema(conn)

    assert (_logical_database_hash(conn), conn.total_changes) == before


def test_preexisting_transaction_semantics_are_phase_aware() -> None:
    v4 = _fixture("paper_schema_v4_fresh.sql")
    v4.execute("BEGIN")
    ensure_paper_trade_schema(v4)
    assert v4.in_transaction
    v4.rollback()

    v3 = _fixture("paper_schema_v4_fresh.sql")
    v3.execute("UPDATE paper_trade_schema_meta SET schema_version=3 WHERE id=1")
    with pytest.raises(RuntimeError, match="clean transaction boundary"):
        ensure_paper_trade_schema(v3)
    assert v3.in_transaction
    v3.rollback()

    v5 = _fixture("paper_schema_v5_target.sql")
    v5.execute("BEGIN")
    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        ensure_paper_trade_schema(v5)
    assert raised.value.diagnostics == ({"code": "transaction_already_active"},)
    assert v5.in_transaction
    v5.rollback()


def test_repeated_current_v4_checks_are_bounded_read_only_queries() -> None:
    conn = _fixture("paper_schema_v4_fresh.sql")
    statements: list[str] = []
    conn.set_trace_callback(statements.append)

    ensure_paper_trade_schema(conn)
    ensure_paper_trade_schema(conn)

    lowered = [statement.strip().lower() for statement in statements]
    assert not any("integrity_check" in statement for statement in lowered)
    assert not any(statement.startswith(
        ("create ", "insert ", "update ", "delete ", "drop ", "alter ",
         "commit", "rollback")
    ) for statement in lowered)
    assert len(statements) < 40


def test_on_disk_v5_cache_skips_repeat_deep_scan_and_invalidates_on_ddl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v5.db"
    conn = _file_fixture(path, "paper_schema_v5_target.sql")
    _verified_v5_paths.clear()
    first: list[str] = []
    conn.set_trace_callback(first.append)
    ensure_paper_trade_schema(conn)
    assert any("integrity_check" in statement.lower() for statement in first)

    second: list[str] = []
    conn.set_trace_callback(second.append)
    ensure_paper_trade_schema(conn)
    assert not any("integrity_check" in statement.lower() for statement in second)
    assert len(second) < 25

    conn.execute("DROP INDEX idx_paper_trades_status")
    conn.commit()
    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        ensure_paper_trade_schema(conn)
    assert "missing_index" in {item["code"] for item in raised.value.diagnostics}


def test_v5_cache_invalidates_on_identity_drift(tmp_path: Path) -> None:
    path = tmp_path / "v5-identity.db"
    conn = _file_fixture(path, "paper_schema_v5_target.sql")
    _verified_v5_paths.clear()
    ensure_paper_trade_schema(conn)
    conn.execute(
        "UPDATE paper_trade_schema_meta SET schema_fingerprint=? WHERE id=1",
        ("0" * 64,),
    )
    conn.commit()

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        ensure_paper_trade_schema(conn)

    assert "v5_identity_mismatch" in {
        item["code"] for item in raised.value.diagnostics
    }


def test_temp_overlap_is_checked_before_v5_cache_hit(tmp_path: Path) -> None:
    path = tmp_path / "v5-temp.db"
    conn = _file_fixture(path, "paper_schema_v5_target.sql")
    _verified_v5_paths.clear()
    ensure_paper_trade_schema(conn)
    conn.execute("CREATE TEMP TABLE paper_trades(id INTEGER)")

    with pytest.raises(PaperSchemaInitializationError) as raised:
        ensure_paper_trade_schema(conn)

    assert raised.value.diagnostics == ({
        "code": "temp_schema_overlap",
        "object_type": "table",
        "name": "paper_trades",
        "table": "paper_trades",
    },)


def test_cached_v4_path_recognizes_explicit_offline_transition_to_v5(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transition.db"
    with _file_fixture(path, "paper_schema_v4_fresh.sql") as conn:
        ensure_paper_trade_schema(conn)
    with sqlite3.connect(path) as maintenance:
        migrate_paper_schema_v4_to_v5(maintenance)
    with sqlite3.connect(path) as runtime:
        ensure_paper_trade_schema(runtime)
        assert runtime.execute(
            "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone() == (5,)


def test_initializer_never_calls_offline_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline contraction must never be automatic")

    monkeypatch.setattr(
        "trader_koo.paper_trade.schema_v5_migration."
        "migrate_paper_schema_v4_to_v5",
        forbidden,
    )
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    assert conn.execute(
        "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (4,)
    with pytest.raises(ValueError, match="activation interlock"):
        require_contracted_paper_schema(conn)


def test_release_evidence_records_initializer_rejection_and_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ambiguous-v4.db"
    sql = (FIXTURES / "paper_schema_v4_fresh.sql").read_text(encoding="utf-8")
    sql = sql.replace(
        "id INTEGER PRIMARY KEY CHECK (id=1)", "id INTEGER PRIMARY KEY", 1,
    )
    with sqlite3.connect(source) as conn:
        conn.executescript(sql)
        conn.execute("INSERT INTO paper_trade_schema_meta VALUES (2,4)")
        conn.commit()
    output = tmp_path / "evidence"
    output.mkdir()

    with pytest.raises(RuntimeError, match="migration or accounting invariant"):
        migrate_copy(source, output)

    manifest = json.loads(
        (output / "database-migration-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["passed"] is False
    assert manifest["paper_schema_initialization_diagnostics"] == [{
        "code": "ambiguous_schema_phase", "reason": "invalid_v4_identity",
    }]

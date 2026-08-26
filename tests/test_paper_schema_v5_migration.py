from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from trader_koo.paper_trade.schema import (
    ensure_paper_trade_schema,
    require_contracted_paper_schema,
)
from trader_koo.paper_trade.schema_v5_migration import (
    PaperSchemaV5MigrationError,
    _logical_database_hash,
    migrate_paper_schema_v4_to_v5,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def _connect_fixture(name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript((FIXTURES / name).read_text(encoding="utf-8"))
    return conn


def _insert_canonical_v2_trade(
    conn: sqlite3.Connection,
    *,
    accounting_status: str,
    realized_pnl_usd: float,
    entry_notional: float = 100.0,
) -> None:
    trigger_rows = conn.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name IN ('report_runs','report_run_decisions','paper_trades') "
        "ORDER BY name"
    ).fetchall()
    for name, _sql in trigger_rows:
        conn.execute(f'DROP TRIGGER "{name}"')
    conn.execute(
        """INSERT INTO report_runs (
               run_id,report_kind,status,started_ts,completed_ts,published_ts,
               generated_ts,scanned_universe_json,ranked_candidates_json,
               decisions_json,inputs_json,source_timestamps_json,config_json,
               config_hash,code_version,content_hash,markdown_hash,artifact_path,
               markdown_path,generation_key,is_generation_canonical,
               publication_verified)
           VALUES (
               'v2-run','daily','published','2026-08-25T00:00:00Z',
               '2026-08-25T00:01:00Z','2026-08-25T00:02:00Z',
               '2026-08-25T00:01:00Z','["V2"]','["V2"]','[]','{}','{}','{}',
               ?,?,?,?,?,?,'daily:2026-08-25T00:01:00Z',1,1)""",
        ("a" * 64, "b" * 40, "c" * 64, "d" * 64, "/tmp/report.json", "/tmp/report.md"),
    )
    conn.execute(
        """INSERT INTO report_run_decisions
               (run_id,ticker,selected_rank,decision,reason_codes_json,inputs_json)
           VALUES ('v2-run','V2',1,'accepted','[]','{}')"""
    )
    conn.execute(
        """INSERT INTO paper_trades
               (report_date,generated_ts,report_run_id,campaign_id,ticker,direction,
                entry_price,entry_date,exit_price,exit_date,status,current_price,
                quantity,entry_notional,entry_commission,exit_commission,borrow_cost,
                realized_pnl_usd,accounting_status)
           VALUES ('2026-08-25','2026-08-25T00:01:00Z','v2-run','paper-v2','V2',
                   'long',100.0,'2026-08-25',110.0,'2026-08-26','closed',110.0,
                   1.0,?,0.0,0.0,0.0,?,?)""",
        (entry_notional, realized_pnl_usd, accounting_status),
    )
    for _name, sql in trigger_rows:
        conn.execute(str(sql))
    conn.commit()


def _insert_offsetting_v2_trade(conn: sqlite3.Connection) -> None:
    trigger = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='report_run_decisions_parent_started'"
    ).fetchone()[0]
    conn.execute("DROP TRIGGER report_run_decisions_parent_started")
    conn.execute(
        """INSERT INTO report_run_decisions
               (run_id,ticker,selected_rank,decision,reason_codes_json,inputs_json)
           VALUES ('v2-run','V2B',2,'accepted','[]','{}')"""
    )
    conn.execute(str(trigger))
    conn.execute(
        """INSERT INTO paper_trades
               (report_date,generated_ts,report_run_id,campaign_id,ticker,direction,
                entry_price,entry_date,exit_price,exit_date,status,current_price,
                quantity,entry_notional,entry_commission,exit_commission,borrow_cost,
                realized_pnl_usd,accounting_status)
           VALUES ('2026-08-26','2026-08-25T00:01:00Z','v2-run','paper-v2','V2B',
                   'long',100.0,'2026-08-25',110.0,'2026-08-26','closed',110.0,
                   1.0,100.0,0.0,0.0,0.0,9.995,'reconciled')"""
    )
    conn.commit()


def _schema_objects(conn: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    return [
        (
            str(kind), str(name), str(table),
            " ".join(str(sql or "").lower().split()),
        )
        for kind, name, table, sql in conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    ]


@pytest.mark.parametrize(
    "fixture",
    ["paper_schema_v4_fresh.sql", "paper_schema_v4_legacy_production_like.sql"],
)
def test_migration_reaches_the_exact_frozen_v5_schema(fixture: str) -> None:
    conn = _connect_fixture(fixture)
    target = _connect_fixture("paper_schema_v5_target.sql")

    result = migrate_paper_schema_v4_to_v5(conn)

    assert result["status"] == "migrated"
    assert result["integrity_check"] == "ok"
    assert result["foreign_key_break_count"] == 0
    assert result["accounting_break_count"] == 0
    assert _schema_objects(conn) == _schema_objects(target)
    assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    contract = json.loads((ROOT / "paper-schema-contract-v5.json").read_text())
    assert conn.execute(
        "SELECT schema_version,contract_id,schema_fingerprint "
        "FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (
        5,
        contract["contract_id"],
        contract["fingerprint"]["expected_sha256"],
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM schema_migrations WHERE migration_id=?",
        (contract["migration_ids"]["target"],),
    ).fetchone() == (1,)


def test_production_like_rows_ids_sequences_and_legacy_reads_are_preserved() -> None:
    conn = _connect_fixture("paper_schema_v4_legacy_production_like.sql")
    trades_before = conn.execute(
        "SELECT id,report_date,ticker,direction,status,campaign_id "
        "FROM paper_trades ORDER BY id"
    ).fetchall()
    snapshot_before = conn.execute(
        "SELECT snapshot_date,open_trades,total_unrealized_pnl_pct,snapshot_ts,"
        "campaign_id FROM paper_portfolio_snapshots"
    ).fetchone()
    sequences_before = dict(conn.execute(
        "SELECT name,seq FROM sqlite_sequence"
    ).fetchall())

    migrate_paper_schema_v4_to_v5(conn)

    assert conn.execute(
        "SELECT id,report_date,ticker,direction,status,campaign_id "
        "FROM paper_trades ORDER BY id"
    ).fetchall() == trades_before
    assert conn.execute(
        "SELECT snapshot_date,open_trades,total_unrealized_pnl_pct,snapshot_ts,"
        "campaign_id FROM paper_portfolio_snapshots"
    ).fetchone() == snapshot_before
    assert conn.execute(
        "SELECT id FROM paper_portfolio_snapshots"
    ).fetchone() == (1,)
    sequences_after = dict(conn.execute(
        "SELECT name,seq FROM sqlite_sequence"
    ).fetchall())
    assert sequences_after["paper_trades"] == sequences_before["paper_trades"] == 42
    assert sequences_after["paper_portfolio_snapshots"] == 1
    assert conn.execute(
        "SELECT report_date,ticker,direction,status FROM paper_trades "
        "ORDER BY id LIMIT 1"
    ).fetchone() == trades_before[0][1:5]
    assert conn.execute(
        "SELECT snapshot_date,open_trades,equity_index "
        "FROM paper_portfolio_snapshots"
    ).fetchone() == (snapshot_before[0], snapshot_before[1], 100.0)


@pytest.mark.parametrize(
    "fault_point", ["after_rename", "after_copy", "after_validation", "before_commit"],
)
def test_injected_failure_restores_the_byte_logical_v4_state(fault_point: str) -> None:
    conn = _connect_fixture("paper_schema_v4_legacy_production_like.sql")
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn, inject_failure_at=fault_point)

    assert raised.value.diagnostics == ({
        "code": "injected_failure", "failure_point": fault_point,
    },)
    assert _logical_database_hash(conn) == before
    assert not conn.in_transaction
    assert conn.execute(
        "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (4,)
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%__v4_source'"
    ).fetchone() == (0,)


def test_contract_collision_cases_produce_stable_named_diagnostics() -> None:
    manifest = json.loads((
        FIXTURES / "paper_schema_v5_contract_cases.json"
    ).read_text())
    for case in manifest["cases"]:
        expected = case.get("expected_error")
        expected_many = case.get("expected_errors", [])
        if not expected and not expected_many:
            continue
        messages: list[str] = []
        codes: set[str] = set()
        for _ in range(2):
            conn = sqlite3.connect(":memory:")
            conn.executescript(";".join(case["setup_sql"]) + ";")
            with pytest.raises(PaperSchemaV5MigrationError) as raised:
                migrate_paper_schema_v4_to_v5(conn)
            messages.append(str(raised.value))
            codes = {str(item["code"]) for item in raised.value.diagnostics}
        assert messages[0] == messages[1]
        assert set(expected_many or [expected]) <= codes


def test_valid_legacy_cancelled_order_is_not_diagnosed_as_invalid() -> None:
    manifest = json.loads((
        FIXTURES / "paper_schema_v5_contract_cases.json"
    ).read_text())
    case = next(item for item in manifest["cases"] if item["id"] == "legacy_cancelled_order")
    conn = sqlite3.connect(":memory:")
    conn.executescript(";".join(case["setup_sql"]) + ";")

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn)

    assert "invalid_legacy_order_hash" not in {
        item["code"] for item in raised.value.diagnostics
    }


def test_unreconciled_v2_trade_fails_the_accounting_gate_and_rolls_back() -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    _insert_canonical_v2_trade(
        conn, accounting_status="legacy_unreconciled", realized_pnl_usd=10.0,
    )
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn)

    assert raised.value.diagnostics == ({
        "code": "accounting",
        "reason": "unreconciled_v2_trade",
        "row_count": 1,
        "first_trade_id": 1,
    },)
    assert _logical_database_hash(conn) == before


@pytest.mark.parametrize("delta", [0.005, 0.01, 0.02])
def test_every_nonzero_v2_accounting_delta_fails_exactly(delta: float) -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    _insert_canonical_v2_trade(
        conn, accounting_status="reconciled", realized_pnl_usd=10.0 + delta,
    )
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn)

    assert raised.value.diagnostics[0]["code"] == "accounting"
    assert raised.value.diagnostics[0]["reason"] == "realized_pnl_mismatch"
    assert raised.value.diagnostics[0]["arithmetic"] == (
        "exact_decimal_from_persisted_values"
    )
    assert raised.value.diagnostics[0]["actual_realized_pnl_usd"] != (
        raised.value.diagnostics[0]["expected_realized_pnl_usd"]
    )
    assert _logical_database_hash(conn) == before


def test_offsetting_per_trade_errors_cannot_cancel() -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    _insert_canonical_v2_trade(
        conn, accounting_status="reconciled", realized_pnl_usd=10.005,
    )
    _insert_offsetting_v2_trade(conn)
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn)

    assert [
        (item["trade_id"], item["reason"])
        for item in raised.value.diagnostics
    ] == [
        (1, "realized_pnl_mismatch"),
        (2, "realized_pnl_mismatch"),
    ]
    assert _logical_database_hash(conn) == before


def test_wrong_entry_notional_fails_per_trade_and_rolls_back() -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    _insert_canonical_v2_trade(
        conn,
        accounting_status="reconciled",
        realized_pnl_usd=10.0,
        entry_notional=99.0,
    )
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn)

    assert raised.value.diagnostics == ({
        "code": "accounting",
        "reason": "entry_notional_mismatch",
        "trade_id": 1,
        "expected_entry_notional": "100.00",
        "actual_entry_notional": "99.0",
        "arithmetic": "exact_decimal_from_persisted_values",
    },)
    assert _logical_database_hash(conn) == before


@pytest.mark.parametrize("missing_column", ["exit_commission", "borrow_cost"])
def test_missing_closed_trade_cost_is_not_invented_as_zero(
    missing_column: str,
) -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    _insert_canonical_v2_trade(
        conn, accounting_status="reconciled", realized_pnl_usd=10.0,
    )
    conn.execute(f'UPDATE paper_trades SET "{missing_column}"=NULL WHERE id=1')
    conn.commit()
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn)

    assert raised.value.diagnostics == ({
        "code": "accounting",
        "reason": "missing_close_accounting",
        "trade_id": 1,
    },)
    assert _logical_database_hash(conn) == before


def test_exact_v2_accounting_has_no_hidden_tolerance() -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    _insert_canonical_v2_trade(
        conn, accounting_status="reconciled", realized_pnl_usd=10.0,
    )

    result = migrate_paper_schema_v4_to_v5(conn)

    assert result["status"] == "migrated"
    assert result["accounting_arithmetic"] == (
        "exact_decimal_from_persisted_values"
    )


def test_target_marker_without_v5_identity_is_deterministic_source_drift() -> None:
    messages: list[str] = []
    for _ in range(2):
        conn = _connect_fixture("paper_schema_v4_fresh.sql")
        contract = json.loads((ROOT / "paper-schema-contract-v5.json").read_text())
        target_migration = contract["migration_ids"]["target"]
        conn.execute(
            "INSERT INTO schema_migrations(migration_id) VALUES (?)",
            (target_migration,),
        )
        conn.commit()
        before = _logical_database_hash(conn)

        with pytest.raises(PaperSchemaV5MigrationError) as raised:
            migrate_paper_schema_v4_to_v5(conn)

        messages.append(str(raised.value))
        assert {
            "code": "v4_contract_drift",
            "reason": "target_migration_without_v5_identity",
            "migration_id": target_migration,
        } in raised.value.diagnostics
        assert _logical_database_hash(conn) == before
        assert not conn.in_transaction
        assert conn.execute(
            "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone() == (4,)
    assert messages[0] == messages[1]


def test_exact_v5_identity_is_an_explicit_noop_not_runtime_verification() -> None:
    conn = _connect_fixture("paper_schema_v5_target.sql")
    before = _logical_database_hash(conn)

    result = migrate_paper_schema_v4_to_v5(conn)

    assert result["status"] == "already_v5_identity_only"
    assert _logical_database_hash(conn) == before
    with pytest.raises(ValueError, match="activation interlock"):
        require_contracted_paper_schema(conn)


def test_migration_supports_the_application_row_factory() -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    conn.row_factory = sqlite3.Row

    assert migrate_paper_schema_v4_to_v5(conn)["status"] == "migrated"
    assert migrate_paper_schema_v4_to_v5(conn)["status"] == "already_v5_identity_only"


def test_migration_requires_a_clean_owned_transaction() -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    conn.execute("BEGIN")

    with pytest.raises(PaperSchemaV5MigrationError) as raised:
        migrate_paper_schema_v4_to_v5(conn)

    assert raised.value.diagnostics == ({"code": "transaction_already_active"},)
    conn.rollback()


def test_normal_initializer_does_not_invoke_the_v5_maintenance_migration() -> None:
    conn = _connect_fixture("paper_schema_v4_fresh.sql")
    schema_before = hashlib.sha256(repr(_schema_objects(conn)).encode()).hexdigest()

    ensure_paper_trade_schema(conn)

    assert hashlib.sha256(repr(_schema_objects(conn)).encode()).hexdigest() == schema_before
    assert [row[1] for row in conn.execute(
        "PRAGMA table_info(paper_trade_schema_meta)"
    )] == ["id", "schema_version"]
    with pytest.raises(ValueError, match="activation interlock"):
        require_contracted_paper_schema(conn)

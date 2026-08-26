from __future__ import annotations

import json
import hashlib
import os
import shutil
import sqlite3
import subprocess
import time
import sys
from pathlib import Path

import pytest

from trader_koo.backend.services.maintenance import (
    MaintenanceError,
    decide,
    migrate_verified_copy,
    quiesce_backup,
    request_maintenance,
    state_path,
    status,
    verify_resolution,
    acquire_lease,
)
from trader_koo.paper_trade.schema import (
    ensure_paper_trade_schema,
    require_contracted_paper_schema,
)
from trader_koo.paper_trade.schema_v5_migration import (
    PaperSchemaV5MigrationError,
    _logical_database_hash,
    migrate_paper_schema_v4_to_v5,
)
from trader_koo.paper_trade.schema_v5_verifier import verify_paper_schema_v5

FIXTURE = Path(__file__).parent / "fixtures" / "paper_schema_v4_legacy_production_like.sql"
FRESH_FIXTURE = Path(__file__).parent / "fixtures" / "paper_schema_v4_fresh.sql"


def _production_copy(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(FIXTURE.read_text(encoding="utf-8"))


def _add_reconciled_v2_trade(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        triggers = conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
            "AND tbl_name IN ('report_runs','report_run_decisions','paper_trades')"
        ).fetchall()
        for name, _sql in triggers:
            conn.execute(f'DROP TRIGGER "{name}"')
        if conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE campaign_id='paper-v1'"
        ).fetchone()[0] == 0:
            conn.executemany(
                """INSERT INTO paper_trades
                       (report_date,ticker,direction,entry_price,entry_date,status,
                        accounting_status,campaign_id)
                   VALUES ('2026-03-18',?,'long',100.0,'2026-03-19','closed',
                           'legacy_unreconciled','paper-v1')""",
                [(f"T{index:02d}",) for index in range(42)],
            )
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
                   '2026-08-25T00:01:00Z','[\"V2\"]','[\"V2\"]','[]','{}','{}','{}',
                   ?,?,?,?,?,?,'daily:2026-08-25T00:01:00Z',1,1)""",
            ("a" * 64, "b" * 40, "c" * 64, "d" * 64,
             "/tmp/report.json", "/tmp/report.md"),
        )
        conn.execute(
            "INSERT INTO report_run_decisions "
            "(run_id,ticker,selected_rank,decision,reason_codes_json,inputs_json) "
            "VALUES ('v2-run','V2',1,'accepted','[]','{}')"
        )
        conn.execute(
            """INSERT INTO paper_trades
                   (report_date,generated_ts,report_run_id,campaign_id,ticker,direction,
                    entry_price,entry_date,exit_price,exit_date,status,current_price,
                    quantity,entry_notional,entry_commission,exit_commission,borrow_cost,
                    realized_pnl_usd,accounting_status)
               VALUES ('2026-08-25','2026-08-25T00:01:00Z','v2-run','paper-v2','V2',
                       'long',100.0,'2026-08-25',110.0,'2026-08-26','closed',110.0,
                       1.0,100.0,0.0,0.0,0.0,10.0,'reconciled')"""
        )
        for _name, sql in triggers:
            conn.execute(str(sql))
        conn.commit()


def _prepared(tmp_path: Path, *, decide_complete: bool = True) -> Path:
    db = tmp_path / "copied-production.db"
    _production_copy(db)
    request_maintenance(
        db, run_id="rehearsal-1", boot_id="writer-boot",
        reason="copied production schema v5 rehearsal", timeout_sec=1,
    )
    quiesce_backup(
        db, run_id="rehearsal-1", boot_id="maintenance-boot",
        backup_dir=tmp_path / "backups",
    )
    if decide_complete:
        decide(
            db, run_id="rehearsal-1", decision="complete",
            reason="reviewed copied-production rehearsal",
        )
    return db


def _logical_hash(path: Path) -> str:
    with sqlite3.connect(path) as conn:
        return _logical_database_hash(conn)


def _rehash(value: dict, field: str) -> dict:
    payload = {key: item for key, item in value.items() if key != field}
    value[field] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()
    return value


def test_complete_decision_is_required_before_live_migration(tmp_path: Path) -> None:
    db = _prepared(tmp_path, decide_complete=False)
    source_hash = _logical_hash(db)
    with pytest.raises(MaintenanceError, match="complete_decision_required"):
        migrate_verified_copy(db, run_id="rehearsal-1")
    assert _logical_hash(db) == source_hash
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone() == (4,)


def test_copied_production_rehearsal_preserves_v1_and_paused_v2(
    tmp_path: Path,
) -> None:
    db = _prepared(tmp_path)
    result = migrate_verified_copy(db, run_id="rehearsal-1")
    plan = json.loads(result["migration_plan_json"])
    receipt = json.loads(result["migration_receipt_json"])
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    assert receipt["migration_status"] == "migrated"
    assert receipt["campaign_evidence"]["v1_trade_count"] == 42
    assert receipt["campaign_evidence"]["v1_legacy_unreconciled_count"] == 42
    assert receipt["campaign_evidence"]["v1_snapshot_count"] == 1

    with sqlite3.connect(db) as conn:
        verified = verify_paper_schema_v5(conn)
        assert verified["schema_fingerprint"] == receipt["schema_fingerprint"]
        assert conn.execute(
            "SELECT campaign_id,status FROM paper_campaigns ORDER BY campaign_id"
        ).fetchall() == [("paper-v1", "frozen"), ("paper-v2", "draft")]
        assert conn.execute(
            "SELECT COUNT(*),SUM(accounting_status='legacy_unreconciled') FROM paper_trades "
            "WHERE campaign_id='paper-v1'"
        ).fetchone() == (42, 42)
        ensure_paper_trade_schema(conn)
        with pytest.raises(ValueError, match="activation interlock"):
            require_contracted_paper_schema(conn)

    resolved = verify_resolution(db, run_id="rehearsal-1")
    assert resolved["state"] == "resolved"


def test_reconciled_v2_accounting_survives_copy_rehearsal(tmp_path: Path) -> None:
    db = tmp_path / "copied-production.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(FRESH_FIXTURE.read_text(encoding="utf-8"))
    _add_reconciled_v2_trade(db)
    request_maintenance(
        db, run_id="rehearsal-1", boot_id="writer-boot",
        reason="v2 accounting rehearsal", timeout_sec=1,
    )
    quiesce_backup(
        db, run_id="rehearsal-1", boot_id="maintenance-boot",
        backup_dir=tmp_path / "backups",
    )
    decide(
        db, run_id="rehearsal-1", decision="complete",
        reason="reviewed v2 accounting rehearsal",
    )
    result = migrate_verified_copy(db, run_id="rehearsal-1")
    evidence = json.loads(result["migration_receipt_json"])["campaign_evidence"]
    assert evidence["v1_trade_count"] == 42
    assert evidence["v2_trade_count"] == 1
    assert evidence["campaign_states"] == {
        "paper-v1": "frozen", "paper-v2": "draft",
    }
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT accounting_status,realized_pnl_usd FROM paper_trades "
            "WHERE campaign_id='paper-v2'"
        ).fetchone() == ("reconciled", 10.0)


def test_rehearsal_plan_and_receipt_are_deterministic_and_idempotent(
    tmp_path: Path,
) -> None:
    db = _prepared(tmp_path)
    first = migrate_verified_copy(db, run_id="rehearsal-1")
    first_receipt = first["migration_receipt_json"]
    first_target = json.loads(first["migration_plan_json"])[
        "expected_target_logical_sha256"
    ]
    second = migrate_verified_copy(db, run_id="rehearsal-1")
    assert second["migration_receipt_json"] == first_receipt
    assert _logical_hash(db) == first_target


def test_delayed_migrations_from_same_copy_have_identical_target(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    _production_copy(source)
    shutil.copy2(source, first)
    shutil.copy2(source, second)
    expected_source = _logical_hash(source)
    with sqlite3.connect(first) as conn:
        migrate_paper_schema_v4_to_v5(
            conn, expected_source_logical_sha256=expected_source,
            target_applied_ts="2026-08-26T00:00:00Z",
        )
    time.sleep(1.05)
    with sqlite3.connect(second) as conn:
        migrate_paper_schema_v4_to_v5(
            conn, expected_source_logical_sha256=expected_source,
            target_applied_ts="2026-08-26T00:00:00Z",
        )
    assert _logical_hash(first) == _logical_hash(second)


def test_source_cohort_is_checked_inside_owned_transaction(tmp_path: Path) -> None:
    db = tmp_path / "copy.db"
    _production_copy(db)
    expected = _logical_hash(db)
    with sqlite3.connect(db) as writer:
        writer.execute(
            "UPDATE schema_migrations SET applied_ts='2026-08-26 00:30:00' "
            "WHERE migration_id='paper_campaign_v1_backfill_20260822'"
        )
        writer.commit()
    changed = _logical_hash(db)
    with sqlite3.connect(db, timeout=0) as conn, pytest.raises(
        PaperSchemaV5MigrationError,
    ) as raised:
        migrate_paper_schema_v4_to_v5(
            conn, expected_source_logical_sha256=expected,
            target_applied_ts="2026-08-26T00:00:00Z",
        )
    assert raised.value.diagnostics[0]["code"] == "source_cohort_mismatch"
    assert _logical_hash(db) == changed
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone() == (4,)


def test_exclusive_contention_has_stable_migration_diagnostic(tmp_path: Path) -> None:
    db = tmp_path / "copy.db"
    _production_copy(db)
    blocker = sqlite3.connect(db)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with sqlite3.connect(db, timeout=0) as conn, pytest.raises(
            PaperSchemaV5MigrationError,
        ) as raised:
            migrate_paper_schema_v4_to_v5(conn)
        assert raised.value.diagnostics[0]["code"] == "exclusive_transaction_unavailable"
    finally:
        blocker.rollback()
        blocker.close()


@pytest.mark.parametrize("point", ["after_plan", "after_migration_commit", "before_receipt"])
def test_crash_retry_uses_fsynced_plan_and_exact_target(
    tmp_path: Path, point: str,
) -> None:
    db = _prepared(tmp_path)

    def crash(at: str) -> None:
        if at == point:
            raise MaintenanceError(f"crash_{point}")

    with pytest.raises(MaintenanceError, match=f"crash_{point}"):
        migrate_verified_copy(db, run_id="rehearsal-1", fault=crash)
    after_crash = status(db, "rehearsal-1")
    assert after_crash["migration_plan_json"]
    recovered = migrate_verified_copy(db, run_id="rehearsal-1")
    receipt = json.loads(recovered["migration_receipt_json"])
    assert receipt["migration_status"] == (
        "migrated" if point == "after_plan" else "recovered_commit"
    )
    verify_resolution(db, run_id="rehearsal-1")


def test_resolution_rejects_tampered_hashed_receipt(tmp_path: Path) -> None:
    db = _prepared(tmp_path)
    migrate_verified_copy(db, run_id="rehearsal-1")
    sidecar = state_path(db)
    with sqlite3.connect(sidecar) as state:
        receipt = json.loads(state.execute(
            "SELECT migration_receipt_json FROM maintenance_runs WHERE run_id='rehearsal-1'"
        ).fetchone()[0])
        receipt["backup_sha256"] = "0" * 64
        state.execute(
            "UPDATE maintenance_runs SET migration_receipt_json=? WHERE run_id='rehearsal-1'",
            (json.dumps(receipt, sort_keys=True),),
        )
        state.commit()
    with pytest.raises(MaintenanceError, match="migration_receipt_invalid"):
        verify_resolution(db, run_id="rehearsal-1")


@pytest.mark.parametrize(
    ("record", "removed", "error"),
    [
        ("migration_plan_json", "expected_target_logical_sha256", "migration_plan_invalid"),
        ("migration_receipt_json", "target_inode", "migration_receipt_invalid"),
    ],
)
def test_self_consistent_but_malformed_evidence_fails_stably(
    tmp_path: Path, record: str, removed: str, error: str,
) -> None:
    db = _prepared(tmp_path)
    migrate_verified_copy(db, run_id="rehearsal-1")
    sidecar = state_path(db)
    hash_field = "plan_sha256" if record == "migration_plan_json" else "receipt_sha256"
    with sqlite3.connect(sidecar) as state:
        value = json.loads(state.execute(
            f"SELECT {record} FROM maintenance_runs WHERE run_id='rehearsal-1'"
        ).fetchone()[0])
        value.pop(removed)
        _rehash(value, hash_field)
        state.execute(
            f"UPDATE maintenance_runs SET {record}=? WHERE run_id='rehearsal-1'",
            (json.dumps(value, sort_keys=True),),
        )
        state.commit()
    with pytest.raises(MaintenanceError, match=error):
        verify_resolution(db, run_id="rehearsal-1")


def test_source_inode_swap_during_plan_fails_before_live_migration(tmp_path: Path) -> None:
    db = _prepared(tmp_path)
    replacement = tmp_path / "replacement.db"
    shutil.copy2(db, replacement)

    def replace_source(at: str) -> None:
        if at == "after_plan":
            os.replace(replacement, db)

    with pytest.raises(MaintenanceError, match="database_source_replaced"):
        migrate_verified_copy(db, run_id="rehearsal-1", fault=replace_source)
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone() == (4,)


def test_shared_writer_lease_blocks_before_plan_or_migration(tmp_path: Path) -> None:
    db = _prepared(tmp_path)
    source_hash = _logical_hash(db)
    shared = acquire_lease(db, exclusive=False)
    try:
        with pytest.raises(MaintenanceError, match="writer_lease_timeout"):
            migrate_verified_copy(db, run_id="rehearsal-1")
    finally:
        shared.close()
    assert _logical_hash(db) == source_hash
    assert not status(db, "rehearsal-1")["migration_plan_json"]


def test_offline_cli_runs_the_same_protected_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trader_koo.scripts.paper_schema_maintenance import main

    db = _prepared(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "paper_schema_maintenance", "migrate-copy", "--db-path", str(db),
        "--run-id", "rehearsal-1",
    ])
    main()
    output = json.loads(capsys.readouterr().out)
    assert output["migration_receipt_json"]


def test_restart_recognizes_v5_but_first_admission_remains_interlocked(
    tmp_path: Path,
) -> None:
    db = _prepared(tmp_path)
    migrate_verified_copy(db, run_id="rehearsal-1")
    script = """
import sqlite3, sys
from trader_koo.paper_trade.schema import ensure_paper_trade_schema, require_contracted_paper_schema
from trader_koo.paper_trade.schema_v5_verifier import verify_paper_schema_v5
with sqlite3.connect(sys.argv[1]) as conn:
    ensure_paper_trade_schema(conn)
    verify_paper_schema_v5(conn)
    try:
        require_contracted_paper_schema(conn)
    except ValueError as exc:
        assert 'activation interlock' in str(exc)
    else:
        raise AssertionError('first admission unexpectedly enabled')
print('v5_restart_verified_admission_blocked')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(db)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "v5_restart_verified_admission_blocked"


def test_crash_after_durable_receipt_reuses_identical_receipt(tmp_path: Path) -> None:
    db = _prepared(tmp_path)

    def crash(at: str) -> None:
        if at == "after_receipt":
            raise MaintenanceError("crash_after_receipt")

    with pytest.raises(MaintenanceError, match="crash_after_receipt"):
        migrate_verified_copy(db, run_id="rehearsal-1", fault=crash)
    crashed = status(db, "rehearsal-1")
    receipt = crashed["migration_receipt_json"]
    assert receipt
    assert crashed["error_code"] == "crash_after_receipt"
    retried = migrate_verified_copy(db, run_id="rehearsal-1")
    assert retried["migration_receipt_json"] == receipt
    assert retried["error_code"] is None


def test_inode_swap_before_receipt_cannot_publish_false_evidence(tmp_path: Path) -> None:
    db = _prepared(tmp_path)
    replacement = tmp_path / "replacement.db"

    def replace_source(at: str) -> None:
        if at == "before_receipt":
            shutil.copy2(db, replacement)
            os.replace(replacement, db)

    with pytest.raises(MaintenanceError, match="database_source_replaced"):
        migrate_verified_copy(db, run_id="rehearsal-1", fault=replace_source)
    assert not status(db, "rehearsal-1")["migration_receipt_json"]


def test_wal_reader_blocks_receipt_then_retry_recovers_planned_target(
    tmp_path: Path,
) -> None:
    db = tmp_path / "copied-production.db"
    _production_copy(db)
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    request_maintenance(
        db, run_id="rehearsal-1", boot_id="writer-boot",
        reason="WAL copied production rehearsal", timeout_sec=1,
    )
    quiesce_backup(
        db, run_id="rehearsal-1", boot_id="maintenance-boot",
        backup_dir=tmp_path / "backups",
    )
    decide(
        db, run_id="rehearsal-1", decision="complete",
        reason="reviewed WAL rehearsal",
    )
    readers: list[sqlite3.Connection] = []

    def hold_reader(at: str) -> None:
        if at == "after_migration_commit":
            reader = sqlite3.connect(db)
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM paper_trades").fetchone()
            readers.append(reader)

    try:
        with pytest.raises(MaintenanceError, match="migration_wal_reader_active"):
            migrate_verified_copy(db, run_id="rehearsal-1", fault=hold_reader)
    finally:
        for reader in readers:
            reader.rollback()
            reader.close()
    assert not status(db, "rehearsal-1")["migration_receipt_json"]
    recovered = migrate_verified_copy(db, run_id="rehearsal-1")
    assert json.loads(recovered["migration_receipt_json"])["migration_status"] == (
        "recovered_commit"
    )


def test_migration_fault_rolls_back_to_exact_source_cohort(tmp_path: Path) -> None:
    db = _prepared(tmp_path)
    source_hash = _logical_hash(db)
    with pytest.raises(MaintenanceError, match="injected_failure"):
        migrate_verified_copy(
            db, run_id="rehearsal-1", inject_failure_at="after_rename",
        )
    assert _logical_hash(db) == source_hash
    assert not status(db, "rehearsal-1")["migration_receipt_json"]

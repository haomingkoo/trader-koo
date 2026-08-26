"""Durable writer quiescence for offline paper-schema maintenance.

The recovery record is a sidecar database, so restoring the live database
cannot erase the interlock. Explicit offline helpers either migrate a verified
copy or preserve the failed live database before installing its named backup.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Literal

from starlette.responses import JSONResponse

from trader_koo.paper_trade.schema_v5_migration import (
    PaperSchemaV5MigrationError,
    _logical_database_hash,
    migrate_paper_schema_v4_to_v5,
)
from trader_koo.paper_trade.schema_v5_verifier import verify_paper_schema_v5
from trader_koo.scripts.backup_db import DEFAULT_BACKUP_DIR, backup_database, backup_path_by_name, list_backups

ACTIVE_STATES = frozenset({"draining", "backup_verified", "decision_required"})
_DDL = """
CREATE TABLE IF NOT EXISTS maintenance_runs (
 run_id TEXT PRIMARY KEY, state TEXT NOT NULL, requested_ts TEXT NOT NULL,
 requested_boot_id TEXT NOT NULL, reason TEXT NOT NULL, drain_timeout_sec INTEGER NOT NULL,
 source_path TEXT, source_device INTEGER, source_inode INTEGER, logical_cohort_sha256 TEXT,
 backup_dir TEXT, backup_name TEXT, backup_sha256 TEXT, backup_size_bytes INTEGER,
 retained_backups_json TEXT NOT NULL DEFAULT '[]', decision TEXT, decision_reason TEXT,
 decided_ts TEXT, resolved_ts TEXT, migration_plan_json TEXT,
 migration_receipt_json TEXT,
 restore_plan_json TEXT, restore_receipt_json TEXT,
 error_code TEXT, evidence_json TEXT NOT NULL DEFAULT '{}'
)
"""


class MaintenanceError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ProcessLease:
    def __init__(self, handle: Any, *, exclusive: bool):
        self._handle, self.exclusive = handle, exclusive

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def close(self) -> None:
        if not self.closed:
            # Close, do not LOCK_UN: inherited writer children share this open
            # file description and must keep the lease after the parent exits.
            self._handle.close()

    def fileno(self) -> int:
        return self._handle.fileno()


class MaintenanceInterlockMiddleware:
    """Stop all new HTTP work after intent; recovery continues offline."""

    def __init__(self, app: Any, *, db_path: Path):
        self.app, self.db_path = app, db_path

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        app = scope.get("app")
        maintenance_boot = bool(
            app is not None and getattr(getattr(app, "state", None), "maintenance_mode", False)
        )
        request_scope = scope.get("type") in {"http", "websocket"}
        state_error: str | None = None
        try:
            blocked = request_scope and (maintenance_boot or writers_blocked(self.db_path))
        except MaintenanceError as exc:
            blocked = request_scope
            state_error = exc.code
        if blocked and scope.get("type") == "websocket":
            await send({"type": "websocket.close", "code": 1013, "reason": "maintenance"})
            return
        if blocked and scope.get("path") == "/api/health":
            try:
                current = status(self.db_path)
            except MaintenanceError as exc:
                current = {"state": "invalid", "error_code": exc.code}
            response = JSONResponse({
                "ok": True,
                "maintenance": True,
                "maintenance_state": current["state"],
                "restart_required": (
                    (not maintenance_boot and current["state"] in ACTIVE_STATES)
                    or (maintenance_boot and current["state"] == "resolved")
                ),
                "error_code": current.get("error_code") or state_error,
            })
            await response(scope, receive, send)
            return
        if blocked:
            response = JSONResponse(
                {"detail": state_error or "maintenance_writers_blocked_use_offline_recovery_tool"},
                status_code=503,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def state_path(db_path: Path) -> Path:
    configured = os.getenv("TRADER_KOO_MAINTENANCE_STATE_PATH", "").strip()
    return Path(configured) if configured else db_path.with_name(f".{db_path.name}.maintenance.sqlite3")


def _required_marker(db_path: Path) -> Path:
    path = state_path(db_path)
    return path.with_name(f"{path.name}.required")


def _connect(db_path: Path) -> sqlite3.Connection:
    path = state_path(db_path)
    if path.resolve(strict=False) == db_path.resolve(strict=False):
        raise MaintenanceError("maintenance_state_overlaps_live_database")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MaintenanceError("unsafe_maintenance_state")
    try:
        conn = sqlite3.connect(str(path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute(_DDL)
        columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(maintenance_runs)")
        }
        if "migration_plan_json" not in columns:
            conn.execute(
                "ALTER TABLE maintenance_runs ADD COLUMN migration_plan_json TEXT"
            )
        conn.commit()
        return conn
    except sqlite3.DatabaseError as exc:
        raise MaintenanceError("maintenance_state_invalid") from exc


def _fsync(db_path: Path, conn: sqlite3.Connection) -> None:
    path = state_path(db_path)
    conn.execute("PRAGMA wal_checkpoint(FULL)")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def acquire_lease(
    db_path: Path, *, exclusive: bool, timeout_sec: float = 0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ProcessLease:
    path = db_path.with_name(f".{db_path.name}.writer.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MaintenanceError("unsafe_writer_lease")
    handle = path.open("a+", encoding="utf-8")
    operation = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
    deadline = monotonic() + timeout_sec
    while True:
        try:
            fcntl.flock(handle.fileno(), operation)
            handle.seek(0); handle.truncate()
            handle.write(f"pid={os.getpid()} exclusive={int(exclusive)} ts={_now()}\n"); handle.flush()
            return ProcessLease(handle, exclusive=exclusive)
        except BlockingIOError:
            if monotonic() >= deadline:
                handle.close()
                raise MaintenanceError("writer_lease_timeout")
            sleep(0.05)


def inherited_writer_lease_fds() -> tuple[int, ...]:
    """Propagate the app lease into supported writer subprocess trees."""
    raw = os.getenv("TRADER_KOO_WRITER_LEASE_FD", "").strip()
    if not raw.isdigit():
        return ()
    descriptor = int(raw)
    try:
        os.fstat(descriptor)
    except OSError:
        return ()
    return (descriptor,)


def status(db_path: Path, run_id: str | None = None) -> dict[str, Any]:
    if _required_marker(db_path).exists() and not state_path(db_path).is_file():
        raise MaintenanceError("maintenance_state_missing")
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM maintenance_runs WHERE run_id=?" if run_id else
            "SELECT * FROM maintenance_runs ORDER BY requested_ts DESC LIMIT 1",
            (run_id,) if run_id else (),
        ).fetchone()
    if row is None:
        return {"state": "idle", "writers_blocked": False}
    result = dict(row)
    result["writers_blocked"] = result["state"] in ACTIVE_STATES
    result["evidence"] = json.loads(result.pop("evidence_json") or "{}")
    result["retained_backups"] = json.loads(result.pop("retained_backups_json") or "[]")
    return result


def writers_blocked(db_path: Path) -> bool:
    return bool(status(db_path)["writers_blocked"])


def request_maintenance(db_path: Path, *, run_id: str, boot_id: str, reason: str, timeout_sec: int) -> dict[str, Any]:
    """Persist and fsync intent before any drain/restart begins."""
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT 1 FROM maintenance_runs WHERE run_id=?", (run_id,)).fetchone()
        if existing:
            row = conn.execute("SELECT * FROM maintenance_runs WHERE run_id=?", (run_id,)).fetchone()
            if (row["requested_boot_id"], row["reason"], row["drain_timeout_sec"]) != (
                boot_id, reason.strip(), timeout_sec,
            ):
                conn.rollback(); raise MaintenanceError("maintenance_idempotency_conflict")
        else:
            if conn.execute("SELECT 1 FROM maintenance_runs WHERE state IN ('draining','backup_verified','decision_required')").fetchone():
                conn.rollback(); raise MaintenanceError("maintenance_already_active")
            conn.execute(
                "INSERT INTO maintenance_runs(run_id,state,requested_ts,requested_boot_id,reason,drain_timeout_sec) "
                "VALUES(?,'draining',?,?,?,?)", (run_id, _now(), boot_id, reason.strip(), timeout_sec),
            )
        conn.commit(); _fsync(db_path, conn)
        marker = _required_marker(db_path)
        marker.touch(exist_ok=True)
        with marker.open("rb") as handle:
            os.fsync(handle.fileno())
    return status(db_path, run_id)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _campaign_state_evidence(conn: sqlite3.Connection) -> dict[str, Any]:
    v1_count, v1_unreconciled = conn.execute(
        "SELECT COUNT(*),COALESCE(SUM(accounting_status='legacy_unreconciled'),0) "
        "FROM paper_trades WHERE campaign_id='paper-v1'"
    ).fetchone()
    states = {
        str(campaign_id): str(state)
        for campaign_id, state in conn.execute(
            "SELECT campaign_id,status FROM paper_campaigns "
            "WHERE campaign_id IN ('paper-v1','paper-v2') ORDER BY campaign_id"
        )
    }
    return {
        "v1_trade_count": int(v1_count),
        "v1_legacy_unreconciled_count": int(v1_unreconciled),
        "v1_snapshot_count": int(conn.execute(
            "SELECT COUNT(*) FROM paper_portfolio_snapshots "
            "WHERE campaign_id='paper-v1'"
        ).fetchone()[0]),
        "v2_trade_count": int(conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE campaign_id='paper-v2'"
        ).fetchone()[0]),
        "campaign_states": states,
    }


def _hashed_record(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _canonical_hash(payload)
    return result


def _require_hashed_record(
    value: Any, *, field: str, error_code: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaintenanceError(error_code)
    expected = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    if not isinstance(expected, str) or expected != _canonical_hash(payload):
        raise MaintenanceError(error_code)
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _campaign_evidence_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "v1_trade_count", "v1_legacy_unreconciled_count",
        "v1_snapshot_count", "v2_trade_count", "campaign_states",
    }:
        return False
    if any(
        not isinstance(value.get(field), int) or value[field] < 0
        for field in (
            "v1_trade_count", "v1_legacy_unreconciled_count",
            "v1_snapshot_count", "v2_trade_count",
        )
    ):
        return False
    return isinstance(value.get("campaign_states"), dict) and all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value["campaign_states"].items()
    )


def _migration_plan(current: dict[str, Any], *, required: bool) -> dict[str, Any] | None:
    raw = current.get("migration_plan_json")
    if not raw:
        if required:
            raise MaintenanceError("migration_plan_required")
        return None
    try:
        plan = _require_hashed_record(
            json.loads(str(raw)), field="plan_sha256",
            error_code="migration_plan_invalid",
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MaintenanceError("migration_plan_invalid") from exc
    if set(plan) != {
        "run_id", "backup_name", "backup_sha256", "source_path",
        "source_device", "source_inode", "source_cohort_sha256",
        "target_applied_ts", "expected_target_logical_sha256", "contract_id",
        "schema_fingerprint", "campaign_evidence", "created_ts",
        "plan_sha256",
    }:
        raise MaintenanceError("migration_plan_invalid")
    expected = {
        "run_id": current["run_id"],
        "backup_name": current["backup_name"],
        "backup_sha256": current["backup_sha256"],
        "source_path": current["source_path"],
        "source_device": current["source_device"],
        "source_inode": current["source_inode"],
        "source_cohort_sha256": current["logical_cohort_sha256"],
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise MaintenanceError("migration_plan_binding_mismatch")
    for field in ("target_applied_ts", "contract_id", "created_ts"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            raise MaintenanceError("migration_plan_invalid")
    if any(not _is_sha256(plan.get(field)) for field in (
        "backup_sha256", "source_cohort_sha256", "expected_target_logical_sha256",
        "schema_fingerprint", "plan_sha256",
    )) or not _campaign_evidence_valid(plan.get("campaign_evidence")):
        raise MaintenanceError("migration_plan_invalid")
    return plan


def _migration_receipt(
    current: dict[str, Any], plan: dict[str, Any], *, required: bool,
) -> dict[str, Any] | None:
    raw = current.get("migration_receipt_json")
    if not raw:
        if required:
            raise MaintenanceError("migration_receipt_required")
        return None
    try:
        receipt = _require_hashed_record(
            json.loads(str(raw)), field="receipt_sha256",
            error_code="migration_receipt_invalid",
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MaintenanceError("migration_receipt_invalid") from exc
    if set(receipt) != {
        "run_id", "plan_sha256", "target_logical_sha256", "contract_id",
        "schema_fingerprint", "campaign_evidence", "target_device",
        "target_inode", "migration_status", "verified_ts", "receipt_sha256",
    }:
        raise MaintenanceError("migration_receipt_invalid")
    expected = {
        "run_id": current["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "target_logical_sha256": plan["expected_target_logical_sha256"],
        "contract_id": plan["contract_id"],
        "schema_fingerprint": plan["schema_fingerprint"],
        "campaign_evidence": plan["campaign_evidence"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise MaintenanceError("migration_receipt_mismatch")
    if not isinstance(receipt.get("target_device"), int) or not isinstance(
        receipt.get("target_inode"), int,
    ):
        raise MaintenanceError("migration_receipt_mismatch")
    if receipt.get("migration_status") not in {"migrated", "recovered_commit"}:
        raise MaintenanceError("migration_receipt_invalid")
    if not isinstance(receipt.get("verified_ts"), str) or not receipt["verified_ts"]:
        raise MaintenanceError("migration_receipt_invalid")
    if any(not _is_sha256(receipt.get(field)) for field in (
        "receipt_sha256", "plan_sha256", "target_logical_sha256",
        "schema_fingerprint",
    )) or not _campaign_evidence_valid(receipt.get("campaign_evidence")):
        raise MaintenanceError("migration_receipt_invalid")
    return receipt


def _fsync_file_and_parent(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _begin_exclusive(conn: sqlite3.Connection, code: str) -> None:
    try:
        conn.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as exc:
        raise MaintenanceError(code) from exc


def _backup_logical_hash(path: Path) -> str:
    with gzip.open(path, "rb") as source, tempfile.NamedTemporaryFile(suffix=".db") as target:
        shutil.copyfileobj(source, target); target.flush()
        with sqlite3.connect(target.name) as conn:
            return _logical_database_hash(conn)


def quiesce_backup(db_path: Path, *, run_id: str, boot_id: str, backup_dir: Path = DEFAULT_BACKUP_DIR) -> dict[str, Any]:
    """Offline helper: exclusive process lease, bounded DB probe, fresh backup."""
    current = status(db_path, run_id)
    if current["state"] == "backup_verified":
        return current
    if current["state"] != "draining" or current["requested_boot_id"] == boot_id:
        raise MaintenanceError("restart_required_before_quiescence")
    try:
        lease = acquire_lease(
            db_path, exclusive=True, timeout_sec=int(current["drain_timeout_sec"]),
        )
    except MaintenanceError as exc:
        with _connect(db_path) as conn:
            conn.execute(
                "UPDATE maintenance_runs SET state='draining',error_code=? WHERE run_id=?",
                (exc.code, run_id),
            )
            conn.commit()
            _fsync(db_path, conn)
        raise
    try:
        if db_path.is_symlink() or not db_path.is_file():
            raise MaintenanceError("unsafe_database_source")
        source = db_path.resolve(strict=True); stat = source.stat()
        with sqlite3.connect(str(source), timeout=0) as probe:
            _begin_exclusive(probe, "active_sqlite_transaction")
            queued = probe.execute(
                "SELECT COUNT(*) FROM pipeline_runs WHERE status='queued'"
            ).fetchone()[0] if probe.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
            ).fetchone() else 0
            running = probe.execute(
                "SELECT COUNT(*) FROM pipeline_runs WHERE status='running'"
            ).fetchone()[0] if probe.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
            ).fetchone() else 0
            if running:
                probe.execute(
                    "UPDATE pipeline_runs SET status='interrupted',finished_ts=?,"
                    "error_message='maintenance_quiescence_after_exclusive_lease' "
                    "WHERE status='running'", (_now(),),
                )
                probe.commit()
            else:
                probe.rollback()
        pre_backup = source.stat()
        if (pre_backup.st_dev, pre_backup.st_ino) != (stat.st_dev, stat.st_ino):
            raise MaintenanceError("database_source_replaced")
        backup = backup_database(source, backup_dir)
        post_backup = source.stat()
        if (post_backup.st_dev, post_backup.st_ino) != (stat.st_dev, stat.st_ino):
            raise MaintenanceError("database_source_replaced")
        named = backup_path_by_name(str(backup["backup_name"]), backup_dir)
        if named is None or _hash(named) != backup["sha256"]:
            raise MaintenanceError("backup_hash_verification_failed")
        cohort = _backup_logical_hash(named)
        _fsync_file_and_parent(named)
        retained = list_backups(backup_dir)
        # Acquire real SQLite authority after the online snapshot.  Any unknown
        # writer that slipped between backup and this transaction changes the
        # cohort and fails closed; authority stays held while evidence is fsynced.
        with sqlite3.connect(str(source), timeout=0) as authority:
            _begin_exclusive(authority, "sqlite_authority_unavailable")
            authority_stat = source.stat()
            if (authority_stat.st_dev, authority_stat.st_ino) != (stat.st_dev, stat.st_ino):
                authority.rollback(); raise MaintenanceError("database_source_replaced")
            if _logical_database_hash(authority) != cohort:
                authority.rollback()
                raise MaintenanceError("backup_cohort_mismatch")
            with _connect(db_path) as conn:
                conn.execute(
                    "UPDATE maintenance_runs SET state='backup_verified',source_path=?,source_device=?,source_inode=?,"
                    "logical_cohort_sha256=?,backup_dir=?,backup_name=?,backup_sha256=?,backup_size_bytes=?,retained_backups_json=?,"
                    "evidence_json=? WHERE run_id=? AND state='draining'",
                    (str(source), stat.st_dev, stat.st_ino, cohort, str(backup_dir.resolve()), backup["backup_name"], backup["sha256"],
                     backup["dest_size_bytes"], json.dumps(retained, sort_keys=True),
                     json.dumps({"exclusive_process_lease": True, "exclusive_sqlite_transaction": True,
                                 "queued_pipeline_runs_preserved": queued,
                                 "running_pipeline_runs_interrupted": running,
                                 "backup_verified": True}, sort_keys=True), run_id),
                )
                conn.commit(); _fsync(db_path, conn)
            authority.rollback()
    except MaintenanceError as exc:
        with _connect(db_path) as conn:
            conn.execute("UPDATE maintenance_runs SET state='draining',error_code=? WHERE run_id=?",
                         (exc.code, run_id)); conn.commit(); _fsync(db_path, conn)
        raise
    finally:
        lease.close()
    return status(db_path, run_id)


def migrate_verified_copy(
    db_path: Path,
    *,
    run_id: str,
    inject_failure_at: str | None = None,
    fault: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Migrate one quiesced copy and fsync a receipt for completion review."""
    current = status(db_path, run_id)
    if current["state"] != "decision_required" or current.get("decision") != "complete":
        raise MaintenanceError("complete_decision_required")
    backup_dir = Path(str(current.get("backup_dir") or ""))
    named = backup_path_by_name(str(current.get("backup_name") or ""), backup_dir)
    if named is None or _hash(named) != current.get("backup_sha256"):
        raise MaintenanceError("recovery_backup_invalid")
    if _backup_logical_hash(named) != current.get("logical_cohort_sha256"):
        raise MaintenanceError("backup_cohort_mismatch")

    lease = acquire_lease(
        db_path, exclusive=True, timeout_sec=int(current["drain_timeout_sec"]),
    )
    try:
        if db_path.is_symlink() or not db_path.is_file():
            raise MaintenanceError("unsafe_database_source")
        source = db_path.resolve(strict=True)
        source_stat = source.stat()
        if (source_stat.st_dev, source_stat.st_ino) != (
            current.get("source_device"), current.get("source_inode"),
        ):
            raise MaintenanceError("database_source_replaced")

        plan = _migration_plan(current, required=False)
        if plan is None:
            with tempfile.TemporaryDirectory(prefix="paper-v5-rehearsal-") as tmp_dir:
                rehearsal_path = Path(tmp_dir) / "copied-production.db"
                with gzip.open(named, "rb") as backup, rehearsal_path.open("wb") as target:
                    shutil.copyfileobj(backup, target)
                    target.flush()
                    os.fsync(target.fileno())
                with sqlite3.connect(str(rehearsal_path)) as rehearsal:
                    source_evidence = _campaign_state_evidence(rehearsal)
                    result = migrate_paper_schema_v4_to_v5(
                        rehearsal,
                        expected_source_logical_sha256=current["logical_cohort_sha256"],
                        target_applied_ts=current["requested_ts"],
                    )
                    if result.get("status") != "migrated":
                        raise MaintenanceError("rehearsal_migration_not_fresh")
                    verified = verify_paper_schema_v5(rehearsal)
                    target_evidence = _campaign_state_evidence(rehearsal)
                    if target_evidence != source_evidence:
                        raise MaintenanceError("rehearsal_preservation_mismatch")
                    expected_target = _logical_database_hash(rehearsal)
            plan = _hashed_record({
                "run_id": run_id,
                "backup_name": current["backup_name"],
                "backup_sha256": current["backup_sha256"],
                "source_path": str(source),
                "source_device": source_stat.st_dev,
                "source_inode": source_stat.st_ino,
                "source_cohort_sha256": current["logical_cohort_sha256"],
                "target_applied_ts": current["requested_ts"],
                "expected_target_logical_sha256": expected_target,
                "contract_id": verified["contract_id"],
                "schema_fingerprint": verified["schema_fingerprint"],
                "campaign_evidence": source_evidence,
                "created_ts": _now(),
            }, "plan_sha256")
            with _connect(db_path) as state:
                changed = state.execute(
                    "UPDATE maintenance_runs SET migration_plan_json=?,error_code=NULL "
                    "WHERE run_id=? AND state='decision_required' AND decision='complete' "
                    "AND migration_plan_json IS NULL",
                    (json.dumps(plan, sort_keys=True), run_id),
                ).rowcount
                if changed != 1:
                    state.rollback()
                    raise MaintenanceError("migration_plan_state_changed")
                state.commit()
                _fsync(db_path, state)
            if fault:
                fault("after_plan")

        current = status(db_path, run_id)
        plan = _migration_plan(current, required=True)
        assert plan is not None
        existing_receipt = _migration_receipt(current, plan, required=False)

        with sqlite3.connect(str(source), timeout=0) as live:
            opened_stat = source.stat()
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                current["source_device"], current["source_inode"],
            ):
                raise MaintenanceError("database_source_replaced")
            live_hash = _logical_database_hash(live)
            migrated_fresh = live_hash == current["logical_cohort_sha256"]
            recovered_commit = live_hash == plan["expected_target_logical_sha256"]
            if existing_receipt is not None:
                if not recovered_commit:
                    raise MaintenanceError("migration_receipt_mismatch")
                verified = verify_paper_schema_v5(live)
                if (
                    verified["contract_id"] != existing_receipt["contract_id"]
                    or verified["schema_fingerprint"]
                    != existing_receipt["schema_fingerprint"]
                ):
                    raise MaintenanceError("migration_receipt_mismatch")
                final_stat = source.stat()
                if (final_stat.st_dev, final_stat.st_ino) != (
                    existing_receipt["target_device"],
                    existing_receipt["target_inode"],
                ):
                    raise MaintenanceError("database_source_replaced")
                with _connect(db_path) as state:
                    state.execute(
                        "UPDATE maintenance_runs SET error_code=NULL WHERE run_id=? "
                        "AND migration_receipt_json=?",
                        (run_id, current["migration_receipt_json"]),
                    )
                    state.commit()
                    _fsync(db_path, state)
                return status(db_path, run_id)
            if migrated_fresh:
                result = migrate_paper_schema_v4_to_v5(
                    live,
                    inject_failure_at=inject_failure_at,
                    expected_source_logical_sha256=current["logical_cohort_sha256"],
                    target_applied_ts=plan["target_applied_ts"],
                )
                if result.get("status") != "migrated":
                    raise MaintenanceError("live_migration_not_fresh")
                if fault:
                    fault("after_migration_commit")
            elif not recovered_commit:
                raise MaintenanceError("live_cohort_not_planned")

            checkpoint = live.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is None or int(checkpoint[0]) != 0:
                raise MaintenanceError("migration_wal_reader_active")
            migrated_stat = source.stat()
            if (migrated_stat.st_dev, migrated_stat.st_ino) != (
                current["source_device"], current["source_inode"],
            ):
                raise MaintenanceError("database_source_replaced")
            verified = verify_paper_schema_v5(live)
            target_hash = _logical_database_hash(live)
            if target_hash != plan["expected_target_logical_sha256"]:
                raise MaintenanceError("migration_target_mismatch")
            campaign_evidence = _campaign_state_evidence(live)
            if campaign_evidence != plan["campaign_evidence"]:
                raise MaintenanceError("migration_preservation_mismatch")
            receipt = _hashed_record({
                "run_id": run_id,
                "plan_sha256": plan["plan_sha256"],
                "target_logical_sha256": target_hash,
                "contract_id": verified["contract_id"],
                "schema_fingerprint": verified["schema_fingerprint"],
                "campaign_evidence": campaign_evidence,
                "target_device": migrated_stat.st_dev,
                "target_inode": migrated_stat.st_ino,
                "migration_status": "recovered_commit" if recovered_commit else "migrated",
                "verified_ts": _now(),
            }, "receipt_sha256")
            _begin_exclusive(live, "migration_receipt_authority_unavailable")
            if _logical_database_hash(live) != target_hash:
                raise MaintenanceError("verified_database_changed")
            receipt_stat = source.stat()
            if (receipt_stat.st_dev, receipt_stat.st_ino) != (
                migrated_stat.st_dev, migrated_stat.st_ino,
            ):
                raise MaintenanceError("database_source_replaced")
            if fault:
                fault("before_receipt")
            receipt_stat = source.stat()
            if (receipt_stat.st_dev, receipt_stat.st_ino) != (
                migrated_stat.st_dev, migrated_stat.st_ino,
            ):
                raise MaintenanceError("database_source_replaced")
            with _connect(db_path) as state:
                changed = state.execute(
                    "UPDATE maintenance_runs SET migration_receipt_json=?,error_code=NULL "
                    "WHERE run_id=? AND state='decision_required' AND decision='complete' "
                    "AND migration_plan_json=? AND migration_receipt_json IS NULL",
                    (json.dumps(receipt, sort_keys=True), run_id,
                     json.dumps(plan, sort_keys=True)),
                ).rowcount
                if changed != 1:
                    state.rollback()
                    raise MaintenanceError("migration_receipt_state_changed")
                state.commit()
                _fsync(db_path, state)
            live.rollback()
        if fault:
            fault("after_receipt")
    except PaperSchemaV5MigrationError as exc:
        code = str(exc.diagnostics[0].get("code", "migration_failed"))
        wrapped = MaintenanceError(code)
        with _connect(db_path) as state:
            state.execute(
                "UPDATE maintenance_runs SET error_code=? WHERE run_id=?",
                (wrapped.code, run_id),
            )
            state.commit()
            _fsync(db_path, state)
        raise wrapped from exc
    except MaintenanceError as exc:
        with _connect(db_path) as state:
            state.execute(
                "UPDATE maintenance_runs SET error_code=? WHERE run_id=?",
                (exc.code, run_id),
            )
            state.commit()
            _fsync(db_path, state)
        raise
    finally:
        lease.close()
    return status(db_path, run_id)


def decide(db_path: Path, *, run_id: str, decision: Literal["restore", "complete"], reason: str) -> dict[str, Any]:
    """Record the choice; this alone never clears the writer interlock."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM maintenance_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None: raise MaintenanceError("maintenance_run_not_found")
        if row["state"] not in {"backup_verified", "decision_required"} or not (
            row["backup_name"] and row["backup_sha256"] and row["logical_cohort_sha256"]
        ):
            raise MaintenanceError("verified_backup_required")
        if row["decision"] and (row["decision"] != decision or row["decision_reason"] != reason.strip()):
            raise MaintenanceError("recovery_decision_conflict")
        conn.execute("UPDATE maintenance_runs SET state='decision_required',decision=?,decision_reason=?,decided_ts=? WHERE run_id=?",
                     (decision, reason.strip(), _now(), run_id))
        conn.commit(); _fsync(db_path, conn)
    return status(db_path, run_id)


def restore_backup(
    db_path: Path, *, run_id: str,
    fault: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Install the chosen verified backup and preserve the failed live file."""
    current = status(db_path, run_id)
    if current.get("decision") != "restore" or current["state"] != "decision_required":
        raise MaintenanceError("restore_decision_required")
    backup_dir = Path(str(current.get("backup_dir") or ""))
    named = backup_path_by_name(str(current.get("backup_name") or ""), backup_dir)
    if named is None or _hash(named) != current.get("backup_sha256"):
        raise MaintenanceError("recovery_backup_invalid")
    lease = acquire_lease(db_path, exclusive=True, timeout_sec=int(current["drain_timeout_sec"]))
    failed_path = db_path.with_name(f"{db_path.name}.pre_restore_{run_id}")
    temp_path: Path | None = None
    try:
        if db_path.is_symlink() or (not db_path.is_file() and not failed_path.is_file()):
            raise MaintenanceError("unsafe_database_source")
        if db_path.is_file():
            with sqlite3.connect(str(db_path), timeout=0) as live:
                checkpoint = live.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is None or int(checkpoint[0]) != 0:
                    raise MaintenanceError("active_sqlite_reader")
                _begin_exclusive(live, "restore_sqlite_authority_unavailable")
                live.rollback()
        plan = json.loads(current.get("restore_plan_json") or "{}")
        if not plan:
            if failed_path.exists():
                raise MaintenanceError("unplanned_failed_live_artifact")
            if not db_path.is_file():
                raise MaintenanceError("live_database_missing")
            planned_sidecars = {}
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{db_path}{suffix}")
                if sidecar.is_file() and not sidecar.is_symlink():
                    planned_sidecars[suffix] = _hash(sidecar)
            live_stat = db_path.stat()
            plan = {"live_inode": live_stat.st_ino, "live_sha256": _hash(db_path),
                    "failed_live_path": str(failed_path), "sidecars": planned_sidecars}
            with _connect(db_path) as state:
                state.execute("UPDATE maintenance_runs SET restore_plan_json=? WHERE run_id=?",
                              (json.dumps(plan, sort_keys=True), run_id))
                state.commit(); _fsync(db_path, state)
            if fault: fault("after_plan")
        descriptor, raw_temp = tempfile.mkstemp(prefix=f".{db_path.name}.restore-", dir=db_path.parent)
        os.close(descriptor); temp_path = Path(raw_temp)
        with gzip.open(named, "rb") as source, temp_path.open("wb") as target:
            shutil.copyfileobj(source, target); target.flush(); os.fsync(target.fileno())
        with sqlite3.connect(str(temp_path)) as restored:
            if restored.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise MaintenanceError("restore_integrity_failed")
            if _logical_database_hash(restored) != current["logical_cohort_sha256"]:
                raise MaintenanceError("restore_cohort_mismatch")
        restored_file_sha256 = _hash(temp_path)
        if failed_path.exists():
            if not failed_path.is_file() or failed_path.is_symlink():
                raise MaintenanceError("failed_live_preservation_invalid")
            if _hash(failed_path) != plan.get("live_sha256"):
                raise MaintenanceError("failed_live_plan_mismatch")
            if db_path.is_file():
                with sqlite3.connect(str(db_path)) as installed:
                    if _logical_database_hash(installed) != current["logical_cohort_sha256"]:
                        raise MaintenanceError("ambiguous_restore_state")
                temp_path.unlink(); temp_path = None
        else:
            if _hash(db_path) != plan.get("live_sha256"):
                raise MaintenanceError("live_database_plan_mismatch")
            os.replace(db_path, failed_path)
            if fault:
                fault("after_preserve")
                fault("after_preserve_main")
        # Reconcile sidecars on every retry; a crash can occur between renames.
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{db_path}{suffix}")
            preserved = Path(f"{failed_path}{suffix}")
            expected = (plan.get("sidecars") or {}).get(suffix)
            if preserved.exists() and sidecar.exists():
                raise MaintenanceError("restore_sidecar_collision")
            if sidecar.exists():
                if sidecar.is_symlink() or (expected and _hash(sidecar) != expected):
                    raise MaintenanceError("restore_sidecar_plan_mismatch")
                os.replace(sidecar, preserved)
                if fault: fault(f"after_preserve_{suffix[1:]}")
            elif preserved.exists() and expected and _hash(preserved) != expected:
                raise MaintenanceError("restore_sidecar_plan_mismatch")
        if not db_path.exists():
            os.replace(temp_path, db_path); temp_path = None
            if fault: fault("after_install")
        directory = os.open(str(db_path.parent), os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
        if fault: fault("after_directory_fsync")
        new_stat = db_path.stat()
        if _hash(db_path) != restored_file_sha256:
            raise MaintenanceError("restored_bytes_mismatch")
        receipt = {"failed_live_path": str(failed_path), "failed_live_inode": failed_path.stat().st_ino,
                   "failed_live_sha256": _hash(failed_path),
                   "restored_live_inode": new_stat.st_ino, "backup_name": named.name,
                   "restored_live_sha256": restored_file_sha256,
                   "backup_sha256": current["backup_sha256"],
                   "preserved_sidecars": [str(Path(f"{failed_path}{suffix}")) for suffix in ("-wal", "-shm")
                                           if Path(f"{failed_path}{suffix}").is_file()]}
        with _connect(db_path) as state:
            state.execute("UPDATE maintenance_runs SET restore_receipt_json=? WHERE run_id=?",
                          (json.dumps(receipt, sort_keys=True), run_id))
            state.commit(); _fsync(db_path, state)
        if fault: fault("after_receipt")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        lease.close()
    return status(db_path, run_id)


def verify_resolution(db_path: Path, *, run_id: str) -> dict[str, Any]:
    """Offline helper: clear only after exact-v5 completion or verified restore."""
    current = status(db_path, run_id)
    if not current.get("backup_name") or not current.get("logical_cohort_sha256"):
        raise MaintenanceError("verified_backup_required")
    lease = acquire_lease(db_path, exclusive=True, timeout_sec=int(current["drain_timeout_sec"]))
    try:
        source = db_path.resolve(strict=True)
        backup_dir = Path(str(current.get("backup_dir") or ""))
        named = backup_path_by_name(str(current.get("backup_name") or ""), backup_dir)
        if named is None or _hash(named) != current.get("backup_sha256"):
            raise MaintenanceError("recovery_backup_invalid")
        if _backup_logical_hash(named) != current.get("logical_cohort_sha256"):
            raise MaintenanceError("backup_cohort_mismatch")
        with sqlite3.connect(str(source), timeout=0) as conn:
            if current.get("decision") == "complete":
                if not current.get("migration_receipt_json"):
                    raise MaintenanceError("migration_receipt_required")
                plan = _migration_plan(current, required=True)
                assert plan is not None
                receipt = _migration_receipt(current, plan, required=True)
                assert receipt is not None
                verified = verify_paper_schema_v5(conn)
                verified_hash = _logical_database_hash(conn)
                if (
                    verified_hash != plan["expected_target_logical_sha256"]
                    or receipt["contract_id"] != verified["contract_id"]
                    or receipt["schema_fingerprint"] != verified["schema_fingerprint"]
                ):
                    raise MaintenanceError("migration_receipt_mismatch")
                campaign_evidence = _campaign_state_evidence(conn)
                if campaign_evidence != receipt["campaign_evidence"]:
                    raise MaintenanceError("migration_receipt_mismatch")
                target_stat = source.stat()
                if (target_stat.st_dev, target_stat.st_ino) != (
                    receipt["target_device"], receipt["target_inode"],
                ):
                    raise MaintenanceError("migration_receipt_mismatch")
                _begin_exclusive(conn, "resolution_sqlite_authority_unavailable")
                if _logical_database_hash(conn) != verified_hash:
                    raise MaintenanceError("verified_database_changed")
            elif current.get("decision") == "restore":
                receipt = json.loads(current.get("restore_receipt_json") or "{}")
                failed_live = Path(str(receipt.get("failed_live_path") or ""))
                if not receipt or not failed_live.is_file():
                    raise MaintenanceError("restore_receipt_required")
                if _hash(failed_live) != receipt.get("failed_live_sha256"):
                    raise MaintenanceError("failed_live_receipt_mismatch")
                _begin_exclusive(conn, "resolution_sqlite_authority_unavailable")
                if _logical_database_hash(conn) != current.get("logical_cohort_sha256"):
                    raise MaintenanceError("restored_cohort_mismatch")
                if _hash(source) != receipt.get("restored_live_sha256"):
                    raise MaintenanceError("restored_receipt_mismatch")
                if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise MaintenanceError("restored_integrity_failed")
            else:
                raise MaintenanceError("recovery_decision_required")
            with _connect(db_path) as state:
                changed = state.execute(
                    "UPDATE maintenance_runs SET state='resolved',resolved_ts=?,error_code=NULL "
                    "WHERE run_id=? AND state='decision_required' AND decision=?",
                    (_now(), run_id, current.get("decision")),
                ).rowcount
                if changed != 1:
                    state.rollback()
                    raise MaintenanceError("resolution_state_changed")
                state.commit(); _fsync(db_path, state)
            conn.rollback()
    finally:
        lease.close()
    return status(db_path, run_id)

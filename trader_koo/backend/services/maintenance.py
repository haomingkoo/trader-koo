"""Durable writer quiescence for offline paper-schema maintenance.

The recovery record is a sidecar database, so restoring the live database
cannot erase the interlock. This module never migrates; its explicit restore
helper preserves the failed live database before installing a verified backup.
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

from trader_koo.paper_trade.schema_v5_migration import _logical_database_hash
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
 decided_ts TEXT, resolved_ts TEXT, migration_receipt_json TEXT,
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
        with sqlite3.connect(str(source), timeout=0) as conn:
            if current.get("decision") == "complete":
                receipt = json.loads(current.get("migration_receipt_json") or "{}")
                if not receipt or receipt.get("source_cohort_sha256") != current.get("logical_cohort_sha256"):
                    raise MaintenanceError("migration_receipt_required")
                verified = verify_paper_schema_v5(conn)
                verified_hash = _logical_database_hash(conn)
                if (
                    receipt.get("target_logical_sha256") != verified_hash
                    or receipt.get("schema_fingerprint") != verified["schema_fingerprint"]
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
                state.execute("UPDATE maintenance_runs SET state='resolved',resolved_ts=?,error_code=NULL WHERE run_id=?",
                              (_now(), run_id)); state.commit(); _fsync(db_path, state)
            conn.rollback()
    finally:
        lease.close()
    return status(db_path, run_id)

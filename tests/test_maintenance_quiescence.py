from __future__ import annotations

import gzip
import os
import sqlite3
import subprocess
import sys
import asyncio
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.services.maintenance import (
    MaintenanceInterlockMiddleware,
    MaintenanceError,
    acquire_lease,
    decide,
    quiesce_backup,
    request_maintenance,
    restore_backup,
    state_path,
    status,
)


def _live_db(path: Path, *, pipeline_status: str | None = None) -> None:
    from trader_koo.backend.services.pipeline import ensure_pipeline_runs_schema

    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE payload(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO payload(value) VALUES ('original')")
        ensure_pipeline_runs_schema(conn)
        if pipeline_status:
            conn.execute(
                """INSERT INTO pipeline_runs
                   (run_id,started_ts,mode,source,status,stage,stage_started_ts)
                   VALUES ('run-1','2026-08-26T00:00:00Z','report','admin',?,'queued','2026-08-26T00:00:00Z')""",
                (pipeline_status,),
            )
        conn.commit()


def _request(db: Path, *, run_id: str = "maint-test") -> dict:
    return request_maintenance(
        db, run_id=run_id, boot_id="boot-before", reason="schema v5 rehearsal",
        timeout_sec=1,
    )


def test_sidecar_survives_live_database_replacement(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db); _request(db)
    replacement = tmp_path / "replacement.db"; _live_db(replacement)
    os.replace(replacement, db)
    assert state_path(db) != db
    assert status(db)["state"] == "draining"
    assert status(db)["writers_blocked"] is True


def test_request_is_idempotent_but_conflicting_payload_fails(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db)
    first = _request(db)
    assert _request(db) == first
    with pytest.raises(MaintenanceError, match="maintenance_idempotency_conflict"):
        request_maintenance(db, run_id="maint-test", boot_id="other",
                            reason="schema v5 rehearsal", timeout_sec=1)


def test_exclusive_lease_waits_for_inherited_child_fd(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db)
    shared = acquire_lease(db, exclusive=False)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        pass_fds=(shared.fileno(),),
    )
    shared.close()
    try:
        with pytest.raises(MaintenanceError, match="writer_lease_timeout"):
            acquire_lease(db, exclusive=True, timeout_sec=0.05)
    finally:
        child.wait(timeout=3)
    exclusive = acquire_lease(db, exclusive=True, timeout_sec=0.2)
    exclusive.close()


def test_process_lifetime_lease_covers_direct_writer_thread(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db)
    process_lease = acquire_lease(db, exclusive=False)
    release = threading.Event()

    def direct_writer() -> None:
        release.wait(timeout=1)
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO payload(value) VALUES ('thread write')")
            conn.commit()

    thread = threading.Thread(target=direct_writer)
    thread.start()
    # Lifespan shutdown deliberately retains the process lease.
    with pytest.raises(MaintenanceError, match="writer_lease_timeout"):
        acquire_lease(db, exclusive=True, timeout_sec=0.05)
    release.set(); thread.join(timeout=1)
    assert not thread.is_alive()
    with pytest.raises(MaintenanceError, match="writer_lease_timeout"):
        acquire_lease(db, exclusive=True, timeout_sec=0.05)
    process_lease.close()  # Simulated OS process exit.
    exclusive = acquire_lease(db, exclusive=True, timeout_sec=0.2)
    exclusive.close()


def test_quiesce_preserves_durably_accepted_queued_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db, pipeline_status="queued"); _request(db)
    result = quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    assert result["state"] == "backup_verified"
    assert result["evidence"]["queued_pipeline_runs_preserved"] == 1
    assert result["backup_dir"] == str(backups.resolve())
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT run_id,mode,source,status,error_message FROM pipeline_runs").fetchone() == (
            "run-1", "report", "admin", "queued", None,
        )


def test_resolved_restart_requeues_preserved_run_id_once(tmp_path: Path) -> None:
    from trader_koo.backend.services.maintenance import verify_resolution
    from trader_koo.backend.services.pipeline import queue_reserved_pipeline_run

    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db, pipeline_status="queued"); _request(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    decide(db, run_id="maint-test", decision="restore", reason="queue recovery")
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO payload(value) VALUES ('failed state')"); conn.commit()
    restore_backup(db, run_id="maint-test")
    verify_resolution(db, run_id="maint-test")

    class Scheduler:
        jobs: list[dict] = []

        def add_job(self, _function, **kwargs) -> None:
            self.jobs.append(kwargs)

    scheduler = Scheduler()
    queued = queue_reserved_pipeline_run(scheduler, db_path=db)
    assert queued == {
        "scheduled": True,
        "job_id": "resume_reserved_run-1",
        "run_id": "run-1",
        "mode": "report",
        "source": "admin",
    }
    assert len(scheduler.jobs) == 1
    assert scheduler.jobs[0]["kwargs"]["run_id"] == "run-1"


def test_quiesce_interrupts_restart_orphaned_running_pipeline(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db, pipeline_status="running"); _request(db)
    result = quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    assert result["evidence"]["running_pipeline_runs_interrupted"] == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status,error_message FROM pipeline_runs").fetchone() == (
            "interrupted", "maintenance_quiescence_after_exclusive_lease",
        )


def test_backup_is_fresh_hash_verified_and_bound_to_source(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    result = quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    backup = backups / result["backup_name"]
    assert backup.is_file()
    assert result["source_device"] == db.stat().st_dev
    assert result["source_inode"] == db.stat().st_ino
    with gzip.open(backup, "rb") as stream:
        assert stream.read(16).startswith(b"SQLite format 3")
    assert result["retained_backups"][0]["name"] == backup.name


def test_backup_fsync_precedes_verified_sidecar_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trader_koo.backend.services.maintenance as maintenance

    db = tmp_path / "live.db"; _live_db(db); _request(db)
    ordering: list[str] = []
    real_backup_fsync = maintenance._fsync_file_and_parent
    real_state_fsync = maintenance._fsync

    def backup_fsync(path: Path) -> None:
        ordering.append("backup")
        real_backup_fsync(path)

    def state_fsync(path: Path, conn: sqlite3.Connection) -> None:
        if status(path, "maint-test")["state"] == "backup_verified":
            ordering.append("verified_state")
        real_state_fsync(path, conn)

    monkeypatch.setattr(maintenance, "_fsync_file_and_parent", backup_fsync)
    monkeypatch.setattr(maintenance, "_fsync", state_fsync)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups")
    assert ordering.index("backup") < ordering.index("verified_state")


def test_decision_does_not_clear_interlock(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    result = decide(db, run_id="maint-test", decision="restore", reason="recovery drill")
    assert result["state"] == "decision_required"
    assert result["writers_blocked"] is True


def test_restore_requires_real_install_and_preserves_failed_live(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    decide(db, run_id="maint-test", decision="restore", reason="recovery drill")
    from trader_koo.backend.services.maintenance import verify_resolution
    with pytest.raises(MaintenanceError, match="restore_receipt_required"):
        verify_resolution(db, run_id="maint-test")
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO payload(value) VALUES ('failed migration state')")
        conn.commit()
    restored = restore_backup(db, run_id="maint-test")
    receipt = __import__("json").loads(restored["restore_receipt_json"])
    assert Path(receipt["failed_live_path"]).is_file()
    assert receipt["failed_live_inode"] != receipt["restored_live_inode"]
    resolved = verify_resolution(db, run_id="maint-test")
    assert resolved["state"] == "resolved"


def test_restore_rejects_unplanned_failed_live_artifact(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    decide(db, run_id="maint-test", decision="restore", reason="artifact collision")
    failed = db.with_name(f"{db.name}.pre_restore_maint-test")
    failed.write_bytes(b"unrelated")
    with pytest.raises(MaintenanceError, match="unplanned_failed_live_artifact"):
        restore_backup(db, run_id="maint-test")
    assert db.is_file()


def test_unknown_write_between_backup_and_authority_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trader_koo.backend.services.maintenance as maintenance

    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    real_backup = maintenance.backup_database

    def backup_then_write(*args, **kwargs):
        result = real_backup(*args, **kwargs)
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO payload(value) VALUES ('raced')")
            conn.commit()
        return result

    monkeypatch.setattr(maintenance, "backup_database", backup_then_write)
    with pytest.raises(MaintenanceError, match="backup_cohort_mismatch"):
        quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    assert status(db)["state"] == "draining"
    assert status(db)["writers_blocked"] is True


def test_writer_between_backup_and_authority_has_stable_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trader_koo.backend.services.maintenance as maintenance

    db = tmp_path / "live.db"; _live_db(db); _request(db)
    real_backup = maintenance.backup_database
    blocker: list[sqlite3.Connection] = []

    def backup_then_hold_write(*args, **kwargs):
        result = real_backup(*args, **kwargs)
        connection = sqlite3.connect(db)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT INTO payload(value) VALUES ('uncommitted race')")
        blocker.append(connection)
        return result

    monkeypatch.setattr(maintenance, "backup_database", backup_then_hold_write)
    try:
        with pytest.raises(MaintenanceError, match="sqlite_authority_unavailable"):
            quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups")
    finally:
        blocker[0].rollback(); blocker[0].close()
    current = status(db)
    assert current["state"] == "draining"
    assert current["error_code"] == "sqlite_authority_unavailable"


def test_http_interlock_stops_new_handlers_after_intent(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db)
    calls: list[str] = []
    app = FastAPI()
    app.add_middleware(MaintenanceInterlockMiddleware, db_path=db)

    @app.post("/write")
    def write() -> dict[str, bool]:
        calls.append("write")
        return {"ok": True}

    client = TestClient(app)
    assert client.post("/write").status_code == 200
    _request(db)
    blocked = client.post("/write")
    assert blocked.status_code == 503
    assert calls == ["write"]


def test_health_reports_phase_aware_restart_requirement(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db); _request(db)
    original = FastAPI()
    original.state.maintenance_mode = False
    original.add_middleware(MaintenanceInterlockMiddleware, db_path=db)
    assert TestClient(original).get("/api/health").json()["restart_required"] is True

    maintenance_boot = FastAPI()
    maintenance_boot.state.maintenance_mode = True
    maintenance_boot.add_middleware(MaintenanceInterlockMiddleware, db_path=db)
    client = TestClient(maintenance_boot)
    assert client.get("/api/health").json()["restart_required"] is False

    with sqlite3.connect(state_path(db)) as conn:
        conn.execute("UPDATE maintenance_runs SET state='resolved' WHERE run_id='maint-test'")
        conn.commit()
    resolved = client.get("/api/health").json()
    assert resolved["maintenance_state"] == "resolved"
    assert resolved["restart_required"] is True


def test_interlock_rejects_websocket_before_handler(tmp_path: Path) -> None:
    from starlette.websockets import WebSocketDisconnect

    db = tmp_path / "live.db"; _live_db(db); _request(db)
    entered: list[bool] = []
    app = FastAPI()
    app.add_middleware(MaintenanceInterlockMiddleware, db_path=db)

    @app.websocket("/ws/test")
    async def websocket_handler(websocket) -> None:
        entered.append(True)
        await websocket.accept()

    with pytest.raises(WebSocketDisconnect) as exc:
        with TestClient(app).websocket_connect("/ws/test"):
            pass
    assert exc.value.code == 1013
    assert entered == []


def test_admin_request_uses_existing_auth_and_pauses_running_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trader_koo.backend.routers.admin import router as admin_router
    from trader_koo.backend.routers.admin import maintenance as route
    from trader_koo.middleware.auth import require_admin

    assert any(dependency.dependency is require_admin for dependency in admin_router.dependencies)
    db = tmp_path / "live.db"; _live_db(db)
    monkeypatch.setattr(route, "DB_PATH", db)

    class Scheduler:
        running = True
        paused = False

        def pause(self) -> None:
            self.paused = True

    scheduler = Scheduler()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        boot_id="boot-before", maintenance_mode=False, scheduler=scheduler,
    )))
    body = route.MaintenanceRequest(
        reason="schema v5 rehearsal", timeout_sec=1, idempotency_key="request-123",
    )
    result = route.request_database_maintenance(request, body)
    assert result["ok"] is True
    assert scheduler.paused is True
    assert status(db)["state"] == "draining"


def test_admin_request_returns_durable_run_when_scheduler_pause_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trader_koo.backend.routers.admin import maintenance as route

    db = tmp_path / "live.db"; _live_db(db)
    monkeypatch.setattr(route, "DB_PATH", db)

    class Scheduler:
        running = True

        def pause(self) -> None:
            raise RuntimeError("scheduler fault")

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        boot_id="boot-before", maintenance_mode=False, scheduler=Scheduler(),
    )))
    body = route.MaintenanceRequest(
        reason="schema v5 rehearsal", timeout_sec=1, idempotency_key="request-fault",
    )
    result = route.request_database_maintenance(request, body)
    assert result["ok"] is True
    assert result["scheduler_warning"] == "scheduler_pause_failed_restart_required"
    assert result["maintenance"]["run_id"].startswith("maint_")
    assert status(db)["writers_blocked"] is True


def test_lease_timeout_persists_fail_closed_error(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db); _request(db)
    shared = acquire_lease(db, exclusive=False)
    try:
        with pytest.raises(MaintenanceError, match="writer_lease_timeout"):
            quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups")
    finally:
        shared.close()
    current = status(db)
    assert current["state"] == "draining"
    assert current["error_code"] == "writer_lease_timeout"
    recovered = quiesce_backup(
        db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups",
    )
    assert recovered["state"] == "backup_verified"


def test_active_sqlite_transaction_stays_fail_closed(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db); _request(db)
    blocker = sqlite3.connect(db)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(MaintenanceError, match="active_sqlite_transaction"):
            quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups")
    finally:
        blocker.rollback(); blocker.close()
    assert status(db)["error_code"] == "active_sqlite_transaction"


def test_source_inode_swap_during_backup_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trader_koo.backend.services.maintenance as maintenance

    db = tmp_path / "live.db"; _live_db(db); _request(db)
    real_backup = maintenance.backup_database

    def backup_then_swap(*args, **kwargs):
        result = real_backup(*args, **kwargs)
        replacement = tmp_path / "replacement.db"; _live_db(replacement)
        os.replace(replacement, db)
        return result

    monkeypatch.setattr(maintenance, "backup_database", backup_then_swap)
    with pytest.raises(MaintenanceError, match="database_source_replaced"):
        quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups")
    assert status(db)["writers_blocked"] is True


def test_required_sidecar_missing_or_corrupt_fails_closed(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db); _request(db)
    sidecar = state_path(db)
    sidecar.unlink()
    with pytest.raises(MaintenanceError, match="maintenance_state_missing"):
        status(db)
    sidecar.write_bytes(b"not sqlite")
    with pytest.raises(MaintenanceError, match="maintenance_state_invalid"):
        status(db)

    app = FastAPI()
    app.state.maintenance_mode = True
    app.add_middleware(MaintenanceInterlockMiddleware, db_path=db)
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["error_code"] == "maintenance_state_invalid"
    assert client.get("/").status_code == 503


def test_decision_without_verified_backup_rejects(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db); _request(db)
    with pytest.raises(MaintenanceError, match="verified_backup_required"):
        decide(db, run_id="maint-test", decision="complete", reason="not backed up")


def test_complete_resolution_rejects_v4_and_stays_blocked(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; _live_db(db); _request(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups")
    decide(db, run_id="maint-test", decision="complete", reason="migration allegedly complete")
    with pytest.raises(MaintenanceError, match="migration_receipt_required"):
        from trader_koo.backend.services.maintenance import verify_resolution
        verify_resolution(db, run_id="maint-test")
    assert status(db)["writers_blocked"] is True


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_complete_resolution_revalidates_rollback_backup(tmp_path: Path, damage: str) -> None:
    from trader_koo.backend.services.maintenance import verify_resolution

    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    backed = quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    decide(db, run_id="maint-test", decision="complete", reason="migration complete")
    backup = backups / backed["backup_name"]
    if damage == "missing":
        backup.unlink()
    else:
        backup.write_bytes(b"corrupt")
    with pytest.raises(MaintenanceError, match="recovery_backup_invalid"):
        verify_resolution(db, run_id="maint-test")
    assert status(db)["writers_blocked"] is True


@pytest.mark.parametrize("stage", ["after_preserve", "after_install", "after_directory_fsync", "after_receipt"])
def test_restore_is_resumable_at_each_durable_boundary(tmp_path: Path, stage: str) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    decide(db, run_id="maint-test", decision="restore", reason="crash drill")
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO payload(value) VALUES ('failed state')")
        conn.commit()

    def crash(boundary: str) -> None:
        if boundary == stage:
            raise RuntimeError(f"injected:{stage}")

    with pytest.raises(RuntimeError, match=f"injected:{stage}"):
        restore_backup(db, run_id="maint-test", fault=crash)
    recovered = restore_backup(db, run_id="maint-test")
    receipt = __import__("json").loads(recovered["restore_receipt_json"])
    assert Path(receipt["failed_live_path"]).is_file()
    assert all(Path(path).is_file() for path in receipt["preserved_sidecars"])


def test_maintenance_boot_touches_no_live_database_or_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADER_KOO_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TRADER_KOO_LOGOS_DIR", str(tmp_path / "logos"))
    monkeypatch.setenv("TRADER_KOO_DB_PATH", str(tmp_path / "live.db"))
    import trader_koo.backend.main as main
    import trader_koo.backend.services.maintenance as maintenance

    db = tmp_path / "live.db"; _live_db(db); _request(db)
    monkeypatch.setattr(main, "DB_PATH", db)
    monkeypatch.setattr(
        main, "create_scheduler",
        lambda: (_ for _ in ()).throw(AssertionError("scheduler constructed")),
    )
    opened: list[str] = []
    real_connect = sqlite3.connect

    def tracked_connect(database, *args, **kwargs):
        opened.append(str(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(maintenance.sqlite3, "connect", tracked_connect)
    app = FastAPI()

    async def exercise() -> None:
        async with main.lifespan(app):
            assert app.state.maintenance_mode is True

    asyncio.run(exercise())
    assert str(db) not in opened


def test_wal_reader_blocks_restore_before_any_rename(tmp_path: Path) -> None:
    db = tmp_path / "live.db"; backups = tmp_path / "backups"
    _live_db(db); _request(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=backups)
    decide(db, run_id="maint-test", decision="restore", reason="reader drill")
    with sqlite3.connect(db) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO payload(value) VALUES ('before reader')")
        writer.commit()
    reader = sqlite3.connect(db)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM payload").fetchall()
    with sqlite3.connect(db) as writer:
        writer.execute("INSERT INTO payload(value) VALUES ('held in wal')")
        writer.commit()
    failed = db.with_name(f"{db.name}.pre_restore_maint-test")
    try:
        with pytest.raises(MaintenanceError, match="active_sqlite_reader"):
            restore_backup(db, run_id="maint-test")
        assert db.is_file()
        assert not failed.exists()
    finally:
        reader.rollback(); reader.close()

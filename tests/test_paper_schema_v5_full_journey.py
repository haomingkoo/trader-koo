"""Production-faithful paper-v5 completion and rollback journeys on copies."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from trader_koo.backend.services.maintenance import MaintenanceError, acquire_lease, status
from trader_koo.db.price_contract import (
    record_price_series_revision,
    research_price_contract,
)
from trader_koo.paper_trade.schema_v5_migration import _logical_database_hash
from trader_koo.paper_trade.schema_v5_verifier import verify_paper_schema_v5


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "paper_schema_v4_legacy_production_like.sql"
WORKER = ROOT / "tests" / "fixtures" / "paper_v5_journey_worker.py"
API_KEY = "copied-database-rehearsal-key-0001"
# A test-process watchdog, not a maintenance lock/drain timeout. Large external
# copies can spend minutes compressing and verifying the named rollback backup.
REHEARSAL_COMMAND_TIMEOUT_SEC = 10 * 60


def _fresh_v4(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(FIXTURE.read_text())
        conn.execute(
            """CREATE TABLE price_daily (
                   ticker TEXT NOT NULL,date TEXT NOT NULL,open REAL,high REAL,low REAL,
                   close REAL,volume REAL,data_source TEXT,fetch_timestamp TEXT,
                   adjustment_basis TEXT,adjustment_version TEXT,basis_status TEXT,
                   unresolved_reason TEXT,UNIQUE(ticker,date)
               )"""
        )
        conn.executemany(
            """INSERT INTO price_daily VALUES (
                   ?,?,?,?,?,?,?,'fixture','2026-08-25T00:00:00Z',
                   'split_adjusted_price_only','fixture-actions-v1','verified',NULL
               )""",
            [
                ("SPY", "2026-08-24", 498, 501, 497, 500, 1_000_000),
                ("SPY", "2026-08-25", 500, 503, 499, 502, 1_100_000),
            ],
        )
        record_price_series_revision(
            conn,
            "SPY",
            evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
            fetch_timestamp="2026-08-25T00:00:00Z",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_source(source: Path, target: Path) -> dict:
    source_hash = _sha256(source)
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as incoming, target.open("xb") as outgoing:
            while block := incoming.read(1024 * 1024):
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    else:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as incoming:
            with sqlite3.connect(target) as outgoing:
                incoming.backup(outgoing)
        with target.open("rb") as copied:
            os.fsync(copied.fileno())
    if _sha256(source) != source_hash:
        raise AssertionError("external rehearsal source changed while copying")
    with sqlite3.connect(target) as conn:
        cohort = _logical_database_hash(conn)
    return {
        "source_sha256": source_hash,
        "source_format": "gzip" if source.suffix == ".gz" else "sqlite",
        "copied_v4_logical_cohort_sha256": cohort,
    }


def _campaign_evidence(path: Path) -> dict:
    with sqlite3.connect(path) as conn:
        v1 = conn.execute(
            "SELECT id,accounting_status FROM paper_trades "
            "WHERE campaign_id='paper-v1' ORDER BY id"
        ).fetchall()
        v2 = conn.execute(
            "SELECT status,policy_version,policy_hash,replay_live_parity "
            "FROM paper_campaigns WHERE campaign_id='paper-v2'"
        ).fetchone()
        v2_trade_ids = [
            int(row[0]) for row in conn.execute(
                "SELECT id FROM paper_trades WHERE campaign_id='paper-v2' ORDER BY id"
            )
        ]
        snapshots = conn.execute(
            "SELECT COUNT(*) FROM paper_portfolio_snapshots WHERE campaign_id='paper-v1'"
        ).fetchone()[0]
    return {
        "v1_trade_ids": [int(row[0]) for row in v1],
        "v1_accounting": {
            status: sum(1 for row in v1 if row[1] == status)
            for status in sorted({str(row[1]) for row in v1})
        },
        "v1_snapshot_count": int(snapshots),
        "v2_state": list(v2) if v2 else None,
        "v2_trade_ids": v2_trade_ids,
    }


def _spy_price_evidence(path: Path, *, exclude_date: str | None = None) -> dict:
    with sqlite3.connect(path) as conn:
        where = "ticker='SPY'" + (" AND date<>?" if exclude_date else "")
        params = (exclude_date,) if exclude_date else ()
        rows = conn.execute(
            "SELECT * FROM price_daily WHERE " + where + " ORDER BY date",
            params,
        ).fetchall()
        actions = (
            conn.execute(
                "SELECT * FROM price_corporate_actions WHERE ticker='SPY' ORDER BY action_date",
            ).fetchall()
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='price_corporate_actions'"
            ).fetchone()
            else []
        )
    return {
        "row_count": len(rows),
        "rows_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
        "action_count": len(actions),
        "actions_sha256": hashlib.sha256(
            json.dumps(actions, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
    }


def _source_database(tmp_path: Path, target: Path) -> tuple[Path, dict]:
    configured = os.getenv("TRADER_KOO_V5_REHEARSAL_SOURCE", "").strip()
    if configured:
        source = Path(configured).resolve(strict=True)
    else:
        source = tmp_path / "source-production-like-v4.db"
        _fresh_v4(source)
    return source, _copy_source(source, target)


def test_external_source_copy_supports_gzip_and_live_wal(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _fresh_v4(source)
    with sqlite3.connect(source) as conn:
        expected_cohort = _logical_database_hash(conn)

    compressed = tmp_path / "source.db.gz"
    with source.open("rb") as incoming, gzip.open(compressed, "xb") as outgoing:
        while block := incoming.read(1024 * 1024):
            outgoing.write(block)
    compressed_hash = _sha256(compressed)
    gzip_target = tmp_path / "gzip-copy.db"
    gzip_evidence = _copy_source(compressed, gzip_target)
    assert gzip_evidence == {
        "source_sha256": compressed_hash,
        "source_format": "gzip",
        "copied_v4_logical_cohort_sha256": expected_cohort,
    }
    assert _sha256(compressed) == compressed_hash

    wal_source = tmp_path / "wal-source.db"
    _fresh_v4(wal_source)
    with sqlite3.connect(wal_source) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE price_fixture (id INTEGER PRIMARY KEY, value TEXT)")
        writer.execute("INSERT INTO price_fixture(value) VALUES ('committed-in-wal')")
        writer.commit()
        wal_path = Path(f"{wal_source}-wal")
        assert wal_path.exists()
        source_hash = _sha256(wal_source)
        wal_hash = _sha256(wal_path)
        wal_cohort = _logical_database_hash(writer)

        wal_target = tmp_path / "wal-copy.db"
        wal_evidence = _copy_source(wal_source, wal_target)
        assert writer.execute("SELECT value FROM price_fixture").fetchone() == (
            "committed-in-wal",
        )
        assert _sha256(wal_source) == source_hash
        assert _sha256(wal_path) == wal_hash

    assert wal_evidence["copied_v4_logical_cohort_sha256"] == wal_cohort
    with sqlite3.connect(wal_target) as copied:
        assert copied.execute("SELECT value FROM price_fixture").fetchone() == (
            "committed-in-wal",
        )
        assert _logical_database_hash(copied) == wal_cohort


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    port: int, path: str, *, payload: dict | None = None, authenticated: bool = False,
) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["X-API-Key"] = API_KEY
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=(json.dumps(payload).encode() if payload is not None else None),
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can report EPERM for a just-emptied process group. Every member
        # of this dedicated session was launched by the current test user, so
        # a live member would remain signalable.
        return False
    return True


def _stop_process_group(
    process: subprocess.Popen,
    *,
    term_timeout_sec: float = 20,
    kill_timeout_sec: float = 10,
) -> dict[str, float | bool]:
    """Stop the local server's container-like process group, including children."""
    started = time.monotonic()
    process_group_id = process.pid
    escalated = False
    if _process_group_exists(process_group_id):
        os.killpg(process_group_id, signal.SIGTERM)
    term_deadline = time.monotonic() + term_timeout_sec
    while _process_group_exists(process_group_id) and time.monotonic() < term_deadline:
        process.poll()
        time.sleep(0.05)
    if _process_group_exists(process_group_id):
        escalated = True
        os.killpg(process_group_id, signal.SIGKILL)
        kill_deadline = time.monotonic() + kill_timeout_sec
        while _process_group_exists(process_group_id) and time.monotonic() < kill_deadline:
            process.poll()
            time.sleep(0.05)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("server process group leader did not exit") from exc
    if _process_group_exists(process_group_id):
        raise AssertionError("server process group still exists after bounded shutdown")
    return {
        "elapsed_sec": round(time.monotonic() - started, 3),
        "escalated": escalated,
    }


def test_server_group_shutdown_reaps_inherited_lease_child(tmp_path: Path) -> None:
    db_path = tmp_path / "paper.db"
    lease = acquire_lease(db_path, exclusive=False)
    ready = tmp_path / "child-ready"
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        "fd=int(sys.argv[1]); ready=sys.argv[2]; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[3]],pass_fds=(fd,)); "
        "open(ready,'w').write('ready'); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader_code, str(lease.fileno()), str(ready), child_code],
        pass_fds=(lease.fileno(),),
        start_new_session=True,
    )
    lease.close()
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(MaintenanceError, match="writer_lease_timeout"):
            acquire_lease(db_path, exclusive=True, timeout_sec=0.1)

        shutdown = _stop_process_group(
            process, term_timeout_sec=0.1, kill_timeout_sec=2
        )
        assert shutdown["escalated"] is True
        exclusive = acquire_lease(db_path, exclusive=True, timeout_sec=0.2)
        exclusive.close()
    finally:
        if _process_group_exists(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


@contextlib.contextmanager
def _server(tmp_path: Path, db_path: Path, *, writes: bool):
    port = _free_port()
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT),
        "TRADER_KOO_DB_PATH": str(db_path),
        "TRADER_KOO_REPORT_DIR": str(tmp_path / "reports"),
        "TRADER_KOO_LOG_DIR": str(tmp_path / "logs"),
        "TRADER_KOO_LOGOS_DIR": str(tmp_path / "logos"),
        "TRADER_KOO_API_KEY": API_KEY,
        "ADMIN_STRICT_API_KEY": "1",
        "TRADER_KOO_PAPER_TRADE_ENABLED": "1" if writes else "0",
        "FINNHUB_API_KEY": "",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": "",
    })
    log_path = tmp_path / f"uvicorn-{port}.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "trader_koo.backend.main:app",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(log_path.read_text(errors="replace"))
                try:
                    code, health = _request(port, "/api/health")
                    if code == 200:
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                raise AssertionError(f"server did not become healthy: {log_path.read_text(errors='replace')}")
            yield port, env
        finally:
            shutdown = _stop_process_group(process)
            log.write(
                ("\nrehearsal_process_group_shutdown=" + json.dumps(shutdown) + "\n").encode()
            )
            log.flush()
            lease = acquire_lease(db_path, exclusive=True, timeout_sec=2)
            lease.close()


def _cli(db_path: Path, backup_dir: Path, action: str, run_id: str, *extra: str) -> dict:
    command = [
        sys.executable, "-m", "trader_koo.scripts.paper_schema_maintenance",
        action, "--db-path", str(db_path), "--backup-dir", str(backup_dir),
        "--run-id", run_id, *extra,
    ]
    result = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=REHEARSAL_COMMAND_TIMEOUT_SEC,
    )
    return json.loads(result.stdout)


def _request_maintenance(port: int, key: str) -> str:
    payload = {
        "reason": "non-production copied-database schema rehearsal",
        "timeout_sec": 10,
        "idempotency_key": key,
    }
    assert _request(port, "/api/admin/maintenance/request", payload=payload)[0] == 401
    code, body = _request(
        port, "/api/admin/maintenance/request", payload=payload, authenticated=True,
    )
    assert code == 200 and body["ok"] is True
    return "maint_" + hashlib.sha256(key.encode()).hexdigest()[:20]


def _publish_and_admit(
    env: dict[str, str], db_path: Path, report_dir: Path, *, candidate: bool,
) -> dict:
    command = [
        sys.executable, str(WORKER), "--db-path", str(db_path),
        "--report-dir", str(report_dir),
    ]
    if candidate:
        command.append("--candidate")
    result = subprocess.run(
        command, cwd=ROOT, env=env, check=True, text=True,
        capture_output=True, timeout=30,
    )
    return json.loads(result.stdout)


def _prepare_prices(env: dict[str, str], db_path: Path, report_dir: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--db-path",
            str(db_path),
            "--report-dir",
            str(report_dir),
            "--prepare-prices",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_copied_v4_migrate_activate_admit_restart_and_persist(tmp_path: Path) -> None:
    db_path, backup_dir = tmp_path / "paper.db", tmp_path / "backups"
    report_dir = tmp_path / "reports"
    source, evidence = _source_database(tmp_path, db_path)
    evidence["code_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    evidence["pre_migration"] = _campaign_evidence(db_path)

    with _server(tmp_path, db_path, writes=False) as (port, _):
        run_id = _request_maintenance(port, "journey-complete-001")
        assert _request(port, "/api/health")[1]["restart_required"] is True

    with _server(tmp_path, db_path, writes=False) as (port, _):
        health = _request(port, "/api/health")[1]
        assert health["maintenance"] is True and health["restart_required"] is False
        _cli(db_path, backup_dir, "quiesce-backup", run_id, "--boot-id", "offline-copy")
        _cli(db_path, backup_dir, "decide", run_id, "--decision", "complete",
             "--reason", "approve migration of non-production copy")
        _cli(db_path, backup_dir, "migrate-copy", run_id)
        resolved = _cli(db_path, backup_dir, "verify-resolution", run_id)
        assert resolved["state"] == "resolved"
        evidence["maintenance"] = {
            "run_id": run_id, "backup_name": resolved["backup_name"],
            "backup_sha256": resolved["backup_sha256"],
            "target_receipt_sha256": json.loads(resolved["migration_receipt_json"])["receipt_sha256"],
        }
        assert _request(port, "/api/health")[1]["restart_required"] is True

    with _server(tmp_path, db_path, writes=False) as (port, _):
        summary = _request(port, "/api/paper-trades/summary")[1]
        assert summary["campaign_health"]["status"] == "draft"
        assert summary["campaign_health"]["write_state"] == "paused"

    with _server(tmp_path, db_path, writes=True) as (port, enabled_env):
        summary = _request(port, "/api/paper-trades/summary")[1]
        assert summary["campaign_health"]["status"] == "draft"
        assert summary["campaign_health"]["write_state"] == "enabled"
    paper_before_price_preparation = _campaign_evidence(db_path)
    spy_before = _spy_price_evidence(db_path)
    price_preparation = _prepare_prices(enabled_env, db_path, report_dir)
    assert _campaign_evidence(db_path) == paper_before_price_preparation
    spy_after_existing = _spy_price_evidence(
        db_path,
        exclude_date=price_preparation["spy_appended_date"],
    )
    assert spy_after_existing == spy_before
    evidence["price_preparation"] = {
        **price_preparation,
        "preexisting_spy_row_count": spy_before["row_count"],
        "preexisting_spy_rows_sha256": spy_before["rows_sha256"],
        "spy_action_count": spy_before["action_count"],
        "spy_actions_sha256": spy_before["actions_sha256"],
    }
    observation = _publish_and_admit(
        enabled_env, db_path, report_dir, candidate=True,
    )
    assert observation["admitted"]["paper_trades"] == 0

    with _server(tmp_path, db_path, writes=True) as (port, _):
        transition = {
            "action": "activate", "reason": "human approves copied-database observation",
            "idempotency_key": "journey-activation-001",
        }
        code, first = _request(
            port, "/api/admin/paper-campaigns/paper-v2/transition",
            payload=transition, authenticated=True,
        )
        assert code == 200 and first["transition"]["idempotent"] is False
        code, retry = _request(
            port, "/api/admin/paper-campaigns/paper-v2/transition",
            payload=transition, authenticated=True,
        )
        assert code == 200 and retry["transition"]["idempotent"] is True
        changed = {**transition, "reason": "changed payload must fail"}
        assert _request(
            port, "/api/admin/paper-campaigns/paper-v2/transition",
            payload=changed, authenticated=True,
        )[0] == 409
    admitted = _publish_and_admit(enabled_env, db_path, report_dir, candidate=True)
    assert admitted["admitted"]["paper_trades"] == 1
    evidence["application"] = {
        "observation_report_run_id": observation["run_id"],
        "admitted_report_run_id": admitted["run_id"],
        "activation_idempotency_key": transition["idempotency_key"],
    }

    with _server(tmp_path, db_path, writes=True) as (port, _):
        code, trades = _request(port, "/api/paper-trades")
        assert code == 200
        assert trades["count"] == len(evidence["pre_migration"]["v2_trade_ids"]) + 1
        summary = _request(port, "/api/paper-trades/summary")[1]
        assert summary["campaign_health"]["status"] == "active"
        assert summary["campaign_health"]["write_state"] == "enabled"

    with sqlite3.connect(db_path) as conn:
        verified = verify_paper_schema_v5(conn)
        assert verified["status"] == "verified"
        new_runs = (
            evidence["application"]["observation_report_run_id"],
            evidence["application"]["admitted_report_run_id"],
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE campaign_id='paper-v2' "
            "AND report_run_id=?", (new_runs[1],),
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT status,publication_verified,is_generation_canonical "
            "FROM report_runs WHERE run_id=?", (new_runs[1],),
        ).fetchone() == ("published", 1, 1)
        assert conn.execute(
            "SELECT COUNT(*) FROM report_run_decisions WHERE run_id=? "
            "AND decision='accepted'", (new_runs[1],),
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT status,report_complete,is_canonical FROM paper_decision_sets "
            "WHERE report_run_id=? AND campaign_id='paper-v2'", (new_runs[1],),
        ).fetchone() == ("sealed", 1, 1)
        assert conn.execute(
            "SELECT COUNT(*) FROM paper_decision_sets WHERE report_run_id IN (?,?)",
            new_runs,
        ).fetchone() == (2,)
        assert conn.execute(
            "SELECT COUNT(*) FROM report_admission_attempts "
            "WHERE run_id IN (?,?) AND status='succeeded'", new_runs,
        ).fetchone() == (2,)
        assert conn.execute(
            "SELECT COUNT(*) FROM paper_campaign_audit WHERE idempotency_key=?",
            (evidence["application"]["activation_idempotency_key"],),
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT COUNT(*) FROM audit_logs WHERE request_path LIKE '%paper-campaigns%'"
        ).fetchone()[0] >= 3
        assert conn.execute(
            "SELECT COUNT(*) FROM paper_portfolio_snapshots WHERE campaign_id='paper-v2'"
        ).fetchone()[0] >= 1
        assert conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE campaign_id='paper-v2' "
            "AND accounting_status!='reconciled'"
        ).fetchone() == (0,)
        evidence["contract"] = {
            "contract_id": verified["contract_id"],
            "schema_fingerprint": verified["schema_fingerprint"],
            "final_logical_sha256": _logical_database_hash(conn),
        }
        evidence["application"]["trade_ids"] = [
            int(row[0]) for row in conn.execute(
                "SELECT id FROM paper_trades WHERE campaign_id='paper-v2' "
                "AND report_run_id=? ORDER BY id", (new_runs[1],),
            )
        ]
    evidence["post_restart"] = _campaign_evidence(db_path)
    assert evidence["post_restart"]["v1_trade_ids"] == evidence["pre_migration"]["v1_trade_ids"]
    assert evidence["post_restart"]["v1_accounting"] == evidence["pre_migration"]["v1_accounting"]
    assert evidence["post_restart"]["v1_snapshot_count"] == evidence["pre_migration"]["v1_snapshot_count"]
    assert _sha256(source) == evidence["source_sha256"]
    evidence_path = Path(os.getenv(
        "TRADER_KOO_V5_REHEARSAL_EVIDENCE", str(tmp_path / "paper-v5-rehearsal-evidence.json"),
    ))
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(str(evidence_path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    assert evidence_path.is_file()


def test_copied_v4_restore_path_recovers_named_backup_cohort(tmp_path: Path) -> None:
    db_path, backup_dir = tmp_path / "restore.db", tmp_path / "restore-backups"
    source, source_evidence = _source_database(tmp_path, db_path)
    with _server(tmp_path, db_path, writes=False) as (port, _):
        run_id = _request_maintenance(port, "journey-restore-001")

    with _server(tmp_path, db_path, writes=False) as (port, _):
        _cli(db_path, backup_dir, "quiesce-backup", run_id, "--boot-id", "offline-restore")
        cohort = status(db_path, run_id)["logical_cohort_sha256"]
        _cli(db_path, backup_dir, "decide", run_id, "--decision", "complete",
             "--reason", "approve migration before causal rollback drill")
        _cli(db_path, backup_dir, "migrate-copy", run_id)
        with sqlite3.connect(db_path) as damaged:
            damaged.execute("DROP INDEX idx_paper_trades_report_run")
            damaged.commit()
        damaged_hash = _sha256(db_path)
        _cli(db_path, backup_dir, "decide", run_id, "--decision", "restore",
             "--reason", "verifier-invalid post-commit copy requires rollback")
        restored = _cli(db_path, backup_dir, "restore-backup", run_id)
        assert restored["restore_receipt_json"]
        resolved = _cli(db_path, backup_dir, "verify-resolution", run_id)
        assert resolved["state"] == "resolved"
        assert _request(port, "/api/health")[1]["restart_required"] is True

    with sqlite3.connect(db_path) as conn:
        assert _logical_database_hash(conn) == cohort
        assert conn.execute(
            "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone() == (4,)
    assert db_path.with_name(f"{db_path.name}.pre_restore_{run_id}").is_file()
    assert _sha256(db_path.with_name(f"{db_path.name}.pre_restore_{run_id}")) == damaged_hash
    assert _sha256(source) == source_evidence["source_sha256"]

    with _server(tmp_path, db_path, writes=False) as (port, _):
        code, health = _request(port, "/api/health")
        assert code == 200 and health.get("maintenance") is not True


def test_report_worker_holds_shared_writer_lease(tmp_path: Path) -> None:
    db_path, ready, release = tmp_path / "worker.db", tmp_path / "ready", tmp_path / "release"
    _fresh_v4(db_path)
    process = subprocess.Popen(
        [
            sys.executable, str(WORKER), "--db-path", str(db_path),
            "--report-dir", str(tmp_path / "reports"), "--lease-only",
            "--lease-ready", str(ready), "--lease-release", str(release),
        ],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(MaintenanceError, match="writer_lease_timeout"):
            acquire_lease(db_path, exclusive=True, timeout_sec=0.1)
        release.write_text("release\n")
        assert process.wait(timeout=5) == 0
        exclusive = acquire_lease(db_path, exclusive=True, timeout_sec=0.1)
        exclusive.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

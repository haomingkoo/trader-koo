"""Boot preflight run by start.sh before any database write."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from trader_koo.scripts.start_preflight import PreflightError, preflight

PROD = {"RAILWAY_ENVIRONMENT": "production"}


def _db(path: Path, rows: int = 1, *, table: bool = True) -> Path:
    conn = sqlite3.connect(path)
    if table:
        conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, close REAL)")
        conn.executemany(
            "INSERT INTO price_daily VALUES ('SPY', ?, 1.0)",
            [(f"d{i}",) for i in range(rows)],
        )
    conn.commit()
    conn.close()
    return path


def _mounted(mount: Path):
    real = os.path.realpath(mount)
    return lambda path: os.path.realpath(path) == real


def _prod_env(mount: Path, **extra: str) -> dict[str, str]:
    return {**PROD, "RAILWAY_VOLUME_MOUNT_PATH": str(mount), **extra}


# -- local development ------------------------------------------------------

def test_local_missing_db_bootstraps(tmp_path):
    assert preflight(tmp_path / "new.db", environ={}) == "bootstrap"


def test_local_existing_db_is_ready(tmp_path):
    assert preflight(_db(tmp_path / "t.db"), environ={}) == "ready"


def test_local_empty_price_daily_is_ready_with_warning(tmp_path, capsys):
    assert preflight(_db(tmp_path / "t.db", rows=0), environ={}) == "ready"
    assert "0 price_daily rows" in capsys.readouterr().err


def test_corrupt_db_fails_closed_even_locally(tmp_path):
    db = tmp_path / "bad.db"
    db.write_bytes(b"not a database" * 512)

    with pytest.raises(PreflightError, match="cannot read"):
        preflight(db, environ={})


def test_db_path_that_is_a_directory_fails(tmp_path):
    (tmp_path / "dir.db").mkdir()

    with pytest.raises(PreflightError, match="not a regular file"):
        preflight(tmp_path / "dir.db", environ={})


def test_preflight_does_not_write_existing_db(tmp_path):
    db = _db(tmp_path / "t.db")
    before = (db.stat().st_mtime_ns, db.read_bytes())

    preflight(db, environ={})

    assert (db.stat().st_mtime_ns, db.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.db"]


# -- production -------------------------------------------------------------

def test_production_missing_db_fails_without_opt_in(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    with pytest.raises(PreflightError, match="TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP"):
        preflight(mount / "trader_koo.db", environ=_prod_env(mount), ismount=_mounted(mount))


def test_production_missing_db_bootstraps_with_opt_in(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    result = preflight(
        mount / "trader_koo.db",
        environ=_prod_env(mount, TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP="1"),
        ismount=_mounted(mount),
    )

    assert result == "bootstrap"


def test_production_detected_from_volume_variable_alone(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    with pytest.raises(PreflightError, match="TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP"):
        preflight(
            mount / "trader_koo.db",
            environ={"RAILWAY_VOLUME_MOUNT_PATH": str(mount)},
            ismount=_mounted(mount),
        )


def test_production_without_volume_variable_fails(tmp_path):
    with pytest.raises(PreflightError, match="no volume"):
        preflight(
            tmp_path / "trader_koo.db",
            environ={**PROD, "TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP": "1"},
        )


def test_production_unmounted_volume_path_fails_even_with_opt_in(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    with pytest.raises(PreflightError, match="not a mount point"):
        preflight(
            mount / "trader_koo.db",
            environ=_prod_env(mount, TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP="1"),
            ismount=lambda path: False,
        )


def test_production_db_outside_volume_fails(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(PreflightError, match="outside the volume"):
        preflight(
            _db(elsewhere / "trader_koo.db"),
            environ=_prod_env(mount),
            ismount=_mounted(mount),
        )


def test_production_db_dir_on_other_device_fails(tmp_path, monkeypatch):
    mount = tmp_path / "data"
    (mount / "sub").mkdir(parents=True)
    db = _db(mount / "sub" / "trader_koo.db")
    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if os.path.realpath(path) == os.path.realpath(mount / "sub"):
            values = list(result)
            values[2] = result.st_dev + 1  # st_dev
            return os.stat_result(values)
        return result

    monkeypatch.setattr("trader_koo.scripts.start_preflight.os.stat", fake_stat)

    with pytest.raises(PreflightError, match="different device"):
        preflight(db, environ=_prod_env(mount), ismount=_mounted(mount))


def test_production_existing_db_on_volume_is_ready(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    result = preflight(
        _db(mount / "trader_koo.db"), environ=_prod_env(mount), ismount=_mounted(mount)
    )

    assert result == "ready"


def test_production_empty_price_daily_fails_without_opt_in(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    with pytest.raises(PreflightError, match="0 price_daily rows"):
        preflight(
            _db(mount / "trader_koo.db", rows=0), environ=_prod_env(mount), ismount=_mounted(mount)
        )


def test_production_missing_price_daily_table_fails_without_opt_in(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    with pytest.raises(PreflightError, match="no price_daily table"):
        preflight(
            _db(mount / "trader_koo.db", table=False),
            environ=_prod_env(mount),
            ismount=_mounted(mount),
        )


def test_production_empty_price_daily_allowed_with_opt_in(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()

    result = preflight(
        _db(mount / "trader_koo.db", rows=0),
        environ=_prod_env(mount, TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP="1"),
        ismount=_mounted(mount),
    )

    assert result == "ready"


# -- maintenance-only boot ---------------------------------------------------

def _request_maintenance(db: Path) -> None:
    from trader_koo.backend.services.maintenance import request_maintenance

    request_maintenance(db, run_id="maint-test", boot_id="boot", reason="test", timeout_sec=1)


def _corrupt(db: Path) -> Path:
    db.write_bytes(b"not a database" * 512)
    return db


@pytest.fixture(autouse=True)
def _no_state_path_override(monkeypatch):
    monkeypatch.delenv("TRADER_KOO_MAINTENANCE_STATE_PATH", raising=False)


def test_active_maintenance_skips_checks_on_corrupt_db(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()
    db = mount / "trader_koo.db"
    _request_maintenance(db)
    _corrupt(db)

    result = preflight(db, environ=_prod_env(mount), ismount=_mounted(mount))

    assert result == "maintenance"


def test_active_maintenance_with_missing_db_after_preserve_crash(tmp_path):
    """restore_backup can crash after moving the live DB aside; boot must allow the retry."""
    from trader_koo.backend.services.maintenance import decide, quiesce_backup, restore_backup
    from trader_koo.backend.services.pipeline import ensure_pipeline_runs_schema

    mount = tmp_path / "data"
    mount.mkdir()
    db = _db(mount / "trader_koo.db")
    with sqlite3.connect(db) as conn:
        ensure_pipeline_runs_schema(conn)
    _request_maintenance(db)
    quiesce_backup(db, run_id="maint-test", boot_id="boot-after", backup_dir=tmp_path / "backups")
    decide(db, run_id="maint-test", decision="restore", reason="crash drill")

    def crash(boundary: str) -> None:
        if boundary == "after_preserve":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        restore_backup(db, run_id="maint-test", fault=crash)
    assert not db.exists()

    result = preflight(db, environ=_prod_env(mount), ismount=_mounted(mount))

    assert result == "maintenance"
    assert not db.exists()
    assert restore_backup(db, run_id="maint-test")["restore_receipt_json"]


def test_maintenance_boot_leaves_live_db_files_untouched(tmp_path):
    db = _db(tmp_path / "t.db")
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    _request_maintenance(db)
    before = sorted(p.name for p in tmp_path.iterdir())
    stamp = (db.stat().st_mtime_ns, db.read_bytes())

    assert preflight(db, environ={}) == "maintenance"

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert (db.stat().st_mtime_ns, db.read_bytes()) == stamp


def test_required_marker_without_state_is_maintenance(tmp_path):
    from trader_koo.backend.services.maintenance import state_path

    db = _corrupt(tmp_path / "t.db")
    marker = state_path(db)
    marker.with_name(f"{marker.name}.required").touch()

    assert preflight(db, environ={}) == "maintenance"


def test_unreadable_maintenance_state_is_maintenance(tmp_path):
    from trader_koo.backend.services.maintenance import state_path

    db = _corrupt(tmp_path / "t.db")
    state_path(db).write_bytes(b"garbage" * 512)

    assert preflight(db, environ={}) == "maintenance"


def test_resolved_maintenance_runs_normal_checks(tmp_path):
    from trader_koo.backend.services.maintenance import state_path

    db = _db(tmp_path / "t.db")
    _request_maintenance(db)
    conn = sqlite3.connect(state_path(db))
    conn.execute("UPDATE maintenance_runs SET state='resolved'")
    conn.commit()
    conn.close()
    _corrupt(db)

    with pytest.raises(PreflightError, match="cannot read"):
        preflight(db, environ={})


def test_maintenance_does_not_bypass_volume_check(tmp_path):
    mount = tmp_path / "data"
    mount.mkdir()
    db = _db(mount / "trader_koo.db")
    _request_maintenance(db)

    with pytest.raises(PreflightError, match="not a mount point"):
        preflight(db, environ=_prod_env(mount), ismount=lambda path: False)

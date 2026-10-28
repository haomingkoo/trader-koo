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

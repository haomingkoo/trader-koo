from __future__ import annotations

import datetime as dt
import gzip
import sqlite3
from pathlib import Path

import pytest

from trader_koo.scripts import backup_db
from trader_koo.scripts.backup_db import backup_database, backup_path_by_name


def test_backup_database_creates_consistent_compressed_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO evidence(value) VALUES ('preserved')")
        conn.commit()

    result = backup_database(source, tmp_path / "backups")
    restored = tmp_path / "restored.db"
    with gzip.open(result["backup_path"], "rb") as compressed:
        restored.write_bytes(compressed.read())

    with sqlite3.connect(restored) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT value FROM evidence").fetchone()[0] == "preserved"
    assert len(str(result["sha256"])) == 64
    assert backup_path_by_name(str(result["backup_name"]), tmp_path / "backups") == Path(
        str(result["backup_path"])
    )
    assert backup_path_by_name("../source.db", tmp_path / "backups") is None
    symlink = tmp_path / "backups" / "trader_koo_backup_symlink.db.gz"
    symlink.symlink_to(source)
    assert backup_path_by_name(symlink.name, tmp_path / "backups") is None


def _wal_db(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO evidence(value) VALUES ('preserved')")
        conn.commit()
    return path


def test_backup_records_quick_check_ok(tmp_path: Path) -> None:
    result = backup_database(_wal_db(tmp_path / "source.db"), tmp_path / "backups")

    assert result["quick_check"] == "ok"


def test_backup_fails_and_writes_nothing_when_quick_check_is_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _wal_db(tmp_path / "source.db")
    backups = tmp_path / "backups"
    monkeypatch.setattr(
        backup_db,
        "_quick_check",
        lambda path: ["*** in database main ***", "Page 7: btreeInitPage() returns error code 11"],
    )

    with pytest.raises(backup_db.BackupIntegrityError, match="btreeInitPage"):
        backup_database(source, backups)

    assert list(backups.glob("*")) == []


def test_quick_check_reads_the_real_pragma(tmp_path: Path) -> None:
    assert backup_db._quick_check(_wal_db(tmp_path / "source.db")) == ["ok"]


def test_quick_check_reports_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany("INSERT INTO t(v) VALUES (?)", [("x" * 200,) for _ in range(500)])
        conn.commit()
    raw = bytearray(path.read_bytes())
    page_size = int.from_bytes(raw[16:18], "big")
    raw[page_size * 2 : page_size * 2 + 64] = b"\xff" * 64
    path.write_bytes(bytes(raw))

    assert backup_db._quick_check(path) != ["ok"]

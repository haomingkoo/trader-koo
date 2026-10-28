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


def test_failed_compression_leaves_no_backup_and_keeps_the_latest_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    source = _wal_db(tmp_path / "source.db")
    backups = tmp_path / "backups"
    good = backup_database(source, backups)
    before = backup_db.latest_backup_summary(backups)
    real_copy = backup_db.shutil.copyfileobj

    def copy_then_fill_disk(f_in, f_out, length=0):
        f_out.write(f_in.read(1024))
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(backup_db.shutil, "copyfileobj", copy_then_fill_disk)
    with pytest.raises(OSError, match="No space left"):
        backup_database(source, backups)
    monkeypatch.setattr(backup_db.shutil, "copyfileobj", real_copy)

    assert sorted(p.name for p in backups.iterdir()) == [good["backup_name"]]
    assert backup_db.latest_backup_summary(backups) == before


def test_backup_is_published_under_its_generated_name_only_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _wal_db(tmp_path / "source.db")
    backups = tmp_path / "backups"
    seen: list[list[str]] = []
    real_copy = backup_db.shutil.copyfileobj

    def copy_and_look(f_in, f_out, length=0):
        real_copy(f_in, f_out, length)
        seen.append(sorted(p.name for p in backups.glob("trader_koo_backup_*.db.gz")))

    monkeypatch.setattr(backup_db.shutil, "copyfileobj", copy_and_look)
    result = backup_database(source, backups)

    assert seen == [[]]
    with gzip.open(result["backup_path"], "rb") as handle:
        assert handle.read(16) == b"SQLite format 3\x00"


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


def _touch_backups(backup_dir: Path, stamps: list[str]) -> list[str]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{backup_db.BACKUP_PREFIX}{stamp}{backup_db.BACKUP_SUFFIX}" for stamp in stamps]
    for name in names:
        (backup_dir / name).write_bytes(b"x")
    return names


def _remaining(backup_dir: Path) -> set[str]:
    return {p.name for p in backup_dir.iterdir()}


def test_prune_keeps_older_restore_points_after_a_burst_of_same_day_backups(tmp_path: Path) -> None:
    # Production on 2026-10-28 held only Oct 25-26 copies after repeated
    # dark-deploy POSTs; older weekly points must survive such a burst.
    backups = tmp_path / "backups"
    weekly = _touch_backups(
        backups,
        ["20261004T030000000000Z", "20261011T030000000000Z", "20261018T030000000000Z"],
    )
    burst = _touch_backups(backups, [f"20261026T{h:02d}0000000000Z" for h in range(10, 18)])

    pruned = backup_db._prune_old_backups(backups, backup_db.MAX_BACKUPS)

    remaining = _remaining(backups)
    assert set(weekly) <= remaining
    assert burst[-1] in remaining and burst[-2] in remaining
    assert len(remaining) <= backup_db.MAX_BACKUPS
    assert set(pruned) == (set(weekly) | set(burst)) - remaining


def test_prune_steady_daily_cadence_keeps_recent_days_and_weekly_points(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    names = _touch_backups(backups, [f"202610{d:02d}T030000000000Z" for d in range(1, 29)])

    backup_db._prune_old_backups(backups, backup_db.MAX_BACKUPS)

    # Newest three days plus the newest copy of each of the four newest ISO weeks.
    prefix, suffix = backup_db.BACKUP_PREFIX, backup_db.BACKUP_SUFFIX
    assert _remaining(backups) == {
        names[-1],
        names[-2],
        names[-3],
        f"{prefix}20261025T030000000000Z{suffix}",
        f"{prefix}20261018T030000000000Z{suffix}",
        f"{prefix}20261011T030000000000Z{suffix}",
    }


def test_prune_never_exceeds_the_budget_and_keeps_the_newest(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    start = dt.datetime(2026, 8, 1, tzinfo=dt.timezone.utc)
    names = _touch_backups(
        backups,
        [(start + dt.timedelta(hours=13 * i)).strftime("%Y%m%dT%H%M%S%fZ") for i in range(150)],
    )

    for budget in (backup_db.MAX_BACKUPS, 3, 1):
        backup_db._prune_old_backups(backups, budget)
        remaining = _remaining(backups)
        assert len(remaining) <= budget
        assert names[-1] in remaining


def test_prune_leaves_files_outside_the_generated_pattern(tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    _touch_backups(backups, [f"202610{d:02d}T030000000000Z" for d in range(1, 29)])
    manual = f"{backup_db.BACKUP_PREFIX}manual{backup_db.BACKUP_SUFFIX}"
    (backups / "pre_split_rebase.db.gz").write_bytes(b"x")
    (backups / manual).write_bytes(b"x")

    backup_db._prune_old_backups(backups, backup_db.MAX_BACKUPS)

    remaining = _remaining(backups)
    assert "pre_split_rebase.db.gz" in remaining
    assert manual in remaining


def test_max_backups_budget_stays_at_least_seven() -> None:
    # maintenance.py re-hashes a named backup across stages and dark-deploy
    # POSTs share this budget.
    assert backup_db.MAX_BACKUPS >= 7

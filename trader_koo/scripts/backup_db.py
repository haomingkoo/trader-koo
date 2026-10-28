"""SQLite database backup — quick-checked, compressed, tiered retention.

Usage (standalone)::

    python -m trader_koo.scripts.backup_db --db-path /data/trader_koo.db

Or called programmatically by the APScheduler daily job in
``trader_koo.backend.services.scheduler``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import logging
import os
import sqlite3
import shutil
import tempfile
import time
from pathlib import Path

LOG = logging.getLogger("trader_koo.scripts.backup_db")

DEFAULT_BACKUP_DIR = Path("/data/backups")
# Hard ceiling on generated backups. The scheduled job, dark-deploy POSTs and
# maintenance quiesce backups all share it, and every copy lives on the same
# 5 GB Railway volume as the database. A maintenance run's named backup is
# not pruned under it: while maintenance blocks writers the scheduled backup
# skips and every HTTP path except /api/health answers 503.
MAX_BACKUPS = 7
# Retention tiers. The union is at most 7 files, because the newest backup is
# also the newest of its day and of its ISO week.
KEEP_RECENT = 2
KEEP_DAILY = 3
KEEP_WEEKLY = 4
BACKUP_PREFIX = "trader_koo_backup_"
BACKUP_SUFFIX = ".db.gz"
PARTIAL_SUFFIX = ".partial"
STAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
QUICK_CHECK_DETAIL_ROWS = 5


class BackupIntegrityError(RuntimeError):
    """The online snapshot failed ``PRAGMA quick_check``; nothing was written."""


def _quick_check(path: Path) -> list[str]:
    """Return the ``PRAGMA quick_check`` rows for *path* (``["ok"]`` when clean)."""
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            return [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
    except sqlite3.DatabaseError as exc:
        return [f"{type(exc).__name__}: {exc}"]


def backup_database(
    db_path: Path,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    max_backups: int = MAX_BACKUPS,
) -> dict[str, str | int | float]:
    """Compress *db_path* into a timestamped ``.db.gz`` in *backup_dir*.

    Returns a summary dict with backup path, size, hash, quick_check result
    and duration. Raises ``FileNotFoundError`` if the source DB does not exist
    and ``BackupIntegrityError`` if the snapshot fails ``PRAGMA quick_check``.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now(dt.timezone.utc).strftime(STAMP_FORMAT)
    dest = backup_dir / f"{BACKUP_PREFIX}{stamp}{BACKUP_SUFFIX}"

    t0 = time.monotonic()
    src_size = db_path.stat().st_size

    # A raw file copy can capture SQLite between WAL writes.  Take a consistent
    # online snapshot first, then compress that immutable file.
    with tempfile.TemporaryDirectory(prefix="trader-koo-backup-") as tmp_dir:
        snapshot = Path(tmp_dir) / "snapshot.db"
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source:
            with sqlite3.connect(snapshot) as target:
                source.backup(target)
        quick_check = _quick_check(snapshot)
        if quick_check != ["ok"]:
            detail = "; ".join(quick_check[:QUICK_CHECK_DETAIL_ROWS])
            LOG.error("Backup aborted: snapshot quick_check failed: %s", detail)
            raise BackupIntegrityError(f"snapshot quick_check failed: {detail}")
        # Compress under a name the backup glob does not match, so a write cut
        # short (for example ENOSPC) is never listed, served or counted as
        # the latest backup.
        partial = dest.with_name(dest.name + PARTIAL_SUFFIX)
        try:
            with partial.open("wb") as raw:
                with snapshot.open("rb") as f_in, gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=6
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
                raw.flush()
                os.fsync(raw.fileno())
            dest_size = partial.stat().st_size
            digest = hashlib.sha256()
            with partial.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            os.replace(partial, dest)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    elapsed = round(time.monotonic() - t0, 2)
    ratio = round(dest_size / src_size * 100, 1) if src_size else 0.0

    LOG.info(
        "Backup complete: %s (%.1f MB -> %.1f MB, %.1f%%, %.2fs)",
        dest.name,
        src_size / 1_048_576,
        dest_size / 1_048_576,
        ratio,
        elapsed,
    )

    # Prune old backups beyond max_backups
    pruned = _prune_old_backups(backup_dir, max_backups)
    if pruned:
        LOG.info("Pruned %d old backup(s): %s", len(pruned), ", ".join(pruned))

    return {
        "backup_path": str(dest),
        "backup_name": dest.name,
        "src_size_bytes": src_size,
        "dest_size_bytes": dest_size,
        "sha256": digest.hexdigest(),
        "quick_check": "ok",
        "compression_ratio_pct": ratio,
        "elapsed_sec": elapsed,
        "pruned_count": len(pruned),
    }


def _backup_time(path: Path) -> dt.datetime | None:
    stamp = path.name[len(BACKUP_PREFIX):-len(BACKUP_SUFFIX)]
    try:
        return dt.datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _prune_old_backups(backup_dir: Path, max_backups: int) -> list[str]:
    """Delete generated backups outside the retention tiers.

    Keeps the ``KEEP_RECENT`` newest copies, the newest copy of each of the
    ``KEEP_DAILY`` most recent UTC days and of each of the ``KEEP_WEEKLY`` most
    recent ISO weeks, capped at *max_backups*. A burst of same-day backups (for
    example repeated dark-deploy POSTs) therefore cannot evict every older
    restore point. Files whose name does not parse are left alone.
    """
    dated: list[tuple[dt.datetime, Path]] = []
    for path in backup_dir.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"):
        stamp = _backup_time(path)
        if stamp is None:
            LOG.warning("Backup retention ignores unrecognised file %s", path.name)
            continue
        dated.append((stamp, path))
    dated.sort(key=lambda item: item[0], reverse=True)

    keep: set[Path] = {path for _, path in dated[:KEEP_RECENT]}
    for tier_key, limit in (
        (lambda ts: ts.date(), KEEP_DAILY),
        (lambda ts: ts.isocalendar()[:2], KEEP_WEEKLY),
    ):
        seen: set[object] = set()
        for stamp, path in dated:
            key = tier_key(stamp)
            if key in seen:
                continue
            if len(seen) == limit:
                break
            seen.add(key)
            keep.add(path)
    kept_newest_first = [path for _, path in dated if path in keep][: max(max_backups, 1)]
    keep = set(kept_newest_first)

    pruned: list[str] = []
    for _, old in dated:
        if old in keep:
            continue
        try:
            old.unlink()
            pruned.append(old.name)
        except OSError as exc:
            LOG.warning("Failed to delete old backup %s: %s", old.name, exc)
    return pruned


def list_backups(backup_dir: Path = DEFAULT_BACKUP_DIR) -> list[dict[str, str | int]]:
    """Return metadata for all backups in *backup_dir*, newest first."""
    if not backup_dir.exists():
        return []

    pattern = f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"
    files = sorted(backup_dir.glob(pattern), key=lambda p: p.name, reverse=True)
    result: list[dict[str, str | int]] = []
    for f in files:
        stat = f.stat()
        modified_utc = (
            dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )
        result.append({
            "name": f.name,
            "size_bytes": stat.st_size,
            "modified_ts": modified_utc,
        })
    return result


def latest_backup_path(backup_dir: Path = DEFAULT_BACKUP_DIR) -> Path | None:
    """Return the generated backup with the newest name stamp, or ``None``.

    Files whose name does not parse and symlinks are ignored.
    """
    newest: tuple[dt.datetime, Path] | None = None
    if backup_dir.is_dir():
        for path in backup_dir.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"):
            stamp = _backup_time(path)
            if stamp is None or path.is_symlink():
                continue
            if newest is None or stamp > newest[0]:
                newest = (stamp, path)
    return newest[1] if newest else None


def latest_backup_summary(backup_dir: Path | None = None) -> dict[str, str | int | None]:
    """Return the newest generated backup's UTC time and size, or nulls.

    The time comes from the generated name, not the file mtime, so a copied
    or touched file cannot make an old backup look fresh. *backup_dir*
    defaults to ``DEFAULT_BACKUP_DIR`` read at call time.
    """
    path = latest_backup_path(DEFAULT_BACKUP_DIR if backup_dir is None else backup_dir)
    empty: dict[str, str | int | None] = {"latest_backup_utc": None, "latest_backup_size_bytes": None}
    if path is None:
        return empty
    try:
        size = path.stat().st_size
    except OSError as exc:
        LOG.warning("Latest backup %s unreadable: %s", path.name, exc)
        return empty
    stamp = _backup_time(path)
    assert stamp is not None
    return {
        "latest_backup_utc": stamp.replace(microsecond=0).isoformat(),
        "latest_backup_size_bytes": size,
    }


def backup_path_by_name(
    backup_name: str,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
) -> Path | None:
    """Resolve one generated backup name without accepting arbitrary paths."""
    if (
        Path(backup_name).name != backup_name
        or not backup_name.startswith(BACKUP_PREFIX)
        or not backup_name.endswith(BACKUP_SUFFIX)
    ):
        return None
    candidate = backup_dir / backup_name
    if candidate.is_symlink() or not candidate.is_file():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    return resolved if resolved.parent == backup_dir.resolve() else None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backup trader_koo SQLite database")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/data/trader_koo.db"),
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=DEFAULT_BACKUP_DIR,
        help="Directory to store backups",
    )
    parser.add_argument(
        "--max-backups",
        type=int,
        default=MAX_BACKUPS,
        help="Maximum number of backups to retain",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = backup_database(args.db_path, args.backup_dir, args.max_backups)
    print(f"Backup saved: {result['backup_name']} ({result['dest_size_bytes']} bytes, {result['elapsed_sec']}s)")


if __name__ == "__main__":
    main()

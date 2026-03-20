"""SQLite database backup — compress to timestamped .gz, retain last N copies.

Usage (standalone)::

    python -m trader_koo.scripts.backup_db --db-path /data/trader_koo.db

Or called programmatically by the APScheduler weekly job in
``trader_koo.backend.services.scheduler``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import logging
import shutil
import time
from pathlib import Path

LOG = logging.getLogger("trader_koo.scripts.backup_db")

DEFAULT_BACKUP_DIR = Path("/data/backups")
MAX_BACKUPS = 7
BACKUP_PREFIX = "trader_koo_backup_"
BACKUP_SUFFIX = ".db.gz"


def backup_database(
    db_path: Path,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    max_backups: int = MAX_BACKUPS,
) -> dict[str, str | int | float]:
    """Compress *db_path* into a timestamped ``.db.gz`` in *backup_dir*.

    Returns a summary dict with backup path, size, and duration.
    Raises ``FileNotFoundError`` if the source DB does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"{BACKUP_PREFIX}{stamp}{BACKUP_SUFFIX}"

    t0 = time.monotonic()
    src_size = db_path.stat().st_size

    # Stream-compress to avoid loading entire DB into memory
    with db_path.open("rb") as f_in, gzip.open(dest, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)

    elapsed = round(time.monotonic() - t0, 2)
    dest_size = dest.stat().st_size
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
        "compression_ratio_pct": ratio,
        "elapsed_sec": elapsed,
        "pruned_count": len(pruned),
    }


def _prune_old_backups(backup_dir: Path, max_backups: int) -> list[str]:
    """Delete oldest backups when count exceeds *max_backups*."""
    pattern = f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"
    existing = sorted(backup_dir.glob(pattern), key=lambda p: p.name, reverse=True)
    pruned: list[str] = []
    for old in existing[max_backups:]:
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
    """Return the path to the most recent backup, or ``None``."""
    if not backup_dir.exists():
        return None
    pattern = f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"
    files = sorted(backup_dir.glob(pattern), key=lambda p: p.name, reverse=True)
    return files[0] if files else None


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

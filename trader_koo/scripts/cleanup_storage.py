"""One-off and recurring storage cleanup for the /data persistent volume.

Safe to run at any time — only deletes ephemeral/diagnostic data,
never touches price_daily, finviz_fundamentals, yolo_patterns,
crypto_bars, or any ML/feature data.

Usage:
    python -m trader_koo.scripts.cleanup_storage --db-path /data/trader_koo.db
    python -m trader_koo.scripts.cleanup_storage --db-path /data/trader_koo.db --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path

LOG = logging.getLogger("trader_koo.cleanup")

RETENTION_DAYS = {
    "polymarket_snapshots": 7,
    "external_data_cache": 0,  # delete all expired rows
    "ingest_runs": 90,
    "ingest_ticker_status": 90,
    "audit_logs": 90,
    "yolo_run_events": 90,
}

# VACUUM rewrites the whole database. In WAL mode that rewrite stages through the
# -wal file inside a single transaction, so wal_autocheckpoint cannot interrupt it
# and journal_size_limit cannot cap it: it needs roughly the database size in free
# space, plus headroom. Starting one without that space fails after doing all the
# work, and on 2026-10-18 the margin on /data was 168 MiB against a 1297 MiB file.
VACUUM_FREE_SPACE_MULTIPLIER: float = 1.3

# VACUUM is also usually not worth its cost. Over 17 observed nights it reclaimed
# 108.6 MB in total and exactly nothing on 10 of them, at 26-57s each. Only run it
# when the freelist holds enough to repay the rewrite.
VACUUM_MIN_FREELIST_BYTES: int = 50 * 1024 * 1024

# The transient cost of a VACUUM is (current usage + database size), because the
# second copy exists alongside the original until the swap. Expressing the limit
# as a share of the volume is therefore the only rule that actually bounds it; a
# multiple of the database size does not, since it ignores how full the volume is.
# Measured on 2026-10-24: a three-day retention backlog left a large freelist, the
# nightly VACUUM rewrote all 1348 MB, and a 4.4 GB volume peaked at 3.99 GB = 89%,
# which raised a Railway alert. Steady state was 60% before and after. The old
# free >= 1.3x database rule was satisfied and permitted exactly that.
VACUUM_PEAK_MAX_PCT: float = 80.0


# The volume reached 68% with no alarm of any kind: there was no disk monitoring
# anywhere in the repository. At 100% SQLite returns SQLITE_FULL, every scheduler
# job swallows its own exception, the API keeps serving reads and /health stays
# green, so the first symptom is silently stale data.
DISK_ALERT_THRESHOLD_PCT: float = 85.0


def disk_pressure_alert(
    *,
    free_bytes: int,
    total_bytes: int,
    db_bytes: int,
) -> str | None:
    """Return an operator-facing alert when the volume is too full, else None.

    Reports whether a VACUUM can still run, because "full" and "full and unable
    to dig out" need different responses.
    """
    if total_bytes <= 0:
        return None

    used_bytes = total_bytes - free_bytes
    used_pct = used_bytes / total_bytes * 100.0
    if used_pct < DISK_ALERT_THRESHOLD_PCT:
        return None

    required = int(db_bytes * VACUUM_FREE_SPACE_MULTIPLIER)
    if free_bytes >= required:
        vacuum_note = f"VACUUM still fits (needs {required / 1e6:.0f} MB)."
    else:
        vacuum_note = (
            f"VACUUM is BLOCKED: it needs {required / 1e6:.0f} MB free to stage the "
            f"rewrite, so the volume cannot reclaim its own space."
        )

    # Name the ruler: df reports usable bytes, the Railway dashboard reports the
    # raw device, and the two disagree by about 3 percentage points.
    return (
        f"Volume {used_pct:.0f}% full per df "
        f"({used_bytes / 1e9:.2f} GB of {total_bytes / 1e9:.2f} GB, "
        f"{free_bytes / 1e6:.0f} MB free). Database {db_bytes / 1e6:.0f} MB. "
        f"{vacuum_note}"
    )


def vacuum_headroom_alert(
    *,
    free_bytes: int,
    total_bytes: int,
    db_bytes: int,
) -> str | None:
    """Return an alert when VACUUM can no longer start, whatever the volume shows.

    This is the alarm a percentage threshold cannot raise. VACUUM's requirement
    scales with the database, not with the volume, so a growing database starves
    it long before the disk looks full: on 2026-10-18 the volume was two thirds
    full, which is quiet, while VACUUM was already blocked.
    """
    required = int(db_bytes * VACUUM_FREE_SPACE_MULTIPLIER)
    if free_bytes >= required:
        return None

    used_pct = (total_bytes - free_bytes) / total_bytes * 100.0 if total_bytes > 0 else 0.0
    return (
        f"VACUUM headroom gone: {free_bytes / 1e6:.0f} MB free against the "
        f"{required / 1e6:.0f} MB that a {db_bytes / 1e6:.0f} MB database needs to stage "
        f"its rewrite, while the volume is only {used_pct:.0f}% full per df. "
        f"Deletions can no longer reclaim space."
    )


def storage_health_alert(db_path: Path) -> str | None:
    """Return the most urgent storage alert for *db_path*'s volume, or None.

    Thin filesystem wrapper over the two pure checks above, in severity order.
    """
    try:
        usage = shutil.disk_usage(db_path.parent)
        db_bytes = db_path.stat().st_size
    except OSError as exc:
        return f"Storage health check could not read {db_path.parent}: {exc}"

    return disk_pressure_alert(
        free_bytes=usage.free, total_bytes=usage.total, db_bytes=db_bytes,
    ) or vacuum_headroom_alert(
        free_bytes=usage.free, total_bytes=usage.total, db_bytes=db_bytes,
    )


def _vacuum_decision(
    *,
    free_bytes: int,
    db_bytes: int,
    freelist_pages: int,
    page_size: int,
    total_bytes: int = 0,
) -> tuple[bool, str]:
    """Return (should_vacuum, reason).

    Space is checked before worth, so an operator reading the log hears about the
    disk before hearing about tidiness. ``total_bytes`` is optional: callers that
    cannot report the volume size fall back to the free-space floor alone.
    """
    if total_bytes > 0:
        used_bytes = total_bytes - free_bytes
        peak_pct = (used_bytes + db_bytes) / total_bytes * 100.0
        if peak_pct > VACUUM_PEAK_MAX_PCT:
            return False, (
                f"unsafe: the rewrite would peak the volume at {peak_pct:.0f}% "
                f"(now {used_bytes / total_bytes * 100.0:.0f}%, plus a "
                f"{db_bytes / 1e6:.0f} MB second copy), above the "
                f"{VACUUM_PEAK_MAX_PCT:.0f}% ceiling"
            )

    required = int(db_bytes * VACUUM_FREE_SPACE_MULTIPLIER)
    if free_bytes < required:
        return False, (
            f"unsafe: {free_bytes / 1e6:.0f} MB free is under the "
            f"{VACUUM_FREE_SPACE_MULTIPLIER:.1f}x database size ({required / 1e6:.0f} MB) "
            f"that the rewrite has to stage"
        )

    reclaimable = freelist_pages * page_size
    if reclaimable < VACUUM_MIN_FREELIST_BYTES:
        return False, (
            f"not worth it: freelist holds {reclaimable / 1e6:.1f} MB, under the "
            f"{VACUUM_MIN_FREELIST_BYTES / 1e6:.0f} MB threshold"
        )

    return True, (
        f"ok: {free_bytes / 1e6:.0f} MB free, freelist {reclaimable / 1e6:.1f} MB reclaimable"
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    return conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]


def _report_sizes(conn: sqlite3.Connection) -> dict[str, int]:
    """Report row counts for all tables."""
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    sizes: dict[str, int] = {}
    for table in tables:
        sizes[table] = _count_rows(conn, table)
    return sizes


def _prune_polymarket(
    conn: sqlite3.Connection, retention_days: int, *, dry_run: bool,
) -> int:
    if not _table_exists(conn, "polymarket_snapshots"):
        return 0
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=retention_days))
    cutoff_iso = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    count = conn.execute(
        "SELECT COUNT(*) FROM polymarket_snapshots WHERE snapshot_ts < ?",
        (cutoff_iso,),
    ).fetchone()[0]
    if count > 0 and not dry_run:
        conn.execute(
            "DELETE FROM polymarket_snapshots WHERE snapshot_ts < ?",
            (cutoff_iso,),
        )
        conn.commit()
    return count


def _prune_expired_cache(conn: sqlite3.Connection, *, dry_run: bool) -> int:
    if not _table_exists(conn, "external_data_cache"):
        return 0
    now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    count = conn.execute(
        "SELECT COUNT(*) FROM external_data_cache WHERE expires_ts < ?",
        (now_iso,),
    ).fetchone()[0]
    if count > 0 and not dry_run:
        conn.execute(
            "DELETE FROM external_data_cache WHERE expires_ts < ?",
            (now_iso,),
        )
        conn.commit()
    return count


def _prune_by_ts_column(
    conn: sqlite3.Connection,
    table: str,
    ts_column: str,
    retention_days: int,
    *,
    dry_run: bool,
) -> int:
    if not _table_exists(conn, table):
        return 0
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=retention_days))
    cutoff_iso = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    count = conn.execute(
        f"SELECT COUNT(*) FROM [{table}] WHERE [{ts_column}] < ?",
        (cutoff_iso,),
    ).fetchone()[0]
    if count > 0 and not dry_run:
        conn.execute(
            f"DELETE FROM [{table}] WHERE [{ts_column}] < ?",
            (cutoff_iso,),
        )
        conn.commit()
    return count


def _prune_unreferenced_reports(
    conn: sqlite3.Connection,
    report_dir: Path,
    *,
    grace_days: float = 2.0,
    dry_run: bool = False,
) -> int:
    """Delete report artifacts that no ``report_runs`` row references.

    A run-scoped artifact is immutable registry evidence: ``report/runs.py``
    re-reads and re-hashes it on every resolve, and a missing file makes paper
    admission raise ``ReportLineageError``. So retention cannot be by age --
    it has to ask the ledger. Files younger than *grace_days* are always kept,
    because an artifact is written moments before its run row commits.

    Returns the number of files deleted (or that would be, when *dry_run*).
    """
    if not report_dir.is_dir() or not _table_exists(conn, "report_runs"):
        return 0

    referenced: set[str] = set()
    for artifact, markdown in conn.execute(
        "SELECT artifact_path, markdown_path FROM report_runs"
    ):
        for value in (artifact, markdown):
            if value:
                referenced.add(os.path.realpath(str(value)))

    cutoff = dt.datetime.now(dt.timezone.utc).timestamp() - grace_days * 86400.0
    removed = 0
    for path in sorted(report_dir.glob("daily_report_*")):
        if path.suffix not in {".json", ".md"} or path.name.startswith("daily_report_latest"):
            continue
        if os.path.realpath(path) in referenced or path.stat().st_mtime >= cutoff:
            continue
        if not dry_run:
            try:
                path.unlink()
            except OSError as exc:
                LOG.warning("[CLEANUP] could not remove %s: %s", path.name, exc)
                continue
        removed += 1
    LOG.info(
        "[CLEANUP] report artifacts: %d unreferenced %s (%d referenced paths retained)",
        removed, "would be removed" if dry_run else "removed", len(referenced),
    )
    return removed


def run_cleanup(db_path: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Run all cleanup tasks. Returns dict of table → rows deleted."""
    mode = "DRY RUN" if dry_run else "LIVE"
    LOG.info("[CLEANUP] Starting storage cleanup (%s) on %s", mode, db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    results: dict[str, int] = {}

    try:
        # Report current sizes
        sizes = _report_sizes(conn)
        LOG.info("[CLEANUP] Current table sizes:")
        for table, count in sorted(sizes.items(), key=lambda x: -x[1]):
            if count > 0:
                LOG.info("  %-40s %10d rows", table, count)

        # DB file size
        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        wal_path = db_path.with_suffix(".db-wal")
        wal_size_mb = wal_path.stat().st_size / (1024 * 1024) if wal_path.exists() else 0
        LOG.info("[CLEANUP] DB file: %.1f MB, WAL: %.1f MB", db_size_mb, wal_size_mb)

        # 1. Polymarket snapshots (biggest offender)
        n = _prune_polymarket(conn, RETENTION_DAYS["polymarket_snapshots"], dry_run=dry_run)
        results["polymarket_snapshots"] = n
        LOG.info("[CLEANUP] polymarket_snapshots: %d rows to delete (keep %dd)", n, RETENTION_DAYS["polymarket_snapshots"])

        # 2. Expired cache entries
        n = _prune_expired_cache(conn, dry_run=dry_run)
        results["external_data_cache"] = n
        LOG.info("[CLEANUP] external_data_cache: %d expired rows to delete", n)

        # 3. Old ingest runs
        n = _prune_by_ts_column(conn, "ingest_runs", "started_ts", RETENTION_DAYS["ingest_runs"], dry_run=dry_run)
        results["ingest_runs"] = n
        LOG.info("[CLEANUP] ingest_runs: %d rows to delete (keep %dd)", n, RETENTION_DAYS["ingest_runs"])

        # 4. Old ingest ticker status
        n = _prune_by_ts_column(conn, "ingest_ticker_status", "started_ts", RETENTION_DAYS["ingest_ticker_status"], dry_run=dry_run)
        results["ingest_ticker_status"] = n
        LOG.info("[CLEANUP] ingest_ticker_status: %d rows to delete (keep %dd)", n, RETENTION_DAYS["ingest_ticker_status"])

        # 5. Old audit logs
        n = _prune_by_ts_column(conn, "audit_logs", "timestamp", RETENTION_DAYS["audit_logs"], dry_run=dry_run)
        results["audit_logs"] = n
        LOG.info("[CLEANUP] audit_logs: %d rows to delete (keep %dd)", n, RETENTION_DAYS["audit_logs"])

        # 6. Old YOLO run events
        n = _prune_by_ts_column(conn, "yolo_run_events", "created_ts", RETENTION_DAYS["yolo_run_events"], dry_run=dry_run)
        results["yolo_run_events"] = n
        LOG.info("[CLEANUP] yolo_run_events: %d rows to delete (keep %dd)", n, RETENTION_DAYS["yolo_run_events"])

        # 7. Report artifacts with no owning run (the ledger decides, not age)
        n = _prune_unreferenced_reports(
            conn, Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports")), dry_run=dry_run,
        )
        results["report_artifacts"] = n

        # 8. WAL checkpoint + VACUUM (only in live mode)
        if not dry_run:
            LOG.info("[CLEANUP] Running WAL checkpoint...")
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as exc:
                LOG.warning("[CLEANUP] WAL checkpoint failed: %s", exc)
            # Measure after the checkpoint: a live WAL makes the file size misleading,
            # and the gate below depends on an accurate one.
            usage = shutil.disk_usage(db_path.parent)
            free_bytes = usage.free
            db_bytes = db_path.stat().st_size
            freelist_pages = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
            should_vacuum, reason = _vacuum_decision(
                free_bytes=free_bytes,
                db_bytes=db_bytes,
                freelist_pages=freelist_pages,
                page_size=page_size,
                total_bytes=usage.total,
            )
            if not should_vacuum:
                # ERROR, not warning: a skipped VACUUM on a filling volume is the
                # signal that the volume needs attention, and it used to be silent.
                LOG.error(
                    "[CLEANUP] VACUUM SKIPPED — %s (db %.0f MB, free %.0f MB)",
                    reason, db_bytes / 1e6, free_bytes / 1e6,
                )
            else:
                LOG.info("[CLEANUP] VACUUM gate %s; running (this may take a minute)...", reason)
                try:
                    conn.execute("VACUUM")
                    new_size_mb = db_path.stat().st_size / (1024 * 1024)
                    LOG.info(
                        "[CLEANUP] DB size after VACUUM: %.1f MB (was %.1f MB)",
                        new_size_mb, db_bytes / (1024 * 1024),
                    )
                except Exception as exc:
                    LOG.error("[CLEANUP] VACUUM FAILED after doing the work: %s", exc)

        total = sum(results.values())
        LOG.info("[CLEANUP] %s complete: %d total rows %s", mode, total, "would be deleted" if dry_run else "deleted")

    finally:
        conn.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up /data persistent volume storage")
    parser.add_argument("--db-path", type=str, default="/data/trader_koo.db")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    db = Path(args.db_path)
    if not db.exists():
        LOG.error("Database not found: %s", db)
        sys.exit(1)

    run_cleanup(db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

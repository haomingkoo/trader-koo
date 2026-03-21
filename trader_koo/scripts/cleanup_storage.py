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

        # 7. WAL checkpoint + VACUUM (only in live mode)
        if not dry_run:
            LOG.info("[CLEANUP] Running WAL checkpoint...")
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as exc:
                LOG.warning("[CLEANUP] WAL checkpoint failed: %s", exc)
            LOG.info("[CLEANUP] Running VACUUM (this may take a minute)...")
            try:
                conn.execute("VACUUM")
                new_size_mb = db_path.stat().st_size / (1024 * 1024)
                LOG.info("[CLEANUP] DB size after VACUUM: %.1f MB (was %.1f MB)", new_size_mb, db_size_mb)
            except Exception as exc:
                LOG.warning("[CLEANUP] VACUUM failed (disk may be too full): %s", exc)

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

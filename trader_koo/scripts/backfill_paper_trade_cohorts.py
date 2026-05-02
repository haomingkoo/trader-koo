"""Backfill ``bot_version`` on legacy paper_trades using the deploy timeline.

The diagnostics work made ``bot_version`` auto-derive from the live git
SHA, but the existing 27+ trades all carry the old static ``"v1.0.0"``
tag — useless for cohort analysis. This script maps each historical
trade onto the fix wave it actually traded under.

Mapping (entry_date thresholds, UTC):

    < 2026-03-26 14:18    -> "pre-fix-frankenstein"
    < 2026-04-07 18:24    -> "c9322ed-regime-fail-closed"
    < 2026-04-07 22:18    -> "c317c56-graduated-trails"
    < 2026-04-10 05:46    -> "a605924-family-edge-gate"
    < 2026-04-10 15:26    -> "4ada5f6-7-root-causes"
    >= 2026-04-10 15:26   -> "e841eb8-calibration-pulse"

Run with --dry-run first to preview reassignments.

Usage:
    python -m trader_koo.scripts.backfill_paper_trade_cohorts \\
        --db-path /data/trader_koo.db --dry-run
    python -m trader_koo.scripts.backfill_paper_trade_cohorts \\
        --db-path /data/trader_koo.db
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sqlite3
from pathlib import Path

LOG = logging.getLogger("trader_koo.scripts.backfill_paper_trade_cohorts")

# Each fix wave that materially changed risk/sizing/regime logic. Ordered
# chronologically so we pick the LATEST cohort whose timestamp is <= entry.
COHORT_TIMELINE: list[tuple[dt.datetime, str]] = [
    (dt.datetime(2026, 3, 26, 14, 18, tzinfo=dt.timezone.utc),
     "c9322ed-regime-fail-closed"),
    (dt.datetime(2026, 4, 7, 18, 24, tzinfo=dt.timezone.utc),
     "c317c56-graduated-trails"),
    (dt.datetime(2026, 4, 7, 22, 18, tzinfo=dt.timezone.utc),
     "a605924-family-edge-gate"),
    (dt.datetime(2026, 4, 10, 5, 46, tzinfo=dt.timezone.utc),
     "4ada5f6-7-root-causes"),
    (dt.datetime(2026, 4, 10, 15, 26, tzinfo=dt.timezone.utc),
     "e841eb8-calibration-pulse"),
]
PRE_FIX_LABEL = "pre-fix-frankenstein"


def _resolve_cohort(entry_date: str) -> str:
    """Pick the cohort label active at ``entry_date``.

    ``entry_date`` is the ``YYYY-MM-DD`` string from paper_trades. We
    treat it as midnight UTC for the comparison; the fix-wave timestamps
    above are in UTC.
    """
    try:
        dt_obj = dt.datetime.strptime(entry_date, "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc,
        )
    except (ValueError, TypeError):
        return PRE_FIX_LABEL

    label = PRE_FIX_LABEL
    for threshold, name in COHORT_TIMELINE:
        if dt_obj >= threshold:
            label = name
        else:
            break
    return label


def backfill(db_path: Path, *, dry_run: bool, only_legacy: bool = True) -> dict[str, int]:
    """Reassign bot_version for trades matching the legacy "v1.0.0" tag.

    Returns a counter of updates per cohort label. With ``dry_run=True``
    nothing is written, but the same counts are returned for preview.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    counts: dict[str, int] = {}
    try:
        if only_legacy:
            where_clause = "WHERE bot_version IN ('v1.0.0', 'unknown') OR bot_version IS NULL"
        else:
            where_clause = ""

        rows = conn.execute(
            f"SELECT id, entry_date, bot_version FROM paper_trades {where_clause}"
        ).fetchall()

        for row in rows:
            cohort = _resolve_cohort(row["entry_date"])
            counts[cohort] = counts.get(cohort, 0) + 1
            if not dry_run:
                conn.execute(
                    "UPDATE paper_trades SET bot_version = ? WHERE id = ?",
                    (cohort, row["id"]),
                )

        if not dry_run:
            conn.commit()
    finally:
        conn.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/data/trader_koo.db"),
        help="Path to the SQLite DB (default: /data/trader_koo.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the reassignment without writing to the DB.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Reassign every paper_trade row (not just legacy v1.0.0/unknown). "
            "Use with caution — overwrites existing per-deploy SHAs."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.db_path.exists():
        raise SystemExit(f"DB not found: {args.db_path}")

    counts = backfill(args.db_path, dry_run=args.dry_run, only_legacy=not args.all)
    total = sum(counts.values())
    mode = "WOULD UPDATE" if args.dry_run else "UPDATED"
    LOG.info("%s %d paper_trades:", mode, total)
    for cohort, n in sorted(counts.items()):
        LOG.info("  %-35s -> %d trades", cohort, n)


if __name__ == "__main__":
    main()

"""Research database seeding and resume state for the price backfill."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from trader_koo.price_backfill import config as cfg

SEED_TABLES = ("price_daily", "price_corporate_actions", "price_series_revisions")


def seed_research_db(conn: sqlite3.Connection, source_path: Path) -> dict[str, dict]:
    """Copy price rows from a source database, opened read-only. Existing rows win.

    Returns, per table: rows copied, rows in the source, rows ignored because the
    research database already had them, and any column mismatch (a table missing
    from the source is reported, not skipped quietly).
    """
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    report: dict[str, dict] = {}
    try:
        for table in SEED_TABLES:
            theirs = [r[1] for r in source.execute(f"PRAGMA table_info({table})")]
            ours = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
            if not ours:
                raise LookupError(f"research database has no {table} table; run ensure_schema first")
            columns = [c for c in ours if c in theirs]
            entry: dict = {
                "copied": 0,
                "source_rows": 0,
                "ignored_existing": 0,
                "source_table_missing": not theirs,
                "target_columns_absent_in_source": [c for c in ours if c not in theirs] if theirs else [],
                "source_columns_not_copied": [c for c in theirs if c not in ours],
            }
            report[table] = entry
            if not columns:
                continue
            names = ",".join(columns)
            marks = ",".join("?" * len(columns))
            before = conn.total_changes
            entry["source_rows"] = source.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cursor = source.execute(f"SELECT {names} FROM {table}")
            while batch := cursor.fetchmany(cfg.SEED_BATCH):
                conn.executemany(
                    f"INSERT OR IGNORE INTO {table} ({names}) VALUES ({marks})", batch
                )
            entry["copied"] = conn.total_changes - before
            entry["ignored_existing"] = entry["source_rows"] - entry["copied"]
        conn.commit()
    finally:
        source.close()
    return report


def load_state(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(path: Optional[Path], state: dict) -> None:
    if path is not None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True, indent=1))
        tmp.replace(path)


def existing_snapshot(conn: sqlite3.Connection, ticker: str, earliest: Optional[str]) -> tuple:
    if earliest is None:
        return ((), ())
    prices = conn.execute(
        "SELECT * FROM price_daily WHERE ticker=? AND date>=? ORDER BY date", (ticker, earliest)
    ).fetchall()
    actions = conn.execute(
        "SELECT * FROM price_corporate_actions WHERE ticker=? AND action_date>=? "
        "ORDER BY action_date, action_type, provider",
        (ticker, earliest),
    ).fetchall()
    return tuple(prices), tuple(actions)


def save_yolo(conn: sqlite3.Connection, ticker: str) -> tuple[list[str], list[tuple]]:
    """record_price_series_revision clears a ticker's chart patterns on a new start date.

    Older history does not change recent bars, so the patterns stay valid.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='yolo_patterns'"
    ).fetchone()
    if not exists:
        return [], []
    columns = [row[1] for row in conn.execute("PRAGMA table_info(yolo_patterns)")]
    rows = conn.execute("SELECT * FROM yolo_patterns WHERE ticker=?", (ticker,)).fetchall()
    return columns, rows


def restore_yolo(conn: sqlite3.Connection, ticker: str, saved: tuple[list[str], list[tuple]]) -> None:
    columns, rows = saved
    if not columns:
        return
    conn.execute("DELETE FROM yolo_patterns WHERE ticker=?", (ticker,))
    marks = ",".join("?" * len(columns))
    conn.executemany(
        f"INSERT INTO yolo_patterns ({','.join(columns)}) VALUES ({marks})", rows
    )

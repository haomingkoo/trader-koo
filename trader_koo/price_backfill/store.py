"""Research database seeding and resume state for the price backfill."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from trader_koo.price_backfill import config as cfg

SEED_TABLES = ("price_daily", "price_corporate_actions", "price_series_revisions")


def seed_research_db(conn: sqlite3.Connection, source_path: Path) -> dict[str, int]:
    """Copy price rows from a source database, opened read-only. Existing rows win."""
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    copied: dict[str, int] = {}
    try:
        for table in SEED_TABLES:
            theirs = {r[1] for r in source.execute(f"PRAGMA table_info({table})")}
            columns = [r[1] for r in conn.execute(f"PRAGMA table_info({table})") if r[1] in theirs]
            if not columns:
                copied[table] = 0
                continue
            names = ",".join(columns)
            marks = ",".join("?" * len(columns))
            before = conn.total_changes
            cursor = source.execute(f"SELECT {names} FROM {table}")
            while batch := cursor.fetchmany(cfg.SEED_BATCH):
                conn.executemany(
                    f"INSERT OR IGNORE INTO {table} ({names}) VALUES ({marks})", batch
                )
            copied[table] = conn.total_changes - before
        conn.commit()
    finally:
        source.close()
    return copied


def load_state(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(path: Optional[Path], state: dict) -> None:
    if path is not None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True, indent=1))
        tmp.replace(path)

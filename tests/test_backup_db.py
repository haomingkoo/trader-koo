from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

from trader_koo.scripts.backup_db import backup_database


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

"""Report artifacts are registry evidence: retention asks the ledger, not the clock."""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from trader_koo.scripts.cleanup_storage import _prune_unreferenced_reports

_OLD = 10 * 86400  # comfortably past the 2-day grace period


def _seed(tmp_path: Path) -> tuple[sqlite3.Connection, Path, Path, Path]:
    reports = tmp_path / "reports"
    reports.mkdir()
    referenced = reports / "daily_report_20260101T000000Z_aaaaaaaa-0000-0000-0000-000000000000.json"
    orphan = reports / "daily_report_20260101T000000Z_bbbbbbbb-0000-0000-0000-000000000000.json"
    for path in (referenced, orphan):
        path.write_text("{}")
        old = time.time() - _OLD
        os.utime(path, (old, old))

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE report_runs (artifact_path TEXT, markdown_path TEXT)")
    conn.execute("INSERT INTO report_runs VALUES (?, ?)", (str(referenced), None))
    conn.commit()
    return conn, reports, referenced, orphan


def test_prune_removes_orphans_and_keeps_referenced(tmp_path: Path):
    conn, reports, referenced, orphan = _seed(tmp_path)
    try:
        removed = _prune_unreferenced_reports(conn, reports)
    finally:
        conn.close()

    assert removed == 1
    assert referenced.exists(), "an artifact a report_run points at must never be deleted"
    assert not orphan.exists()


def test_prune_keeps_recent_orphans_within_the_grace_period(tmp_path: Path):
    """An artifact is written moments before its run row commits."""
    conn, reports, _referenced, orphan = _seed(tmp_path)
    os.utime(orphan, None)  # touch to now
    try:
        removed = _prune_unreferenced_reports(conn, reports)
    finally:
        conn.close()

    assert removed == 0
    assert orphan.exists()


def test_prune_never_touches_the_latest_pointer(tmp_path: Path):
    conn, reports, _referenced, _orphan = _seed(tmp_path)
    latest = reports / "daily_report_latest.json"
    latest.write_text("{}")
    old = time.time() - _OLD
    os.utime(latest, (old, old))
    try:
        _prune_unreferenced_reports(conn, reports)
    finally:
        conn.close()

    assert latest.exists()


def test_dry_run_deletes_nothing(tmp_path: Path):
    conn, reports, _referenced, orphan = _seed(tmp_path)
    try:
        removed = _prune_unreferenced_reports(conn, reports, dry_run=True)
    finally:
        conn.close()

    assert removed == 1
    assert orphan.exists()

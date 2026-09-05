"""The diagnostic must not silently mislabel a stale feed as healthy."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

from trader_koo.scripts.system_check import (
    FAIL,
    PASS,
    WARN,
    Report,
    _age_hours,
    check_gates,
)


def test_age_hours_parses_both_stamp_shapes():
    now = dt.datetime.now(dt.timezone.utc)

    iso_z = (now - dt.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    offset = (now - dt.timedelta(hours=5)).isoformat()
    date_only = (now - dt.timedelta(days=2)).strftime("%Y-%m-%d")

    assert 4.9 < _age_hours(iso_z) < 5.1
    assert 4.9 < _age_hours(offset) < 5.1
    assert 47 < _age_hours(date_only) < 73  # date-only lands on midnight
    assert _age_hours(None) is None
    assert _age_hours("not a date") is None


def test_report_fails_only_on_fail_state():
    report = Report()
    report.add("A", "ok", PASS, "fine")
    report.add("A", "slow", WARN, "slow but up")
    assert report.failed == []

    report.add("A", "down", FAIL, "broken")
    assert [r["check"] for r in report.failed] == ["down"]


def _conn_with_sector_coverage(*, with_sector: int, total: int) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE finviz_fundamentals (snapshot_ts TEXT NOT NULL, ticker TEXT NOT NULL,"
        " raw_json TEXT, PRIMARY KEY (snapshot_ts, ticker))"
    )
    for index in range(total):
        raw = {"Price": "1"}
        if index < with_sector:
            raw["Sector"] = "Healthcare"
        conn.execute(
            "INSERT INTO finviz_fundamentals (snapshot_ts, ticker, raw_json) VALUES (?, ?, ?)",
            ("2026-09-05T00:00:00Z", f"T{index}", json.dumps(raw)),
        )
    conn.commit()
    return conn


def test_sector_coverage_fails_when_the_gate_cannot_evaluate():
    """0% coverage is the production state that let three same-sector trades open."""
    conn = _conn_with_sector_coverage(with_sector=0, total=100)
    report = Report()
    try:
        check_gates(report, conn)
    finally:
        conn.close()

    row = next(r for r in report.rows if r["check"] == "sector coverage")
    assert row["state"] == FAIL
    assert "0%" in row["detail"]


def test_sector_coverage_passes_when_populated():
    conn = _conn_with_sector_coverage(with_sector=95, total=100)
    report = Report()
    try:
        check_gates(report, conn)
    finally:
        conn.close()

    row = next(r for r in report.rows if r["check"] == "sector coverage")
    assert row["state"] == PASS

"""A delisted ticker is out of scope, not stale coverage.

price_daily is append-only, so every ticker ever ingested stays in it forever.
The Green Barrier scan read the whole table, so rotated-out symbols aged past
max_age_days, raised green_barrier_incomplete_coverage, set the report's ok flag
to False and blocked publication. Production on 2026-10-18: the latest ingest run
covered 538 tickers with 0 failures while price_daily held 559 distinct tickers,
leaving 21 orphans, and report publication had been dead for 23 days.

The gate itself is correct and stays: a ticker that IS in the current universe and
goes stale must still raise the warning. Only the universe was wrong.
"""
from __future__ import annotations

import sqlite3
import datetime as dt
from pathlib import Path

import pandas as pd

from trader_koo.analysis.green_barrier import scan_green_barrier_snapshot
from trader_koo.db.price_contract import record_price_series_revision

_AS_OF = dt.date(2026, 10, 18)


def _seed(path: Path, *, universe: tuple[str, ...]) -> sqlite3.Connection:
    """Price history for CURRENT plus ORPHAN, with an ingest run covering *universe*.

    Both get 18 completed month-end bars so the scan has enough history. CURRENT
    additionally gets recent daily bars so its latest observation is inside
    max_age_days; ORPHAN's history simply stops, which is what a rotated-out
    symbol looks like.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            data_source TEXT DEFAULT 'yfinance',
            fetch_timestamp TEXT DEFAULT '2026-10-18T00:00:00Z',
            adjustment_basis TEXT NOT NULL DEFAULT 'split_adjusted_price_only',
            adjustment_version TEXT NOT NULL DEFAULT 'test-v1',
            basis_status TEXT NOT NULL DEFAULT 'verified',
            unresolved_reason TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ingest_runs (
            run_id TEXT PRIMARY KEY, started_ts TEXT, finished_ts TEXT, status TEXT,
            tickers_total INTEGER, tickers_ok INTEGER, tickers_failed INTEGER,
            args_json TEXT, error_message TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ingest_ticker_status (
            run_id TEXT NOT NULL, ticker TEXT NOT NULL, started_ts TEXT,
            finished_ts TEXT, status TEXT
        )
        """
    )

    monthly = pd.date_range(end=pd.Timestamp("2026-08-31"), periods=18, freq="ME")
    for ticker in ("CURRENT", "ORPHAN"):
        conn.executemany(
            "INSERT INTO price_daily (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(ticker, d.date().isoformat(), 110, 120, 80, 110, 1000) for d in monthly],
        )
    # Only CURRENT keeps reporting: latest 2026-10-16, two days before as_of.
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, open, high, low, close, volume) "
        "VALUES ('CURRENT', ?, ?, ?, ?, ?, ?)",
        [(d.date().isoformat(), 110, 120, 80, 110, 1000)
         for d in pd.date_range("2026-10-12", "2026-10-16", freq="D")],
    )
    conn.commit()

    for ticker in ("CURRENT", "ORPHAN"):
        record_price_series_revision(
            conn,
            ticker,
            evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
            fetch_timestamp="2026-10-18T00:00:00Z",
        )

    conn.execute(
        "INSERT INTO ingest_runs (run_id, started_ts, status, tickers_total, tickers_ok, "
        "tickers_failed) VALUES ('20261016T220001Z-test', '2026-10-16T22:00:01Z', 'ok', ?, ?, 0)",
        (len(universe), len(universe)),
    )
    conn.executemany(
        "INSERT INTO ingest_ticker_status (run_id, ticker, status) VALUES "
        "('20261016T220001Z-test', ?, 'ok')",
        [(t,) for t in universe],
    )
    conn.commit()
    return conn


def test_a_rotated_out_ticker_is_not_counted_as_stale(tmp_path: Path):
    """The production bug: ORPHAN lingers in price_daily and blocks publication."""
    conn = _seed(tmp_path / "p.db", universe=("CURRENT",))
    try:
        coverage = scan_green_barrier_snapshot(conn, as_of=_AS_OF)["coverage"]
    finally:
        conn.close()

    assert coverage["stale_skipped_count"] == 0, coverage
    assert coverage["out_of_universe_count"] == 1
    assert coverage["out_of_universe_tickers"] == ["ORPHAN"]


def test_a_ticker_still_in_the_universe_that_goes_stale_still_raises():
    """The gate is correct and must survive the universe fix."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # ORPHAN is now part of the ingest universe, so its staleness is real.
        conn = _seed(Path(tmp) / "p.db", universe=("CURRENT", "ORPHAN"))
        try:
            coverage = scan_green_barrier_snapshot(conn, as_of=_AS_OF)["coverage"]
        finally:
            conn.close()

    assert coverage["stale_skipped_count"] == 1, coverage
    assert coverage["stale_skipped_tickers"] == ["ORPHAN"]
    assert coverage["out_of_universe_count"] == 0


def test_the_universe_origin_is_recorded(tmp_path: Path):
    """Coverage must say which universe it scanned, not leave it implied."""
    conn = _seed(tmp_path / "p.db", universe=("CURRENT",))
    try:
        coverage = scan_green_barrier_snapshot(conn, as_of=_AS_OF)["coverage"]
    finally:
        conn.close()

    assert coverage["source_universe_origin"] == "ingest_runs:20261016T220001Z-test"
    assert coverage["scanned_ticker_count"] == 1


def test_a_missing_ingest_ledger_scans_everything_rather_than_nothing(tmp_path: Path):
    """No ingest ledger must not silently empty the scan.

    Falling back to the whole table keeps today's conservative behaviour: the
    orphan is counted stale and the warning blocks publication. Scanning nothing
    would publish a green report on no evidence, which is the worse failure.
    """
    conn = _seed(tmp_path / "p.db", universe=("CURRENT",))
    try:
        conn.execute("DROP TABLE ingest_ticker_status")
        conn.commit()
        coverage = scan_green_barrier_snapshot(conn, as_of=_AS_OF)["coverage"]
    finally:
        conn.close()

    assert coverage["source_universe_origin"] == "price_daily"
    assert coverage["stale_skipped_count"] == 1
    assert coverage["out_of_universe_count"] == 0

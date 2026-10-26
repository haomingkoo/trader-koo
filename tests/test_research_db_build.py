"""Research database build: failure detection, stage ordering and end-of-build checks."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from ops import build_research_db as build
from ops import research_db_checks as chk

SCHEMA = """
CREATE TABLE price_daily (ticker TEXT, date TEXT, adjustment_basis TEXT, basis_status TEXT, unresolved_reason TEXT,
  PRIMARY KEY (ticker, date));
CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, provider TEXT,
  PRIMARY KEY (ticker, action_date, action_type, provider));
CREATE TABLE price_series_revisions (ticker TEXT PRIMARY KEY, managed_start TEXT, managed_end TEXT, evidence_json TEXT);
CREATE TABLE sp500_membership (ticker TEXT, start_date TEXT, end_date TEXT, PRIMARY KEY (ticker, start_date));
CREATE TABLE sec_fundamentals (cik INT, tag TEXT, unit TEXT, period_start TEXT, period_end TEXT, accession TEXT,
  value REAL, PRIMARY KEY (cik, tag, unit, period_start, period_end, accession));
CREATE TABLE sec_cik_map (ticker TEXT, cik INT, PRIMARY KEY (ticker, cik));
CREATE TABLE sec_fetch_status (cik INT PRIMARY KEY, status TEXT);
CREATE TABLE earnings_events (ticker TEXT, report_date TEXT, source TEXT, PRIMARY KEY (ticker, report_date, source));
"""
SESSIONS = ["2010-01-04", "2010-01-05", "2010-01-06"]


def make_db(path: Path, spy_start: str = "2010-01-04") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for sym in chk.CONTEXT_SYMBOLS:
        for d in SESSIONS:
            if sym in chk.BENCHMARK_SYMBOLS and d < spy_start:
                continue
            conn.execute("INSERT INTO price_daily VALUES (?,?,'split_adjusted_price_only','verified',NULL)", (sym, d))
    conn.execute("INSERT INTO sp500_membership VALUES ('AAA','2009-01-01',NULL)")
    conn.execute("INSERT INTO sp500_membership VALUES ('BRK-B','2009-01-01',NULL)")
    for t in ("AAA", "BRK-B"):
        for d in SESSIONS:
            conn.execute("INSERT INTO price_daily VALUES (?,?,'split_adjusted_price_only','verified',NULL)", (t, d))
    conn.execute("INSERT INTO sec_cik_map VALUES ('AAA',1),('BRK-B',2)")
    conn.execute("INSERT INTO sec_fetch_status VALUES (1,'ok'),(2,'ok')")
    conn.execute("INSERT INTO sec_fundamentals VALUES (1,'CommonStockSharesOutstanding','shares','','2010-01-01','a1',1e9)")
    conn.execute("INSERT INTO sec_fundamentals VALUES (2,'EntityCommonStockSharesOutstanding','shares','','2010-01-01','b1',0)")
    conn.commit()
    return conn


def by_name(checks: chk.Checks) -> dict[str, chk.Result]:
    return {r.name.split(" (")[0]: r for r in checks.results}


def test_clean_db_passes_and_reports_real_values(tmp_path):
    make_db(tmp_path / "r.db").close()
    checks = chk.run_checks(tmp_path / "r.db")
    assert checks.failed == []
    text = chk.render(checks)
    assert "2010  member_days=      6  with_price=      6  100.0%" in text
    assert "universe tickers=2  with facts=2 (100.0%)" in text
    assert "non_positive  BRK-B" in text  # off-scale share count is reported, not hidden
    assert "BRK tickers in universe=['BRK-B']" in text


def test_late_spy_history_fails_build(tmp_path):
    make_db(tmp_path / "r.db", spy_start="2010-01-06").close()
    failed = chk.run_checks(tmp_path / "r.db").failed
    assert [r.name.startswith("SPY/QQQ") for r in failed] == [True]


def test_logical_duplicate_key_fails_build(tmp_path):
    conn = make_db(tmp_path / "r.db")
    conn.execute("INSERT INTO price_daily VALUES ('aaa','2010-01-04','x','verified',NULL)")  # case variant of AAA
    conn.commit()
    conn.close()
    failed = chk.run_checks(tmp_path / "r.db").failed
    assert [r.name for r in failed] == ["duplicate keys = 0"]
    assert "price_daily (upper(ticker), date): 1" in failed[0].lines[0]


def test_low_price_coverage_fails_build(tmp_path):
    conn = make_db(tmp_path / "r.db")
    conn.execute("DELETE FROM price_daily WHERE ticker = 'AAA'")
    conn.execute("DELETE FROM price_daily WHERE ticker = 'BRK-B' AND date > '2010-01-04'")
    conn.commit()
    conn.close()
    failed = chk.run_checks(tmp_path / "r.db").failed
    assert any(r.name.startswith("member-day price coverage") for r in failed)


def test_missing_vendor_split_is_listed(tmp_path):
    conn = make_db(tmp_path / "r.db")
    ledger = {"vendor_action_ledger": [{"action_date": "2010-01-05", "action_type": "split", "value": 2.0},
                                       {"action_date": "1999-01-05", "action_type": "split", "value": 2.0}],
              "normalization_actions": []}
    conn.execute("INSERT INTO price_series_revisions VALUES ('AAA','2010-01-04','2010-01-06',?)", (json.dumps(ledger),))
    conn.commit()
    conn.close()
    res = by_name(chk.run_checks(tmp_path / "r.db"))
    lines = res["vendor splits in price_series_revisions but not in price_corporate_actions"].lines
    assert "missing from price_corporate_actions=1" in lines[0]  # the 1999 split is before the managed range
    assert "('AAA', '2010-01-05', 'split', 2.0)" in lines[1]


def test_scan_line_detects_database_failures_not_yfinance_noise():
    assert build.scan_line("sqlite3.OperationalError: database is locked")
    assert build.scan_line("Traceback (most recent call last):")
    assert build.scan_line("2026 | ERROR | cik 1 failed: database is locked")
    assert build.scan_line("2026 | ERROR | SLV: No earnings dates found, symbol may be delisted") is None


def stage(name: str, code: str) -> build.Stage:
    return build.Stage(name, [sys.executable, "-c", code])


def run(stages, tmp_path, capsys):
    with open(tmp_path / "log", "w") as log:
        return build.run_stages(stages, log, tmp_path, tmp_path / "x.db", {})


def test_first_failing_stage_stops_the_build(tmp_path, capsys):
    marker = tmp_path / "ran_third"
    stages = [stage("one", "print('ok')"), stage("two", "import sys; sys.exit(3)"),
              stage("three", f"open({str(marker)!r}, 'w')")]
    assert run(stages, tmp_path, capsys) == "two exited with status 3"
    assert not marker.exists()


def test_logged_lock_error_fails_even_with_exit_zero(tmp_path, capsys):
    reason = run([stage("sec", "print('sqlite3.OperationalError: database is locked')")], tmp_path, capsys)
    assert reason and "database is locked" in reason


def test_price_backfill_error_status_fails(tmp_path):
    (tmp_path / "backfill_report.json").write_text(json.dumps({"status_counts": {"ok": 3, "error": 2}}))
    assert "2 ticker(s)" in build.verify_backfill_report(tmp_path)
    (tmp_path / "backfill_report.json").write_text(json.dumps({"status_counts": {"ok": 3, "no_data": 9}}))
    assert build.verify_backfill_report(tmp_path) is None


def test_stage_order_and_subset_context_symbols(tmp_path):
    stages = build.build_stages(tmp_path / "r.db", tmp_path / "s.db", ["AAPL", "XOM"], "yfinance", tmp_path)
    assert [s.name for s in stages] == ["price backfill", "membership load", "SEC fundamentals load",
                                        "earnings backfill"]
    price, _, sec, _ = (s.argv for s in stages)
    assert "AAPL,XOM,SPY,QQQ,^VIX,^IRX" in price
    assert "--context-symbols" in price and "SPY,QQQ,^VIX,^IRX" in price
    assert "XOM=34088" in sec
    assert "--tickers" in sec and "AAPL,XOM" in sec


def test_missing_finnhub_key_is_refused_not_skipped(tmp_path, monkeypatch):
    src = tmp_path / "s.db"
    src.write_bytes(b"")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        build.main(["--db-path", str(tmp_path / "r.db"), "--source-db", str(src)])

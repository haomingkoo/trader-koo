"""Focused tests for report-generation guardrails."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from trader_koo.report.generator import (
    _report_stale_price_guard_error,
    fetch_report_payload,
)


def _create_minimal_report_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE price_daily (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                data_source TEXT
            );

            CREATE TABLE finviz_fundamentals (
                ticker TEXT NOT NULL,
                snapshot_ts TEXT NOT NULL,
                price REAL
            );

            CREATE TABLE options_iv (
                ticker TEXT NOT NULL,
                snapshot_ts TEXT NOT NULL,
                option_type TEXT
            );

            CREATE TABLE yolo_patterns (
                ticker TEXT NOT NULL,
                timeframe TEXT,
                detected_ts TEXT,
                as_of_date TEXT,
                confidence REAL
            );

            CREATE TABLE ingest_runs (
                run_id TEXT PRIMARY KEY,
                started_ts TEXT,
                finished_ts TEXT,
                status TEXT DEFAULT 'running',
                tickers_total INTEGER DEFAULT 0,
                tickers_ok INTEGER DEFAULT 0,
                tickers_failed INTEGER DEFAULT 0,
                error_message TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume, data_source)
            VALUES ('SPY', '2026-04-04', 500, 505, 495, 502, 1000000, 'test')
            """
        )
        conn.execute(
            """
            INSERT INTO ingest_runs (
                run_id, started_ts, finished_ts, status, tickers_total, tickers_ok, tickers_failed, error_message
            ) VALUES (
                'run_1', '2026-04-07T22:00:00Z', '2026-04-07T22:05:00Z', 'failed', 536, 0, 536, 'network failure'
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_report_stale_price_guard_blocks_failed_ingest_with_stale_prices():
    message = _report_stale_price_guard_error(
        latest_run={
            "status": "failed",
            "started_ts": "2026-04-07T22:00:00Z",
            "finished_ts": "2026-04-07T22:05:00Z",
        },
        latest_price_date="2026-04-06",
        market_date="2026-04-07",
    )

    assert message is not None
    assert "trading_days_behind=1" in message
    assert "Aborting report generation" in message


def test_report_stale_price_guard_allows_failed_ingest_with_current_prices():
    message = _report_stale_price_guard_error(
        latest_run={
            "status": "failed",
            "started_ts": "2026-04-07T22:00:00Z",
            "finished_ts": "2026-04-07T22:05:00Z",
        },
        latest_price_date="2026-04-07",
        market_date="2026-04-07",
    )

    assert message is None


def test_fetch_report_payload_aborts_before_signal_build_when_prices_are_stale(tmp_path: Path):
    db_path = tmp_path / "report.db"
    run_log = tmp_path / "cron.log"
    _create_minimal_report_db(db_path)

    with pytest.raises(RuntimeError, match="Aborting report generation"):
        fetch_report_payload(
            db_path=db_path,
            run_log=run_log,
            tail_lines=40,
        )


def test_unlinked_report_generation_is_read_only_for_calls_and_trades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from trader_koo.report import generator

    db_path = tmp_path / "report.db"
    run_log = tmp_path / "cron.log"
    _create_minimal_report_db(db_path)
    now = dt.datetime.now(dt.timezone.utc)
    asof_date = now.date().isoformat()
    snapshot_ts = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM price_daily")
    conn.execute("DELETE FROM ingest_runs")
    conn.execute(
        "INSERT INTO price_daily (ticker, date, open, high, low, close, volume, data_source) "
        "VALUES ('AAA', ?, 100, 101, 99, 100, 1000000, 'test')",
        (asof_date,),
    )
    conn.execute(
        "INSERT INTO finviz_fundamentals (ticker, snapshot_ts, price) VALUES ('AAA', ?, 100)",
        (snapshot_ts,),
    )
    conn.execute(
        "INSERT INTO yolo_patterns (ticker, timeframe, detected_ts, as_of_date, confidence) "
        "VALUES ('AAA', 'daily', ?, ?, 0.9)",
        (snapshot_ts, asof_date),
    )
    conn.execute(
        "INSERT INTO ingest_runs (run_id, started_ts, finished_ts, status, tickers_total, tickers_ok) "
        "VALUES ('fresh', ?, ?, 'ok', 1, 1)",
        (snapshot_ts, snapshot_ts),
    )
    conn.commit()
    conn.close()

    setup = {
        "ticker": "AAA",
        "setup_family": "bullish_continuation",
        "setup_tier": "A",
        "signal_bias": "bullish",
        "actionability": "higher-probability",
        "score": 90.0,
        "close": 100.0,
        "atr_pct_14": 2.0,
        "debate_agreement_score": 90.0,
    }
    monkeypatch.setattr(
        generator,
        "fetch_signals",
        lambda _conn: {
            "setup_quality_top": [dict(setup)],
            "setup_quality_all": [dict(setup)],
            "setup_quality_lookup": {"AAA": dict(setup)},
            "report_decisions": [],
            "scanned_universe": [],
            "regime_context": {},
        },
    )
    monkeypatch.setattr(generator, "llm_enabled", lambda: False)
    monkeypatch.setattr(generator, "build_earnings_calendar_payload", lambda *args, **kwargs: {})
    monkeypatch.setattr(generator, "build_tonight_key_changes", lambda *args, **kwargs: [])
    monkeypatch.setattr(generator, "fetch_yolo_delta", lambda *args, **kwargs: {})
    monkeypatch.setattr(generator, "fetch_yolo_pattern_persistence", lambda *args, **kwargs: {})

    payload = fetch_report_payload(
        db_path=db_path,
        run_log=run_log,
        tail_lines=0,
        report_run_id=None,
    )

    verify = sqlite3.connect(db_path)
    assert verify.execute("SELECT COUNT(*) FROM setup_call_evaluations").fetchone() == (0,)
    assert verify.execute("SELECT COUNT(*) FROM paper_trades").fetchone() == (0,)
    assert payload["signals"]["setup_evaluation"]["admission_state"] == (
        "requires_report_run_publication"
    )
    assert payload["paper_trades"]["admission_state"] == "requires_report_run_publication"
    verify.close()

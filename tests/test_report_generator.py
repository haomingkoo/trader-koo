"""Focused tests for report-generation guardrails."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pytest
import time_machine

from trader_koo.report.generator import (
    _green_barrier_coverage_requires_warning,
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


def test_green_barrier_insufficient_timeframe_history_is_not_fatal() -> None:
    coverage = {
        "stale_skipped_count": 0,
        "invalid_date_skipped_count": 0,
        "insufficient_history_skipped_count": 5,
    }

    assert _green_barrier_coverage_requires_warning(coverage) is False


@pytest.mark.parametrize("coverage_gap", ["stale_skipped_count", "invalid_date_skipped_count"])
def test_green_barrier_stale_or_invalid_dates_fail_closed(coverage_gap: str) -> None:
    coverage = {
        "stale_skipped_count": 0,
        "invalid_date_skipped_count": 0,
        "insufficient_history_skipped_count": 0,
    }
    coverage[coverage_gap] = 1

    assert _green_barrier_coverage_requires_warning(coverage) is True


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


def _fresh_report_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_yolo: bool) -> dict:
    from trader_koo.report import generator

    db_path = tmp_path / "report.db"
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
    if with_yolo:
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
        "ticker": "AAA", "setup_family": "bullish_continuation", "setup_tier": "A",
        "signal_bias": "bullish", "actionability": "higher-probability",
        "score": 90.0, "close": 100.0, "atr_pct_14": 2.0, "debate_agreement_score": 90.0,
    }
    monkeypatch.setattr(generator, "fetch_signals", lambda _conn: {
        "setup_quality_top": [dict(setup)], "setup_quality_all": [dict(setup)],
        "setup_quality_lookup": {"AAA": dict(setup)}, "report_decisions": [],
        "scanned_universe": [], "regime_context": {},
    })
    monkeypatch.setattr(generator, "build_earnings_calendar_payload", lambda *a, **k: {})
    monkeypatch.setattr(generator, "build_tonight_key_changes", lambda *a, **k: [])
    monkeypatch.setattr(generator, "fetch_yolo_delta", lambda *a, **k: {})
    monkeypatch.setattr(generator, "fetch_yolo_pattern_persistence", lambda *a, **k: {})
    return {"db_path": db_path, "run_log": tmp_path / "cron.log", "tail_lines": 0, "report_run_id": None}


def test_data_warning_still_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trader_koo.report.runs import _assert_publishable_report

    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=False)

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_missing" in payload["warnings"]
    assert not any(code.startswith("llm_") for code in payload["warnings"])
    assert payload["ok"] is False
    with pytest.raises(ValueError, match="degraded|partial-quality"):
        _assert_publishable_report(payload)


def _set_yolo_rows(db_path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Replace yolo_patterns with (timeframe, detected_ts, as_of_date) rows."""
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM yolo_patterns")
    conn.executemany(
        "INSERT INTO yolo_patterns (ticker, timeframe, detected_ts, as_of_date, confidence) "
        "VALUES ('AAA', ?, ?, ?, 0.9)",
        rows,
    )
    conn.commit()
    conn.close()


def _hours_ago_ts(hours: float) -> str:
    ts = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def test_yolo_current_on_latest_price_date_is_fresh_despite_old_detected_ts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=False)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _set_yolo_rows(kwargs["db_path"], [("daily", _hours_ago_ts(40), today)])

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_stale" not in payload["warnings"]
    assert payload["warnings"] == []
    assert payload["ok"] is True
    assert payload["latest_data"]["yolo_as_of_date"] == today
    assert payload["freshness"]["yolo_age_hours"] > 30


def test_yolo_daily_behind_latest_price_date_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=False)
    yesterday = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat()
    _set_yolo_rows(kwargs["db_path"], [("daily", _hours_ago_ts(1), yesterday)])

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_stale" in payload["warnings"]
    assert payload["ok"] is False


def test_yolo_without_rows_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=False)

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_missing" in payload["warnings"]
    assert "yolo_data_stale" not in payload["warnings"]


def test_yolo_weekly_rows_alone_do_not_count_as_current_daily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=False)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _set_yolo_rows(kwargs["db_path"], [("weekly", _hours_ago_ts(1), today)])

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_missing" in payload["warnings"]
    assert payload["ok"] is False


def _add_price_ticker(db_path: Path, ticker: str, date: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO price_daily (ticker, date, open, high, low, close, volume, data_source) "
        "VALUES (?, ?, 100, 101, 99, 100, 1000000, 'test')",
        (ticker, date),
    )
    conn.commit()
    conn.close()


def _add_yolo_row(db_path: Path, ticker: str, timeframe: str, as_of: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO yolo_patterns (ticker, timeframe, detected_ts, as_of_date, confidence) "
        "VALUES (?, ?, ?, ?, 0.9)",
        (ticker, timeframe, _hours_ago_ts(1), as_of),
    )
    conn.commit()
    conn.close()


def _add_yolo_event(db_path: Path, ticker: str, status: str, as_of: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS yolo_run_events (run_id TEXT, timeframe TEXT, "
        "ticker TEXT, status TEXT, as_of_date TEXT)"
    )
    conn.execute(
        "INSERT INTO yolo_run_events VALUES ('r1', 'daily', ?, ?, ?)", (ticker, status, as_of)
    )
    conn.commit()
    conn.close()


def test_yolo_missing_reports_no_as_of_date_and_no_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=False)

    payload = fetch_report_payload(**kwargs)

    assert payload["latest_data"]["yolo_as_of_date"] is None
    assert payload["freshness"]["yolo_age_hours"] is None


# Monday before the US close, so Friday is the latest completed session. The
# wall clock made this test fail every Monday evening after the close.
@time_machine.travel(dt.datetime(2026, 10, 26, 14, 0, tzinfo=dt.timezone.utc), tick=False)
def test_yolo_current_over_long_weekend_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=False)
    friday = "2026-10-23"
    conn = sqlite3.connect(kwargs["db_path"])
    conn.execute("UPDATE price_daily SET date = ?", (friday,))
    conn.execute(
        "INSERT INTO yolo_patterns (ticker, timeframe, detected_ts, as_of_date, confidence) "
        "VALUES ('AAA', 'daily', ?, ?, 0.9)",
        (_hours_ago_ts(80), friday),
    )
    conn.commit()
    conn.close()

    payload = fetch_report_payload(**kwargs)

    assert payload["freshness"]["yolo_age_hours"] > 72
    assert payload["warnings"] == []


def test_yolo_current_daily_with_old_weekly_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=True)
    _add_yolo_row(kwargs["db_path"], "AAA", "weekly", "2000-01-07")

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_stale" not in payload["warnings"]
    assert "yolo_data_missing" not in payload["warnings"]


def test_yolo_partial_run_below_coverage_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _add_price_ticker(kwargs["db_path"], "BBB", today)

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_stale" in payload["warnings"]


def test_yolo_zero_detection_ticker_counts_via_ok_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _add_price_ticker(kwargs["db_path"], "BBB", today)
    _add_yolo_event(kwargs["db_path"], "BBB", "ok", today)

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_stale" not in payload["warnings"]


def test_yolo_failed_event_does_not_count_as_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _add_price_ticker(kwargs["db_path"], "BBB", today)
    _add_yolo_event(kwargs["db_path"], "BBB", "failed", today)

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_stale" in payload["warnings"]


def test_yolo_coverage_ignores_tickers_excluded_from_research_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trader_koo.report import generator

    kwargs = _fresh_report_inputs(tmp_path, monkeypatch, with_yolo=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _add_price_ticker(kwargs["db_path"], "BBB", today)
    base_fetch = generator.fetch_signals
    monkeypatch.setattr(
        generator,
        "fetch_signals",
        lambda conn: {**base_fetch(conn), "price_contract": {"excluded_tickers": ["BBB"]}},
    )

    payload = fetch_report_payload(**kwargs)

    assert "yolo_data_stale" not in payload["warnings"]


LATEST_SESSION = "2026-10-23"
STALE_SESSION = "2026-07-17"


def _add_bars(conn: sqlite3.Connection, ticker: str, prev: tuple[str, float], last: tuple[str, float]) -> None:
    for date, close in (prev, last):
        conn.execute(
            "INSERT INTO price_daily (ticker, date, open, high, low, close, volume, data_source) "
            "VALUES (?, ?, ?, ?, ?, ?, 1000, 'test')",
            (ticker, date, close, close, close, close),
        )


def _signals_with_stale_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    stray_future_bar: bool = False, lone_future_bar: bool = False,
) -> dict:
    from trader_koo.report import generator

    db_path = tmp_path / "signals.db"
    _create_minimal_report_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM price_daily")
    conn.execute("DROP TABLE ingest_runs")
    conn.executescript(
        """
        CREATE TABLE ingest_runs (
            run_id TEXT PRIMARY KEY, started_ts TEXT, status TEXT,
            tickers_failed INTEGER, args_json TEXT
        );
        CREATE TABLE ingest_ticker_status (run_id TEXT, ticker TEXT, status TEXT);
        INSERT INTO ingest_runs VALUES ('r1', '2026-10-23T22:00:00Z', 'ok', 0, '{"use_sp500": 1}');
        """
    )
    tickers = {"UPP": 103.0, "DWN": 98.0, "STALE": 88.56}
    for ticker in tickers:
        conn.execute("INSERT INTO ingest_ticker_status VALUES ('r1', ?, 'ok')", (ticker,))
    _add_bars(conn, "UPP", ("2026-10-22", 100.0), (LATEST_SESSION, 103.0))
    _add_bars(conn, "DWN", ("2026-10-22", 100.0), (LATEST_SESSION, 98.0))
    _add_bars(conn, "STALE", ("2026-07-16", 100.0), (STALE_SESSION, 88.56))
    if stray_future_bar:
        _add_bars(conn, "UPP", ("2026-10-29", 103.0), ("2026-10-30", 104.0))
    if lone_future_bar:
        conn.execute(
            "INSERT INTO price_daily (ticker, date, open, high, low, close, volume, data_source) "
            "VALUES ('NEW', '2026-10-30', 1, 1, 1, 1, 1, 'test')"
        )
    conn.commit()
    monkeypatch.setattr(generator, "research_eligible_tickers", lambda _conn: set(tickers))
    monkeypatch.setattr(generator, "research_price_contract", lambda *_a, **_k: {"eligible": True})
    try:
        return generator._fetch_signals_snapshot(conn)
    finally:
        conn.close()


def test_movers_exclude_tickers_whose_latest_bar_is_not_the_latest_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signals = _signals_with_stale_ticker(tmp_path, monkeypatch)

    losers = [m["ticker"] for m in signals["movers_down_today"]]
    assert losers == ["DWN"]
    assert [m["ticker"] for m in signals["movers_up_today"]] == ["UPP"]


def test_breadth_counts_only_tickers_on_the_latest_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    breadth = _signals_with_stale_ticker(tmp_path, monkeypatch)["market_breadth"]

    assert (breadth["total_tickers"], breadth["advancers"], breadth["decliners"]) == (2, 1, 1)


def test_large_moves_exclude_stale_tickers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signals = _signals_with_stale_ticker(tmp_path, monkeypatch)

    assert "STALE" not in [m["ticker"] for m in signals["large_moves_today"]]


def test_sector_heatmap_excludes_stale_tickers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signals = _signals_with_stale_ticker(tmp_path, monkeypatch)

    assert sum(row["tickers"] for row in signals["sector_heatmap"]) == 2


def test_near_52w_lists_exclude_stale_tickers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signals = _signals_with_stale_ticker(tmp_path, monkeypatch)

    near = [r["ticker"] for r in signals["near_52w_high"] + signals["near_52w_low"]]
    assert "STALE" not in near
    assert "UPP" in near


def test_breadth_warns_when_latest_session_covers_few_tickers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trader_koo.report import generator

    monkeypatch.setattr(generator, "_report_warnings", [])

    signals = _signals_with_stale_ticker(tmp_path, monkeypatch, stray_future_bar=True)

    assert signals["market_breadth"]["total_tickers"] == 1
    assert "market_breadth_latest_session_thin" in generator._report_warnings


def test_breadth_does_not_warn_when_latest_session_is_fully_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trader_koo.report import generator

    monkeypatch.setattr(generator, "_report_warnings", [])

    _signals_with_stale_ticker(tmp_path, monkeypatch)

    assert "market_breadth_latest_session_thin" not in generator._report_warnings


def test_breadth_warns_and_is_empty_when_latest_session_has_no_comparable_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trader_koo.report import generator

    monkeypatch.setattr(generator, "_report_warnings", [])

    signals = _signals_with_stale_ticker(tmp_path, monkeypatch, lone_future_bar=True)

    assert signals["market_breadth"]["total_tickers"] == 0
    assert "market_breadth_latest_session_thin" in generator._report_warnings

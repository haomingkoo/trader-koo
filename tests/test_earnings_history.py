from __future__ import annotations

import sqlite3

import pandas as pd

from trader_koo.collection.earnings_history import (
    backfill_earnings_history,
    ensure_earnings_events_schema,
    session_from_timestamp,
    upsert_earnings_events,
    yfinance_rows,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_schema_is_idempotent():
    conn = _conn()
    ensure_earnings_events_schema(conn)
    ensure_earnings_events_schema(conn)
    pk = [r[1] for r in conn.execute("PRAGMA table_info(earnings_events)") if r[5]]
    assert set(pk) == {"ticker", "report_date", "source"}


def test_upsert_is_idempotent_and_keeps_known_values():
    conn = _conn()
    row = {
        "ticker": "aapl", "earnings_date": "2026-07-30", "earnings_session": "AMC",
        "estimate_eps": 1.9, "actual_eps": None, "source": "finnhub",
    }
    assert upsert_earnings_events(conn, [row]).written == 1
    assert upsert_earnings_events(conn, [row]).written == 1
    assert conn.execute("SELECT COUNT(*) FROM earnings_events").fetchone()[0] == 1
    later = dict(row, actual_eps=2.02, earnings_session="TBD", estimate_eps=None)
    upsert_earnings_events(conn, [later])
    r = conn.execute("SELECT * FROM earnings_events").fetchone()
    assert r["ticker"] == "AAPL"
    assert r["eps_actual"] == 2.02
    assert r["eps_estimate"] == 1.9  # NULL never overwrites a known value
    assert r["session"] == "AMC"  # TBD is stored as unknown, never overwrites


def test_missing_fields_are_null_and_bad_rows_skipped():
    conn = _conn()
    result = upsert_earnings_events(
        conn,
        [
            {"ticker": "MSFT", "earnings_date": "2026-07-29", "source": "alpha_vantage"},
            {"ticker": "", "earnings_date": "2026-07-29", "source": "x"},
            {"ticker": "KO", "earnings_date": "", "source": "x"},
            {"ticker": "PEP", "earnings_date": "2026-07-29", "earnings_session": "TBD", "source": "x"},
        ],
    )
    assert result.written == 2
    assert result.skipped_invalid == 2  # counted, not silently dropped
    assert result.session_unrecognised == 1  # TBD is stored as NULL and counted
    r = conn.execute("SELECT * FROM earnings_events").fetchone()
    assert r["session"] is None and r["eps_actual"] is None and r["revenue_actual"] is None


def test_session_from_timestamp():
    et = "America/New_York"
    assert session_from_timestamp(pd.Timestamp("2026-07-30 16:00", tz=et)) == "AMC"
    assert session_from_timestamp(pd.Timestamp("2026-07-30 07:00", tz=et)) == "BMO"
    assert session_from_timestamp(pd.Timestamp("2026-07-30 12:00", tz=et)) is None
    assert session_from_timestamp(pd.Timestamp("2026-07-30 00:00", tz=et)) is None


def test_yfinance_rows_maps_nan_to_none():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-07-30 16:00", tz="America/New_York"),
         pd.Timestamp("2026-11-02 00:00", tz="America/New_York")]
    )
    frame = pd.DataFrame(
        {"EPS Estimate": [1.89, 1.98], "Reported EPS": [2.02, float("nan")],
         "Surprise(%)": [6.74, float("nan")]},
        index=idx,
    )
    rows = yfinance_rows("AAPL", frame)
    assert rows[0]["earnings_date"] == "2026-07-30"
    assert rows[0]["earnings_session"] == "AMC"
    assert rows[0]["actual_eps"] == 2.02 and rows[0]["surprise_pct"] == 6.74
    assert rows[1]["actual_eps"] is None and rows[1]["surprise_pct"] is None


def _finnhub(symbol: str):
    return [{"ticker": symbol, "earnings_date": "2026-04-30", "earnings_session": "AMC",
             "estimate_eps": 1.9, "actual_eps": 2.0, "source": "finnhub"}]


def _yf(symbol: str):
    return [{"ticker": symbol, "earnings_date": "2026-04-30", "earnings_session": "AMC",
             "estimate_eps": 1.94, "actual_eps": 2.01, "surprise_pct": 3.46, "source": "yfinance"}]


def test_backfill_rate_limits_and_records_sources():
    conn = _conn()
    sleeps: list[float] = []
    summary = backfill_earnings_history(
        conn, ["AAPL", "MSFT"], finnhub_fetch=_finnhub, yfinance_fetch=_yf,
        finnhub_min_interval_sec=1.0, sleep=sleeps.append,
    )
    assert summary["tickers"] == 2 and summary["rows"] == 4
    assert {r[0] for r in conn.execute("SELECT source FROM earnings_events")} == {"finnhub", "yfinance"}
    assert sleeps and all(0 < s <= 1.0 for s in sleeps)
    backfill_earnings_history(conn, ["AAPL", "MSFT"], finnhub_fetch=_finnhub,
                              yfinance_fetch=_yf, sleep=lambda s: None)
    assert conn.execute("SELECT COUNT(*) FROM earnings_events").fetchone()[0] == 4


def test_backfill_records_failures_without_stopping():
    conn = _conn()

    def boom(symbol: str):
        raise RuntimeError("429")

    summary = backfill_earnings_history(
        conn, ["AAPL"], finnhub_fetch=boom, yfinance_fetch=_yf, sleep=lambda s: None,
    )
    assert summary["failures"] == {"AAPL:finnhub": "429"}
    assert summary["rows"] == 1


def test_finnhub_calendar_fetch_is_persisted(monkeypatch):
    from trader_koo import catalyst_data

    rows = [{"ticker": "AAPL", "earnings_date": "2026-11-02", "earnings_session": "AMC",
             "estimate_eps": 1.98, "actual_eps": None, "estimate_revenue": None,
             "actual_revenue": None, "source": "finnhub"}]
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setattr(catalyst_data, "_fetch_finnhub_calendar_rows", lambda *a, **k: rows)
    conn = _conn()
    import datetime as dt

    from trader_koo.scripts.update_market_db import ensure_schema

    ensure_schema(conn)
    catalyst_data.build_earnings_calendar_payload(conn, market_date=dt.date(2026, 10, 26))
    r = conn.execute("SELECT ticker, report_date, session, source FROM earnings_events").fetchone()
    assert tuple(r) == ("AAPL", "2026-11-02", "AMC", "finnhub")


def test_backfill_reports_invalid_rows_unrecognised_sessions_and_empty_fetches():
    conn = _conn()

    def junk(symbol: str):
        return [{"ticker": symbol, "earnings_date": "2026-04-30", "earnings_session": "TBD", "source": "finnhub"},
                {"ticker": symbol, "earnings_date": "", "source": "finnhub"}]

    summary = backfill_earnings_history(
        conn, ["AAPL"], finnhub_fetch=junk, yfinance_fetch=lambda s: [], sleep=lambda s: None)
    assert summary["rows"] == 1 and summary["rows_skipped_invalid"] == 1
    assert summary["sessions_unrecognised"] == 1
    assert summary["no_rows"] == ["AAPL:yfinance"]


def test_a_failed_archive_is_surfaced_in_the_meta(monkeypatch):
    from trader_koo.collection import earnings_history as eh

    conn = _conn()

    def broken(*a, **k):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(eh, "upsert_earnings_events", broken)
    rows, meta = eh.persist_fetched_calendar(conn, ([{"ticker": "AAPL"}], {"used_cache": False}))
    assert rows == [{"ticker": "AAPL"}] and meta["archive_error"] == "disk full"
    assert "archive_error" not in eh.persist_fetched_calendar(conn, ([], {"used_cache": False}))[1]

from __future__ import annotations

import datetime as dt
import sqlite3
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest

from trader_koo.collection.options_universe import (
    ensure_options_daily_metrics_schema,
    latest_session_date,
    load_ingest_universe,
    select_metric_expiries,
    snapshot_universe,
    universe_session_date,
)

TODAY = dt.date(2026, 10, 26)
SPOT = 100.0
RATE = 0.04


def _exps(*offsets: int) -> list[str]:
    return [(TODAY + dt.timedelta(days=d)).isoformat() for d in offsets]


def test_select_expiries_keeps_nearest_and_brackets_each_target():
    exps = _exps(0, 2, 4, 7, 14, 21, 28, 35, 49, 56, 63, 91, 120, 180)
    got = select_metric_expiries(exps, TODAY, nearest=2, target_days=(30, 60, 90))
    # nearest 0,2; bracket 30 -> 28,35; 60 -> 56,63; 90 -> 63,91
    assert got == _exps(0, 2, 28, 35, 56, 63, 91)


def test_select_expiries_exact_hit_and_dedup():
    exps = _exps(30, 60, 90)
    assert select_metric_expiries(exps, TODAY, nearest=2, target_days=(30, 60, 90)) == exps


def test_select_expiries_without_far_expiries_and_past_dates():
    assert select_metric_expiries(_exps(-3, 5), TODAY, nearest=1, target_days=(30,)) == _exps(5)
    assert select_metric_expiries([], TODAY, nearest=2, target_days=(30,)) == []


def test_session_date_is_the_evening_the_run_started_on():
    utc = dt.timezone.utc
    assert universe_session_date(dt.datetime(2026, 10, 26, 23, 45, tzinfo=utc)) == dt.date(2026, 10, 26)
    # after midnight UTC the run still belongs to the previous evening's session
    assert universe_session_date(dt.datetime(2026, 10, 27, 1, 30, tzinfo=utc)) == dt.date(2026, 10, 26)
    # Friday evening session may run into Saturday UTC morning
    assert universe_session_date(dt.datetime(2026, 10, 24, 1, 0, tzinfo=utc)) == dt.date(2026, 10, 23)


def test_window_excludes_nightly_report_and_weekend_sessions():
    utc = dt.timezone.utc
    for hour, minute in ((21, 30), (22, 0), (23, 30), (23, 44), (3, 0), (12, 0)):
        assert universe_session_date(dt.datetime(2026, 10, 27, hour, minute, tzinfo=utc)) is None
    assert universe_session_date(dt.datetime(2026, 10, 24, 23, 45, tzinfo=utc)) is None  # Saturday evening
    assert universe_session_date(dt.datetime(2026, 10, 25, 23, 45, tzinfo=utc)) is None  # Sunday evening
    assert universe_session_date(dt.datetime(2026, 10, 25, 1, 0, tzinfo=utc)) is None  # Sat evening's tail
    assert universe_session_date(dt.datetime(2026, 10, 26, 1, 0, tzinfo=utc)) is None  # Sun evening's tail


def test_latest_session_date_for_manual_runs():
    utc = dt.timezone.utc
    assert latest_session_date(dt.datetime(2026, 10, 26, 21, 0, tzinfo=utc)) == dt.date(2026, 10, 26)
    assert latest_session_date(dt.datetime(2026, 10, 26, 11, 45, tzinfo=utc)) == dt.date(2026, 10, 23)
    assert latest_session_date(dt.datetime(2026, 10, 25, 15, 0, tzinfo=utc)) == dt.date(2026, 10, 23)


def _ledger() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE ingest_runs (run_id TEXT, started_ts TEXT);
        CREATE TABLE ingest_ticker_status (run_id TEXT, ticker TEXT);
        INSERT INTO ingest_runs VALUES ('old','2026-10-01'),('new','2026-10-20');
        INSERT INTO ingest_ticker_status VALUES ('old','ZZZ'),('new','AAPL'),('new','BRK.B'),('new','MSFT');
        """
    )
    return conn


def test_universe_comes_from_latest_ingest_run_and_is_capped():
    conn = _ledger()
    assert load_ingest_universe(conn) == ["AAPL", "BRK-B", "MSFT"]
    assert load_ingest_universe(conn, max_tickers=2) == ["AAPL", "BRK-B"]


def test_universe_fails_closed_without_ledger():
    assert load_ingest_universe(sqlite3.connect(":memory:")) == []


# End to end through the real fetch, metrics and persist path with a yfinance-shaped fake.

def _frame(kind: str, base: float) -> pd.DataFrame:
    strikes = [float(k) for k in range(80, 121)]
    ivs = [base + 0.005 if kind == "call" else base - 0.005 + 0.002 * max(0.0, SPOT - k) for k in strikes]
    return pd.DataFrame(
        {"strike": strikes, "lastPrice": 1.0, "bid": 0.9, "ask": 1.1, "impliedVolatility": ivs,
         "openInterest": 10.0, "volume": 5.0 if kind == "call" else 7.0}
    )


class _FakeTicker:
    bases = {5: 0.18, 28: 0.20, 63: 0.30, 91: 0.25}

    def __init__(self, symbol):
        self.options = _exps(*self.bases)

    def history(self, period="5d"):
        return pd.DataFrame({"Close": [SPOT]})

    def option_chain(self, exp):
        base = self.bases[(dt.date.fromisoformat(exp) - TODAY).days]
        return SimpleNamespace(calls=_frame("call", base), puts=_frame("put", base))


snap = partial(snapshot_universe, ticker_factory=_FakeTicker, today=TODAY, rate=RATE)


def _row(conn, ticker="AAA"):
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM options_daily_metrics WHERE ticker = ?", (ticker,)).fetchone()


def test_snapshot_writes_one_metrics_row_per_ticker_and_no_raw_chain():
    conn = sqlite3.connect(":memory:")
    summary = snap(conn, ["AAA", "BBB", "CCC"], snapshot_ts="2026-10-26T20:10:00Z", workers=2)
    assert summary["rows_written"] == 3 and summary["quality"] == {"ok": 3}
    assert conn.execute("SELECT COUNT(*) FROM options_daily_metrics").fetchone()[0] == 3
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'options_iv'").fetchone() is None
    r = _row(conn)
    assert r["asof_date"] == TODAY.isoformat() and r["spot"] == SPOT and r["rate"] == RATE
    assert r["n_expiries"] == 4 and r["quality"] == "ok"
    assert r["call_volume"] == 5.0 * 41 * 4 and r["put_volume"] == 7.0 * 41 * 4
    assert r["pc_volume_ratio"] == pytest.approx(7 / 5)
    assert r["pc_oi_ratio"] == pytest.approx(1.0)


def test_rerun_same_day_is_skipped_and_force_replaces_in_place():
    conn = sqlite3.connect(":memory:")
    snap(conn, ["AAA"], snapshot_ts="2026-10-26T20:10:00Z")
    again = snap(conn, ["AAA"], snapshot_ts="2026-10-26T20:20:00Z")
    assert again["skipped_existing"] == 1 and again["rows_written"] == 0
    forced = snap(conn, ["AAA"], snapshot_ts="2026-10-26T20:30:00Z", force=True)
    assert forced["rows_written"] == 1
    assert conn.execute("SELECT COUNT(*) FROM options_daily_metrics").fetchone()[0] == 1
    assert _row(conn)["snapshot_ts"] == "2026-10-26T20:30:00Z"


def test_schema_migration_is_idempotent():
    conn = sqlite3.connect(":memory:")
    ensure_options_daily_metrics_schema(conn)
    ensure_options_daily_metrics_schema(conn)
    pk = [r[1] for r in conn.execute("PRAGMA table_info(options_daily_metrics)") if r[5]]
    assert set(pk) == {"asof_date", "ticker"}


def test_missing_rate_gives_null_skew_never_a_guess():
    conn = sqlite3.connect(":memory:")
    snap(conn, ["AAA"], rate=None)
    r = _row(conn)
    assert r["put_skew_25d"] is None and r["rate"] is None
    assert r["iv_30d"] is not None and r["quality"] == "partial"


def test_snapshot_stops_when_should_stop():
    conn = sqlite3.connect(":memory:")
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 1

    summary = snap(conn, ["AAA", "BBB", "CCC", "DDD"], workers=2, should_stop=stop)
    assert summary["stopped_early"] is True and summary["rows_written"] == 2


@pytest.mark.parametrize("workers", [1, 2])
def test_failures_are_recorded_not_raised(workers):
    class Boom(_FakeTicker):
        def option_chain(self, exp):
            raise RuntimeError("rate limited")

    conn = sqlite3.connect(":memory:")
    summary = snapshot_universe(
        conn, ["AAA"], workers=workers, ticker_factory=Boom, today=TODAY, rate=RATE
    )
    assert "rate limited" in summary["errors"]["AAA"] and summary["rows_written"] == 0

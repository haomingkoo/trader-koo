"""Backfill of pre-2018 daily prices: range selection, safety and idempotency."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from trader_koo.db.sources import DataSourceManager, PriceFetchError
from trader_koo.price_backfill.core import (
    Spell,
    backfill_ticker,
    load_membership,
    plan_window,
    run_backfill,
    usable_spells,
    to_db_ticker,
)
from trader_koo.price_backfill.store import seed_research_db
from trader_koo.scripts.update_market_db import (
    ensure_schema,
    reconcile_vendor_action_ledger,
    write_price_daily,
)

FLOOR = "2010-01-01"
EXISTING_START = "2018-01-02"
DIVIDEND_DATE = "2012-03-15"
SPLIT_DATE = "2014-05-12"


def _raw(start: str, end: str, *, base: float = 50.0) -> pd.DataFrame:
    index = pd.bdate_range(start, end, name="Date")
    close = pd.Series(
        [base + (day - pd.Timestamp('2008-01-01')).days * 0.002 for day in index], index=index
    )
    frame = pd.DataFrame(
        {
            "Open": close, "High": close + 1, "Low": close - 1, "Close": close,
            "Adj Close": close * 0.9, "Volume": 1000.0,
            "Dividends": 0.0, "Stock Splits": 0.0,
        },
        index=index,
    )
    if pd.Timestamp(DIVIDEND_DATE) in index:
        frame.loc[DIVIDEND_DATE, "Dividends"] = 0.25
    # Yahoo already rebased history: Adj Close/Close does not change across the split.
    if pd.Timestamp(SPLIT_DATE) in index:
        frame.loc[SPLIT_DATE, "Stock Splits"] = 2.0
    return frame


def _frame(start: str, end: str | None, *, avail: tuple[str, str]) -> pd.DataFrame:
    raw = _raw(*avail)
    mask = raw.index >= pd.Timestamp(start)
    if end:
        mask &= raw.index < pd.Timestamp(end)
    raw = raw[mask]
    if raw.empty:
        raise PriceFetchError("yfinance returned no data: Empty response")
    return DataSourceManager._normalize_ohlcv(raw.reset_index())


def make_fetch(avail: tuple[str, str] = ("2008-01-01", "2018-06-29")):
    calls: list[tuple[str, str, str | None]] = []

    def fetch(ticker: str, start: str, end: str | None):
        calls.append((ticker, start, end))
        return _frame(start, end, avail=avail), "yfinance"

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def split_ledger(_ticker: str) -> list[dict]:
    return [{"action_date": SPLIT_DATE, "action_type": "split", "value": 2.0}]


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    ensure_schema(connection)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS yolo_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL,
            pattern TEXT NOT NULL)"""
    )
    return connection


def seed_existing(conn: sqlite3.Connection, ticker: str = "AAA") -> None:
    df = _frame(EXISTING_START, None, avail=("2018-01-01", "2018-06-29"))
    reconcile_vendor_action_ledger(df, [], managed_start=EXISTING_START)
    write_price_daily(conn, ticker, df)
    conn.commit()


def existing_rows(conn: sqlite3.Connection, ticker: str = "AAA") -> list[tuple]:
    return conn.execute(
        "SELECT * FROM price_daily WHERE ticker=? AND date>=? ORDER BY date",
        (ticker, EXISTING_START),
    ).fetchall()


CURRENT = [Spell("2005-01-03", None)]


def run_one(conn, fetch, spells=CURRENT, ticker="AAA", **kw):
    kw.setdefault("actions_fn", split_ledger)
    kw.setdefault("probe_alive_fn", lambda t: False)
    return backfill_ticker(
        conn, ticker, spells, fetch_fn=fetch, floor=FLOOR, warmup_days=400, **kw
    )


# --- range selection ---------------------------------------------------------

def test_window_ends_at_existing_earliest_and_starts_at_floor() -> None:
    assert plan_window(CURRENT, EXISTING_START, floor=FLOOR, warmup_days=400) == (
        FLOOR, EXISTING_START,
    )


def test_window_nothing_to_do_when_history_already_reaches_floor() -> None:
    assert plan_window(CURRENT, "2010-01-04", floor=FLOOR, warmup_days=400) is None


def test_window_starts_one_warmup_before_a_late_joiner() -> None:
    start, _ = plan_window([Spell("2016-03-01", None)], EXISTING_START, floor=FLOOR, warmup_days=400)
    assert start == "2015-01-26"


def test_window_for_departed_ticker_stops_after_last_spell() -> None:
    start, end = plan_window([Spell("2011-01-03", "2014-06-30")], EXISTING_START, floor=FLOOR, warmup_days=400)
    assert end == "2014-07-11"
    assert start == FLOOR


def test_window_ignores_spells_that_ended_before_floor() -> None:
    assert plan_window([Spell("1999-01-01", "2009-06-01")], None, floor=FLOOR, warmup_days=400) is None


def test_spells_before_a_long_gap_are_dropped_unless_same_company() -> None:
    spells = [Spell("1996-01-02", "2013-10-29"), Spell("2024-09-23", None)]
    assert usable_spells(spells) == [Spell("2024-09-23", None)]
    assert usable_spells(spells, same_company=True) == spells
    short_gap = [Spell("2011-01-03", "2014-06-30"), Spell("2014-12-01", None)]
    assert usable_spells(short_gap) == short_gap


def test_recycled_current_symbol_gets_no_bars_from_the_old_holders_spell(conn) -> None:
    spells = [Spell("2000-01-03", "2011-04-01"), Spell("2025-11-03", None)]
    fetch = make_fetch()
    out = run_one(conn, fetch, spells=spells, ticker="NEWCO")
    assert out["dropped_spells"] == [["2000-01-03", "2011-04-01"]]
    assert fetch.calls[0][1] == "2024-09-29"


def test_ticker_symbols_map_to_price_daily_convention() -> None:
    assert to_db_ticker("BRK.B") == "BRK-B"
    assert to_db_ticker("BF.B") == "BF-B"
    assert to_db_ticker("AAPL") == "AAPL"


def test_membership_csv_keeps_only_spells_overlapping_floor(tmp_path: Path) -> None:
    csv = tmp_path / "m.csv"
    csv.write_text(
        "ticker,start_date,end_date\n"
        "OLD,1999-01-01,2009-12-31\n"
        "AAL,1996-01-02,1997-01-15\n"
        "AAL,2015-03-23,2024-09-23\n"
        "BRK.B,2010-02-16,\n"
    )
    dropped: list = []
    membership = load_membership(csv, floor=FLOOR, dropped_before_floor=dropped)
    assert set(membership) == {"AAL", "BRK-B"}
    assert membership["AAL"] == [Spell("2015-03-23", "2024-09-23")]
    assert dropped == [("OLD", "1999-01-01", "2009-12-31"), ("AAL", "1996-01-02", "1997-01-15")]  # reported, not lost


def test_renamed_symbol_spells_merge_under_the_new_symbol(tmp_path: Path) -> None:
    csv = tmp_path / "m.csv"
    csv.write_text("ticker,start_date,end_date\nFB,2013-12-23,2022-06-09\nMETA,2022-06-09,\n")
    membership = load_membership(csv, floor=FLOOR, dropped_before_floor=[])
    assert list(membership) == ["META"]
    assert membership["META"] == [Spell("2013-12-23", "2022-06-09"), Spell("2022-06-09", None)]


# --- safety ------------------------------------------------------------------

def test_adds_only_earlier_rows_and_leaves_existing_rows_identical(conn) -> None:
    seed_existing(conn)
    before = existing_rows(conn)
    out = run_one(conn, make_fetch())
    assert out["status"] == "ok"
    assert existing_rows(conn) == before
    earliest = conn.execute("SELECT MIN(date) FROM price_daily WHERE ticker='AAA'").fetchone()[0]
    assert earliest == "2010-01-01"
    assert out["rows_added"] == conn.execute(
        "SELECT COUNT(*) FROM price_daily WHERE ticker='AAA' AND date<?", (EXISTING_START,)
    ).fetchone()[0]


def test_fetch_never_requests_existing_dates(conn) -> None:
    seed_existing(conn)
    fetch = make_fetch()
    run_one(conn, fetch)
    assert fetch.calls == [("AAA", FLOOR, EXISTING_START)]


def test_backfilled_rows_carry_the_existing_price_basis(conn) -> None:
    seed_existing(conn)
    run_one(conn, make_fetch())
    bases = conn.execute(
        "SELECT DISTINCT adjustment_basis, adjustment_version, basis_status FROM price_daily WHERE ticker='AAA'"
    ).fetchall()
    assert len(bases) == 1
    assert bases[0][0] == "split_adjusted_price_only"
    assert bases[0][2] == "verified"


def test_corporate_actions_recorded_for_backfilled_range(conn) -> None:
    seed_existing(conn)
    run_one(conn, make_fetch())
    kinds = {
        row[0] for row in conn.execute(
            "SELECT action_type FROM price_corporate_actions WHERE ticker='AAA' AND action_date<?",
            (EXISTING_START,),
        )
    }
    assert kinds == {"dividend", "split"}


def test_second_run_adds_nothing(conn) -> None:
    seed_existing(conn)
    run_one(conn, make_fetch())
    snapshot = conn.execute("SELECT * FROM price_daily ORDER BY ticker, date").fetchall()
    actions = conn.execute("SELECT * FROM price_corporate_actions ORDER BY 1,2,3").fetchall()
    fetch = make_fetch()
    out = run_one(conn, fetch)
    assert out["status"] == "up_to_date"
    assert fetch.calls == []
    assert conn.execute("SELECT * FROM price_daily ORDER BY ticker, date").fetchall() == snapshot
    assert conn.execute("SELECT * FROM price_corporate_actions ORDER BY 1,2,3").fetchall() == actions


def test_row_that_would_change_existing_rows_is_rolled_back(conn) -> None:
    seed_existing(conn)
    conn.execute(
        "UPDATE price_daily SET basis_status='unresolved', unresolved_reason='x' WHERE ticker='AAA'"
    )
    conn.commit()
    before = conn.execute("SELECT * FROM price_daily ORDER BY ticker, date").fetchall()
    out = run_one(conn, make_fetch())
    assert out["status"] == "skipped_would_modify_existing_rows"
    assert conn.execute("SELECT * FROM price_daily ORDER BY ticker, date").fetchall() == before


def test_yolo_patterns_survive_the_revision_reseal(conn) -> None:
    seed_existing(conn)
    conn.execute("INSERT INTO yolo_patterns (ticker, pattern) VALUES ('AAA', 'triangle')")
    conn.commit()
    run_one(conn, make_fetch())
    assert conn.execute("SELECT ticker, pattern FROM yolo_patterns").fetchall() == [("AAA", "triangle")]


def test_other_tickers_untouched(conn) -> None:
    seed_existing(conn, "AAA")
    seed_existing(conn, "BBB")
    other = conn.execute("SELECT * FROM price_daily WHERE ticker='BBB' ORDER BY date").fetchall()
    run_one(conn, make_fetch())
    assert conn.execute("SELECT * FROM price_daily WHERE ticker='BBB' ORDER BY date").fetchall() == other


# --- recycled and unserved tickers --------------------------------------------

def test_recycled_symbol_is_skipped_and_nothing_written(conn) -> None:
    spells = [Spell("2011-01-03", "2014-06-30")]
    # The symbol trades today, but its only bars start after the S&P 500 spell ended.
    fetch = make_fetch(avail=("2016-01-04", "2018-06-29"))
    alive_calls: list[str] = []
    out = run_one(
        conn, fetch, spells=spells, ticker="OLDCO",
        probe_alive_fn=lambda t: alive_calls.append(t) or True,
    )
    assert out["status"] == "skipped_recycled_symbol"
    assert alive_calls == ["OLDCO"]
    assert conn.execute("SELECT COUNT(*) FROM price_daily").fetchone()[0] == 0


def test_series_starting_years_after_the_member_joined_is_another_holder(conn) -> None:
    seed_existing(conn)
    before = conn.execute("SELECT COUNT(*) FROM price_daily").fetchone()[0]
    out = run_one(conn, make_fetch(avail=("2016-01-04", "2018-06-29")), spells=[Spell("2010-11-17", None)])
    assert out["status"] == "skipped_history_starts_after_spell"
    assert conn.execute("SELECT COUNT(*) FROM price_daily").fetchone()[0] == before


def test_departed_ticker_never_receives_bars_after_its_last_spell(conn) -> None:
    spells = [Spell("2011-01-03", "2014-06-30")]
    out = run_one(conn, make_fetch(avail=("2008-01-01", "2018-06-29")), spells=spells, ticker="GONE")
    assert out["status"] == "ok"
    latest = conn.execute("SELECT MAX(date) FROM price_daily WHERE ticker='GONE'").fetchone()[0]
    assert latest <= "2014-07-11"


def test_unserved_departed_ticker_is_reported_not_served(conn) -> None:
    spells = [Spell("2011-01-03", "2014-06-30")]
    out = run_one(
        conn, make_fetch(avail=("2030-01-01", "2030-02-01")), spells=spells,
        ticker="DEAD", probe_alive_fn=lambda t: False,
    )
    assert out["status"] == "no_data"
    assert conn.execute("SELECT COUNT(*) FROM price_daily").fetchone()[0] == 0


# --- resumability and backoff ---------------------------------------------------

def test_transient_error_is_retried_with_backoff(conn) -> None:
    seed_existing(conn)
    inner = make_fetch()
    attempts = {"n": 0}
    sleeps: list[float] = []

    def flaky(ticker, start, end):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PriceFetchError("yfinance returned no data: Too Many Requests. Rate limited")
        return inner(ticker, start, end)

    out = run_one(conn, flaky, retries=3, backoff_sec=2.0, sleep_fn=sleeps.append)
    assert out["status"] == "ok"
    assert sleeps == [2.0, 4.0]


def test_persistent_error_is_reported_and_not_terminal(conn) -> None:
    seed_existing(conn)

    def broken(ticker, start, end):
        raise PriceFetchError("yfinance returned no data: Too Many Requests. Rate limited")

    out = run_one(conn, broken, retries=1, backoff_sec=0.0, sleep_fn=lambda s: None)
    assert out["status"] == "error"


def test_run_resumes_from_state_file(conn, tmp_path: Path) -> None:
    seed_existing(conn, "AAA")
    seed_existing(conn, "BBB")
    membership = {"AAA": CURRENT, "BBB": CURRENT}
    state = tmp_path / "state.json"
    fetch = make_fetch()
    common = dict(
        fetch_fn=fetch, actions_fn=split_ledger, probe_alive_fn=lambda t: False,
        floor=FLOOR, warmup_days=400, sleep_fn=lambda s: None, pace_sec=0.0, state_path=state,
    )
    first = run_backfill(conn, membership, limit=1, **common)
    assert [c[0] for c in fetch.calls] == ["AAA"]
    assert [(r["ticker"], r["status"]) for r in first] == [("AAA", "ok"), ("BBB", "not_attempted_limit")]
    second = run_backfill(conn, membership, **common)
    assert [c[0] for c in fetch.calls] == ["AAA", "BBB"]
    assert [(r["ticker"], r["status"]) for r in second] == [("AAA", "skipped_settled_in_state"), ("BBB", "ok")]


# --- research database seeding ----------------------------------------------------

def _file_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    ensure_schema(connection)
    return connection


def test_seed_copies_price_rows_and_never_overwrites(tmp_path: Path) -> None:
    source = _file_db(tmp_path / "source.db")
    seed_existing(source, "AAA")
    source.close()
    research = _file_db(tmp_path / "research.db")
    report = seed_research_db(research, tmp_path / "source.db")
    assert report["price_daily"]["copied"] > 100
    assert report["price_daily"]["copied"] == report["price_daily"]["source_rows"]
    research.execute("UPDATE price_daily SET close=999 WHERE ticker='AAA' AND date=?", (EXISTING_START,))
    research.commit()
    again = seed_research_db(research, tmp_path / "source.db")["price_daily"]
    assert again["copied"] == 0 and again["ignored_existing"] == again["source_rows"]  # existing rows win, counted
    assert research.execute(
        "SELECT close FROM price_daily WHERE ticker='AAA' AND date=?", (EXISTING_START,)
    ).fetchone()[0] == 999


def test_source_database_is_opened_read_only(tmp_path: Path) -> None:
    source = _file_db(tmp_path / "source.db")
    seed_existing(source, "AAA")
    source.close()
    digest = (tmp_path / "source.db").read_bytes()
    research = _file_db(tmp_path / "research.db")
    seed_research_db(research, tmp_path / "source.db")
    run_one(research, make_fetch())
    assert (tmp_path / "source.db").read_bytes() == digest
    assert research.execute("SELECT MIN(date) FROM price_daily WHERE ticker='AAA'").fetchone()[0] == "2010-01-01"


def test_context_symbols_are_backfilled_without_touching_existing_rows(conn) -> None:
    from trader_koo.price_backfill.core import add_context_symbols

    seed_existing(conn, "SPY")
    before = existing_rows(conn, "SPY")
    already: list = []
    membership = add_context_symbols({"AAA": CURRENT}, ["spy", "AAA"], floor=FLOOR, already_members=already)
    assert membership["SPY"] == [Spell(FLOOR, None)] and membership["AAA"] == CURRENT
    assert already == ["AAA"]
    with pytest.raises(ValueError):  # a blank symbol is an input error, not skipped
        add_context_symbols({}, [" "], floor=FLOOR, already_members=[])
    out = run_one(conn, make_fetch(), spells=membership["SPY"], ticker="SPY")
    assert out["status"] == "ok" and out["rows_added"] > 0
    assert out["earliest_after"] < EXISTING_START
    assert existing_rows(conn, "SPY") == before

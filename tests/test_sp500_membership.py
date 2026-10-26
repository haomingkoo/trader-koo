from __future__ import annotations

import datetime as dt
import sqlite3

from trader_koo.db import membership as m
from trader_koo.db import membership_check as chk


def _db(rows, source="t", version="v1"):
    conn = sqlite3.connect(":memory:")
    m.replace_membership(conn, m.build_intervals(rows), source, version)
    return conn


def test_schema_is_idempotent_and_additive():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE paper_trades (id INTEGER)")
    before = conn.execute("SELECT sql FROM sqlite_master WHERE name='paper_trades'").fetchone()
    m.ensure_membership_schema(conn)
    m.ensure_membership_schema(conn)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == {"paper_trades", "sp500_membership"}
    assert conn.execute("SELECT sql FROM sqlite_master WHERE name='paper_trades'").fetchone() == before


def test_reload_replaces_instead_of_duplicating():
    rows = [("AAA", "2019-01-01", None)]
    conn = _db(rows)
    m.replace_membership(conn, m.build_intervals(rows), "t", "v2")
    assert conn.execute("SELECT COUNT(*), MIN(source_version) FROM sp500_membership").fetchone() == (1, "v2")


def test_interval_is_half_open():
    conn = _db([("OLD", "2019-01-01", "2020-03-02"), ("NEW", "2020-03-02", None)])
    assert m.members_on(conn, "2020-03-01") == {"OLD"}
    assert m.members_on(conn, "2020-03-02") == {"NEW"}
    assert m.members_on(conn, "2018-12-31") == set()
    assert m.is_member(conn, "OLD", "2019-01-01")
    assert not m.is_member(conn, "OLD", "2020-03-02")


def test_ticker_format_matches_price_daily():
    conn = _db([("BRK.B", "2010-02-16", None)])
    assert m.members_on(conn, "2024-01-02") == {"BRK-B"}
    assert m.is_member(conn, "BRK.B", "2024-01-02")
    assert m.is_member(conn, "BRK-B", "2024-01-02")


def test_rename_is_folded_and_merged():
    conn = _db([("FB", "2013-12-23", "2022-06-09"), ("META", "2022-06-09", None)])
    rows = conn.execute("SELECT ticker, start_date, end_date FROM sp500_membership").fetchall()
    assert rows == [("META", "2013-12-23", None)]
    assert m.is_member(conn, "FB", "2020-01-02")


def test_recycled_ticker_with_gap_stays_separate():
    conn = _db([("AAL", "1996-01-02", "1997-01-15"), ("AAL", "2015-03-23", "2024-09-23")])
    assert conn.execute("SELECT COUNT(*) FROM sp500_membership").fetchone()[0] == 2
    assert not m.is_member(conn, "AAL", "2000-01-03")


def test_month_ends_cap_at_as_of():
    out = chk.month_ends(2024, dt.date(2024, 3, 10))
    assert out == ["2024-01-31", "2024-02-29", "2024-03-10"]


def test_month_end_counts_flag_out_of_range():
    rows = [(f"T{i}", "2024-01-01", None) for i in range(496)]
    conn = _db(rows)
    out = chk.month_end_counts(conn, dt.date(2024, 2, 29), 2024)
    assert [r["ok"] for r in out] == [True, True]
    assert chk.month_end_counts(_db(rows[:10]), dt.date(2024, 1, 31), 2024)[0]["ok"] is False


def test_compare_current_reports_both_directions():
    conn = _db([("AAA", "2020-01-01", None), ("BBB", "2020-01-01", None)])
    diff = chk.compare_current(conn, dt.date(2024, 1, 2), {"BBB", "CCC"})
    assert diff == {"only_in_db": ["AAA"], "only_in_wiki": ["CCC"]}


def test_compare_events_tolerance_mismatch_and_missing():
    conn = _db([
        ("AAA", "2020-01-02", None),
        ("BBB", "2019-06-01", "2020-05-05"),
        ("DDD", "2021-01-01", None),
    ])
    wiki = [
        {"kind": "add", "ticker": "AAA", "date": "2020-01-06"},  # within 5 days
        {"kind": "remove", "ticker": "BBB", "date": "2020-05-20"},  # 15 days off
        {"kind": "add", "ticker": "CCC", "date": "2020-07-01"},  # missing from db
    ]
    out = chk.compare_events(conn, wiki, since="2020-01-01")
    got = {(p["issue"], p["ticker"], p["kind"]) for p in out}
    assert got == {
        ("date_mismatch", "BBB", "remove"),
        ("wiki_only", "CCC", "add"),
        ("db_only", "DDD", "add"),
    }


def test_wiki_rename_pair_is_cancelled():
    conn = _db([("CDAY", "2021-09-20", "2024-02-01"), ("DAY", "2024-02-01", None)])
    wiki = [
        {"kind": "add", "ticker": "DAY", "date": "2024-02-01"},
        {"kind": "remove", "ticker": "DAY", "date": "2024-02-01"},  # CDAY normalised to DAY
    ]
    out = chk.compare_events(conn, wiki, since="2020-01-01")
    assert [p for p in out if p["ticker"] == "DAY" and p["issue"] == "wiki_only"] == []


WIKI_HTML = """
<table id="constituents"><tr><th>Symbol</th><th>Security</th></tr>
<tr><td>BRK.B</td><td>Berkshire</td></tr><tr><td>AAA</td><td>A</td></tr></table>
<table id="changes">
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th><th rowspan="2">Refs</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>March 2, 2020</td><td>NEW</td><td>New Co</td><td>OLD</td><td>Old Co</td><td>x</td><td>[1]</td></tr>
<tr><td>April 3, 2020</td><td></td><td></td><td>GONE</td><td>Gone Co</td><td>y</td><td>[2]</td></tr>
</table>
"""


def test_parse_wiki_tables():
    assert chk.parse_wiki_constituents(WIKI_HTML) == {"BRK-B", "AAA"}
    events = chk.parse_wiki_changes(WIKI_HTML)
    assert {(e["kind"], e["ticker"], e["date"]) for e in events} == {
        ("add", "NEW", "2020-03-02"),
        ("remove", "OLD", "2020-03-02"),
        ("remove", "GONE", "2020-04-03"),
    }


def test_coverage_and_survivorship_gap():
    conn = _db([("HAS", "2024-01-01", None), ("GONE", "2024-01-01", "2024-01-03")])
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT)")
    conn.executemany(
        "INSERT INTO price_daily VALUES (?, ?)",
        [("HAS", "2024-01-02"), ("HAS", "2024-01-03"), ("SPY", "2024-01-02")],
    )
    cov = chk.coverage_by_year(conn, "2024-01-01", "2024-12-31")
    # days: 01-02 (HAS, GONE are members), 01-03 (HAS only)
    assert cov == [{"year": "2024", "member_days": 3, "with_price": 2}]
    gap = chk.survivorship_gap(conn, "2024-01-01", "2024-12-31")
    assert gap == [{"ticker": "GONE", "member_days": 1, "last_end": "2024-01-03"}]

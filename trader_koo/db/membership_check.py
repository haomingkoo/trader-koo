"""Cross-checks for the point-in-time S&P 500 membership table.

Every function reports; none repairs. Disagreements go to a report so a human
decides, instead of being silently patched.
"""
from __future__ import annotations

import calendar
import datetime as dt
import io
import sqlite3
from typing import Any

import pandas as pd

from trader_koo.db.membership import members_on, normalize_ticker

MEMBERSHIP_COUNT_MIN = 495  # members; the index holds ~500 names, fewer means missing rows
MEMBERSHIP_COUNT_MAX = 510  # members; more means overlapping or unclosed intervals
MEMBERSHIP_DATE_TOLERANCE_DAYS = 5  # days; Wikipedia and the CSV may differ by effective vs announced date
MEMBERSHIP_FIRST_YEAR = 2018  # calendar year; start of the price history the bench uses
FETCH_TIMEOUT_SEC = 60  # seconds; bound on each download so the CLI cannot hang

Event = dict[str, str]  # {"kind": "add"|"remove", "ticker": ..., "date": ISO}


def month_ends(first_year: int, as_of: dt.date) -> list[str]:
    """Last calendar day of every month from January ``first_year`` up to ``as_of``."""
    out: list[str] = []
    day = dt.date(first_year, 1, 1)
    while day <= as_of:
        last = day.replace(day=calendar.monthrange(day.year, day.month)[1])
        out.append(min(last, as_of).isoformat())
        day = last + dt.timedelta(days=1)
    return out


def month_end_counts(
    conn: sqlite3.Connection, as_of: dt.date, first_year: int = MEMBERSHIP_FIRST_YEAR
) -> list[dict[str, Any]]:
    rows = []
    for day in month_ends(first_year, as_of):
        n = len(members_on(conn, day))
        rows.append({
            "date": day, "count": n,
            "ok": MEMBERSHIP_COUNT_MIN <= n <= MEMBERSHIP_COUNT_MAX,
        })
    return rows


def compare_current(
    conn: sqlite3.Connection, as_of: dt.date, wiki_current: set[str]
) -> dict[str, list[str]]:
    ours = members_on(conn, as_of.isoformat())
    wiki = {normalize_ticker(t) for t in wiki_current}
    return {"only_in_db": sorted(ours - wiki), "only_in_wiki": sorted(wiki - ours)}


def parse_wiki_constituents(html: str) -> set[str]:
    table = pd.read_html(io.StringIO(html), attrs={"id": "constituents"}, flavor="lxml")[0]
    return {normalize_ticker(s) for s in table["Symbol"].astype(str)}


def parse_wiki_changes(html: str) -> list[Event]:
    """Flatten the 'Selected changes' table into add/remove events."""
    table = pd.read_html(io.StringIO(html), attrs={"id": "changes"}, flavor="lxml")[0]
    table.columns = ["date", "add_t", "add_s", "rem_t", "rem_s", "reason", "refs"]
    events: list[Event] = []
    for row in table.itertuples(index=False):
        when = pd.to_datetime(row.date, format="%B %d, %Y", errors="coerce")
        if pd.isna(when):
            continue
        for kind, ticker in (("add", row.add_t), ("remove", row.rem_t)):
            if isinstance(ticker, str) and ticker.strip():
                events.append({
                    "kind": kind, "ticker": normalize_ticker(ticker),
                    "date": when.date().isoformat(),
                })
    return events


def _gap(a: str, b: str) -> int:
    return abs((dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days)


def _cancel_renames(events: list[Event]) -> list[Event]:
    """Drop adds and removes of the same (normalised) ticker within tolerance.

    A rename (or a merger of two share classes into one symbol) shows up as a
    remove and an add of the same symbol once tickers are normalised.
    """
    def paired(event: Event) -> bool:
        other = "remove" if event["kind"] == "add" else "add"
        return any(
            o["kind"] == other and o["ticker"] == event["ticker"]
            and _gap(o["date"], event["date"]) <= MEMBERSHIP_DATE_TOLERANCE_DAYS
            for o in events
        )

    return [e for e in events if not paired(e)]


def _table_events(conn: sqlite3.Connection, since: str) -> list[Event]:
    out: list[Event] = []
    for ticker, start, end in conn.execute("SELECT ticker, start_date, end_date FROM sp500_membership"):
        if start >= since:
            out.append({"kind": "add", "ticker": ticker, "date": start})
        if end is not None and end >= since:
            out.append({"kind": "remove", "ticker": ticker, "date": end})
    return out


def compare_events(
    conn: sqlite3.Connection, wiki_events: list[Event], since: str
) -> list[dict[str, str]]:
    """List disagreements between stored membership events and Wikipedia's.

    ``date_mismatch``: both sides have that ticker/kind but no pair within the
    tolerance. ``wiki_only`` / ``db_only``: one side has no matching event.
    """
    ours = _table_events(conn, since)
    wiki = _cancel_renames([e for e in wiki_events if e["date"] >= since])
    problems: list[dict[str, str]] = []

    def same_key(event: Event, pool: list[Event]) -> list[Event]:
        return [p for p in pool if p["kind"] == event["kind"] and p["ticker"] == event["ticker"]]

    def agrees(event: Event, same: list[Event]) -> bool:
        return any(_gap(event["date"], p["date"]) <= MEMBERSHIP_DATE_TOLERANCE_DAYS for p in same)

    for event in wiki:
        same = same_key(event, ours)
        if agrees(event, same):
            continue
        if same:
            closest = min(same, key=lambda p: _gap(event["date"], p["date"]))
            problems.append({**event, "issue": "date_mismatch", "other_date": closest["date"]})
        else:
            problems.append({**event, "issue": "wiki_only", "other_date": ""})
    for event in ours:
        same = same_key(event, wiki)
        if not same:
            problems.append({**event, "issue": "db_only", "other_date": ""})
    return problems


COVERAGE_SQL = """
WITH days AS (SELECT DISTINCT date FROM {px} WHERE date >= :since AND date <= :until),
member_days AS (
    SELECT d.date AS date, m.ticker AS ticker FROM days d
    JOIN sp500_membership m ON m.start_date <= d.date AND (m.end_date IS NULL OR d.date < m.end_date)
)
SELECT substr(md.date, 1, 4) AS year, COUNT(*) AS member_days,
       SUM(CASE WHEN p.ticker IS NULL THEN 0 ELSE 1 END) AS with_price
FROM member_days md LEFT JOIN {px} p ON p.ticker = md.ticker AND p.date = md.date
GROUP BY year ORDER BY year
"""

GAP_SQL = """
WITH days AS (SELECT DISTINCT date FROM {px} WHERE date >= :since AND date <= :until),
member_days AS (
    SELECT d.date AS date, m.ticker AS ticker FROM days d
    JOIN sp500_membership m ON m.start_date <= d.date AND (m.end_date IS NULL OR d.date < m.end_date)
)
SELECT md.ticker AS ticker, COUNT(*) AS member_days,
       SUM(CASE WHEN p.ticker IS NULL THEN 0 ELSE 1 END) AS with_price,
       (SELECT MAX(end_date) FROM sp500_membership s WHERE s.ticker = md.ticker) AS last_end
FROM member_days md LEFT JOIN {px} p ON p.ticker = md.ticker AND p.date = md.date
GROUP BY md.ticker
HAVING with_price = 0
ORDER BY md.ticker
"""


def coverage_by_year(
    conn: sqlite3.Connection, since: str, until: str, price_table: str = "price_daily"
) -> list[dict[str, Any]]:
    """Member-days (membership x trading days seen in the price table) with a price row."""
    cur = conn.execute(COVERAGE_SQL.format(px=price_table), {"since": since, "until": until})
    return [{"year": y, "member_days": m, "with_price": w or 0} for y, m, w in cur]


def survivorship_gap(
    conn: sqlite3.Connection, since: str, until: str, price_table: str = "price_daily"
) -> list[dict[str, Any]]:
    """Tickers that were members in the window but have no price row on any member-day.

    ``last_end`` is NULL for current members (a gap for them is a data hole, not
    a departure).
    """
    cur = conn.execute(GAP_SQL.format(px=price_table), {"since": since, "until": until})
    return [{"ticker": t, "member_days": m, "last_end": e or ""} for t, m, _, e in cur]

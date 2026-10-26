"""Point-in-time accessors over sec_fundamentals.

Every read filters on ``filed_date <= as_of_date``: a value is knowable only
from the day its filing was accepted. Restated values carry their own
accession and filed_date, so a restatement is invisible before that date.

TTM rule (flow items). Companies file 10-Qs with cumulative year-to-date
periods and a 10-K with the full year; Q4 is never filed on its own. Using only
facts filed by ``as_of_date``:

    TTM = FY(last full year) + YTD(current) - YTD(same span, prior year)

where YTD(current) starts the day after the last fiscal year end and ends at
the newest reported quarter. Algebraically this equals the sum of the latest
four discrete quarters, with Q4 = FY - (Q1+Q2+Q3) implicit. If the newest
filing is the 10-K, TTM is that fiscal year. Any missing component returns
None; nothing is estimated. EPS is summed the same way, which is approximate
when share counts moved.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from typing import Sequence

from trader_koo.fundamentals import config as cfg

_DAYS = "julianday(period_end) - julianday(period_start)"
_KIND_SQL = {
    "instant": "period_start = ''",
    "quarter": f"period_start <> '' AND {_DAYS} BETWEEN {cfg.QUARTER_DAYS[0]} AND {cfg.QUARTER_DAYS[1]}",
    "ytd": f"period_start <> '' AND {_DAYS} BETWEEN {cfg.YTD_DAYS[0]} AND {cfg.YTD_DAYS[1]}",
    "annual": f"period_start <> '' AND {_DAYS} BETWEEN {cfg.ANNUAL_DAYS[0]} AND {cfg.ANNUAL_DAYS[1]}",
}
_DAYS_PER_YEAR = cfg.DAYS_PER_YEAR


@dataclass(frozen=True)
class Fact:
    tag: str
    value: float
    period_start: str
    period_end: str
    filed_date: str
    form: str | None
    accession: str


def _tags(tag: str | Sequence[str]) -> list[str]:
    if isinstance(tag, str):
        return [t for _, t in cfg.CONCEPTS[tag]] if tag in cfg.CONCEPTS else [tag]
    return list(tag)


def _one(conn: sqlite3.Connection, ticker: str, as_of: str, tag: str, kind: str | None) -> Fact | None:
    where = f" AND {_KIND_SQL[kind]}" if kind else ""
    row = conn.execute(
        f"""SELECT f.tag, f.value, f.period_start, f.period_end, f.filed_date, f.form, f.accession
            FROM sec_fundamentals f JOIN sec_cik_map m ON m.cik = f.cik
            WHERE m.ticker = ? AND f.tag = ? AND f.filed_date <= ?{where}
            ORDER BY f.period_end DESC, f.filed_date DESC, f.accession DESC LIMIT 1""",
        (ticker.upper(), tag, as_of),
    ).fetchone()
    return Fact(*row) if row else None


def fundamentals_as_of(
    conn: sqlite3.Connection,
    ticker: str,
    as_of_date: str,
    tag: str | Sequence[str],
    kind: str | None = None,
) -> Fact | None:
    """Latest fact (by period_end, then by filing) with filed_date <= as_of_date.

    ``tag`` is a tag name, a concept name from config.CONCEPTS (candidates are
    compared and the newest period wins, earlier priority breaking ties), or a
    sequence of tags. ``kind`` limits to instant/quarter/ytd/annual periods.
    """
    best: Fact | None = None
    for candidate in _tags(tag):
        fact = _one(conn, ticker, as_of_date, candidate, kind)
        if fact and (best is None or fact.period_end > best.period_end):
            best = fact
    return best


def _day(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def _known_durations(conn: sqlite3.Connection, ticker: str, as_of: str, tag: str) -> list[Fact]:
    rows = conn.execute(
        """SELECT f.tag, f.value, f.period_start, f.period_end, f.filed_date, f.form, f.accession
           FROM sec_fundamentals f JOIN sec_cik_map m ON m.cik = f.cik
           WHERE m.ticker = ? AND f.tag = ? AND f.filed_date <= ? AND f.period_start <> ''
           ORDER BY f.filed_date, f.accession""",
        (ticker.upper(), tag, as_of),
    ).fetchall()
    latest: dict[tuple[str, str], Fact] = {}
    for r in rows:  # later filings overwrite earlier ones for the same period
        fact = Fact(*r)
        latest[(fact.period_start, fact.period_end)] = fact
    return list(latest.values())


def _span(f: Fact) -> int:
    return (_day(f.period_end) - _day(f.period_start)).days


def _near(a: str, b: dt.date) -> bool:
    return abs((_day(a) - b).days) <= cfg.MATCH_TOLERANCE_DAYS


def _ttm_one(conn: sqlite3.Connection, ticker: str, as_of: str, tag: str,
             period_end_max: str | None = None) -> Fact | None:
    facts = _known_durations(conn, ticker, as_of, tag)
    if period_end_max is not None:
        facts = [f for f in facts if f.period_end <= period_end_max]
    annual = [f for f in facts if cfg.ANNUAL_DAYS[0] <= _span(f) <= cfg.ANNUAL_DAYS[1]]
    if not annual:
        return None
    fy = max(annual, key=lambda f: f.period_end)
    partial = [f for f in facts if cfg.QUARTER_DAYS[0] <= _span(f) <= cfg.YTD_DAYS[1] and f.period_end > fy.period_end]
    if not partial:
        return fy
    newest = max(f.period_end for f in partial)
    fy_next_start = _day(fy.period_end) + dt.timedelta(days=1)
    current = [f for f in partial if f.period_end == newest and _near(f.period_start, fy_next_start)]
    if not current:
        return None
    cur = max(current, key=_span)
    prior_start = _day(cur.period_start) - dt.timedelta(days=_DAYS_PER_YEAR)
    prior_end = _day(cur.period_end) - dt.timedelta(days=_DAYS_PER_YEAR)
    prior = [f for f in facts if _near(f.period_start, prior_start) and _near(f.period_end, prior_end)]
    if not prior:
        return None
    pri = min(prior, key=lambda f: abs(_span(f) - _span(cur)))
    return Fact(
        tag, fy.value + cur.value - pri.value, pri.period_start, cur.period_end,
        max(fy.filed_date, cur.filed_date, pri.filed_date), cur.form, cur.accession,
    )


def ttm_as_of(
    conn: sqlite3.Connection, ticker: str, as_of_date: str, tag: str | Sequence[str],
    period_end_max: str | None = None,
) -> Fact | None:
    """Trailing-twelve-month flow value knowable at as_of_date (rule in module doc).

    ``period_end_max`` asks for the TTM ending at or before that date, using only
    facts filed by ``as_of_date``: the TTM of an earlier year as the filings known
    now report it (restatements and split-adjusted comparatives included).
    """
    best: Fact | None = None
    for candidate in _tags(tag):
        fact = _ttm_one(conn, ticker, as_of_date, candidate, period_end_max)
        if fact and (best is None or fact.period_end > best.period_end):
            best = fact
    return best

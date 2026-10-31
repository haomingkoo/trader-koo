"""Code 33 (variant H1a_code33 of docs/research/hypotheses-v9-champions.md), point in time by filing date.

Only facts filed before the signal session's date are read (the bench's filing cutoff), with the latest value per
period as known then. A discrete quarter is a duration fact of a quarter's length, or the difference of two
cumulative facts with the same start whose ends are one quarter apart (Q4 = the year minus nine months).
"""
from __future__ import annotations

import datetime as dt
import sqlite3

from trader_koo.fundamentals import config as cfg
from trader_koo.fundamentals.pit import _known_durations

from .data import filing_cutoff

EPS = ("EarningsPerShareDiluted",)
REVENUE = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
           "RevenueFromContractWithCustomerIncludingAssessedTax")
NET_INCOME = ("NetIncomeLoss",)
QUARTERS_NEEDED = 8
YOY = 4
ACCELERATIONS = 3                 # g0 > g1 > g2 > g3
YEAR_DAYS = (cfg.ANNUAL_DAYS[0], cfg.ANNUAL_DAYS[1])


def _days(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def _quarter_gap(a: str, b: str) -> bool:
    return cfg.QUARTER_DAYS[0] <= _days(a, b) <= cfg.QUARTER_DAYS[1]


def quarters(conn: sqlite3.Connection, ticker: str, tag: str, as_of: str) -> dict[str, float]:
    """Discrete quarterly values by period end, from facts filed before ``as_of``."""
    facts = _known_durations(conn, ticker, filing_cutoff(as_of), tag)
    out = {f.period_end: f.value for f in facts if _quarter_gap(f.period_start, f.period_end)}
    by_start: dict[str, list] = {}
    for f in facts:
        by_start.setdefault(f.period_start, []).append(f)
    for group in by_start.values():
        group.sort(key=lambda f: f.period_end)
        for a, b in zip(group, group[1:]):
            if b.period_end not in out and _quarter_gap(a.period_end, b.period_end):
                out[b.period_end] = b.value - a.value
    return out


def _first_of(conn, ticker: str, tags: tuple[str, ...], as_of: str) -> dict[str, float]:
    merged: dict[str, float] = {}
    for tag in tags:
        for end, value in quarters(conn, ticker, tag, as_of).items():
            merged.setdefault(end, value)
    return merged


def _accelerating(x: list[float]) -> bool:
    return all(a > b for a, b in zip(x, x[1:]))


def code33(conn: sqlite3.Connection, ticker: str, as_of: str) -> bool:
    """Three straight accelerations of YoY EPS growth, YoY revenue growth and net margin at the latest quarter."""
    eps, rev, ni = (_first_of(conn, ticker, t, as_of) for t in (EPS, REVENUE, NET_INCOME))
    ends = sorted(set(eps) & set(rev), reverse=True)[:QUARTERS_NEEDED]
    if not eps or not ends or ends[0] != max(eps):  # q0 must be the latest quarter, not an older one with revenue
        return False
    if len(ends) < QUARTERS_NEEDED or not all(_quarter_gap(b, a) for a, b in zip(ends, ends[1:])):
        return False
    growth = {}
    for name, x in (("eps", eps), ("rev", rev)):
        g = []
        for k in range(ACCELERATIONS + 1):
            base = x[ends[k + YOY]]
            if base == 0 or not YEAR_DAYS[0] <= _days(ends[k + YOY], ends[k]) <= YEAR_DAYS[1]:
                return False
            g.append((x[ends[k]] - base) / abs(base))
        growth[name] = g
    if any(ends[k] not in ni or rev[ends[k]] == 0 for k in range(ACCELERATIONS + 1)):
        return False
    margin = [ni[ends[k]] / rev[ends[k]] for k in range(ACCELERATIONS + 1)]
    return _accelerating(growth["eps"]) and _accelerating(growth["rev"]) and _accelerating(margin)

"""Signal interface, reference candidates and the look-ahead guard.

A signal is ``signal(as_of_date, universe, data) -> pd.Series`` of scores per
ticker (higher is better, NaN means unscorable). ``data`` is an ``AsOfView``: it
holds prices up to ``as_of_date`` and SEC facts filed before it (usable from the next session). A signal
may use nothing else. ``assert_no_lookahead`` re-scores after corrupting every
price after ``as_of_date`` and every SEC fact filed on or after it; any change in the
scores is look-ahead.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import hypotheses
from .data import AsOfView, Market

DEFAULT_LOOKBACK = 252
DEFAULT_SKIP = 21
NOISE_RANGE = (0.2, 5.0)

Signal = Callable[[str, list[str], AsOfView], pd.Series]


class LookaheadError(AssertionError):
    pass


def equal_weight(params: dict) -> Signal:
    """Plumbing check: every name scores the same, so the portfolio is the equal-weight universe."""
    return lambda as_of, universe, data: pd.Series(1.0, index=universe)


def momentum_12_1(params: dict) -> Signal:
    """Total return from ``lookback`` sessions ago to ``skip`` sessions ago (default 252 and 21)."""
    lookback, skip = int(params.get("lookback", DEFAULT_LOOKBACK)), int(params.get("skip", DEFAULT_SKIP))

    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        px = data.close.reindex(columns=universe)
        if len(px) <= lookback:
            return pd.Series(np.nan, index=universe)
        return px.iloc[-1 - skip] / px.iloc[-1 - lookback] - 1.0

    return signal


CANDIDATES: dict[str, Callable[[dict], Signal]] = {
    "equal_weight": equal_weight,
    "momentum_12_1": momentum_12_1,
    "low_volatility": hypotheses.low_volatility,
    "h1_quality_fair_price": hypotheses.quality_at_fair_price,
    "h2_accelerating_fundamentals": hypotheses.accelerating_fundamentals,
    "h3_momentum_confirmed": hypotheses.momentum_confirmed_by_earnings,
    "h4_market_cap_leaders": hypotheses.market_cap_leaders,
}
# Candidates that rebalance on the last session of each calendar year instead of every N sessions.
ANNUAL_CANDIDATES = frozenset({"h4_market_cap_leaders"})


def make_signal(name: str, params: dict) -> Signal:
    if name not in CANDIDATES:
        raise KeyError(f"unknown candidate {name!r}; known: {sorted(CANDIDATES)}")
    return CANDIDATES[name](params)


def _main_db_file(conn: sqlite3.Connection) -> str | None:
    for _, name, path in conn.execute("PRAGMA database_list"):
        if name == "main":
            return path or None
    return None


def _view_sql(conn: sqlite3.Connection, table: str, as_of: str) -> str:
    """SELECT over ``src.<table>`` with the later-dated numeric columns corrupted."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if table == "price_daily":
        scale = {c: f"{c} * 3.0" for c in ("open", "high", "low", "close", "adj_close") if c in cols}
        when = f"date > '{as_of}'"
    elif table == "sec_fundamentals":
        scale = {"value": "value * 7.0 + 13.0"}
        when = f"filed_date >= '{as_of}'"  # a filing dated as_of is not usable until the next session
    else:
        return f"SELECT * FROM src.{table}"
    sel = ", ".join(f"CASE WHEN {when} THEN {scale[c]} ELSE {c} END AS {c}" if c in scale else c for c in cols)
    return f"SELECT {sel} FROM src.{table}"


def _perturbed_conn(conn: sqlite3.Connection, as_of: str) -> sqlite3.Connection:
    """Read-only window on the database in which later prices and later-filed SEC facts are corrupted.

    Nothing is copied: the database is attached read-only and every table is
    re-exposed as a TEMP view of the same name, with the perturbation inside the
    view for price_daily and sec_fundamentals. This works on a multi-GB file.
    An in-memory source (tests) falls back to a backup copy.
    """
    dt.date.fromisoformat(as_of)  # the date is inlined into view SQL
    path = _main_db_file(conn)
    if path is None:  # in-memory source: no file to attach, so corrupt a backup copy in place
        mem = sqlite3.connect(":memory:")
        conn.backup(mem)
        for (name,) in mem.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if name == "sec_fundamentals":
                mem.execute("UPDATE sec_fundamentals SET value = value * 7.0 + 13.0 WHERE filed_date >= ?", (as_of,))
            elif name == "price_daily":
                mem.execute("UPDATE price_daily SET open = open * 3.0, close = close * 3.0 WHERE date > ?", (as_of,))
        return mem
    out = sqlite3.connect(":memory:", uri=True)
    out.execute("ATTACH DATABASE ? AS src", (f"file:{path}?mode=ro",))
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        out.execute(f'CREATE TEMP VIEW "{name}" AS {_view_sql(conn, name, as_of)}')
    return out


def assert_no_lookahead(signal: Signal, market: Market, as_of: str, universe: list[str],
                        seed: int = 0) -> None:
    """Raise LookaheadError if scores change when data after ``as_of`` is corrupted.

    The corruption is done by views over a read-only attach, so it costs no
    memory proportional to the database and works on the full research file.
    """
    base = signal(as_of, universe, market.view(as_of))
    rng = np.random.default_rng(seed)
    later = market.adj_close.index > as_of
    close, open_ = market.adj_close.copy(), market.adj_open.copy()
    noise = rng.uniform(*NOISE_RANGE, size=(int(later.sum()), close.shape[1]))
    close.loc[later] = close.loc[later].to_numpy() * noise
    open_.loc[later] = open_.loc[later].to_numpy() * noise
    mem = _perturbed_conn(market.conn, as_of)
    fake = Market(open_, close, mem, price_tickers=market.price_tickers, splits=market.splits)
    try:
        pert = signal(as_of, universe, fake.view(as_of))
    finally:
        mem.close()
    a, b = base.sort_index(), pert.sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True):
        raise LookaheadError(f"signal output on {as_of} changed when later data was perturbed")

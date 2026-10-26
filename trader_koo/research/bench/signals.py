"""Signal interface, reference candidates and the look-ahead guard.

A signal is ``signal(as_of_date, universe, data) -> pd.Series`` of scores per
ticker (higher is better, NaN means unscorable). ``data`` is an ``AsOfView``: it
holds prices up to ``as_of_date`` and SEC facts filed on or before it. A signal
may use nothing else. ``assert_no_lookahead`` re-scores after corrupting every
price after ``as_of_date`` and every SEC fact filed after it; any change in the
scores is look-ahead.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable

import numpy as np
import pandas as pd

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
}


def make_signal(name: str, params: dict) -> Signal:
    if name not in CANDIDATES:
        raise KeyError(f"unknown candidate {name!r}; known: {sorted(CANDIDATES)}")
    return CANDIDATES[name](params)


def _perturbed_conn(conn: sqlite3.Connection, as_of: str) -> sqlite3.Connection:
    """In-memory copy of the database with later prices and later-filed SEC facts corrupted."""
    mem = sqlite3.connect(":memory:")
    conn.backup(mem)
    tables = {r[0] for r in mem.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "sec_fundamentals" in tables:
        mem.execute("UPDATE sec_fundamentals SET value = value * 7.0 + 13.0 WHERE filed_date > ?", (as_of,))
    if "price_daily" in tables:
        mem.execute("UPDATE price_daily SET open = open * 3.0, close = close * 3.0 WHERE date > ?", (as_of,))
    return mem


def assert_no_lookahead(signal: Signal, market: Market, as_of: str, universe: list[str],
                        seed: int = 0) -> None:
    """Raise LookaheadError if scores change when data after ``as_of`` is corrupted.

    Copies the database into memory to corrupt it, so run it on fixtures or a
    modest database rather than in a loop over a large one.
    """
    base = signal(as_of, universe, market.view(as_of))
    rng = np.random.default_rng(seed)
    later = market.adj_close.index > as_of
    close, open_ = market.adj_close.copy(), market.adj_open.copy()
    noise = rng.uniform(*NOISE_RANGE, size=(int(later.sum()), close.shape[1]))
    close.loc[later] = close.loc[later].to_numpy() * noise
    open_.loc[later] = open_.loc[later].to_numpy() * noise
    mem = _perturbed_conn(market.conn, as_of)
    fake = Market(open_, close, mem, price_tickers=market.price_tickers)
    try:
        pert = signal(as_of, universe, fake.view(as_of))
    finally:
        mem.close()
    a, b = base.sort_index(), pert.sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True):
        raise LookaheadError(f"signal output on {as_of} changed when later data was perturbed")

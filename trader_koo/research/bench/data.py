"""Market data for the research bench: back-adjusted prices, universe, as-of access.

Prices are built from the dividend/split total-return chain in
``trader_koo.db.total_return``: ``adj_close`` compounds the total-return series
and ``adj_open = adj_close * open / close`` so opens and closes share one basis
and dividends are inside the price. Rows the chain flags (special dividends,
unexplained jumps) are treated as a 0% day rather than letting a data error move
the result; the count is reported. Off-calendar dividends are the exception: their day keeps
the real close-to-close move plus the dividend. Nothing here writes to the database.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trader_koo.db.membership import members_on
from trader_koo.db.total_return import load_total_return_inputs, total_return_series
from trader_koo.fundamentals.pit import Fact, fundamentals_as_of, ttm_as_of

BENCHMARK_TICKER = "SPY"


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


@dataclass
class AsOfView:
    """What a signal may see on ``as_of``: data knowable at that day's close.

    ``close`` and ``open`` are back-adjusted frames (dates x tickers) cut at
    ``as_of`` inclusive. ``conn`` is exposed for read-only queries, but SEC facts
    must go through ``fundamentals``/``ttm``, which enforce filed_date <= as_of.
    """

    as_of: str
    close: pd.DataFrame
    open: pd.DataFrame
    conn: sqlite3.Connection | None = None

    def fundamentals(self, ticker: str, tag, kind: str | None = None) -> Fact | None:
        return fundamentals_as_of(self._need_conn(), ticker, self.as_of, tag, kind)

    def ttm(self, ticker: str, tag) -> Fact | None:
        return ttm_as_of(self._need_conn(), ticker, self.as_of, tag)

    def _need_conn(self) -> sqlite3.Connection:
        if self.conn is None:
            raise LookupError("no fundamentals connection attached")
        return self.conn


class Market:
    """All sessions, adjusted prices and point-in-time membership of one DB."""

    def __init__(self, adj_open: pd.DataFrame, adj_close: pd.DataFrame,
                 conn: sqlite3.Connection, flagged_rows: int = 0,
                 price_tickers: set[str] | None = None):
        self.adj_open, self.adj_close, self.conn = adj_open, adj_close, conn
        self.sessions: list[str] = list(adj_close.index)
        self.flagged_rows = flagged_rows
        self.price_tickers = price_tickers if price_tickers is not None else set(adj_close.columns)
        # Forward-filled copies value held positions across data gaps only.
        self.open_ff, self.close_ff = adj_open.ffill(), adj_close.ffill()
        self.columns = {t: i for i, t in enumerate(adj_close.columns)}
        self._members: dict[str, set[str]] = {}

    def view(self, as_of: str) -> AsOfView:
        return AsOfView(as_of, self.adj_close.loc[:as_of], self.adj_open.loc[:as_of], self.conn)

    def universe(self, idx: int) -> list[str]:
        """Point-in-time members on session ``idx`` that have a close that day."""
        d = self.sessions[idx]
        if d not in self._members:
            self._members[d] = members_on(self.conn, d)
        row = self.adj_close.iloc[idx]
        return sorted(t for t in self._members[d] if t in self.columns and np.isfinite(row[t]))

    def members_without_prices(self, lo: int, hi: int) -> tuple[int, int]:
        """(members in the window with no price rows at all, members in the window)."""
        a, b = self.sessions[lo], self.sessions[hi]
        rows = self.conn.execute(
            "SELECT DISTINCT ticker FROM sp500_membership "
            "WHERE start_date <= ? AND (end_date IS NULL OR end_date > ?)", (b, a)).fetchall()
        names = {r[0] for r in rows}
        return len(names - self.price_tickers), len(names)


def load_market(db_path: str | Path) -> Market:
    conn = connect_ro(db_path)
    if not _has_table(conn, "sp500_membership"):
        conn.close()
        raise LookupError("database has no sp500_membership table; the bench needs point-in-time membership")
    prices, actions = load_total_return_inputs(db_path)
    tr = total_return_series(prices, actions)
    opens = pd.read_sql_query(
        "SELECT ticker, date, open FROM price_daily "
        "WHERE COALESCE(adjustment_basis,'split_adjusted_price_only')='split_adjusted_price_only' "
        "AND ticker NOT LIKE '^%'", conn)
    # A dividend dated off the trading calendar lands on the next session and the chain
    # leaves that day's return empty; the move is real, so rebuild it from close and dividend.
    prev = tr.groupby("ticker")["close"].shift(1)
    off = (tr["flag"] == "dividend_off_calendar") & (prev > 0)
    ret = tr["total_return"].where(~off, (tr["close"] + tr["dividend"]) / prev - 1.0)
    tr = tr.assign(growth=1.0 + ret.fillna(0.0))
    tr["adj_close"] = tr.groupby("ticker")["growth"].cumprod()
    tr.loc[~(tr["close"] > 0), "adj_close"] = np.nan
    adj_close = tr.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    raw_close = tr.pivot(index="date", columns="ticker", values="close").sort_index()
    raw_open = opens.pivot(index="date", columns="ticker", values="open").reindex(
        index=adj_close.index, columns=adj_close.columns)
    raw_open = raw_open.where(raw_open > 0)
    adj_open = adj_close * (raw_open / raw_close)
    if BENCHMARK_TICKER not in adj_close.columns:
        conn.close()
        raise LookupError(f"{BENCHMARK_TICKER} prices are required for the benchmark")
    calendar = adj_close.index[adj_close[BENCHMARK_TICKER].notna()]
    return Market(adj_open.loc[calendar], adj_close.loc[calendar], conn,
                  flagged_rows=int(((tr["flag"] != "") & ~off).sum()),
                  price_tickers=set(tr["ticker"].unique()))

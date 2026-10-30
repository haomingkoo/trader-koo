"""Signals for docs/research/hypotheses-v6-insider.md (pre-registered, implemented as written).

Insider rows come from ``sec_insider_trades`` and are used only when filed before the signal date
(``filing_date <= data.filing_cutoff``), the same cutoff as SEC facts.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import hypotheses as hyp
from .data import AsOfView

# Pre-registered definitions (docs/research/hypotheses-v6-insider.md). Changing one is a new version.
WINDOW_DAYS = 90          # H1: purchases filed in the last 90 calendar days
MIN_BUYERS = 2            # H1: at least 2 distinct insiders
ROUTINE_YEARS = 3         # H2: bought in the same calendar month in each of the 3 prior years


def insider_trades(data: AsOfView, universe: list[str], since: str) -> pd.DataFrame:
    """Officer and director open-market trades of universe tickers filed in (since, filing cutoff]."""
    tickers = [t.upper() for t in universe]
    rows = data.conn.execute(
        f"""SELECT m.ticker, t.owner_ciks, t.filing_date, t.trans_code, t.shares, t.price
            FROM sec_insider_trades t JOIN sec_cik_map m ON m.cik = t.issuer_cik
            WHERE m.ticker IN ({','.join('?' * len(tickers))})
              AND t.filing_date > ? AND t.filing_date <= ? AND (t.is_officer = 1 OR t.is_director = 1)""",
        (*tickers, since, data.filing_cutoff)).fetchall()
    df = pd.DataFrame(rows, columns=["ticker", "owner_ciks", "filing_date", "code", "shares", "price"])
    df["value"] = df["shares"] * df["price"]
    data.diag.count("v6_trades_without_value", int(df["value"].isna().sum()))
    return df


def _buyers(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, insider) purchase; a joint filing counts each of its owners."""
    buys = df[df["code"] == "P"].assign(owner=lambda x: x["owner_ciks"].str.split(","))
    return buys.explode("owner")


def _drop_routine(buys: pd.DataFrame, data: AsOfView) -> pd.DataFrame:
    if buys.empty:
        return buys
    year = dt.date.fromisoformat(data.as_of).year
    owners = set(buys["owner"])
    rows = data.conn.execute(
        """SELECT owner_ciks, CAST(substr(filing_date, 1, 4) AS INTEGER), CAST(substr(filing_date, 6, 2) AS INTEGER)
           FROM sec_insider_trades WHERE trans_code = 'P' AND filing_date <= ?
             AND CAST(substr(filing_date, 1, 4) AS INTEGER) BETWEEN ? AND ?""",
        (data.filing_cutoff, year - ROUTINE_YEARS - 1, year)).fetchall()
    seen: dict[tuple[str, int], set[int]] = {}
    for ciks, y, m in rows:
        for owner in ciks.split(","):
            if owner in owners:
                seen.setdefault((owner, m), set()).add(y)
    by = pd.to_datetime(buys["filing_date"])
    routine = [
        all(y in seen.get((o, mo), set()) for y in range(py - ROUTINE_YEARS, py))
        for o, py, mo in zip(buys["owner"], by.dt.year, by.dt.month)
    ]
    data.diag.count("v6h2_routine_purchases_dropped", int(sum(routine)))
    return buys[~np.array(routine, dtype=bool)]


def _score(data: AsOfView, universe: list[str], df: pd.DataFrame, buys: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=universe)
    if buys.empty:
        return out
    n_buyers = buys.groupby("ticker")["owner"].nunique()
    signed = np.where(df["code"] == "P", df["value"], -df["value"])
    net = pd.Series(signed, index=df.index).groupby(df["ticker"]).sum(min_count=1)
    ok = n_buyers[n_buyers >= MIN_BUYERS].index
    ok = [t for t in ok if net.get(t, np.nan) > 0]
    data.diag.count("v6_qualifying_names", len(ok))
    if not ok:
        return out
    mcap = hyp.market_caps(data, ok)
    intensity = (net.reindex(ok) / mcap).where(mcap > 0)
    data.diag.count("v6_qualifying_without_market_cap", int(intensity.isna().sum()))
    frac = intensity.rank() / (intensity.notna().sum() + 1)  # below 1, so the buyer count ranks first
    out.loc[ok] = n_buyers.reindex(ok) + frac.fillna(0.0)
    return out


def _since(as_of: str) -> str:
    return (dt.date.fromisoformat(as_of) - dt.timedelta(days=WINDOW_DAYS)).isoformat()


def insider_clusters(params: dict):
    """H1: distinct buying insiders in 90 days (at least 2, net buying), ties broken by net value / market cap."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        df = insider_trades(data, universe, _since(as_of))
        return _score(data, universe, df, _buyers(df))

    return signal


def insider_opportunistic(params: dict):
    """H2: H1 after dropping purchases by routine insiders (Cohen, Malloy and Pomorski 2012)."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        df = insider_trades(data, universe, _since(as_of))
        return _score(data, universe, df, _drop_routine(_buyers(df), data))

    return signal

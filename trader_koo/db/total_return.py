"""Research-only dividend-adjusted daily return series, computed on the fly.

price_daily closes are split-adjusted only. This module adds dividends on their
ex-date to build r_t = (close_t + dividend_t) / close_{t-1} - 1 and a cumulative
index, without modifying any stored price. Rows that cannot be trusted (special
dividends, unexplained jumps, bad closes) return NaN and carry a flag; the
index restarts in a new segment afterwards rather than bridging the gap.

Dividend basis (validated against yfinance auto_adjust=True for 22 tickers,
2018-2026): stored dividends equal yfinance's current Dividends series exactly
(split-adjusted as of fetch) and stored closes equal yfinance's unadjusted-for-
dividends Close, so both share one split-adjusted basis. The one known gap is
spin-offs: yfinance adjusts dividends (MMM 1.137 vs 1.175 paid, Solventum) but
not the close, which costs about 1bp/yr for MMM. Tracking vs yfinance adjusted
close is within 4bp/yr for every sampled ticker.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from trader_koo.research.settings import get as _settings

_T = _settings().total_return
# The two thresholds below are part of TR_VERSION; changing either means a new version (config/research.toml).
TR_VERSION = _T.version
MAX_ABS_DAILY_MOVE = _T.max_abs_daily_move  # fraction of prior close; larger one-day moves are unexplained
MAX_DIVIDEND_FRACTION = _T.max_dividend_fraction  # fraction of prior close; larger dividends are special
DIVIDEND = "dividend"
SPLIT_TYPES = ("split", "reverse_split")
OUT_COLUMNS = ["ticker", "date", "close", "dividend", "total_return", "tr_index", "segment", "flag"]


def _input_hash(prices: pd.DataFrame, actions: pd.DataFrame) -> str:
    p = prices[["ticker", "date", "close"]].sort_values(["ticker", "date"]).astype(str)
    a = actions[["ticker", "action_date", "action_type", "value"]].sort_values(
        ["ticker", "action_date", "action_type", "value"]
    ).astype(str)
    h = hashlib.sha256()
    h.update(p.to_csv(index=False).encode())
    h.update(b"|actions|")
    h.update(a.to_csv(index=False).encode())
    return h.hexdigest()


def _one_ticker(px: pd.DataFrame, acts: pd.DataFrame, unapplied: list) -> pd.DataFrame:
    px = px.sort_values("date").reset_index(drop=True)
    n = len(px)
    dates = px["date"].astype(str).tolist()
    close = pd.to_numeric(px["close"], errors="coerce").to_numpy(dtype=float, copy=True)  # written below; pandas 3 views are read-only
    div = np.zeros(n)
    flags = [""] * n

    def flag(i: int, reason: str) -> None:
        if not flags[i]:
            flags[i] = reason

    for row in acts.itertuples(index=False):
        if row.action_type != DIVIDEND:
            continue
        d = str(row.action_date)
        idx = next((i for i, x in enumerate(dates) if x >= d), None)
        if idx is None:  # ex-date after the last stored price: no return to attach it to; reported, not dropped
            unapplied.append((str(px["ticker"].iloc[0]), d, float(row.value)))
            continue
        if dates[idx] != d:
            flag(idx, "dividend_off_calendar")
        div[idx] += float(row.value)
    split_dates = set(
        acts.loc[acts["action_type"].isin(SPLIT_TYPES), "action_date"].astype(str)
    )

    ret = np.full(n, np.nan)
    valid = np.isfinite(close) & (close > 0)
    for i in range(n):
        if not valid[i]:
            flag(i, "bad_close")
            continue
        if i == 0 or not valid[i - 1]:
            continue
        prev = close[i - 1]
        if div[i] > MAX_DIVIDEND_FRACTION * prev:
            flag(i, "special_dividend")
        if abs(close[i] / prev - 1) > MAX_ABS_DAILY_MOVE:
            flag(i, "move_on_split_date" if dates[i] in split_dates else "unexplained_move")
        if not flags[i]:
            ret[i] = (close[i] + div[i]) / prev - 1

    # An invalid return starts a new segment; the index is 1.0 at each segment start.
    segment = np.cumsum(np.isnan(ret))
    segment = segment - segment[0]
    growth = np.where(np.isnan(ret), 1.0, 1.0 + ret)
    index = pd.Series(growth).groupby(segment).cumprod().to_numpy(copy=True)
    index[~valid] = np.nan
    return pd.DataFrame({
        "ticker": px["ticker"].iloc[0], "date": dates, "close": close, "dividend": div,
        "total_return": ret, "tr_index": index, "segment": segment, "flag": flags,
    })


def total_return_series(
    prices_df: pd.DataFrame, actions_df: pd.DataFrame, version: str = TR_VERSION
) -> pd.DataFrame:
    """Per-ticker total-return series.

    attrs: `version`, `input_sha256`, `dividends_after_last_price` (dividends with no price row on or after their
    ex-date, not applied) and `ignored_action_types` (action types that are neither dividend nor split, with counts).
    """
    if actions_df is None or actions_df.empty:
        actions_df = pd.DataFrame(columns=["ticker", "action_date", "action_type", "value"])
    unapplied: list = []
    parts = [
        _one_ticker(px, actions_df[actions_df["ticker"] == ticker], unapplied)
        for ticker, px in prices_df.groupby("ticker", sort=True)
    ]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OUT_COLUMNS)
    out.attrs["version"] = version
    out.attrs["dividends_after_last_price"] = unapplied
    known = (DIVIDEND, *SPLIT_TYPES)
    out.attrs["ignored_action_types"] = {
        str(k): int(v) for k, v in actions_df.loc[~actions_df["action_type"].isin(known), "action_type"].value_counts().items()
    }
    out.attrs["input_sha256"] = _input_hash(prices_df, actions_df)
    return out


def total_return_quality_report(series: pd.DataFrame) -> pd.DataFrame:
    """Flagged (ticker, date, reason) rows; these have NaN total_return."""
    flagged = series.loc[series["flag"] != "", ["ticker", "date", "flag"]]
    return flagged.rename(columns={"flag": "reason"}).reset_index(drop=True)


def load_total_return_inputs(
    db_path: str | Path, tickers: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read split-adjusted closes and corporate actions through a read-only connection."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Index symbols (^VIX, ^TNX) have no dividends and are not investable series.
        where, params = " AND ticker NOT LIKE '^%'", []
        if tickers:
            where += f" AND ticker IN ({','.join('?' * len(tickers))})"
            params = list(tickers)
        # Rows already on a total-return basis would double count dividends.
        prices = pd.read_sql_query(
            "SELECT ticker, date, close FROM price_daily "
            "WHERE COALESCE(adjustment_basis,'split_adjusted_price_only')='split_adjusted_price_only'"
            + where + " ORDER BY ticker, date",
            conn, params=params,
        )
        actions = pd.read_sql_query(
            "SELECT ticker, action_date, action_type, value FROM price_corporate_actions WHERE 1=1"
            + where + " ORDER BY ticker, action_date",
            conn, params=params,
        )
    finally:
        conn.close()
    return prices, actions

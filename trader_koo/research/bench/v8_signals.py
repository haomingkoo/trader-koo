"""Signals of hypotheses v8 (docs/research/hypotheses-v8-swing.md): earnings drift, earnings gaps, 52-week high.

The numbers below are the registration, not settings; each names its source in the document. Changing one is
a new version. Matrices are (sessions x tickers); an earnings event is placed on the first session whose open
comes after the report was public, so nothing here reads a value from after that open.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .data import AsOfView, Market
from .v2_signals import Prices
from .v2_trades import Grid, finish_grid

# H1 (Bernard and Thomas 1989; Livnat and Mendenhall 2006)
H1_DECILE = 0.9            # top decile of SUE, cut on the previous calendar quarter's events
H1_HOLD = 60               # sessions from the entry open to the exit open
H1_VARIANT_HOLD = 20       # reported only
# H2 (Qullamaggie gap and trail; Stockbee volume)
H2_GAP = 0.10              # open / previous close - 1
H2_VOLUME_MULTIPLE = 3.0   # times the mean volume of the 50 sessions before the gap day
H2_VOLUME_WINDOW = 50
H2_MIN_VOLUME = 300_000    # shares on the gap day
H2_CLOSE_IN_RANGE = 0.5    # set here: close in the upper half of the day's range
H2_TRAIL = 10              # sessions in the trailing simple average of closes
H2_VARIANT_TRAIL = 20      # reported only
H2_MAX_HOLD = 60
# H3 (George and Hwang 2004)
H3_WINDOW = 252            # sessions in the 52-week high, including the review session
MONTHS_PER_QUARTER = 3
QUARTERS_PER_YEAR = 4
MONTH_SLICE = slice(5, 7)  # ISO date characters of the month
YEAR_SLICE = slice(0, 4)
BMO = "BMO"
PRICE_BASIS = "COALESCE(adjustment_basis,'split_adjusted_price_only')='split_adjusted_price_only'"


def load_events(conn: sqlite3.Connection, before: str) -> pd.DataFrame:
    """Every earnings row reported before ``before`` (the seal): ticker, report date, session and EPS fields."""
    return pd.read_sql_query(
        "SELECT ticker, report_date, session, eps_estimate, eps_actual FROM earnings_events "
        "WHERE report_date < ? ORDER BY ticker, report_date", conn, params=(before,))


def entry_rows(report_dates, session_labels, sessions: list[str]) -> np.ndarray:
    """First session whose open follows the report: BMO on or after the date, AMC or unknown after it; -1 if none."""
    dates = np.asarray(report_dates, dtype=object)
    bmo = np.array([s == BMO for s in session_labels], dtype=bool)
    on_or_after = np.searchsorted(sessions, dates.astype(str), side="left")
    after = np.searchsorted(sessions, dates.astype(str), side="right")
    row = np.where(bmo, on_or_after, after)
    return np.where(row < len(sessions), row, -1)


def _quarter(date: str) -> int:
    return int(date[YEAR_SLICE]) * QUARTERS_PER_YEAR + (int(date[MONTH_SLICE]) - 1) // MONTHS_PER_QUARTER


def quarter_cutoffs(entry_dates, sue: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Per event, the 90th percentile of SUE over eligible events entered in the previous calendar quarter (NaN if none)."""
    q = np.array([_quarter(d) for d in entry_dates])
    cut_by_q = {}
    for k in np.unique(q):
        pool = sue[(q == k) & eligible & np.isfinite(sue)]
        cut_by_q[k] = float(np.quantile(pool, H1_DECILE)) if pool.size else np.nan
    return np.array([cut_by_q.get(k - 1, np.nan) for k in q])


def surprise(actual: np.ndarray, estimate: np.ndarray, rows: np.ndarray, cols: np.ndarray,
             stored_close: np.ndarray) -> np.ndarray:
    """SUE = (actual - estimate) / stored close of the session before the entry session."""
    px = np.where(rows >= 1, stored_close[np.maximum(rows - 1, 0), cols], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(px > 0, (actual - estimate) / px, np.nan)


def gap_qualifies(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, v: np.ndarray,
                  rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """H2 at the close of reaction session g = rows: gap, volume against the 50 sessions before g, close in range."""
    ok = rows >= H2_VOLUME_WINDOW
    g, k = np.where(ok, rows, H2_VOLUME_WINDOW), cols
    window = np.stack([v[g - d, k] for d in range(1, H2_VOLUME_WINDOW + 1)])
    complete = np.isfinite(window).all(axis=0)
    avg = np.nanmean(np.where(np.isfinite(window), window, np.nan), axis=0) if window.size else np.array([])
    rng = h[g, k] - lo[g, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = o[g, k] / c[g - 1, k] - 1.0
        placed = (c[g, k] - lo[g, k]) / rng
    vol = v[g, k]
    return (ok & complete & (gap >= H2_GAP) & (vol >= H2_VOLUME_MULTIPLE * avg) & (vol >= H2_MIN_VOLUME)
            & (rng > 0) & (placed >= H2_CLOSE_IN_RANGE))


def stop_trail_grid(market: Market, p: Prices, trail: int, max_hold: int) -> Grid:
    """H2 exits for every (entry session e, ticker) cell; the stop is the low of e - 1.

    Each session s from e: open at or below the stop exits at that open; else a low at or below it exits at the
    stop; else a close below the ``trail``-session average, while that average is above the stop, exits at the
    next open. With no trigger the exit is the open of e + max_hold. A missing bar triggers nothing.
    """
    o, lo, c = p.open.to_numpy(), p.low.to_numpy(), p.close.to_numpy()
    sma = p.close.rolling(trail, min_periods=trail).mean().to_numpy()
    n, t = o.shape
    stop = np.vstack([np.full((1, t), np.nan), lo[:-1]])
    rows = np.arange(n)[:, None]
    exit_row = np.broadcast_to(rows + max_hold, (n, t)).copy()
    fixed = np.full((n, t), np.nan)
    open_ = np.ones((n, t), dtype=bool)
    for k in range(min(max_hold, n)):
        m = n - k
        st, live = stop[:m], open_[:m]
        at_open = live & (o[k:] <= st)
        at_stop = live & ~at_open & (lo[k:] <= st)
        trailed = live & ~at_open & ~at_stop & (sma[k:] > st) & (c[k:] < sma[k:])
        exit_row[:m][at_open] = (rows[:m] + k + np.zeros((1, t), int))[at_open]
        exit_row[:m][at_stop | trailed] = (rows[:m] + k + 1 + np.zeros((1, t), int))[at_stop | trailed]
        fixed[:m][at_stop] = st[at_stop]
        open_[:m] &= ~(at_open | at_stop | trailed)
    return finish_grid(market, p.open, p.close, exit_row, max_hold, fixed)


def near_52_week_high(params: dict):
    """H3: stored close / highest stored close of the 252 sessions ending at the review; complete windows only.

    Stored closes are split-adjusted, so the ratio carries no dividend. Sessions the bench's quality rules
    removed (NaN in the as-of view) count as missing.
    """
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        px = data.close.reindex(columns=universe)
        if len(px) < H3_WINDOW or not universe:
            return pd.Series(np.nan, index=universe)
        dates = px.index[-H3_WINDOW:]
        raw = pd.read_sql_query(
            f"SELECT ticker, date, close FROM price_daily WHERE {PRICE_BASIS} AND close > 0 AND date >= ? AND date <= ? "
            f"AND ticker IN ({','.join('?' * len(universe))})", data.conn, params=(dates[0], as_of, *universe))
        closes = raw.pivot(index="date", columns="ticker", values="close").reindex(index=dates, columns=universe)
        closes = closes.where(px.iloc[-H3_WINDOW:].notna())
        complete = closes.notna().all()
        return (closes.iloc[-1] / closes.max()).where(complete)

    return signal

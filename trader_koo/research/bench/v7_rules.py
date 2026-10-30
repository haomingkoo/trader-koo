"""Exposure rules for docs/research/hypotheses-v7-regime.md (pre-registered, implemented as written).

Each rule maps the aligned inputs to an exposure per session (1 = in SPY, 0 = in cash), decided at that
session's close from data known by then. NaN means the rule has no value yet; the runner holds 1 on
those days and counts them.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Pre-registered definitions (hypotheses-v7-regime.md). Changing one is a new version.
VIX_BREAKOUT = 20.0
VIX_SMA = 20
RV_WINDOW = 21
TRADING_DAYS = 252
VOL_POINTS = 100.0
DIST_DROP = -0.002
DIST_WINDOW = 25
DIST_LIMIT = 5
FABER_MONTHS = 10
TSMOM_MONTHS = 12
INFLATION_LEVEL = 0.05
INFLATION_LOOKBACK_MONTHS = 3
DOLLAR_WINDOW = 126
DOLLAR_RISE = 0.05
MONTHS_PER_YEAR = 12
MONTH_CHARS = 7  # "YYYY-MM"


@dataclass
class Inputs:
    """Arrays over the window's sessions (index i is ``dates[i]``)."""

    dates: list[str]
    spy_close: np.ndarray      # total-return adjusted
    spy_open: np.ndarray       # total-return adjusted
    spy_raw_close: np.ndarray  # as quoted, for the distribution-day rule
    spy_volume: np.ndarray
    vix: np.ndarray
    rf_daily: np.ndarray       # cash return per session, from the previous close's ^IRX
    conn: sqlite3.Connection   # for macro_series, read as known on each date


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _known_daily(conn: sqlite3.Connection, series: str, dates: list[str]) -> np.ndarray:
    """Latest first-published value of a daily series known by each date (NaN before the first)."""
    rows = conn.execute(
        """SELECT obs_date, value, MIN(known_from) FROM macro_series WHERE series = ?
           GROUP BY obs_date ORDER BY obs_date""", (series,)).fetchall()
    df = pd.DataFrame(rows, columns=["obs", "value", "known"])
    df = df[df["known"] <= dates[-1]].sort_values(["known", "obs"])  # same publish day: newest obs wins
    known = pd.Series(df["value"].to_numpy(), index=df["known"].to_numpy())
    known = known[~known.index.duplicated(keep="last")]
    return known.reindex(sorted(set(known.index) | set(dates))).ffill().reindex(dates).to_numpy(float)


def _month_ends(dates: list[str]) -> np.ndarray:
    months = pd.Index([d[:MONTH_CHARS] for d in dates])
    return np.append(months[1:] != months[:-1], True)


def _monthly_hold(dates: list[str], decide) -> np.ndarray:
    """Exposure decided at each month's last session and held until the next one."""
    ends = np.flatnonzero(_month_ends(dates))
    out = np.full(len(dates), np.nan)
    for k, i in enumerate(ends):
        out[i:ends[k + 1] if k + 1 < len(ends) else len(dates)] = decide(ends[: k + 1])
    return out


def _yoy_as_known(conn: sqlite3.Connection, series: str, day: str, months_back: int = 0) -> float:
    """Year-on-year change of a monthly series, latest observation known by ``day``, one vintage."""
    rows = conn.execute(
        """SELECT obs_date, value FROM macro_series WHERE series = ? AND known_from <= ? AND known_until >= ?
           ORDER BY obs_date""", (series, day, day)).fetchall()
    if len(rows) <= MONTHS_PER_YEAR + months_back:
        return np.nan
    vals = dict(rows)
    last = pd.Timestamp(rows[-1][0]) - pd.DateOffset(months=months_back)
    now, base = vals.get(last.date().isoformat()), vals.get((last - pd.DateOffset(years=1)).date().isoformat())
    return now / base - 1.0 if now and base else np.nan


def _monthly_inputs_change_days(inp: Inputs, series: tuple[str, ...]) -> list[int]:
    """Indexes of sessions on or after which a new vintage of any of ``series`` became known."""
    known = sorted({d for s in series for (d,) in inp.conn.execute(
        "SELECT DISTINCT known_from FROM macro_series WHERE series = ?", (s,))})
    idx = sorted({int(np.searchsorted(inp.dates, k)) for k in known if k <= inp.dates[-1]})
    return [0, *[i for i in idx if 0 < i < len(inp.dates)]]


def _eval_on_changes(inp: Inputs, series: tuple[str, ...], fn) -> np.ndarray:
    out = np.full(len(inp.dates), np.nan)
    marks = _monthly_inputs_change_days(inp, series)
    for k, i in enumerate(marks):
        out[i:marks[k + 1] if k + 1 < len(marks) else len(inp.dates)] = fn(inp.dates[i])
    return out


def h1_term_structure(inp: Inputs) -> np.ndarray:
    vix3m = _known_daily(inp.conn, "VIX3M", inp.dates)
    ratio = inp.vix / vix3m
    return np.where(np.isnan(ratio), np.nan, (ratio <= 1.0).astype(float))


def h2_vix_breakout(inp: Inputs) -> np.ndarray:
    sma = _sma(inp.vix, VIX_SMA)
    off = (inp.vix >= VIX_BREAKOUT) & (inp.vix > sma)
    return np.where(np.isnan(sma), np.nan, (~off).astype(float))


def h3_variance_premium(inp: Inputs) -> np.ndarray:
    ret = np.diff(np.log(inp.spy_close), prepend=np.nan)
    rv = pd.Series(ret).rolling(RV_WINDOW, min_periods=RV_WINDOW).std(ddof=1).to_numpy()
    vrp = inp.vix - rv * np.sqrt(TRADING_DAYS) * VOL_POINTS
    return np.where(np.isnan(vrp), np.nan, (vrp >= 0).astype(float))


def h4_distribution_days(inp: Inputs) -> np.ndarray:
    chg = np.diff(inp.spy_raw_close, prepend=np.nan) / np.roll(inp.spy_raw_close, 1)
    vol_up = np.diff(inp.spy_volume, prepend=np.nan) > 0
    day = ((chg <= DIST_DROP) & vol_up).astype(float)
    day[0] = np.nan
    count = pd.Series(day).rolling(DIST_WINDOW, min_periods=DIST_WINDOW).sum().to_numpy()
    return np.where(np.isnan(count), np.nan, (count < DIST_LIMIT).astype(float))


def h5_ten_month_trend(inp: Inputs) -> np.ndarray:
    def decide(ends: np.ndarray) -> float:
        if len(ends) < FABER_MONTHS:
            return np.nan
        closes = inp.spy_close[ends[-FABER_MONTHS:]]
        return float(closes[-1] >= closes.mean())
    return _monthly_hold(inp.dates, decide)


def h6_time_series_momentum(inp: Inputs) -> np.ndarray:
    cash = np.cumprod(1.0 + np.nan_to_num(inp.rf_daily))

    def decide(ends: np.ndarray) -> float:
        if len(ends) <= TSMOM_MONTHS:
            return np.nan
        a, b = ends[-1 - TSMOM_MONTHS], ends[-1]
        return float(inp.spy_close[b] / inp.spy_close[a] >= cash[b] / cash[a])
    return _monthly_hold(inp.dates, decide)


def h7_inflation_regime(inp: Inputs) -> np.ndarray:
    def at(day: str) -> float:
        now = _yoy_as_known(inp.conn, "CPIAUCSL", day)
        before = _yoy_as_known(inp.conn, "CPIAUCSL", day, INFLATION_LOOKBACK_MONTHS)
        if np.isnan(now) or np.isnan(before):
            return np.nan
        return float(not (now >= INFLATION_LEVEL and now > before))
    return _eval_on_changes(inp, ("CPIAUCSL",), at)


def h8_real_money(inp: Inputs) -> np.ndarray:
    def at(day: str) -> float:
        m2, cpi = _yoy_as_known(inp.conn, "M2SL", day), _yoy_as_known(inp.conn, "CPIAUCSL", day)
        return np.nan if np.isnan(m2) or np.isnan(cpi) else float(m2 >= cpi)
    return _eval_on_changes(inp, ("M2SL", "CPIAUCSL"), at)


def h9_strong_dollar(inp: Inputs) -> np.ndarray:
    usd = _known_daily(inp.conn, "DTWEXBGS", inp.dates)
    past = np.roll(usd, DOLLAR_WINDOW)
    past[:DOLLAR_WINDOW] = np.nan
    rise = usd / past - 1.0
    return np.where(np.isnan(rise), np.nan, (rise <= DOLLAR_RISE).astype(float))


def h10_yield_curve(inp: Inputs) -> np.ndarray:
    spread = _known_daily(inp.conn, "T10Y2Y", inp.dates)
    return np.where(np.isnan(spread), np.nan, (spread >= 0).astype(float))


RULES = {
    "H1": h1_term_structure, "H2": h2_vix_breakout, "H3": h3_variance_premium, "H4": h4_distribution_days,
    "H5": h5_ten_month_trend, "H6": h6_time_series_momentum, "H7": h7_inflation_regime, "H8": h8_real_money,
    "H9": h9_strong_dollar, "H10": h10_yield_curve,
}


def h11_vote(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Share of H1 to H10 at exposure 1, each NaN day counted as 1 (the common rule)."""
    stacked = np.vstack([np.nan_to_num(raw[k], nan=1.0) for k in RULES])
    return stacked.mean(axis=0)

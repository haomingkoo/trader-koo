"""Signals of hypotheses v2 (docs/research/hypotheses-v2-saty.md), S1 to S3, on whole price matrices.

Every value at row t uses rows up to t only (rolling and exponential windows look back), so one pass over
the matrix equals a signal re-scored on each date; the look-ahead guard test corrupts every row after t and
compares. Prices are the bench's as-of total-return-adjusted open, high, low and close (dates x tickers).

The numbers are the registration, not settings; each names its section of the document. Readings of text
the document leaves open are listed in the PR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# S1: stack EMA8 > EMA21 > EMA48 > EMA200 and close > EMA200; pullback low <= EMA21 <= close; exit below EMA48 or 20 sessions.
S1_SPANS, S1_EXIT_SPAN, S1_MAX_HOLD = (8, 21, 48, 200), 48, 20
# S2: EMA8 crosses above EMA13 with close > EMA21; first retest of EMA13 within 10 sessions; exit below EMA21 or 20 sessions.
S2_FAST, S2_SLOW, S2_TREND, S2_RETEST_WINDOW, S2_EXIT_SPAN, S2_MAX_HOLD = 8, 13, 21, 10, 21, 20
# S3: Bollinger(21, 2) inside pivot +/- 2 ATR14 is compression; expansion in an uptrend EMA21 > EMA48; no compression in
# the previous 20 sessions for the baseline; exit after 10 sessions.
S3_BB_WINDOW, S3_BB_K, S3_ATR_WINDOW, S3_ATR_K, S3_QUIET_LOOKBACK, S3_TREND, S3_HOLD = 21, 2.0, 14, 2.0, 20, (21, 48), 10


@dataclass
class Prices:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    member: pd.DataFrame  # bool: point-in-time index member with a close that session

    def cut(self, last: int) -> "Prices":
        """Rows 0..last only (used by the look-ahead guard)."""
        return Prices(*(getattr(self, f).iloc[:last + 1] for f in ("open", "high", "low", "close", "member")))


def ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Span-n exponential average, ``adjust=False``; undefined until n observations exist."""
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(p: Prices, n: int) -> pd.DataFrame:
    """Wilder average true range over n sessions."""
    prev = p.close.shift(1)
    tr = np.maximum(np.maximum(p.high - p.low, (p.high - prev).abs()), (p.low - prev).abs())
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def next_true(mask: np.ndarray) -> np.ndarray:
    """Row index of the first True at or after each row, per column; len(rows) when there is none."""
    n = mask.shape[0]
    idx = np.where(mask, np.arange(n)[:, None], n)
    return np.minimum.accumulate(idx[::-1], axis=0)[::-1]


def s1(p: Prices) -> dict:
    e = {n: ema(p.close, n) for n in (*S1_SPANS, S1_EXIT_SPAN)}
    stack = (e[8] > e[21]) & (e[21] > e[48]) & (e[48] > e[200]) & (p.close > e[200])
    pull = (p.low <= e[21]) & (p.close >= e[21])
    return {"signal": stack & pull & p.member, "state": stack & ~pull & p.member, "exit_ema": e[S1_EXIT_SPAN]}


def s2(p: Prices) -> dict:
    e = {n: ema(p.close, n) for n in (S2_FAST, S2_SLOW, S2_TREND)}
    fast, slow = e[S2_FAST], e[S2_SLOW]
    cross = (fast > slow) & (fast.shift(1) <= slow.shift(1)) & (p.close > e[S2_TREND])
    retest = ((p.low <= slow) & (p.close > slow)).to_numpy()
    nxt = next_true(retest)
    n = retest.shape[0]
    first = np.full(retest.shape, n)
    first[:-1] = nxt[1:]  # first retest strictly after the cross day
    cross_a = cross.to_numpy()
    within = cross_a & (first <= np.arange(n)[:, None] + S2_RETEST_WINDOW) & (first < n)
    return {"cross": cross & p.member, "retest_day": np.where(within, first, -1), "exit_ema": e[S2_TREND]}


def s3(p: Prices) -> dict:
    roll = p.close.rolling(S3_BB_WINDOW)
    pivot, sd = roll.mean(), roll.std(ddof=0)  # Bollinger's own population deviation, as in Pine's ta.stdev
    band = S3_ATR_K * atr(p, S3_ATR_WINDOW)
    comp = ((pivot + S3_BB_K * sd <= pivot + band) & (pivot - S3_BB_K * sd >= pivot - band))
    valid = pivot.notna() & band.notna()
    comp = comp & valid
    up = ema(p.close, S3_TREND[0]) > ema(p.close, S3_TREND[1])
    expansion = ~comp & valid & comp.shift(1, fill_value=False).astype(bool) & up
    quiet = comp.astype(float).where(valid).rolling(S3_QUIET_LOOKBACK).sum().shift(1) == 0
    return {"signal": expansion & p.member, "state": up & quiet & p.member}

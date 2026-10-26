"""Causal building blocks for the v3 specs: every function returns, at index t, a value computed from data through t.

``fwd`` is the one function that looks ahead, and it is only used to build outcomes, never signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eventstudy import Spec

TERCILES = 3


def fwd(x: np.ndarray, k: int) -> np.ndarray:
    """x[t+k] at index t (NaN past the end of the window). Outcomes only."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[:len(x) - k] = x[k:]
    return out


def lag(x: np.ndarray, k: int) -> np.ndarray:
    """x[t-k] at index t (NaN before the start)."""
    out = np.full(len(x), np.nan)
    out[k:] = x[:len(x) - k]
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).mean().to_numpy()


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average, span n, ``adjust=False``; undefined until n observations exist."""
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def rmax(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).max().to_numpy()


def rmin(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).min().to_numpy()


def rquantile(x: np.ndarray, n: int, q: float) -> np.ndarray:
    return pd.Series(x).rolling(n).quantile(q).to_numpy()


def pct_change(x: np.ndarray) -> np.ndarray:
    return x / lag(x, 1) - 1.0


def holds(cond: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
    """``cond`` as bool, False wherever any input is missing."""
    ok = np.ones(len(cond), dtype=bool)
    for a in inputs:
        ok &= np.isfinite(a)
    return np.asarray(cond, dtype=bool) & ok


def outcome(cond: np.ndarray, *inputs: np.ndarray) -> np.ndarray:
    """1.0 / 0.0 for ``cond``, NaN wherever any input is missing (an unchanged close is a miss: callers use strict tests)."""
    ok = np.ones(len(cond), dtype=bool)
    for a in inputs:
        ok &= np.isfinite(a)
    return np.where(ok, np.asarray(cond, dtype=float), np.nan)


def lt_after(c: np.ndarray, k: int) -> np.ndarray:
    return outcome(fwd(c, k) < c, fwd(c, k), c)


def gt_after(c: np.ndarray, k: int) -> np.ndarray:
    return outcome(fwd(c, k) > c, fwd(c, k), c)


def terciles(x: np.ndarray, window: int) -> np.ndarray:
    """0, 1, 2 for x(t) at or below, between and above the trailing-window 1/3 and 2/3 quantiles (window includes t); -1 if undefined."""
    lo, hi = rquantile(x, window, 1 / TERCILES), rquantile(x, window, 2 / TERCILES)
    t = (x > lo).astype(int) + (x > hi).astype(int)
    return np.where(np.isfinite(lo) & np.isfinite(hi) & np.isfinite(x), t, -1)


def bands_width(c: np.ndarray, n: int, k: float) -> np.ndarray:
    """(upper - lower) / SMA for Bollinger(n, k sample standard deviations)."""
    s = pd.Series(c).rolling(n)
    return (2.0 * k * s.std() / s.mean()).to_numpy()


def single(key: str, title: str, role: str, signal: np.ndarray, out: np.ndarray, base: np.ndarray,
           claim: float | None = None, note: str = "") -> Spec:
    """A one-stratum spec."""
    n = len(signal)
    return Spec(key, title, role, signal, np.zeros(n, dtype=int), out[None, :], base[None, :], claim, note)

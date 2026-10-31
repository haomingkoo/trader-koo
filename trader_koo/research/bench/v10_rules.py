"""Roy Mattox's stated exposure ladder (docs/research/hypotheses-v10-exposure.md, pre-registered, implemented as written).

``market_states`` turns one index's bars into a target exposure per session, decided at that session's close.
``run_book`` applies the drawdown ladder and the pace to a book, session by session, because the drawdown depends on
the exposure already held. Session t's exposure is held over the interval from the open of t+1 to the open of t+2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .v2_trades import BPS

# Pre-registered values (hypotheses-v10-exposure.md). Changing one is a new version.
DIST_DROP = -0.002          # distribution day: down 0.20% or more on higher volume
FTD_GAIN = 0.017            # follow-through day: up 1.7% or more on higher volume
FTD_FIRST_DAY = 4           # day 4 or later of a rally attempt
FTD_FAIL_DAYS = 3           # a distribution day on session 1 to 3 after the FTD fails it
EMA_FAST = 10
EMA_MID = 21
SMA_SLOW = 50
WEEKS = 30                  # 30-week SMA of weekly closes
WEEKS_IN_ROW = 2            # two consecutive weekly closes beyond it
STEP = 0.25                 # one ladder step
LT_CAP = 0.25               # cap while in a long-term downtrend
DD_STEPS = ((-0.10, 0.0), (-0.07, 0.25), (-0.05, 0.5))   # drawdown at or below -> multiplier

UP, CORRECTION = 1, 0


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
    out[: n - 1] = np.nan
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def distribution_days(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(len(c), bool)
    out[1:] = (c[1:] / c[:-1] - 1.0 <= DIST_DROP) & (v[1:] > v[:-1])
    return out


def _week_ends(dates: list[str]) -> np.ndarray:
    """True on the last session of a calendar week, known only once the next session opens a new week."""
    weeks = [pd.Timestamp(d).isocalendar()[:2] for d in dates]
    return np.array([i + 1 < len(dates) and weeks[i + 1] != weeks[i] for i in range(len(dates))])


def long_term_downtrend(dates: list[str], c: np.ndarray) -> np.ndarray:
    """On from the second consecutive weekly close below the 30-week SMA, off from the second above it."""
    ends = np.flatnonzero(_week_ends(dates))
    weekly = c[ends]
    avg = sma(weekly, WEEKS)
    out = np.zeros(len(dates), bool)
    on, below, above = False, 0, 0
    for k, i in enumerate(ends):
        if np.isnan(avg[k]):
            continue
        below, above = (below + 1, 0) if weekly[k] < avg[k] else (0, above + 1)
        if not on and below >= WEEKS_IN_ROW:
            on = True
        elif on and above >= WEEKS_IN_ROW:
            on = False
        out[i:ends[k + 1] if k + 1 < len(ends) else len(dates)] = on
    return out


def market_states(c, lo, v, ema10, ema21, sma50, lt, warm: int) -> tuple[np.ndarray, np.ndarray]:
    """Target exposure per session (rules 1 to 6 and 9) and the sessions whose FTD started an uptrend."""
    n = len(c)
    target, ftd_start = np.zeros(n), np.zeros(n, bool)
    dist = distribution_days(c, v)
    state, ftd_i, ftd_low, rally_low, day1 = CORRECTION, -1, np.nan, np.nan, -1
    for t in range(warm, n):
        if t == warm:
            state = UP if ema21[t] > sma50[t] else CORRECTION
            rally_low = lo[t]
        elif state == UP:
            cross = ema21[t] < sma50[t] and ema21[t - 1] >= sma50[t - 1]
            fresh = c[t] < ftd_low or cross or (lt[t] and not lt[t - 1])
            if fresh:
                state, rally_low, day1 = CORRECTION, lo[t], -1
            elif ftd_i >= 0 and 1 <= t - ftd_i <= FTD_FAIL_DAYS and dist[t]:
                state = CORRECTION        # failed FTD: the rally attempt keeps its low and day count
        else:
            up_close = c[t] > c[t - 1]
            if lo[t] < rally_low:
                rally_low, day1 = lo[t], t if up_close else -1
            elif day1 < 0 and up_close:
                day1 = t
            if (day1 >= 0 and t - day1 + 1 >= FTD_FIRST_DAY and c[t] / c[t - 1] - 1.0 >= FTD_GAIN
                    and v[t] > v[t - 1]):
                state, ftd_i, ftd_low, ftd_start[t] = UP, t, lo[t], True
        if state == UP:
            steps = 1 + int(c[t] > ema21[t]) + int(c[t] > sma50[t]) + int(ema10[t] > ema21[t])
            target[t] = min(STEP * steps, LT_CAP) if lt[t] else STEP * steps
    return target, ftd_start


def ladder_from_bars(dates: list[str], c, lo, v) -> tuple[np.ndarray, np.ndarray, int]:
    """Indicators of one index, then ``market_states``. Returns target, FTD-start sessions and the warm-up session."""
    e10, e21, s50 = ema(c, EMA_FAST), ema(c, EMA_MID), sma(c, SMA_SLOW)
    lt = long_term_downtrend(dates, c)
    weekly_ready = np.zeros(len(dates), bool)
    ends = np.flatnonzero(_week_ends(dates))
    if len(ends) >= WEEKS:
        weekly_ready[ends[WEEKS - 1]:] = True
    ready = np.isfinite(e10) & np.isfinite(e21) & np.isfinite(s50) & weekly_ready
    if not ready.any():
        raise LookupError("not enough sessions to warm up the v10 indicators")
    warm = int(np.argmax(ready))
    target, ftd = market_states(c, lo, v, e10, e21, s50, lt, warm)
    return target, ftd, warm


@dataclass
class BookPath:
    exposure: np.ndarray   # decided at each close
    hold: np.ndarray       # held over the interval from that session's open to the next open
    returns: np.ndarray    # per interval, net of cost


def dd_multiplier(dd: float) -> float:
    for level, mult in DD_STEPS:
        if dd <= level:
            return mult
    return 1.0


def hold_returns(hold: np.ndarray, R: np.ndarray, cash: np.ndarray, cost_bps: float) -> np.ndarray:
    prev = np.concatenate([[0.0], hold[:-1]])
    return hold * R + (1.0 - hold) * cash - cost_bps / BPS * np.abs(hold - prev)


def run_book(target, ftd_start, R, intra, cash, cost_bps: float, drawdown: bool = True,
             pace: bool = True) -> BookPath:
    """Rules 7 and 8 over a book whose open-to-open return is ``R`` and open-to-close return is ``intra``."""
    n = len(target)
    e, hold, r = np.zeros(n), np.zeros(n), np.zeros(n)
    eq_open, peak, prev = 1.0, 1.0, 0.0
    fee = cost_bps / BPS
    for t in range(n):
        hold[t] = e[t - 1] if t > 0 else 0.0
        mark = eq_open * (1.0 + hold[t] * intra[t])
        if ftd_start[t]:
            peak = mark
        peak = max(peak, mark)
        want = target[t] * (dd_multiplier(mark / peak - 1.0) if drawdown else 1.0)
        e[t] = min(want, prev + STEP) if pace and want > prev else want
        prev = e[t]
        before = hold[t - 1] if t > 0 else 0.0
        r[t] = hold[t] * R[t] + (1.0 - hold[t]) * cash[t] - fee * abs(hold[t] - before)
        eq_open *= 1.0 + r[t]
    return BookPath(e, hold, r)

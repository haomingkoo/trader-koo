"""v3 specs built on price bars: H5, H6, H8, H9, H10, H13 (docs/research/hypotheses-v3-market-timing.md).

The numbers below are the registration, not settings; each names its section of the document.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eventstudy import Spec
from .v3_data import Frame
from .v3_ops import ema, fwd, gt_after, holds, lag, lt_after, outcome, pct_change, rmax, rmin, single, sma

# H5: range of the closes over t-20..t-1 no wider than 5% of their mean, held 10 sessions; outcome at 5 sessions;
# variants QQQ and horizons 3 and 10. Claim tested: a wick break is at least 5 points weaker than a close break.
H5_WINDOW = 20
H5_MAX_WIDTH = 0.05
H5_HELD = 10
H5_HORIZON = 5
H5_HORIZONS = (3, 10)
H5_WEAKER = 0.05
# H6: tail at least 2x the body and half the range, close in the far half, within 2% of the 60-session extreme.
H6_BODY_MULT = 2.0
H6_RANGE_SHARE = 0.5
H6_EXTREME_WINDOW = 60
H6_EXTREME_BAND = 0.02
H6_HORIZON = 2
H6_HORIZONS = (1, 3)
H6_MULTS = (1.5, 3.0)
# H8: close above the 5 DMA, 8 EMA, 10 DMA, 20 DMA, 21 EMA and 50 DMA; outcome SPY lower after 5; matched: below >= 4 of 6.
H8_SMA_WINDOWS = (5, 10, 20, 50)
H8_EMA_SPANS = (8, 21)
H8_HORIZON = 5
H8_MATCH_BELOW = 4
H8_FALSIFY_MARGIN = 0.03
# H9: close below the 8 EMA and 5 DMA on volume above its 20-session average, the day after the H8 state.
H9_EMA = 8
H9_SMA = 5
H9_VOLUME_WINDOW = 20
H9_HORIZON = 5
H9_SECONDARY = 10
# H10: FOMC decision day with a return of -0.5% or lower; outcome next close higher; stated claim 83%.
H10_DROP = -0.005
H10_CLAIM = 0.83
# H13: range under 0.60% of the previous close on t and t-1; pivot = highest high of sessions 4-5; outcomes at
# sessions 8-9 (closes) and 8-19 (lows). Published (trigger, matched) rates: (a) 44.2% vs 30.1%; (b) 38.4% vs 31.6% at 2.36%.
H13_RANGE = 0.0060
H13_PIVOT = (4, 5)
H13_CLOSE_DAYS = (8, 9)
H13_LOW_DAYS = (8, 19)
H13_DEPTH = 0.0236
H13_ALT_DEPTH = 0.02124
H13_PUBLISHED_LIFT = {"a": 0.442 - 0.301, "b": 0.384 - 0.316}
H13_MONTHS = 12


def _bars(f: Frame, t: str):
    b = f.bars[t]
    return b["open"], b["high"], b["low"], b["close"], b["volume"]


def averages(c: np.ndarray) -> np.ndarray:
    """(6, N): the 5, 10, 20 and 50 DMA and the 8 and 21 EMA."""
    return np.stack([*(sma(c, n) for n in H8_SMA_WINDOWS), *(ema(c, n) for n in H8_EMA_SPANS)])


def all_above(c: np.ndarray) -> np.ndarray:
    a = averages(c)
    return holds((c > a).all(axis=0), c, *a)


def h5(f: Frame, ticker: str = "SPY", horizons: tuple = (H5_HORIZON,)) -> list[Spec]:
    _, h, l, c, _ = _bars(f, ticker)
    hi, lo = lag(rmax(c, H5_WINDOW), 1), lag(rmin(c, H5_WINDOW), 1)
    mean = lag(pd.Series(c).rolling(H5_WINDOW).mean().to_numpy(), 1)
    ok = holds((hi - lo) / mean <= H5_MAX_WIDTH, hi, lo, mean)
    held = lag(pd.Series(ok.astype(float)).rolling(H5_HELD).min().to_numpy(), 1) == 1.0
    qual = ok & held
    a_long, a_short = qual & holds(c > hi, c, hi), qual & holds(c < lo, c, lo)
    b_long, b_short = qual & holds((h > hi) & (c <= hi), h, c, hi), qual & holds((l < lo) & (c >= lo), l, c, lo)
    none = qual & ~(a_long | a_short | b_long | b_short)
    specs = []
    for k in horizons:
        out = np.stack([gt_after(c, k), lt_after(c, k)])
        key = "H5" if (ticker, k) == ("SPY", H5_HORIZON) else f"H5-{ticker}-h{k}"
        role = "primary" if key == "H5" else "variant"
        sa, sb = np.where(a_long, 0, 1), np.where(b_long, 0, 1)
        specs.append(Spec(f"{key}-A", f"{ticker}: close beyond the range, same-direction close after {k} sessions "
                          "(long and short pooled)", role, a_long | a_short, sa, out, np.stack([none, none])))
        specs.append(Spec(f"{key}-B", f"{ticker}: wick beyond the range with the close back inside, same-direction close "
                          f"after {k}", "secondary", b_long | b_short, sb, out, np.stack([none, none])))
        specs.append(Spec(f"{key}-AvsB", f"{ticker}: close break against wick break, horizon {k}", "secondary",
                          a_long | a_short, sa, out, np.stack([b_long, b_short]),
                          note="effect = rate after a close break minus rate after a wick break"))
    return specs


def h6(f: Frame, ticker: str, horizon: int = H6_HORIZON, mult: float = H6_BODY_MULT, variant: bool = False) -> list[Spec]:
    o, h, l, c, _ = _bars(f, ticker)
    rng, body = h - l, np.abs(c - o)
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    top_ref, bot_ref = rmax(c, H6_EXTREME_WINDOW), rmin(c, H6_EXTREME_WINDOW)
    near_hi = holds(c >= (1 - H6_EXTREME_BAND) * top_ref, c, top_ref)
    near_lo = holds(c <= (1 + H6_EXTREME_BAND) * bot_ref, c, bot_ref)
    top = holds((upper >= mult * body) & (upper >= H6_RANGE_SHARE * rng) & (c - l <= H6_RANGE_SHARE * rng) & (rng > 0)
                & near_hi, o, h, l, c)
    bottom = holds((lower >= mult * body) & (lower >= H6_RANGE_SHARE * rng) & (c - l >= H6_RANGE_SHARE * rng) & (rng > 0)
                   & near_lo, o, h, l, c)
    key = f"H6-{ticker}" + (f"-h{horizon}-x{mult:g}" if variant else "")
    role = "variant" if variant else "primary"
    return [single(f"{key}-top", f"{ticker}: topping tail near the 60-session high: close(t+{horizon}) < close(t)", role,
                   top, lt_after(c, horizon), near_hi & ~top),
            single(f"{key}-bottom", f"{ticker}: bottoming tail near the 60-session low: close(t+{horizon}) > close(t)", role,
                   bottom, gt_after(c, horizon), near_lo & ~bottom)]


def h8(f: Frame, ticker: str) -> list[Spec]:
    c = f.bars[ticker]["close"]
    a = averages(c)
    below = holds((c < a).sum(axis=0) >= H8_MATCH_BELOW, c, *a)
    return [single(f"H8-{ticker}", f"{ticker}: close above all six averages: close(t+5) < close(t) (a short would have worked)",
                   "primary", all_above(c), lt_after(c, H8_HORIZON), below,
                   note="falsified if the down frequency exceeds the unconditional by 3 points or more")]


def h9(f: Frame) -> list[Spec]:
    _, _, _, c, v = _bars(f, "SPY")
    e8, s5, vol_avg = ema(c, H9_EMA), sma(c, H9_SMA), sma(v, H9_VOLUME_WINDOW)
    below = holds((c < e8) & (c < s5), c, e8, s5)
    prev = lag(all_above(c).astype(float), 1) == 1.0
    sig = below & holds(v > vol_avg, v, vol_avg) & prev
    base = below & ~prev
    return [single("H9", "SPY: close below 8 EMA and 5 DMA on high volume after the all-above state: close(t+5) < close(t)",
                   "primary", sig, lt_after(c, H9_HORIZON), base),
            single("H9-sec", "secondary: close(t+10) < close(t)", "secondary", sig, lt_after(c, H9_SECONDARY), base)]


def h10(f: Frame) -> list[Spec] | None:
    """None when the FOMC list is incomplete: the hypothesis is then blocked, not guessed."""
    if f.fomc is None:
        return None
    c = f.bars["SPY"]["close"]
    drop = holds(pct_change(c) <= H10_DROP, c)
    return [single("H10", "SPY: down FOMC decision day: close(t+1) > close(t)", "primary", f.fomc & drop, gt_after(c, 1),
                   ~f.fomc & drop, H10_CLAIM)]


def quiet_days(f: Frame, ticker: str) -> np.ndarray:
    _, h, l, c, _ = _bars(f, ticker)
    rng = (h - l) / lag(c, 1)
    return holds(rng < H13_RANGE, rng)


def h13_outcomes(f: Frame, ticker: str, depth: float = H13_DEPTH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outcomes if a trigger were at t: (a) below the trigger close on session 8 or 9, (a, both) on 8 and 9, (b) pivot reach."""
    _, h, l, c, _ = _bars(f, ticker)
    need = [fwd(c, d) for d in H13_CLOSE_DAYS]
    a = outcome((need[0] < c) | (need[1] < c), *need, c)
    a_both = outcome((need[0] < c) & (need[1] < c), *need, c)
    pivot = np.fmax(fwd(h, H13_PIVOT[0]), fwd(h, H13_PIVOT[1]))
    lows = np.stack([fwd(l, d) for d in range(H13_LOW_DAYS[0], H13_LOW_DAYS[1] + 1)])
    b = outcome(lows.min(axis=0) <= pivot * (1 - depth), pivot, *lows)
    return a, a_both, b


def h13(f: Frame, ticker: str) -> list[Spec]:
    q = quiet_days(f, ticker)
    sig = q & (lag(q.astype(float), 1) == 1.0)  # quiet on t and t-1, from data through t
    ended = sig & ~np.concatenate([q[1:], [True]])  # published convention: last session of the streak (needs t+1)
    month = f.month()
    a, a_both, b = h13_outcomes(f, ticker)
    _, _, b_alt = h13_outcomes(f, ticker, H13_ALT_DEPTH)
    base = np.stack([(month == m) & ~sig for m in range(H13_MONTHS)])

    def build(key, title, role, signal, o, note=""):
        return Spec(key, title, role, signal, month, np.tile(o, (H13_MONTHS, 1)), base, note=note)

    t = f"H13-{ticker}"
    last = "published convention: the trigger is the last session of the streak, known only once the next session is quiet or not"
    return [build(f"{t}-a", f"{ticker}: quiet days: close below the trigger close on session 8 or 9", "primary", sig, a),
            build(f"{t}-b", f"{ticker}: quiet days: low reaches pivot high - 2.36% in sessions 8-19", "primary", sig, b),
            build(f"{t}-a-both", f"{ticker}: (a) read as below on both session 8 and 9", "secondary", sig, a_both),
            build(f"{t}-b-2124", f"{ticker}: (b) at 2.124%", "secondary", sig, b_alt),
            build(f"{t}-a-lastday", f"{ticker}: (a) with the published last-session trigger", "variant", ended, a, last),
            build(f"{t}-b-lastday", f"{ticker}: (b) with the published last-session trigger", "variant", ended, b, last)]

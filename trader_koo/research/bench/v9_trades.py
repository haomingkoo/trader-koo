"""Daily-bar fills and exits of hypotheses v9 (docs/research/hypotheses-v9-champions.md), for many trades at once.

A trade enters at ``entry`` on session ``row`` and is walked one session at a time, every trade together. In each
session the order is fixed by the registration's fill rules:

1. an exit flagged at the previous close fills at this open (the next valid open; the last close if none);
2. the stop: an open at or below it fills at the open (gap through, not on the entry session), else a low at or
   below it fills at the stop;
3. a profit target fills at max(open, target) on part of the position;
4. close rules exit everything at the close: a listed spin-off, the day-one red close, the close before an
   earnings reaction session, a fixed exit day;
5. a partial sale at the close, after which the stop is raised to the entry;
6. rules that exit at the next open are flagged (trail, band, average, RSI divergence, volume reversal, hold);
7. stop raises that apply from the next session (breakeven).

Missing bars trigger nothing. A trade still open after the last session closes at the last close. Every price is
read at or before the session it is used on, and the arrays end at the last development session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AT_OPEN = 0
INTRADAY = 1
AT_CLOSE = 2
KIND_NAMES = ("stop_gap", "stop", "day_one_red", "earnings", "spinoff", "fixed_day", "next_open_rule", "max_hold",
              "window_end")
STOP_GAP, STOP, DAY_ONE, EARNINGS, SPINOFF, FIXED, NEXT_OPEN, MAX_HOLD, WINDOW_END = range(len(KIND_NAMES))
HALF = 0.5
SHORT_SMA = 10
LONG_SMA = 20
BAND_BUFFER = 0.98          # J Law: below the 10/20 band with a buffer (draft's 2%)
RSI_LEN = 14                # Wilder's RSI
HIGH_LOOKBACK = 19          # a 20-session closing high: above the 19 closes before it
DIV_MIN_GAP = 5
DIV_MAX_GAP = 60
REV_VOLUME_MULTIPLE = 2.0
REV_VOLUME_WINDOW = 50
REV_HIGH_WINDOW = 20


@dataclass(frozen=True)
class Exits:
    """One rule's exits; ``None`` or ``False`` switches a part off. Days count the entry session as day 1."""

    max_hold: int
    partial_day: int | None = None
    partial_frac: float = HALF
    trail: int | None = None             # SMA length; next open after a close below it, while it is above the stop
    band: bool = False                   # next open after a close below min(SMA10, SMA20) x BAND_BUFFER
    target: float | None = None          # multiple of the entry; sells ``target_frac`` there
    target_frac: float = 0.0
    day_one_red: bool = False
    breakeven_day: int | None = None     # stop at max(stop, entry) from this day on
    breakeven_gain: float | None = None  # ... or from the session after a close at entry x this
    sma_exit: int | None = None          # next open after a close below this SMA
    rsi_div: bool = False
    vol_rev: bool = False
    fixed_close_day: int | None = None
    earnings_exit: bool = False


@dataclass
class Bars:
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    v: np.ndarray
    o_next: np.ndarray       # next valid open at or after the row
    c_last: np.ndarray       # last valid close at or before the row
    sma: dict
    div: np.ndarray
    vrev: np.ndarray
    earn_before: np.ndarray  # the next session is an earnings reaction session
    spin: np.ndarray         # a listed spin-off exits at this close
    last: int


def wilder_rsi(c: np.ndarray, n: int = RSI_LEN) -> np.ndarray:
    d = pd.DataFrame(c).diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100.0 - 100.0 / (1.0 + up / down)
    return rsi.where(down > 0, 100.0).where(up.notna()).to_numpy()


def closing_high(c: np.ndarray, lookback: int = HIGH_LOOKBACK) -> np.ndarray:
    """Close above each of the ``lookback`` closes before it, all present."""
    f = pd.DataFrame(c)
    prior = f.shift(1).rolling(lookback, min_periods=lookback).max()
    return ((f > prior) & f.notna()).to_numpy()


def rsi_divergence(c: np.ndarray, rsi: np.ndarray | None = None) -> np.ndarray:
    """A 20-session closing high whose RSI is below the RSI of the latest such high 5 to 60 sessions earlier."""
    rsi = wilder_rsi(c) if rsi is None else rsi
    high = closing_high(c)
    n = c.shape[0]
    idx = pd.DataFrame(np.where(high, np.arange(n)[:, None], np.nan)).ffill().to_numpy()
    p = np.full(c.shape, np.nan)
    p[DIV_MIN_GAP:] = idx[:-DIV_MIN_GAP]
    rows = np.arange(n)[:, None]
    ok = high & np.isfinite(p) & (p >= rows - DIV_MAX_GAP)
    pi = np.where(ok, p, 0).astype(int)
    cols = np.broadcast_to(np.arange(c.shape[1]), c.shape)
    with np.errstate(invalid="ignore"):
        return ok & (rsi < rsi[pi, cols])


def volume_reversal(o: np.ndarray, h: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Volume at least twice the 50 sessions before, the highest high of 20 sessions, a close below open and prior close."""
    fv = pd.DataFrame(v)
    avg = fv.shift(1).rolling(REV_VOLUME_WINDOW, min_periods=REV_VOLUME_WINDOW).mean().to_numpy()
    top = pd.DataFrame(h).rolling(REV_HIGH_WINDOW, min_periods=REV_HIGH_WINDOW).max().to_numpy()
    prev = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    with np.errstate(invalid="ignore"):
        return (v >= REV_VOLUME_MULTIPLE * avg) & (h >= top) & (c < o) & (c < prev)


def make_bars(o, h, lo, c, v, reaction: np.ndarray | None = None, spin: np.ndarray | None = None) -> Bars:
    """Bars plus every exit input that does not depend on the trade. ``reaction`` marks earnings reaction sessions."""
    shape = o.shape
    reaction = np.zeros(shape, bool) if reaction is None else reaction
    earn_before = np.zeros(shape, bool)
    earn_before[:-1] = reaction[1:]
    fc = pd.DataFrame(c)
    sma = {k: fc.rolling(k, min_periods=k).mean().to_numpy() for k in (SHORT_SMA, LONG_SMA)}
    return Bars(o, h, lo, c, v, pd.DataFrame(o).bfill().to_numpy(), fc.ffill().to_numpy(), sma, rsi_divergence(c),
                volume_reversal(o, h, c, v), earn_before, np.zeros(shape, bool) if spin is None else spin,
                shape[0] - 1)


@dataclass
class Legs:
    """Per trade: an optional first (partial) leg and the final leg, each a session, a timing, a price and a share."""

    n1: np.ndarray
    when1: np.ndarray
    p1: np.ndarray
    f1: np.ndarray
    n2: np.ndarray
    when2: np.ndarray
    p2: np.ndarray
    kind: np.ndarray


def _close_price(b: Bars, r: np.ndarray, cols: np.ndarray) -> np.ndarray:
    cc = b.c[r, cols]
    return np.where(np.isfinite(cc), cc, b.c_last[r, cols])


class _Walk:
    """Mutable state of a batch of trades; ``done`` sets the final leg of the masked live trades."""

    def __init__(self, m: int):
        self.live = np.ones(m, bool)
        self.n1, self.when1 = np.full(m, -1), np.full(m, -1)
        self.p1, self.f1 = np.full(m, np.nan), np.zeros(m)
        self.n2, self.when2 = np.full(m, -1), np.full(m, -1)
        self.p2, self.kind = np.full(m, np.nan), np.full(m, -1)

    def done(self, mask, row, when, price, kind):
        mask = mask & self.live
        self.n2[mask] = row[mask] if np.ndim(row) else row
        self.when2[mask] = when[mask] if np.ndim(when) else when
        self.p2[mask] = price[mask] if np.ndim(price) else price
        self.kind[mask] = kind
        self.live &= ~mask

    def part(self, mask, row, when, price, frac):
        mask = mask & self.live & (self.f1 == 0)
        self.n1[mask], self.p1[mask], self.f1[mask] = row[mask], price[mask], frac
        self.when1[mask] = when[mask] if np.ndim(when) else when
        return mask


def simulate(b: Bars, ex: Exits, rows: np.ndarray, cols: np.ndarray, entry: np.ndarray, stop: np.ndarray) -> Legs:
    w = _Walk(len(rows))
    st = stop.astype(float).copy()
    pend = np.zeros(len(rows), bool)
    pend_kind = np.full(len(rows), NEXT_OPEN)
    for k in range(ex.max_hold + 1):
        r = rows + k
        over = r > b.last
        rr = np.minimum(r, b.last)
        w.done(over, b.last, AT_CLOSE, b.c_last[b.last, cols], WINDOW_END)
        # 1. exits flagged at the last close
        nxt = b.o_next[rr, cols]
        has = np.isfinite(nxt)
        for kind in (NEXT_OPEN, MAX_HOLD):
            m = pend & (pend_kind == kind)
            w.done(m, rr, np.where(has, AT_OPEN, AT_CLOSE), np.where(has, nxt, b.c_last[rr, cols]), kind)
        if k == ex.max_hold or not w.live.any():
            break
        oo, hh, ll = b.o[rr, cols], b.h[rr, cols], b.lo[rr, cols]
        with np.errstate(invalid="ignore"):
            # 2. the stop
            if k > 0:
                w.done(oo <= st, rr, AT_OPEN, oo, STOP_GAP)
            w.done(ll <= st, rr, INTRADAY, st, STOP)
            # 3. a profit target
            if ex.target is not None:
                tgt = entry * ex.target
                above = (oo >= tgt) if k > 0 else np.zeros(len(rows), bool)
                w.part(hh >= tgt, rr, np.where(above, AT_OPEN, INTRADAY), np.where(above, oo, tgt), ex.target_frac)
            # 4. close rules
            cc = _close_price(b, rr, cols)
            w.done(b.spin[rr, cols], rr, AT_CLOSE, cc, SPINOFF)
            if ex.day_one_red and k == 0:
                w.done(b.c[rr, cols] < entry, rr, AT_CLOSE, cc, DAY_ONE)
            if ex.earnings_exit:
                w.done(b.earn_before[rr, cols], rr, AT_CLOSE, cc, EARNINGS)
            if ex.fixed_close_day is not None and k == ex.fixed_close_day - 1:
                w.done(np.ones(len(rows), bool), rr, AT_CLOSE, cc, FIXED)
            # 5. a partial sale at the close
            raise_stop = np.zeros(len(rows), bool)
            if ex.partial_day is not None and k == ex.partial_day - 1:
                raise_stop |= w.part(np.ones(len(rows), bool), rr, AT_CLOSE, cc, ex.partial_frac)
            # 6. exits at the next open
            c0 = b.c[rr, cols]
            flag = np.zeros(len(rows), bool)
            if ex.trail is not None:
                s = b.sma[ex.trail][rr, cols]
                flag |= (s > st) & (c0 < s)
            if ex.band:
                flag |= c0 < np.minimum(b.sma[SHORT_SMA][rr, cols], b.sma[LONG_SMA][rr, cols]) * BAND_BUFFER
            if ex.sma_exit is not None:
                flag |= c0 < b.sma[ex.sma_exit][rr, cols]
            if ex.rsi_div:
                flag |= b.div[rr, cols]
            if ex.vol_rev:
                flag |= b.vrev[rr, cols]
            pend = w.live & flag
            pend_kind = np.full(len(rows), NEXT_OPEN)
            hold = w.live & ~pend & (k == ex.max_hold - 1)
            pend |= hold
            pend_kind[hold] = MAX_HOLD
            # 7. stop raises from the next session
            if ex.breakeven_day is not None and k == ex.breakeven_day - 2:
                raise_stop |= w.live
            if ex.breakeven_gain is not None:
                raise_stop |= w.live & (c0 >= entry * ex.breakeven_gain)
            st = np.where(raise_stop, np.maximum(st, entry), st)
    return Legs(w.n1, w.when1, w.p1, w.f1, w.n2, w.when2, w.p2, w.kind)


def buy_stop(b: Bars, sig: np.ndarray, cols: np.ndarray, trigger: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """First session in sig+1..sig+window whose high reaches the trigger, filled at max(open, trigger); -1 if none."""
    row, price = np.full(len(sig), -1), np.full(len(sig), np.nan)
    for j in range(1, window + 1):
        r = sig + j
        ok = (row < 0) & (r <= b.last)
        rr = np.minimum(r, b.last)
        with np.errstate(invalid="ignore"):
            hit = ok & (b.h[rr, cols] >= trigger)
        oo = b.o[rr, cols]
        row[hit] = rr[hit]
        price[hit] = np.where(np.isfinite(oo[hit]), np.maximum(oo[hit], trigger[hit]), trigger[hit])
    return row, price


def r_multiple(legs: Legs, entry: np.ndarray, stop: np.ndarray, cost: float) -> np.ndarray:
    """Net R of each trade: (sale proceeds after the fee - cost with the fee) / initial risk, per share."""
    p1 = np.where(legs.f1 > 0, legs.p1, 0.0)
    proceeds = (legs.f1 * p1 + (1.0 - legs.f1) * legs.p2) * (1.0 - cost)
    return (proceeds - entry * (1.0 + cost)) / (entry - stop)

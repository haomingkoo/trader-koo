"""The risk-sized portfolio of hypotheses v9, trade statistics and the sleeve blend.

Trades are decided by their rule alone (``v9_trades``); the book only chooses which to take and how large, from
the equity at the previous close. Within a session: exits at the open, then entries in priority order, then
intraday and close exits, then the mark at the close (adjusted close carried over a missing bar). Cash earns 0%.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .v2_trades import BPS
from .v9_trades import AT_OPEN

TIER_UP_MTD = 0.08        # Flanders: grow size once up about 8% for the month
TIER_UP = 2.0
TIER_DOWN = 0.5
BRAKE_R = -5.0            # Flanders: -5 adjusted R a month on trades opened that month
ADJUSTED_R = 0.01         # 1R = 1% of the account
MONTH_CHARS = 7
TOP_SHARE = 0.05
DEFAULT_RISK = 0.005
DEFAULT_CAP = 0.20
MIN_POSITION = 0.01


@dataclass
class Cands:
    sig: np.ndarray
    fill: np.ndarray
    col: np.ndarray
    entry: np.ndarray
    stop: np.ndarray
    n1: np.ndarray
    when1: np.ndarray
    p1: np.ndarray
    f1: np.ndarray
    n2: np.ndarray
    when2: np.ndarray
    p2: np.ndarray
    kind: np.ndarray
    prio: np.ndarray

    def take(self, idx: np.ndarray) -> "Cands":
        return Cands(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})


def concat(parts: list[Cands]) -> Cands:
    return Cands(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in Cands.__dataclass_fields__})


@dataclass(frozen=True)
class Sizing:
    risk: float = DEFAULT_RISK
    cap: float = DEFAULT_CAP
    gross: float = 1.0
    min_pos: float = MIN_POSITION
    tiers: bool = False
    equal_weight: float | None = None


@dataclass
class Book:
    equity: np.ndarray                    # at the closes of first..last
    trades: pd.DataFrame
    skipped: dict = field(default_factory=dict)
    invested: np.ndarray | None = None    # open positions at each close

    def returns(self) -> np.ndarray:
        return self.equity / np.concatenate([[1.0], self.equity[:-1]]) - 1.0


class _Pos:
    __slots__ = ("i", "shares", "rem", "month", "pnl", "notional", "risk_pct")

    def __init__(self, i, shares, month, notional, risk_pct):
        self.i, self.shares, self.rem, self.month, self.pnl = i, shares, 1.0, month, 0.0
        self.notional, self.risk_pct = notional, risk_pct


def run_book(c: Cands, close: np.ndarray, first: int, last: int, sizing: Sizing, cost_bps: float,
             dates: list[str] | None = None) -> Book:
    fee = cost_bps / BPS
    ok = (c.fill >= first) & (c.fill <= last) & np.isfinite(c.entry) & (c.entry > c.stop)
    order = np.flatnonzero(ok)
    order = order[np.lexsort((-np.nan_to_num(c.prio[order], nan=-np.inf), c.fill[order]))]
    by_fill: dict[int, list[int]] = {}
    for i in order:
        by_fill.setdefault(int(c.fill[i]), []).append(int(i))
    legs: dict[int, list[tuple[int, int, int]]] = {}
    held: dict[int, _Pos] = {}
    cash, eq_prev, m_start, m_pnl, month = 1.0, 1.0, 1.0, 0.0, 0
    equity, invested, done = np.zeros(last - first + 1), np.zeros(last - first + 1, int), []
    skipped = {"held": 0, "no_room": 0, "small": 0, "brake": 0}

    def sell(pos: _Pos, price: float, frac: float, final: bool):
        nonlocal cash, m_pnl
        cash += pos.shares * frac * price * (1.0 - fee)
        gain = pos.shares * frac * (price * (1.0 - fee) - c.entry[pos.i] * (1.0 + fee))
        pos.pnl += gain
        pos.rem -= frac
        if pos.month == month:
            m_pnl += gain
        if final:
            i = pos.i
            done.append((i, int(c.fill[i]), int(c.col[i]), pos.notional, pos.risk_pct, pos.pnl,
                         pos.pnl / (pos.shares * (c.entry[i] - c.stop[i])), int(c.n2[i]) - int(c.fill[i]) + 1,
                         int(c.kind[i])))
            del held[int(c.col[i])]

    def run_legs(d: int, at_open: bool):
        for when, i, leg in sorted(legs.get(d, [])):
            if (when == AT_OPEN) != at_open:
                continue
            pos = held[int(c.col[i])]
            if leg == 1:
                sell(pos, c.p1[i], c.f1[i], False)
            else:
                sell(pos, c.p2[i], pos.rem, True)

    for d in range(first, last + 1):
        if dates is not None and d > 0 and dates[d][:MONTH_CHARS] != dates[d - 1][:MONTH_CHARS]:
            month, m_start, m_pnl = month + 1, eq_prev, 0.0
        run_legs(d, at_open=True)
        for i in by_fill.get(d, []):
            col = int(c.col[i])
            if col in held:
                skipped["held"] += 1
                continue
            risk = sizing.risk
            if sizing.tiers:
                if m_pnl <= BRAKE_R * ADJUSTED_R * m_start:
                    skipped["brake"] += 1
                    continue
                mtd = eq_prev / m_start - 1.0
                risk *= TIER_UP if mtd >= TIER_UP_MTD else TIER_DOWN if mtd < 0 else 1.0
            gross_now = sum(p.shares * p.rem * close[d - 1, k] for k, p in held.items())
            room = sizing.gross * eq_prev - gross_now
            if room < sizing.min_pos * eq_prev:
                skipped["no_room"] += 1
                continue
            dist = (c.entry[i] - c.stop[i]) / c.entry[i]
            want = sizing.equal_weight * eq_prev if sizing.equal_weight else risk * eq_prev / dist
            notional = min(want, (sizing.equal_weight or sizing.cap) * eq_prev, room)
            if notional < sizing.min_pos * eq_prev:
                skipped["small"] += 1
                continue
            held[col] = _Pos(i, notional / c.entry[i], month, notional / eq_prev, risk if not sizing.equal_weight else None)
            cash -= notional * (1.0 + fee)
            if c.f1[i] > 0 and c.n1[i] >= 0:
                legs.setdefault(int(c.n1[i]), []).append((int(c.when1[i]), i, 1))
            legs.setdefault(int(c.n2[i]), []).append((int(c.when2[i]), i, 2))
        run_legs(d, at_open=False)
        eq_prev = cash + sum(p.shares * p.rem * close[d, k] for k, p in held.items())
        equity[d - first], invested[d - first] = eq_prev, len(held)
    cols = ["idx", "fill", "col", "notional", "risk_pct", "pnl", "r_net", "hold", "kind"]
    return Book(equity, pd.DataFrame(done, columns=cols), skipped, invested)


def trade_stats(r: np.ndarray, pnl: np.ndarray, hold: np.ndarray) -> dict:
    """Win rate, mean win and loss in R, and the share of profit made by the best 5% of trades (by profit)."""
    n = len(r)
    if n == 0:
        return {"n": 0}
    win = r > 0
    top = np.sort(pnl)[::-1][:math.ceil(TOP_SHARE * n)].sum()
    total, gross = pnl.sum(), pnl[pnl > 0].sum()
    return {"n": int(n), "win_rate": float(win.mean()), "mean_r": float(r.mean()),
            "mean_win_r": float(r[win].mean()) if win.any() else None,
            "mean_loss_r": float(r[~win].mean()) if (~win).any() else None,
            "top5_share_of_net": float(top / total) if total != 0 else None,
            "top5_share_of_gross_profit": float(top / gross) if gross > 0 else None,
            "median_hold": float(np.median(hold))}


def blend(r: np.ndarray, dates: list[str]) -> np.ndarray:
    """Daily return of equal sleeves reset at each month's first session; weights drift with returns in between."""
    k = r.shape[1]
    w = np.full(k, 1.0 / k)
    out = np.zeros(r.shape[0])
    for d in range(r.shape[0]):
        if d > 0 and dates[d][:MONTH_CHARS] != dates[d - 1][:MONTH_CHARS]:
            w = np.full(k, 1.0 / k)
        out[d] = w @ r[d]
        w = w * (1.0 + r[d]) / (1.0 + out[d])
    return out

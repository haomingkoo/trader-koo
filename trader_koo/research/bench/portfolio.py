"""Periodic top-N portfolio: selection rules and a vectorised next-open engine.

Why not ``next_open_baseline.simulate_portfolio``: it executes pre-dated
decisions with share rounding, ADV capacity caps and a fixed exit date, which
suits the paper campaign. This bench holds names for as long as the buffer and
minimum-hold rules say, so exits depend on later scores; the engine below
values each holding period with matrix slices instead. Fills are at the next
session's back-adjusted open; costs are bps of traded notional per side.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Market


DEFAULT_TOP_N = 20
DEFAULT_BUFFER_N = 40
DEFAULT_REVIEW_EVERY = 10
DEFAULT_MIN_HOLD = 20
BPS_PER_UNIT = 10_000
COST_SOLVE_ITERATIONS = 8


@dataclass(frozen=True)
class Rules:
    top_n: int = DEFAULT_TOP_N
    buffer_n: int = DEFAULT_BUFFER_N
    review_every: int = DEFAULT_REVIEW_EVERY
    min_hold: int = DEFAULT_MIN_HOLD


@dataclass
class Schedule:
    """Target holdings per execution session (index into Market.sessions)."""

    exec_idx: list[int]
    targets: list[list[str]]
    dropped_no_open: int = 0
    blocked_spinoff: int = 0  # buys skipped because the name's spin-off ex-date falls in the holding period


@dataclass
class Result:
    first_idx: int
    equity: np.ndarray            # close-of-session equity from first_idx; base 1.0 at that open
    trade_idx: np.ndarray         # session index of every fill
    trade_notional: np.ndarray    # notional of every fill (cost = bps * notional)
    period_returns: list[np.ndarray]  # open-to-open return of each held name, per period
    period_idx: list[int]         # fill session of each period in period_returns
    exits: list[tuple[int, int]] = field(default_factory=list)  # (session, column) of spin-off exits


def select_holdings(scores: pd.Series, held: dict[str, int], exec_idx: int, rules: Rules) -> list[str]:
    """Pick targets: keep holdings still in the top buffer_n or inside min_hold, fill to top_n.

    A holding with no score (left the index or has no price) is dropped even
    inside its minimum hold: it can no longer be ranked or valued.
    """
    s = scores.dropna()
    order = s.index.to_series().sort_values().index  # ticker order breaks ties
    ranked = s.loc[order].sort_values(ascending=False, kind="stable")
    rank = {t: i + 1 for i, t in enumerate(ranked.index)}
    keep = [t for t in held if t in rank and (rank[t] <= rules.buffer_n or exec_idx - held[t] < rules.min_hold)]
    keep = keep[: rules.top_n]
    fill = [t for t in ranked.index if t not in set(keep)][: max(rules.top_n - len(keep), 0)]
    return sorted(keep + fill)


def _next_fill(review_idx: list[int], i: int, n_sessions: int, end_idx: int | None) -> int:
    """First session of the next holding period (exclusive end of this one)."""
    nxt = review_idx[i + 1] + 1 if i + 1 < len(review_idx) else n_sessions
    return nxt if end_idx is None else min(nxt, end_idx + 1)


def _exited_since(market: Market, ticker: str, lo: int, hi: int) -> bool:
    col = market.columns.get(ticker)
    return col is not None and market.exit_between(col, lo, hi) is not None


def build_schedule(market: Market, review_idx: list[int], rules: Rules,
                   score_fn: Callable[[int, list[str]], pd.Series],
                   end_idx: int | None = None) -> Schedule:
    """Run selection at each review (signal at close of idx, fill at the open of idx+1).

    A name whose spin-off ex-date falls inside the coming holding period is not
    bought (the next-ranked name takes its place); a name already held is kept and
    sold at the close before the ex-date by ``simulate``.
    """
    held: dict[str, int] = {}
    out = Schedule([], [])
    prev_e = 0
    for i, s in enumerate(review_idx):
        e = s + 1
        if e >= len(market.sessions):
            break
        nxt = _next_fill(review_idx, i, len(market.sessions), end_idx)
        held = {t: v for t, v in held.items() if not _exited_since(market, t, prev_e, e - 1)}
        scores = score_fn(s, market.universe(s))
        blocked = [t for t in scores.index if t not in held and _exited_since(market, t, e, nxt - 1)]
        out.blocked_spinoff += len(blocked)
        chosen = select_holdings(scores.drop(blocked), held, e, rules)
        open_row = market.adj_open.iloc[e]
        tradable = [t for t in chosen if t in held or np.isfinite(open_row[t])]
        out.dropped_no_open += len(chosen) - len(tradable)
        held = {t: held.get(t, e) for t in tradable}
        out.exec_idx.append(e)
        out.targets.append(tradable)
        prev_e = e
    return out


def equal_weight_schedule(market: Market, review_idx: list[int], end_idx: int | None = None) -> Schedule:
    """Benchmark schedule: every member with a close at the review and an open at the fill."""
    out = Schedule([], [])
    prev: set[str] = set()
    for i, s in enumerate(review_idx):
        e = s + 1
        if e >= len(market.sessions):
            break
        nxt = _next_fill(review_idx, i, len(market.sessions), end_idx)
        open_row = market.adj_open.iloc[e]
        names = [t for t in market.universe(s) if np.isfinite(open_row[t])]
        keep = [t for t in names if t in prev or not _exited_since(market, t, e, nxt - 1)]
        out.blocked_spinoff += len(names) - len(keep)
        out.exec_idx.append(e)
        out.targets.append(keep)
        prev = {t for t in keep if not _exited_since(market, t, e, nxt - 1)}
    return out


def simulate(market: Market, schedule: Schedule, end_idx: int, cost_bps: float) -> Result:
    """Value the schedule, resetting to equal weight across the targets at every fill.

    The design says "equal weight", so each review trades every position back to
    1/len(targets) and pays costs on all of it. Letting winners drift instead
    (buy and hold between name changes) overweights past winners and compares
    badly against a benchmark that does rebalance.
    """
    c = cost_bps / BPS_PER_UNIT
    ex, tg = schedule.exec_idx, schedule.targets
    O, C = market.open_ff.to_numpy(), market.close_ff.to_numpy()
    cols = market.columns
    pos = np.zeros(O.shape[1])
    cash, eq_parts = 1.0, []
    t_idx: list[int] = []
    t_amt: list[float] = []
    period_ret: list[np.ndarray] = []
    period_idx: list[int] = []
    result_exits: list[tuple[int, int]] = []
    for k, e in enumerate(ex):
        if e > end_idx:
            break
        nxt = min(ex[k + 1], end_idx + 1) if k + 1 < len(ex) else end_idx + 1
        want = np.array([cols[t] for t in tg[k]], dtype=int)
        equity = cash + pos.sum()
        target = np.zeros_like(pos)
        if len(want):
            target[want] = equity / len(want)
        scale = 1.0
        for _ in range(COST_SOLVE_ITERATIONS):  # fee is charged on what is actually traded, after the fee
            scale = 1.0 - c * np.abs(scale * target - pos).sum() / equity if len(want) else 1.0
        new_pos = target * scale
        traded = np.abs(new_pos - pos)
        cash = equity - c * traded.sum() - new_pos.sum()
        for i in np.nonzero(traded)[0]:
            t_idx.append(e)
            t_amt.append(float(traded[i]))
        pos = new_pos
        held = np.nonzero(pos)[0]
        if len(held):
            ratio = C[e:nxt, held] / O[e, held]
            sold = []  # (position in held, exit session, gross proceeds multiple)
            for j, col in enumerate(held):
                q = market.exit_between(int(col), e, nxt - 1)
                if q is not None:  # sold at the close before a spin-off ex-date, then idle cash
                    gross = ratio[q - e, j]
                    ratio[q - e:, j] = gross * (1 - c)
                    sold.append((j, int(col), q, gross))
                    t_idx.append(q)
                    t_amt.append(float(pos[col] * gross))
                    result_exits.append((q, int(col)))
            eq_parts.append(cash + (ratio * pos[held]).sum(axis=1))
            pos = pos.copy()
            for j, col, q, gross in sold:
                cash += pos[col] * gross * (1 - c)
                pos[col] = 0.0
            if nxt <= end_idx:
                grow = O[nxt, held] / O[e, held]
                for j, col, q, gross in sold:
                    grow[j] = gross
                period_ret.append(grow - 1.0)
                period_idx.append(e)
                still = pos[held] > 0
                pos[held[still]] = pos[held[still]] * grow[still]
        else:
            eq_parts.append(np.full(nxt - e, cash))
            if nxt <= end_idx:
                period_ret.append(np.array([]))
                period_idx.append(e)
    first = ex[0]
    equity_path = np.concatenate(eq_parts) if eq_parts else np.array([])
    return Result(first, equity_path, np.array(t_idx, dtype=int), np.array(t_amt), period_ret, period_idx, result_exits)


def buy_and_hold(market: Market, ticker: str, first_idx: int, end_idx: int) -> np.ndarray:
    """Equity of holding ``ticker`` from the open of first_idx, closes through end_idx."""
    col = market.columns[ticker]
    return market.close_ff.to_numpy()[first_idx:end_idx + 1, col] / market.open_ff.to_numpy()[first_idx, col]

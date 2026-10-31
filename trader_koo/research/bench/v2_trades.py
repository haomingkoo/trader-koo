"""Entries, exits and net returns of event-study trades, for every (entry session, ticker) cell at once.

A trade enters at the open of session e (the signal was at the close of e-1) and exits at the open of
session X: the session after the first close below the exit average, or after the maximum hold, whichever
comes first (a maximum hold of H sessions means the exit is at the open of e + H at the latest). The same
exit rule is applied to every cell, so state baselines and placebo draws use exactly the hypothesis's rule.

Substitutions, each counted by the caller from the flag matrices returned here:
- the open at the exit session is missing: the next session's valid open is used (``fallback_next_open``),
  and if the ticker has no later open, its last close (``fallback_last_close``);
- a listed spin-off (spinoffs.csv) falls inside the hold: exit at the close of the session before the ex-date,
  as the portfolio bench does (``spinoff_exit``).
Entries whose maximum hold would run past the last development session are ``late`` and are not scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Market
from .v2_signals import next_true

BPS = 10_000.0


@dataclass
class Grid:
    gross: np.ndarray          # (N, T) exit price / entry price; NaN where the cell cannot be scored
    exit_row: np.ndarray       # (N, T) int: session of the exit (open), or of the close for spin-off exits
    late: np.ndarray           # bool: maximum hold runs past the last development session
    no_entry_open: np.ndarray  # bool: no open to enter at
    fallback_next_open: np.ndarray
    fallback_last_close: np.ndarray
    spinoff_exit: np.ndarray
    unscorable: np.ndarray     # bool: entry open exists but no exit price could be found


def build_grid(market: Market, open_: pd.DataFrame, close: pd.DataFrame, exit_ema: pd.DataFrame | None, max_hold: int) -> Grid:
    o, c = open_.to_numpy(), close.to_numpy()
    n, t = o.shape
    rows = np.arange(n)[:, None]
    if exit_ema is None:
        trig = np.broadcast_to(rows + max_hold - 1, (n, t))
    else:
        trig = np.minimum(next_true((c < exit_ema.to_numpy())), rows + max_hold - 1)
    return finish_grid(market, open_, close, (trig + 1).astype(int), max_hold)


def finish_grid(market: Market, open_: pd.DataFrame, close: pd.DataFrame, exit_row: np.ndarray, max_hold: int,
                fixed_price: np.ndarray | None = None) -> Grid:
    """Grid from each cell's exit session: exit open with its fallbacks, spin-off exits, late and unscorable flags.

    ``fixed_price`` (NaN where unused) is an exit price set by the rule itself, such as a stop filled during
    the session before ``exit_row``; it replaces the exit open but not a spin-off exit.
    """
    o, c = open_.to_numpy(), close.to_numpy()
    n, t = o.shape
    rows = np.arange(n)[:, None]
    exit_row = exit_row.copy()
    late = np.broadcast_to(rows + max_hold > n - 1, (n, t)).copy()
    safe = np.minimum(exit_row, n - 1)
    cols = np.broadcast_to(np.arange(t), (n, t))
    o_next = open_.bfill().to_numpy()
    c_last = close.ffill().to_numpy()
    at_exit = o[safe, cols]
    use_next = ~np.isfinite(at_exit) & np.isfinite(o_next[safe, cols])
    use_close = ~np.isfinite(at_exit) & ~use_next & np.isfinite(c_last[safe, cols])
    price = np.where(np.isfinite(at_exit), at_exit, np.where(use_next, o_next[safe, cols], c_last[safe, cols]))
    if fixed_price is not None:
        fixed = np.isfinite(fixed_price)
        use_next, use_close = use_next & ~fixed, use_close & ~fixed
        price = np.where(fixed, fixed_price, price)
    spin = np.zeros((n, t), dtype=bool)
    for col, qs in market.exits.items():
        for q in qs:
            if q >= n:  # an ex-date after the development window
                continue
            hit = (np.arange(n) <= q) & (exit_row[:, col] > q) & ~spin[:, col]
            price[hit, col] = c[q, col] if np.isfinite(c[q, col]) else c_last[q, col]
            exit_row[hit, col] = q
            spin[hit, col] = True
    use_next, use_close = use_next & ~spin, use_close & ~spin  # a spin-off exit replaced the price
    entry = o
    ok = np.isfinite(entry) & (entry > 0) & ~late
    gross = np.where(ok & np.isfinite(price), price / entry, np.nan)
    return Grid(gross, exit_row, late, ~np.isfinite(entry), use_next & ok, use_close & ok, spin & ok,
                ok & ~np.isfinite(gross))


def net(gross: np.ndarray, cost_bps: float) -> np.ndarray:
    """Net return of one round trip: buy 1 including the fee, sell proceeds less the fee (cost per side, bps of notional)."""
    c = cost_bps / BPS
    return gross * (1.0 - c) / (1.0 + c) - 1.0


def one_at_a_time(entries: np.ndarray, grid: Grid) -> np.ndarray:
    """Keep an entry only when the ticker's previous kept trade has exited at or before its entry open."""
    keep = np.zeros_like(entries, dtype=bool)
    for col in np.flatnonzero(entries.any(axis=0)):
        busy = -1
        for r in np.flatnonzero(entries[:, col]):
            if r >= busy:
                keep[r, col] = True
                busy = grid.exit_row[r, col] + (1 if grid.spinoff_exit[r, col] else 0)
    return keep


def entry_counts(signal_rows: np.ndarray, cols: np.ndarray, shape: tuple) -> np.ndarray:
    """Number of trades per cell when a signal on row s enters at row s + 1 (several signals may share a cell)."""
    w = np.zeros(shape, dtype=int)
    inside = signal_rows + 1 < shape[0]
    np.add.at(w, (signal_rows[inside] + 1, cols[inside]), 1)
    return w

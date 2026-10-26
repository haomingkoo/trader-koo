"""Statistics for the event studies of hypotheses v2 and v3 (docs/research/hypotheses-v3-market-timing.md, Common rules).

Everything here is pure numpy. A comparison is four per-session arrays of shape (strata, sessions):
independent-event counts and their outcome sums, baseline-session counts and their outcome sums. The
effect is the event rate minus the baseline rate, where the baseline rate is the stratum baseline rates
weighted by the share of events in each stratum (one stratum means a plain rate difference).

Bootstrap: moving blocks of consecutive sessions (not circular, start uniform over the first
N - block + 1 sessions), resampled with replacement until N sessions are drawn; the last block is cut
short. Events and baseline sessions are resampled together with the sessions that carry them. A
resample in which a stratum has events but no baseline session, or no event at all, has no defined
effect: it is left out of the distribution and the number left out is reported, not hidden.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from trader_koo.research.settings import get as _settings

_E = _settings().event_study
CHUNK_CELLS = 4_000_000  # array cells held at once while bootstrapping; memory bound only, no effect on results
NEVER = np.iinfo(np.int64).max


def independent(signal: np.ndarray, strict: bool = False) -> np.ndarray:
    """Signals with no earlier signal fewer than ``spacing_sessions`` sessions before them.

    The earlier signal can itself be dropped (hypotheses-v3: "signals within 5 sessions of an earlier
    signal are dropped"), so a run of signals counts once. ``strict`` reads "within 5" as 5 or fewer
    sessions apart, for the sensitivity count only.
    """
    idx = np.flatnonzero(signal)
    keep = np.zeros(len(signal), dtype=bool)
    if idx.size:
        gap = np.concatenate([[NEVER], np.diff(idx)])
        keep[idx] = gap > _E.spacing_sessions if strict else gap >= _E.spacing_sessions
    return keep


def estimate(ev: np.ndarray, ev_hit: np.ndarray, base: np.ndarray, base_hit: np.ndarray) -> tuple:
    """(event rate, baseline rate, effect) from sums shaped (..., strata); NaN where undefined."""
    n_ev = ev.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_ev = ev_hit.sum(-1) / n_ev
        base_rate = base_hit / base
        weight = ev / n_ev[..., None]
        needed = (ev > 0) & ~(base > 0)
        rate_base = np.where(needed.any(-1), np.nan, np.nansum(np.where(ev > 0, weight * base_rate, 0.0), -1))
    rate_ev = np.where(n_ev > 0, rate_ev, np.nan)
    rate_base = np.where(n_ev > 0, rate_base, np.nan)
    return rate_ev, rate_base, rate_ev - rate_base


def point(parts: np.ndarray) -> dict:
    """Full-sample estimate from ``parts`` shaped (4, strata, sessions)."""
    s = parts.sum(-1).T  # (strata, 4)
    rate_ev, rate_base, effect = estimate(s[:, 0], s[:, 1], s[:, 2], s[:, 3])
    return {"n_events": float(s[:, 0].sum()), "rate_event": float(rate_ev), "rate_base": float(rate_base),
            "effect": float(effect)}


@dataclass
class Bootstrap:
    effect: np.ndarray       # effect of every resample with a defined effect
    rate_event: np.ndarray   # event rate of the same resamples
    undefined: int           # resamples left out


def bootstrap(parts: np.ndarray, seed: int, block: int = _E.bootstrap_block,
              resamples: int = _E.bootstrap_resamples) -> Bootstrap:
    _, k, n = parts.shape
    if n < block:
        raise ValueError(f"{n} sessions are fewer than one block of {block}")
    n_blocks = -(-n // block)
    rem = n - (n_blocks - 1) * block
    cs = np.concatenate([np.zeros((4, k, 1)), parts.cumsum(-1)], axis=-1)
    rng = np.random.default_rng(seed)
    chunk = max(1, CHUNK_CELLS // (4 * k * n_blocks))
    eff, rate = [], []
    for lo in range(0, resamples, chunk):
        starts = rng.integers(0, n - block + 1, size=(min(chunk, resamples - lo), n_blocks))
        head = starts[:, :-1]
        tot = (cs[:, :, head + block] - cs[:, :, head]).sum(-1) + cs[:, :, starts[:, -1] + rem] - cs[:, :, starts[:, -1]]
        rate_ev, _, effect = estimate(tot[0].T, tot[1].T, tot[2].T, tot[3].T)
        eff.append(effect)
        rate.append(rate_ev)
    effect, rate_event = np.concatenate(eff), np.concatenate(rate)
    ok = np.isfinite(effect)
    return Bootstrap(effect[ok], rate_event[ok], int((~ok).sum()))


def resample_indices(n: int, seed: int, block: int = _E.bootstrap_block,
                     resamples: int = _E.bootstrap_resamples) -> Iterator[np.ndarray]:
    """The same block resampling as ``bootstrap``, as session index vectors (for statistics that need them)."""
    n_blocks = -(-n // block)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    for _ in range(resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        yield (starts[:, None] + offsets).ravel()[:n]


def p_two_sided(dist: np.ndarray) -> float:
    """Share of resamples on the far side of zero, doubled, with the +1 correction; 1.0 for an empty distribution."""
    if dist.size == 0:
        return 1.0
    below, above = (dist <= 0).sum(), (dist >= 0).sum()
    return float(min(1.0, 2.0 * (min(below, above) + 1) / (dist.size + 1)))


def interval(dist: np.ndarray, alpha: float) -> tuple[float, float] | None:
    """Percentile interval of level 1 - alpha, or None when no resample had a defined effect."""
    if dist.size == 0:
        return None
    lo, hi = np.quantile(dist, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def holm(p: list[float], alpha: float = _E.alpha) -> tuple[list[float], list[float], list[bool]]:
    """Holm step-down over all tests registered together.

    Returns (adjusted p, interval alpha, rejected). A test without a p-value (insufficient or blocked)
    enters as 1.0: it counts toward the family size and can never be rejected. The interval alpha of
    the test at rank k (0-based, by ascending p) is alpha / (m - k) up to the first step that fails and
    that failing step's alpha from there on, so an interval excludes zero exactly when its test is rejected.
    """
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    adj, level, rejected = [0.0] * m, [alpha / m] * m, [False] * m
    running, failed_level, still = 0.0, None, True
    for rank, i in enumerate(order):
        step = alpha / (m - rank)
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
        if still and p[i] < step:
            rejected[i], level[i] = True, step
        else:
            if still:
                failed_level, still = step, False
            level[i] = failed_level
    return adj, level, rejected

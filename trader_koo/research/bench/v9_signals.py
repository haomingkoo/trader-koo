"""Signals of hypotheses v9 (docs/research/hypotheses-v9-champions.md). Each number is the registration and
names its source there; changing one is a new version. Matrices are (sessions x tickers) and a signal at row t
reads nothing after the close of t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .v9_trades import closing_high

# H1a / H1b (Flanders, Qullamaggie, Stockbee)
GAP = 0.10
VOLUME_MULTIPLE = 3.0
VOLUME_WINDOW = 50
HIGH_WINDOW = 252
RS_CUT = 90.0
RS_WEIGHTS = ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2))
QUIET_LOOKBACK = 63
QUIET_MAX = 0.30
# H2 (J Law)
SURGE = 0.25
MIN_DOLLAR_VOLUME = 100e6
DOLLAR_WINDOW = 20
PULLBACK_FIRST = 5
PULLBACK_LAST = 30
BAND_TOUCH = 1.01
BAND_HOLD = 0.98
PULLBACK_VOLUME_DAYS = 3
CATALYST_VOLUME_DAYS = 4
STOP_BUFFER = 0.99
SHORT = 10
LONG = 20
# H3 (Sean Ryan, draft rule C)
BASE_FROM = 60          # base over t-60..t-21
BASE_TO = 21
BASE_RANGE = 0.25
BURST_FROM = 20         # burst over t-20..t-6, measured from close(t-21)
BURST_TO = 6
BURST_MIN = 0.20
BURST_MAX = 1.00
BURST_VOLUME = 1.5
REST_DAYS = 6                        # t-5..t
REST_HOLD = 0.85
REST_RANGE = 0.12
SETUP_SPACING = 5
# H4 (Stockbee)
BURST_UP = 1.04
MIN_VOLUME = 100_000
RANGE_MEDIAN_DAYS = 7
PRIOR_RUN_DAYS = 3                   # close(t-1) / close(t-4)
PRIOR_RUN_MAX = 0.08
UP_STREAK = 3
# H7
AGREE_WINDOW = 10


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[:-k] if k else x
    return out


def _roll(x: np.ndarray, n: int, how: str) -> np.ndarray:
    return getattr(pd.DataFrame(x).rolling(n, min_periods=n), how)().to_numpy()


def gap_volume(o, c, v, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Open at least 10% over the prior close, volume at least 3x the mean of the 50 sessions before (all present)."""
    ok = rows >= VOLUME_WINDOW
    g = np.where(ok, rows, VOLUME_WINDOW)
    window = np.stack([v[g - d, cols] for d in range(1, VOLUME_WINDOW + 1)])
    complete = np.isfinite(window).all(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(complete, window.mean(axis=0), np.nan)
        return ok & complete & (o[g, cols] / c[g - 1, cols] - 1.0 >= GAP) & (v[g, cols] >= VOLUME_MULTIPLE * avg)


def new_high(c, rows: np.ndarray, cols: np.ndarray, window: int = HIGH_WINDOW) -> np.ndarray:
    """Close at or above the highest of the ``window - 1`` closes before it, all ``window`` closes present."""
    ok = rows >= window - 1
    g = np.where(ok, rows, window - 1)
    win = np.stack([c[g - d, cols] for d in range(window)])
    with np.errstate(invalid="ignore"):
        return ok & np.isfinite(win).all(axis=0) & (win[0] >= win[1:].max(axis=0))


def high_since_start(c, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Close at or above every earlier stored close in the data (variant: the longest high available)."""
    prior = pd.DataFrame(c).shift(1).expanding().max().to_numpy()
    with np.errstate(invalid="ignore"):
        return np.isfinite(c[rows, cols]) & (c[rows, cols] >= prior[rows, cols])


def quiet(c, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    ok = rows >= QUIET_LOOKBACK
    g = np.where(ok, rows, QUIET_LOOKBACK)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ok & (c[g - 1, cols] / c[g - QUIET_LOOKBACK, cols] - 1.0 <= QUIET_MAX)


def rs_percentile(c: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Percentile (0-100] of the IBD-style weighted return among members with a score; NaN for non-members."""
    score = sum(w * (c / _shift(c, n) - 1.0) for n, w in RS_WEIGHTS)
    score = np.where(member & np.isfinite(score), score, np.nan)
    return pd.DataFrame(score).rank(axis=1, pct=True).to_numpy() * 100.0


def dollar_volume(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    return pd.DataFrame(c * v).rolling(DOLLAR_WINDOW, min_periods=DOLLAR_WINDOW).median().to_numpy()


def volume_surge(v: np.ndarray) -> np.ndarray:
    avg = pd.DataFrame(v).shift(1).rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        return v >= VOLUME_MULTIPLE * avg


def burst(h, lo, c, v) -> np.ndarray:
    """Stockbee's 4% burst at the close of t (rules 1 to 5); ``c`` is the stored close, ranges are adjusted."""
    c1, c2, c3, c4 = (_shift(c, k) for k in range(1, UP_STREAK + 2))
    rng = h - lo
    med = _shift(_roll(rng, RANGE_MEDIAN_DAYS, "median"), 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = c / c1 >= BURST_UP
        vol = (v > _shift(v, 1)) & (v >= MIN_VOLUME)
        streak = (c1 > c2) & (c2 > c3) & (c3 > c4)
        quiet_day = (_shift(rng, 1) < med) | (c1 < c2)
        no_run = c1 / c4 - 1.0 < PRIOR_RUN_MAX
    return up & vol & ~streak & quiet_day & no_run


def second_move(h, lo, c, v) -> np.ndarray:
    """Sean Ryan's setup at the close of t: base, burst on volume, a tight rest that holds (draft rule C)."""
    span_base = BASE_FROM - BASE_TO + 1
    span_burst = BURST_FROM - BURST_TO + 1
    base_hi = _shift(_roll(h, span_base, "max"), BASE_TO)
    base_lo = _shift(_roll(lo, span_base, "min"), BASE_TO)
    base_v = _shift(_roll(v, span_base, "mean"), BASE_TO)
    top = _shift(_roll(c, span_burst, "max"), BURST_TO)
    burst_v = _shift(_roll(v, span_burst, "mean"), BURST_TO)
    start = _shift(c, BURST_FROM + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rise = top / start - 1.0
        return ((base_hi / base_lo - 1.0 <= BASE_RANGE) & (rise >= BURST_MIN) & (rise <= BURST_MAX)
                & (burst_v >= BURST_VOLUME * base_v) & (_roll(c, REST_DAYS, "min") >= REST_HOLD * top)
                & (_roll(h, REST_DAYS, "max") / _roll(lo, REST_DAYS, "min") - 1.0 <= REST_RANGE))


def space_out(mask: np.ndarray, spacing: int = SETUP_SPACING) -> np.ndarray:
    """Keep a setup only if no kept setup of the same stock lies in the ``spacing`` sessions before it."""
    keep = np.zeros_like(mask)
    for col in np.flatnonzero(mask.any(axis=0)):
        last = -spacing - 1
        for r in np.flatnonzero(mask[:, col]):
            if r - last > spacing:
                keep[r, col], last = True, r
    return keep


def pullbacks(o, h, lo, c, v, spy, e_rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """J Law's first pullback after each catalyst session: (pullback session t or -1, stop)."""
    del o  # the entry open is read by the caller
    sma_s, sma_l = _roll(c, SHORT, "mean"), _roll(c, LONG, "mean")
    hi20 = closing_high(c)
    spy_ret = np.concatenate([[np.nan], spy[1:] / spy[:-1] - 1.0])
    ret = np.vstack([np.full((1, c.shape[1]), np.nan), c[1:] / c[:-1] - 1.0])
    last = c.shape[0] - 1
    t_out, stop_out = np.full(len(e_rows), -1), np.full(len(e_rows), np.nan)
    for i, (e, k) in enumerate(zip(e_rows, cols)):
        cat_v = np.nanmean(v[e:e + CATALYST_VOLUME_DAYS, k])
        for t in range(e + PULLBACK_FIRST, min(e + PULLBACK_LAST, last) + 1):
            if not hi20[e + 1:t, k].any():
                continue
            seg = c[e:t, k]
            if not np.isfinite(seg).any():
                continue
            peak = e + int(np.nanargmax(seg))
            win = np.arange(peak + 1, t + 1)
            band_hi, band_lo = max(sma_s[t, k], sma_l[t, k]), min(sma_s[t, k], sma_l[t, k])
            if not (lo[t, k] <= band_hi * BAND_TOUCH and c[t, k] >= band_lo * BAND_HOLD):
                continue
            if not np.nanmean(v[t - PULLBACK_VOLUME_DAYS + 1:t + 1, k]) < cat_v:
                continue
            down = spy_ret[win] < 0
            if not down.any() or not (ret[win[down], k] >= spy_ret[win[down]]).all():
                continue
            t_out[i], stop_out[i] = t, np.nanmin(lo[win, k]) * STOP_BUFFER
            break
    return t_out, stop_out


def agreement(fam: dict[str, list[tuple[int, int]]], window: int = AGREE_WINDOW) -> list[tuple[int, int, str]]:
    """(t, col, triggering family) where a family fires on t and another fired on the same stock in t-window+1..t.

    Families are given in priority order; when several qualify on one session the first one triggers.
    """
    by = {f: {} for f in fam}
    for f, sigs in fam.items():
        for r, col in sigs:
            by[f].setdefault(col, []).append(r)
    out, seen = [], set()
    for f in fam:
        for r, col in sorted(set(fam[f])):
            if (r, col) in seen:
                continue
            others = (g for g in fam if g != f)
            if any(r - window < q <= r for g in others for q in by[g].get(col, ())):
                out.append((r, col, f))
                seen.add((r, col))
    return out

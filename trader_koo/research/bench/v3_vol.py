"""v3 specs built on the VIX: H1, H2, H3, H4, H7, H11 (docs/research/hypotheses-v3-market-timing.md).

The numbers below are the registration, not settings; each names its section of the document. A change
is a new version of the document.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eventstudy import Spec
from .v3_data import Frame
from .v3_ops import (TERCILES, bands_width, fwd, gt_after, holds, lt_after, outcome, pct_change, rmax, rmin, rquantile,
                     single, terciles)

# H1: signal VIX one-day change <= -15%; matched band -15% to -5%; variants -12% and -20%.
H1_DROP = -0.15
H1_BAND_LOW = -0.15
H1_BAND_HIGH = -0.05
H1_VARIANTS = (-0.12, -0.20)
# H2: signal VIX one-day change >= +20%; matched band +8% to +20%; stated claim 85%; variants +25% and +40%.
H2_JUMP = 0.20
H2_BAND_LOW = 0.08
H2_CLAIM = 0.85
H2_VARIANTS = (0.25, 0.40)
# H3: Bollinger(20, 2) width at or below the trailing-252 10th percentile; outcome VIX +5% in 3 sessions;
# secondary max close >= 1.10 x close; variants 5th and 20th percentile, horizons 2 and 5.
BB_WINDOW = 20
BB_K = 2.0
BB_TRAIL = 252
H3_PCT = 0.10
H3_HORIZON = 3
H3_RISE = 0.05
H3_SPIKE = 1.10
H3_PCT_VARIANTS = (0.05, 0.20)
H3_HORIZON_VARIANTS = (2, 5)
# H4: SPY and VIX compressed, outcome a close 3% or more away within 10 sessions; terciles of 20-session realised vol.
H4_EXCURSION = 0.03
H4_HORIZON = 10
H4_VOL_WINDOW = 20
# H7: SPY close at a 10-session low and VIX below its 10-session closing high; outcome SPY higher after 3; variants 5, 20.
H7_WINDOW = 10
H7_HORIZON = 3
H7_VARIANTS = (5, 20)
# H11: Fridays, VIX down on the day; stated claim 85% (the 2024+ claim is checked in the sealed window only).
H11_WEEKDAY = 4
H11_CLAIM = 0.85


def _one_day(f: Frame) -> tuple[np.ndarray, np.ndarray]:
    return pct_change(f.bars["^VIX"]["close"]), f.bars["^VIX"]["close"]


def h1(f: Frame) -> list[Spec]:
    r, _ = _one_day(f)
    spy = f.bars["SPY"]["close"]
    prim, out = [], {}
    for name, thr in (("", H1_DROP), *[(f"-{abs(int(v * 100))}", v) for v in H1_VARIANTS]):
        sig = holds(r <= thr, r)
        base = holds((r >= H1_BAND_LOW) & (r <= H1_BAND_HIGH) & ~sig, r)
        prim.append(single(f"H1{name}", f"VIX one-day change <= {thr:.0%}: SPY close(t+2) < close(t)",
                           "primary" if not name else "variant", sig, lt_after(spy, 2), base))
        if not name:
            any_down = outcome((fwd(spy, 1) < spy) | (fwd(spy, 2) < fwd(spy, 1)), fwd(spy, 1), fwd(spy, 2), spy)
            prim.append(single("H1-sec", "secondary: a down close-to-close day in t+1, t+2", "secondary", sig, any_down, base))
    return prim


def h2(f: Frame) -> list[Spec]:
    r, vix = _one_day(f)
    spy = f.bars["SPY"]["close"]
    specs = []
    for name, thr in (("", H2_JUMP), *[(f"+{int(v * 100)}", v) for v in H2_VARIANTS]):
        sig = holds(r >= thr, r)
        base = holds((r >= H2_BAND_LOW) & (r < H2_JUMP) & ~sig, r)
        role = "primary" if not name else "variant"
        specs.append(single(f"H2a{name}", f"VIX one-day change >= {thr:.0%}: SPY close(t+2) > close(t)", role, sig,
                            gt_after(spy, 2), base, H2_CLAIM if not name else None))
        specs.append(single(f"H2b{name}", f"VIX one-day change >= {thr:.0%}: VIX close(t+2) < close(t)", role, sig,
                            lt_after(vix, 2), base))
        if not name:
            specs.append(single("H2a-sec", "secondary: SPY close(t+1) > close(t)", "secondary", sig, gt_after(spy, 1), base))
            specs.append(single("H2b-sec", "secondary: VIX close(t+1) < close(t)", "secondary", sig, lt_after(vix, 1), base))
    return specs


def compression(c: np.ndarray, pct: float) -> np.ndarray:
    """Band width at or below the ``pct`` quantile of the trailing 252 widths (window includes t)."""
    w = bands_width(c, BB_WINDOW, BB_K)
    return holds(w <= rquantile(w, BB_TRAIL, pct), w, rquantile(w, BB_TRAIL, pct))


def _rise(vix: np.ndarray, k: int) -> np.ndarray:
    return outcome(fwd(vix, k) / vix - 1.0 >= H3_RISE, fwd(vix, k), vix)


def h3(f: Frame) -> list[Spec]:
    vix = f.bars["^VIX"]["close"]
    tier = terciles(vix, BB_TRAIL)
    specs = []

    def build(key, title, role, pct, k, out_fn=_rise):
        comp = compression(vix, pct)
        sig = comp & (tier >= 0)
        base = np.stack([(tier == j) & ~comp for j in range(TERCILES)])
        o = out_fn(vix, k)
        return Spec(key, title, role, sig, np.maximum(tier, 0), np.tile(o, (TERCILES, 1)), base,
                    note="matched baseline: not compressed, same VIX tercile of the trailing 252 sessions")

    specs.append(build("H3", "VIX compressed: VIX +5% or more after 3 sessions", "primary", H3_PCT, H3_HORIZON))
    spike = lambda v, k: outcome(np.fmax.reduce([fwd(v, j) for j in range(1, k + 1)]) >= H3_SPIKE * v,
                                 *[fwd(v, j) for j in range(1, k + 1)], v)
    specs.append(build("H3-sec", "secondary: max VIX close in t+1..t+3 >= 1.10 x close", "secondary", H3_PCT, H3_HORIZON, spike))
    up = lambda v, k: outcome(fwd(v, k) > v, fwd(v, k), v)
    specs.append(build("H3-always-up", "comparison: VIX higher after 3 sessions (an always-up call)", "secondary",
                       H3_PCT, H3_HORIZON, up))
    for pct in H3_PCT_VARIANTS:
        specs.append(build(f"H3-p{int(pct * 100)}", f"variant: width percentile {pct:.0%}", "variant", pct, H3_HORIZON))
    for k in H3_HORIZON_VARIANTS:
        specs.append(build(f"H3-h{k}", f"variant: horizon {k} sessions", "variant", H3_PCT, k))
    return specs


def h4(f: Frame) -> list[Spec]:
    vix, spy = f.bars["^VIX"]["close"], f.bars["SPY"]["close"]
    spy_comp = compression(spy, H3_PCT)
    sig = compression(vix, H3_PCT) & spy_comp
    logret = np.log(spy / np.concatenate([[np.nan], spy[:-1]]))
    vol = pd.Series(logret).rolling(H4_VOL_WINDOW).std().to_numpy()
    tier = terciles(vol, BB_TRAIL)
    fw = np.stack([fwd(spy, j) for j in range(1, H4_HORIZON + 1)])
    move = np.abs(fw / spy - 1.0)
    o = outcome(move.max(axis=0) >= H4_EXCURSION, *fw, spy)
    base = np.stack([(tier == j) & ~spy_comp for j in range(TERCILES)])
    return [Spec("H4", "VIX and SPY compressed: SPY close 3% or more from close(t) within 10 sessions", "primary",
                 sig & (tier >= 0), np.maximum(tier, 0), np.tile(o, (TERCILES, 1)), base,
                 note="matched baseline: SPY not compressed, same tercile of 20-session realised volatility")]


def h7(f: Frame) -> list[Spec]:
    spy, vix = f.bars["SPY"]["close"], f.bars["^VIX"]["close"]
    out = gt_after(spy, H7_HORIZON)
    specs = []
    for w in (H7_WINDOW, *H7_VARIANTS):
        low = holds(spy <= rmin(spy, w), spy, rmin(spy, w))
        vmax = rmax(vix, w)
        sig = low & holds(vix < vmax, vix, vmax)
        base = low & holds(vix >= vmax, vix, vmax)
        specs.append(single("H7" if w == H7_WINDOW else f"H7-w{w}",
                            f"SPY at a {w}-session closing low, VIX not at a {w}-session closing high: SPY close(t+3) > close(t)",
                            "primary" if w == H7_WINDOW else "variant", sig, out, base))
    return specs


def h11(f: Frame) -> list[Spec]:
    vix = f.bars["^VIX"]["close"]
    out = outcome(vix < np.concatenate([[np.nan], vix[:-1]]), vix, np.concatenate([[np.nan], vix[:-1]]))
    fri = f.weekday() == H11_WEEKDAY
    return [single("H11", "Friday: VIX close(t) < close(t-1)", "primary", fri, out, ~fri,
                   note="baseline: Monday to Thursday; the 2024+ claim lies in the sealed window")]

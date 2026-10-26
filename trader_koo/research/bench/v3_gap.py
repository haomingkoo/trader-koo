"""H12 (unfilled gaps are filled), docs/research/hypotheses-v3-market-timing.md.

The matched baseline is a different number for every event: the share of non-gap sessions whose price comes
back within 10 sessions to a level the same distance (as a fraction of the open) below the open for an up
gap, above it for a down gap. So it is bootstrapped with session weights instead of the stratum sums of
``eventstats``: each resample gives every session a weight (how many times its block draws it), and the
baseline share is recomputed from the weighted non-gap sessions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import eventstats as es
from .eventstudy import Evaluation, Spec
from .v3_data import Frame
from .v3_ops import holds, lag, outcome

H12_MAX_GAP = 0.015    # gap under 1.5% of the previous close
H12_WINDOW = 10        # filled within t..t+10


def _ahead(x: np.ndarray, how: str) -> np.ndarray:
    """min / max of x over t..t+10 (NaN if the window leaves the data or has a gap)."""
    s = pd.Series(x[::-1]).rolling(H12_WINDOW + 1)
    return (s.min() if how == "min" else s.max()).to_numpy()[::-1]


def gap_arrays(f: Frame, ticker: str = "SPY") -> dict:
    b = f.bars[ticker]
    o, h, l, c = b["open"], b["high"], b["low"], b["close"]
    pc, ph, pl = lag(c, 1), lag(h, 1), lag(l, 1)
    up, dn = holds(o > ph, o, ph), holds(o < pl, o, pl)
    ex = f.ex_dividend[ticker]
    small_up = up & holds(o / pc - 1 < H12_MAX_GAP, o, pc)
    small_dn = dn & holds(1 - o / pc < H12_MAX_GAP, o, pc)
    min_l, max_h = _ahead(l, "min"), _ahead(h, "max")
    return {
        "sig_up": small_up & ~ex, "sig_dn": small_dn & ~ex, "excluded_ex_dividend": int(((small_up | small_dn) & ex).sum()),
        "out_up": outcome(min_l <= pc, min_l, pc), "out_dn": outcome(max_h >= pc, max_h, pc),
        "dist_up": (o - pc) / o, "dist_dn": (pc - o) / o,
        "depth_up": 1.0 - min_l / o, "depth_dn": max_h / o - 1.0,
        "nongap": ~(up | dn) & ~ex & np.isfinite(o) & np.isfinite(ph) & np.isfinite(pl), "ex_nongap_excluded": int((~(up | dn) & ex).sum()),
    }


def _weighted_rates(w, ev, hits, dist, groups):
    n = w[ev].sum()
    if n <= 0:
        return np.nan, np.nan
    base = 0.0
    for sel, (order, vals) in groups:
        wb = w[order]
        tot = wb.sum()
        if tot <= 0 and sel.any():
            return np.nan, np.nan
        suffix = np.concatenate([np.cumsum(wb[::-1])[::-1], [0.0]])
        pos = np.searchsorted(vals, dist[sel], side="left")
        base += (w[ev[sel]] * (suffix[pos] / tot)).sum() if sel.any() else 0.0
    return (w[ev] * hits).sum() / n, base / n


def h12(f: Frame, seed: int) -> Evaluation:
    g = gap_arrays(f)
    n = f.n
    signal = g["sig_up"] | g["sig_dn"]
    out = np.stack([g["out_up"], g["out_dn"]])
    stratum = np.where(g["sig_dn"], 1, 0)
    cols = np.arange(n)
    ev_out = np.where(signal, out[stratum, cols], np.nan)
    defined = np.isfinite(ev_out)
    indep = es.independent(signal) & defined
    strict = es.independent(signal, strict=True) & defined
    spec = Spec("H12", "SPY: small gap, price back to the previous close within 10 sessions", "primary", signal, stratum,
                out, np.zeros_like(out, dtype=bool),
                note=f"matched baseline per event; {g['excluded_ex_dividend']} gap sessions on ex-dividend dates excluded")
    ev = np.flatnonzero(indep)
    is_dn = stratum[ev] == 1
    hits = ev_out[ev]
    dist = np.where(is_dn, g["dist_dn"][ev], g["dist_up"][ev])
    groups = []
    for dn, val in ((False, g["depth_up"]), (True, g["depth_dn"])):
        ok = np.flatnonzero(g["nongap"] & np.isfinite(val))
        order = ok[np.argsort(val[ok], kind="stable")]
        groups.append((is_dn == dn, (order, val[order])))
    eff_n = dict(spec=spec, n_signals=int(signal.sum()), n_no_outcome=int(signal.sum() - (signal & defined).sum()),
                 n_all=int((signal & defined).sum()), n_independent=int(indep.sum()), n_independent_strict=int(strict.sum()))
    if ev.size == 0:
        return Evaluation(**eff_n, rate_event=None, rate_event_all=None, rate_matched=None, rate_unconditional=None,
                          effect=None, effect_vs_unconditional=None, boot=None, p=None)
    ones = np.ones(n)
    rate_ev, rate_base = _weighted_rates(ones, ev, hits, dist, groups)
    unc = np.where(is_dn, np.nanmean(g["out_dn"]), np.nanmean(g["out_up"])).mean()
    eff, rates = [], []
    undefined = 0
    for idx in es.resample_indices(n, seed):
        r, b = _weighted_rates(np.bincount(idx, minlength=n).astype(float), ev, hits, dist, groups)
        if np.isfinite(r) and np.isfinite(b):
            eff.append(r - b)
            rates.append(r)
        else:
            undefined += 1
    boot = es.Bootstrap(np.array(eff), np.array(rates), undefined)
    return Evaluation(**eff_n, rate_event=float(rate_ev), rate_event_all=float(np.nanmean(ev_out[signal])),
                      rate_matched=float(rate_base), rate_unconditional=float(unc), effect=float(rate_ev - rate_base),
                      effect_vs_unconditional=float(rate_ev - unc), boot=boot, p=es.p_two_sided(boot.effect),
                      extra={"excluded_ex_dividend_gap_sessions": g["excluded_ex_dividend"],
                             "ex_dividend_non_gap_sessions_excluded_from_baseline": g["ex_nongap_excluded"]})

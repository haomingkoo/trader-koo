"""Candidate trades of every v9 rule (docs/research/hypotheses-v9-champions.md) and their random substitutes.

A candidate is one signal turned into a trade by its own rule (entry, stop, exits), before any portfolio limit.
The placebo keeps each candidate's signal session, fill session, stop distance and exit rule and swaps the stock
for a random index member on the signal session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import v8_run, v9_fund
from . import v9_signals as sig
from .data import BENCHMARK_TICKER, Market, connect_ro
from .v2_run import load_prices
from .v9_book import Cands, Sizing
from .v9_trades import Exits, buy_stop, make_bars, simulate

H1_EXITS = Exits(max_hold=120, partial_day=5, trail=10, earnings_exit=True)
H1_TRAIL20 = Exits(max_hold=120, partial_day=5, trail=20, earnings_exit=True)
H2_EXITS = Exits(max_hold=120, band=True, target=1.70, target_frac=0.588)
H3_EXITS = Exits(max_hold=60, day_one_red=True, breakeven_day=3, breakeven_gain=1.10, sma_exit=20, rsi_div=True,
                 vol_rev=True, earnings_exit=True)
H4_EXITS = Exits(max_hold=5, fixed_close_day=5)
H4_DAY3 = Exits(max_hold=3, fixed_close_day=3)
SIZING = {"H1a": Sizing(risk=0.005, cap=0.20, tiers=True), "H1b": Sizing(risk=0.005, cap=0.20),
          "H2": Sizing(risk=0.003, cap=0.20), "H3": Sizing(risk=0.01, cap=0.30), "H4": Sizing(risk=0.005, cap=0.20),
          "H7": Sizing(equal_weight=0.10)}
FLAT_H1A = Sizing(risk=0.005, cap=0.20)
H2_MAX_STOP = 0.05          # J Law: stops of 3-5%
H3_ORDER_DAYS = 5           # buy-stop live on t+1..t+5
H3_TRIGGER_DAYS = 6         # rest high and low over t-5..t
SPY_SMA = 50
FIVE_YEARS = 1260           # sessions of history the since-2010 high needs
FAMILIES = ("F1", "F2", "F3", "F4")


@dataclass
class Data:
    p: object
    last: int
    dates: list[str]
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    sc: np.ndarray            # stored close
    v: np.ndarray             # stored volume
    member: np.ndarray
    bars: object
    dv: np.ndarray
    spy: np.ndarray
    spy_up: np.ndarray        # SPY close above its 50-session average
    ev: pd.DataFrame
    counts: dict
    db_path: str


def load(market: Market, db_path: str, boundary: str) -> Data:
    p, last, counts = load_prices(market, boundary)
    o, h, lo, c = (getattr(p, k).to_numpy() for k in ("open", "high", "low", "close"))
    sc = v8_run.stored_matrix(db_path, "close", boundary, p.close)
    v = v8_run.stored_matrix(db_path, "volume", boundary, p.close)
    ev, ev_counts = v8_run.events_on_grid(db_path, boundary, p, last)
    reaction = np.zeros(o.shape, bool)
    reaction[ev["row"].to_numpy(), ev["col"].to_numpy()] = True
    spin = np.zeros(o.shape, bool)
    for col, qs in market.exits.items():
        for q in qs:
            if q <= last:
                spin[q, col] = True
    spy = c[:, list(p.close.columns).index(BENCHMARK_TICKER)]
    spy_sma = pd.Series(spy).rolling(SPY_SMA, min_periods=SPY_SMA).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        spy_up = spy > spy_sma
    return Data(p, last, list(p.close.index), o, h, lo, c, sc, v, p.member.to_numpy(),
                make_bars(o, h, lo, c, v, reaction=reaction, spin=spin), sig.dollar_volume(sc, v), spy, spy_up, ev,
                {"load": counts, "events": ev_counts}, db_path)


@dataclass
class Rule:
    key: str
    exits: Exits
    sig: np.ndarray
    col: np.ndarray
    fill: np.ndarray
    entry: np.ndarray
    stop: np.ndarray

    def where(self, keep: np.ndarray, key: str | None = None, exits: Exits | None = None) -> "Rule":
        return Rule(key or self.key, exits or self.exits, self.sig[keep], self.col[keep], self.fill[keep],
                    self.entry[keep], self.stop[keep])


def _next_open_rule(d: Data, key: str, exits: Exits, t: np.ndarray, col: np.ndarray, stop: np.ndarray) -> Rule:
    fill = t + 1
    ok = fill <= d.last
    t, col, stop, fill = t[ok], col[ok], stop[ok], fill[ok]
    return Rule(key, exits, t, col, fill, d.o[fill, col], stop)


def _unique(t: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.unique(np.stack([t, col], axis=1), axis=0) if len(t) else np.zeros((0, 2), int)
    return pairs[:, 0].astype(int), pairs[:, 1].astype(int)


def earnings_rules(d: Data) -> tuple[dict[str, Rule], dict]:
    ev = d.ev[d.ev["member"]]
    t, col = ev["row"].to_numpy(), ev["col"].to_numpy()
    surprise = (ev["eps_actual"] > ev["eps_estimate"]).to_numpy()
    gv = sig.gap_volume(d.o, d.c, d.v, t, col)
    rs = sig.rs_percentile(d.c, d.member)[t, col] >= sig.RS_CUT
    high52 = sig.new_high(d.sc, t, col)
    first_close = np.argmax(np.isfinite(d.sc), axis=0)
    long_high = sig.high_since_start(d.sc, t, col) & (first_close[col] <= t - FIVE_YEARS)
    masks = {"H1a": gv & high52 & rs, "H1b": gv & sig.quiet(d.sc, t, col) & surprise,
             "H1a_since2010": gv & long_high & rs}
    out = {}
    for k, m in masks.items():
        tt, cc = _unique(t[m], col[m])
        out[k] = _next_open_rule(d, k, H1_EXITS, tt, cc, d.lo[tt, cc])
    a = out["H1a"]
    conn = connect_ro(d.db_path)
    try:
        tickers = list(d.p.close.columns)
        c33 = np.array([v9_fund.code33(conn, tickers[k], d.dates[s]) for s, k in zip(a.sig, a.col)], bool)
    finally:
        conn.close()
    out["H1a_code33"] = a.where(c33, "H1a_code33")
    out["H1a_trail20"] = a.where(np.ones(len(a.sig), bool), "H1a_trail20", H1_TRAIL20)
    counts = {"member_reaction_sessions": int(len(t)), "gap_and_volume": int(gv.sum()),
              "code33_of_H1a": int(c33.sum())}
    return out, {"counts": counts, "earnings_gap_catalyst": (t, col, surprise)}


def h2_rule(d: Data, earn: dict) -> tuple[Rule, dict]:
    t_e, col_e, surprise = earn["earnings_gap_catalyst"]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = d.o[t_e, col_e] / d.c[np.maximum(t_e - 1, 0), col_e] - 1.0 >= sig.GAP
        surge = d.c / np.vstack([np.full((1, d.c.shape[1]), np.nan), d.c[:-1]]) - 1.0 >= sig.SURGE
    ok = sig.volume_surge(d.v) & (d.dv >= sig.MIN_DOLLAR_VOLUME) & d.member
    pr, pc = np.nonzero(surge & ok)
    er, ecol = t_e[gap & surprise], col_e[gap & surprise]
    keep = ok[er, ecol]
    e, col = _unique(np.concatenate([pr, er[keep]]), np.concatenate([pc, ecol[keep]]))
    t, stop = sig.pullbacks(d.o, d.h, d.lo, d.c, d.v, d.spy, e, col)
    found = t >= 0
    t, col, stop = t[found], col[found], stop[found]
    first = pd.DataFrame({"t": t, "col": col}).drop_duplicates().index.to_numpy()
    t, col, stop = t[first], col[first], stop[first]
    on = d.member[t, col]
    rule = _next_open_rule(d, "H2", H2_EXITS, t[on], col[on], stop[on])
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = (rule.entry - rule.stop) / rule.entry
    counts = {"catalysts_price_surge": int(len(pr)), "catalysts_earnings_gap": int(keep.sum()),
              "catalysts": int(len(e)), "pullbacks": int(found.sum()),
              "skipped_stop_over_5pct": int((dist > H2_MAX_STOP).sum())}
    return rule.where(dist <= H2_MAX_STOP), counts


def h3_rule(d: Data) -> tuple[Rule, dict, tuple]:
    setup = sig.space_out(sig.second_move(d.h, d.lo, d.c, d.v) & d.member)
    t, col = np.nonzero(setup)
    trig = pd.DataFrame(d.h).rolling(H3_TRIGGER_DAYS, min_periods=H3_TRIGGER_DAYS).max().to_numpy()[t, col]
    stop = pd.DataFrame(d.lo).rolling(H3_TRIGGER_DAYS, min_periods=H3_TRIGGER_DAYS).min().to_numpy()[t, col]
    fill, price = buy_stop(d.bars, t, col, trig, H3_ORDER_DAYS)
    ok = fill >= 0
    rule = Rule("H3", H3_EXITS, t[ok], col[ok], fill[ok], price[ok], stop[ok])
    return rule, {"setups": int(len(t)), "filled": int(ok.sum())}, (t, col)


def h4_rule(d: Data) -> Rule:
    t, col = np.nonzero(sig.burst(d.h, d.lo, d.sc, d.v) & d.member)
    return _next_open_rule(d, "H4", H4_EXITS, t, col, d.lo[t, col])


def h7_rule(rules: dict[str, Rule], h3_setups: tuple) -> tuple[list[Rule], dict]:
    fam = {"F1": sorted(set(zip(rules["H1a"].sig, rules["H1a"].col)) | set(zip(rules["H1b"].sig, rules["H1b"].col))),
           "F2": list(zip(rules["H2"].sig, rules["H2"].col)), "F3": list(zip(*h3_setups)),
           "F4": list(zip(rules["H4"].sig, rules["H4"].col))}
    fam = {f: [(int(r), int(c)) for r, c in v] for f, v in fam.items()}
    hits = sig.agreement(fam)
    source = {"F1": ("H1a", "H1b"), "F2": ("H2",), "F3": ("H3",), "F4": ("H4",)}
    parts, counts = [], {f: 0 for f in FAMILIES}
    for f in FAMILIES:
        want = {(r, c) for r, c, g in hits if g == f}
        taken = set()
        for key in source[f]:
            rule = rules[key]
            keep = np.array([(int(r), int(c)) in want and (int(r), int(c)) not in taken
                             for r, c in zip(rule.sig, rule.col)], bool)
            taken |= {(int(r), int(c)) for r, c in zip(rule.sig[keep], rule.col[keep])}
            parts.append(rule.where(keep, f"H7_{f}"))
        counts[f] = len(taken)
    return parts, {"agreements": len(hits), "with_a_trade_by_trigger": counts}


def candidates(d: Data, rule: Rule) -> Cands:
    legs = simulate(d.bars, rule.exits, rule.fill, rule.col, rule.entry, rule.stop)
    return Cands(rule.sig, rule.fill, rule.col, rule.entry, rule.stop, legs.n1, legs.when1, legs.p1, legs.f1,
                 legs.n2, legs.when2, legs.p2, legs.kind, d.dv[rule.sig, rule.col])


class Placebo:
    """Random substitutes: a member on the signal session with an open on the fill session, never the same stock."""

    def __init__(self, d: Data, rule: Rule):
        self.d, self.rule = d, rule
        key = rule.sig.astype(np.int64) * (d.last + 1) + rule.fill
        uniq, self.group = np.unique(key, return_inverse=True)
        pools = [np.flatnonzero(d.member[k // (d.last + 1)] & np.isfinite(d.o[k % (d.last + 1)])) for k in uniq]
        self.size = np.array([len(x) for x in pools])
        self.start = np.concatenate([[0], np.cumsum(self.size)[:-1]]).astype(int)
        self.flat = np.concatenate(pools) if pools else np.zeros(0, int)

    def draw(self, rng: np.random.Generator) -> Cands:
        r = self.rule
        if len(r.sig) == 0:
            return candidates(self.d, r)
        size, start = self.size[self.group], self.start[self.group]
        pos = (rng.random(len(r.sig)) * size).astype(int)
        u = self.flat[start + pos]
        clash = (u == r.col) & (size > 1)
        u[clash] = self.flat[start[clash] + (pos[clash] + 1) % size[clash]]
        entry = self.d.o[r.fill, u]
        sub = Rule(r.key, r.exits, r.sig, u, r.fill, entry, entry * (r.stop / r.entry))
        return candidates(self.d, sub)

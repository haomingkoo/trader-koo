"""Orchestration: one candidate, one config, one ledger entry, one report.

Walk-forward here means each calendar year is scored separately from a single
pass in which the signal sees only data up to each review date. The reference
candidates have no fitted parameters, so no per-fold training is needed; a
candidate that fits parameters must fit them only on data before the year it is
scored on.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import ledger, report
from .data import BENCHMARK_TICKER, Market
from .portfolio import (DEFAULT_BUFFER_N, DEFAULT_MIN_HOLD, DEFAULT_REVIEW_EVERY, DEFAULT_TOP_N,
                        Result, Rules, Schedule, annual_reviews, build_schedule, buy_and_hold,
                        equal_weight_schedule, simulate)
from .signals import ANNUAL_CANDIDATES, make_signal

DEFAULT_COST_BPS = 10.0
DEFAULT_STRESS_BPS = 20.0
DEFAULT_N_RANDOM = 200
ANNUAL_RULES = (5, 5, 0)  # H4: five names, no buffer, no minimum hold
SUMMARY_DIGITS = 10
MOVE_DIGITS = 4
PCT_LOW = 5
PCT_MID = 50
PCT_HIGH = 95

SURVIVORSHIP = (
    "Survivorship: price history is missing for {n} of {m} point-in-time members in the window "
    "(departed names). Results exclude them, which likely flatters long-only and momentum results."
)
CAVEATS = [
    "Research only; a backtest proves software behaviour on this data, not future performance.",
    "Flagged price rows keep their real move except moves on split dates (0% day) and bad closes "
    "(NaN); each is listed under data.flagged. A flagged spin-off or data error can still move results.",
    "Held names that lose their price are valued flat at the last price until the next review.",
]


@dataclass(frozen=True)
class BenchConfig:
    candidate: str
    params: dict = field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    cost_bps: float = DEFAULT_COST_BPS
    stress_bps: float = DEFAULT_STRESS_BPS
    n_random: int = DEFAULT_N_RANDOM
    open_sealed: bool = False
    top_n: int = DEFAULT_TOP_N
    buffer_n: int = DEFAULT_BUFFER_N
    review_every: int = DEFAULT_REVIEW_EVERY
    min_hold: int = DEFAULT_MIN_HOLD
    universe: str = "sp500_pit"

    def rules(self) -> Rules:
        return Rules(self.top_n, self.buffer_n, self.review_every, self.min_hold)

    def validate(self) -> None:
        if self.candidate in ANNUAL_CANDIDATES and (self.top_n, self.buffer_n, self.min_hold) != ANNUAL_RULES:
            raise ValueError(f"{self.candidate} is registered with top_n, buffer_n, min_hold = {ANNUAL_RULES}")

    def seed_basis(self) -> dict:
        return dataclasses.asdict(self)


def _index_at_or_after(sessions: list[str], d: str | None, default: int) -> int:
    if d is None:
        return default
    return int(np.searchsorted(sessions, d, side="left"))


def _window_block(market: Market, first: int, lo: int, hi: int, cand: Result, stress: Result,
                  ew: Result, spy: np.ndarray, rand: list[Result]) -> dict:
    p0, p1 = lo - first, hi - first
    r = {"strategy": report.window_returns(cand.equity, p0, p1),
         "spy": report.window_returns(spy, p0, p1),
         "equal_weight": report.window_returns(ew.equity, p0, p1),
         "stress": report.window_returns(stress.equity, p0, p1)}
    st = {k: report.stats(v) for k, v in r.items()}
    rc = np.array([report.stats(report.window_returns(x.equity, p0, p1))["cagr"] for x in rand])
    cagr = st["strategy"]["cagr"]
    years = report.by_year(market.sessions[lo:hi + 1], r)
    beats = [y["strategy"] > y["spy"] for y in years]
    beats_both = [y["strategy"] > y["spy"] and y["strategy"] > y["equal_weight"] for y in years]
    return {
        "start": market.sessions[lo], "end": market.sessions[hi], "sessions": hi - lo + 1,
        "strategy": st["strategy"], "spy": st["spy"], "equal_weight": st["equal_weight"],
        "excess_vs_spy": cagr - st["spy"]["cagr"],
        "excess_vs_equal_weight": cagr - st["equal_weight"]["cagr"],
        "stress": st["stress"], "stress_excess_vs_spy": st["stress"]["cagr"] - st["spy"]["cagr"],
        "turnover": report.turnover(cand, cand.equity[p0:p1 + 1], lo, hi),
        "hit_rate": report.hit_rate(cand, ew, lo, hi),
        "pct_years_beating_spy": float(np.mean(beats)) if beats else None,
        "pct_years_beating_both": float(np.mean(beats_both)) if beats_both else None,
        "years": years,
        "random": {"n": len(rand), "percentile": float(100.0 * np.mean(rc <= cagr)) if len(rc) else None,
                   "p5": float(np.percentile(rc, PCT_LOW)), "p50": float(np.percentile(rc, PCT_MID)),
                   "p95": float(np.percentile(rc, PCT_HIGH))} if len(rc) else
                  {"n": 0, "percentile": None, "p5": None, "p50": None, "p95": None},
    }


def run_bench(market: Market, cfg: BenchConfig, ledger_path: str | Path) -> dict:
    cfg.validate()
    n = len(market.sessions)
    boundary = int(np.searchsorted(market.sessions, ledger.pinned_boundary(ledger_path, market.sessions), side="left"))
    start = _index_at_or_after(market.sessions, cfg.start, 0)
    default_end = n - 1 if cfg.open_sealed else boundary - 1
    end = default_end if cfg.end is None else int(np.searchsorted(market.sessions, cfg.end, side="right")) - 1
    if not cfg.open_sealed and (end >= boundary or start >= boundary):
        raise ledger.SealedWindowError(
            f"the sealed window starts at {market.sessions[boundary]}; pass open_sealed to use it")
    if start >= end:
        raise ValueError("empty date range")
    config = cfg.seed_basis()
    seed = ledger.seed_from_config(config)
    stamp = datetime.now(timezone.utc).isoformat()
    base = {"candidate": cfg.candidate, "config": config, "seed": seed,
            "code_version": ledger.code_version(), "timestamp": stamp}
    if cfg.open_sealed:
        prior = ledger.sealed_openings(ledger_path, cfg.candidate)
        ledger.append(ledger_path, {**base, "event": "sealed_open", "prior_openings": prior})

    signal = make_signal(cfg.candidate, cfg.params)
    rules = cfg.rules()
    reviews = (annual_reviews(market.sessions, start, end) if cfg.candidate in ANNUAL_CANDIDATES
               else list(range(start, end, rules.review_every)))
    coverage: list[tuple[str, int, int]] = []

    def score(s: int, uni: list[str]) -> pd.Series:
        out_ = signal(market.sessions[s], uni, market.view(market.sessions[s]))
        coverage.append((market.sessions[s], len(uni), int(out_.notna().sum())))
        return out_

    cand_sched = build_schedule(market, reviews, rules, score, end)
    keep = next((i for i, t in enumerate(cand_sched.targets) if t), None)
    if keep is None:
        raise ValueError("the candidate never selected a name in this range")
    cand_sched = Schedule(cand_sched.exec_idx[keep:], cand_sched.targets[keep:], cand_sched.dropped_no_open,
                          cand_sched.blocked_spinoff)
    reviews = [e - 1 for e in cand_sched.exec_idx]
    first = cand_sched.exec_idx[0]

    cand = simulate(market, cand_sched, end, cfg.cost_bps)
    stress = simulate(market, cand_sched, end, cfg.stress_bps)
    ew = simulate(market, equal_weight_schedule(market, reviews, end), end, cfg.cost_bps)
    spy = buy_and_hold(market, BENCHMARK_TICKER, first, end)
    rng = np.random.default_rng(seed)
    rand = [
        simulate(market, build_schedule(
            market, reviews, rules, lambda s, uni: pd.Series(rng.random(len(uni)), index=uni), end),
            end, cfg.cost_bps)
        for _ in range(cfg.n_random)
    ]

    out = {"candidate": cfg.candidate, "params": cfg.params, "config_id": ledger.short_id(config),
           "cost_bps": cfg.cost_bps, "stress_bps": cfg.stress_bps, "n_random": cfg.n_random,
           "seed": seed, "universe": cfg.universe, "dev": None, "sealed": None}
    dev_hi = min(end, boundary - 1)
    out["dev"] = _window_block(market, first, first, dev_hi, cand, stress, ew, spy, rand)
    if cfg.open_sealed and end >= boundary:
        out["sealed"] = _window_block(market, first, max(boundary, first), end, cand, stress, ew, spy, rand)
    missing, members = market.members_without_prices(first, end)
    flagged = market.flagged_in(first, end)
    exits = [{**market.exit_events[(col, q)], "strategy_exit": True}
             for q, col in sorted(cand.exits) if first <= q <= end]
    cov = pd.DataFrame(coverage, columns=["date", "universe", "scored"])
    out["coverage"] = {
        "reviews": len(cov), "mean_universe": float(cov["universe"].mean()), "mean_scored": float(cov["scored"].mean()),
        "by_year": [{"year": int(y), "mean_universe": float(g["universe"].mean()), "mean_scored": float(g["scored"].mean())}
                    for y, g in cov.groupby(cov["date"].str[:4])]}
    out["exclusions"] = market.exclusion_summary(first, end)
    out["data"] = {"entries_dropped_no_open": cand_sched.dropped_no_open,
                   "spinoff_exits": exits, "spinoff_buys_blocked": cand_sched.blocked_spinoff,
                   "flagged_price_rows": len(flagged),
                   "excluded_member_days": int(sum(e["member_days"] for e in out["exclusions"])),
                   "flagged_zeroed_or_nan": int((flagged["treatment"] != "real move kept").sum()),
                   "flagged": flagged.round({"move": MOVE_DIGITS}).to_dict("records"),
                   "sealed_boundary": market.sessions[boundary],
                   "members_without_prices": missing, "members_in_window": members}
    out["caveats"] = [SURVIVORSHIP.format(n=missing, m=members)] + CAVEATS
    out["markdown"] = report.markdown(out)

    summary = {k: out["dev"][k] for k in
               ("excess_vs_spy", "excess_vs_equal_weight", "turnover", "hit_rate", "pct_years_beating_spy",
                "pct_years_beating_both")}
    summary.update(cagr=out["dev"]["strategy"]["cagr"], sharpe=out["dev"]["strategy"]["sharpe"],
                   max_drawdown=out["dev"]["strategy"]["max_drawdown"],
                   random_percentile=out["dev"]["random"]["percentile"],
                   stress_cagr=out["dev"]["stress"]["cagr"])
    if out["sealed"]:
        summary["sealed_cagr"] = out["sealed"]["strategy"]["cagr"]
        summary["sealed_excess_vs_spy"] = out["sealed"]["excess_vs_spy"]
    summary = {k: None if v is None else round(float(v), SUMMARY_DIGITS) for k, v in summary.items()}
    out["summary"] = summary
    ledger.append(ledger_path, {
        **base, "event": "run", "trial_id": ledger.short_id(base),
        "params_hash": ledger.sha(cfg.params), "sealed_opened": cfg.open_sealed, "sealed_boundary": market.sessions[boundary],
        "date_range": [market.sessions[first], market.sessions[end]],
        "costs_bps": {"per_side": cfg.cost_bps, "stress": cfg.stress_bps},
        "universe": cfg.universe, "n_sessions": n, "summary": summary,
    })
    return out

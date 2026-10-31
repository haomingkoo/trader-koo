"""Run hypotheses v9 (docs/research/hypotheses-v9-champions.md) on the development window and append the run to the ledger.

Each rule is a risk-sized portfolio against SPY, the equal-weight universe and 500 placebo portfolios (the same
trades on random index members). H6 blends the five rule sleeves; H7 trades where two rule families agree. The
H1a-vs-H1b test compares mean net R over every signal. Eight tests share one Holm family.

Usage: python -m trader_koo.research.bench.v9_run --db <copy of the research db> [--ledger <path>] [--json-out <file>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trader_koo.research.settings import get as _settings

from . import eventstats as es
from . import ledger, v9_report, verdict
from . import v9_rules as rules_mod
from .data import Market, connect_ro, load_market
from .report import by_year, stats
from .v2_trades import BPS
from .v3_run import sealed_boundary
from .v9_book import Cands, blend, concat, run_book, trade_stats
from .v9_rules import SIZING, Placebo, candidates
from .v9_trades import KIND_NAMES

_E = _settings().event_study
_B = _settings().bench
FAMILY = "v9"
CANDIDATE = "v9_champions"
FIRST_ENTRY = "2015-01-01"   # first entry 2015-01-02
FROM_2011 = "2011-01-01"
REGISTRATION = Path(__file__).resolve().parents[3] / "docs" / "research" / "hypotheses-v9-champions.md"
PRIMARY = ("H1a", "H1b", "H1a_vs_H1b", "H2", "H3", "H4", "H6", "H7")
SLEEVES = ("H1a", "H1b", "H2", "H3", "H4")
DRAWS = 500
PASS_YEARS = 0.60
EXIT_REFUSED = 3
N_PARTS = 4             # event count, event sum, baseline count, baseline sum
PLACEBO_TOP = 0.95


def registration_sha256() -> str:
    return hashlib.sha256(REGISTRATION.read_bytes()).hexdigest()


def benchmarks(d, first: int) -> dict[str, np.ndarray]:
    """SPY and the equal-weight universe (members of the previous session), close to close, sessions first..last."""
    c = d.c
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = c[1:] / c[:-1] - 1.0
    ok = d.member[:-1] & np.isfinite(daily)
    ew = np.where(ok.sum(1) > 0, np.where(ok, daily, 0.0).sum(1) / np.maximum(ok.sum(1), 1), 0.0)
    spy = d.spy[1:] / d.spy[:-1] - 1.0
    return {"spy": spy[first - 1:d.last], "equal_weight": ew[first - 1:d.last]}


def _signal_r(c: Cands, first: int, cost: float) -> tuple[np.ndarray, np.ndarray]:
    ok = (c.fill >= first) & np.isfinite(c.entry) & (c.entry > c.stop)
    p1 = np.where(c.f1 > 0, c.p1, 0.0)
    r = ((c.f1 * p1 + (1.0 - c.f1) * c.p2) * (1.0 - cost) - c.entry * (1.0 + cost)) / (c.entry - c.stop)
    return r[ok], ok


def _book(d, c: Cands, sizing, first: int, bps: float):
    return run_book(c, d.bars.c_last, first, d.last, sizing, bps, d.dates)


def _taken_stats(book) -> dict:
    t = book.trades
    return trade_stats(t["r_net"].to_numpy(), t["pnl"].to_numpy(), t["hold"].to_numpy())


def portfolio_block(key: str, d, c: Cands, sizing, first: int, bench: dict, placebo_cagrs=None,
                    stress_returns=None, returns=None) -> dict:
    """Portfolio statistics of one rule (or the blend when ``returns`` is given), its years and its rubric grade."""
    book = stress = None
    if returns is None:
        book, stress = _book(d, c, sizing, first, _B.cost_bps), _book(d, c, sizing, first, _B.stress_bps)
        returns, stress_returns = book.returns(), stress.returns()
    dates = d.dates[first:d.last + 1]
    series = {"strategy": returns, "stress": stress_returns, **bench}
    st = {k: stats(v) for k, v in series.items()}
    years = by_year(dates, {k: series[k] for k in ("strategy", "spy", "equal_weight")})
    out = {"key": key, "start": dates[0], "end": dates[-1], **st, "years": years,
           "excess_vs_equal_weight": st["strategy"]["cagr"] - st["equal_weight"]["cagr"],
           "pct_years_beating_both": float(np.mean([y["strategy"] > y["spy"] and y["strategy"] > y["equal_weight"]
                                                    for y in years])),
           "share_of_sessions_invested": float(np.mean(returns != 0)), "returns": returns,
           "stress_returns": stress_returns}
    if book is not None:
        out.update(trades=_taken_stats(book), stress_trades=_taken_stats(stress), skipped=book.skipped,
                   mean_open=float(book.invested.mean()), exits=_exit_kinds(book))
        r_all, ok = _signal_r(c, first, _B.cost_bps / BPS)
        out["all_signals"] = trade_stats(r_all, r_all, (c.n2 - c.fill + 1)[ok])
    if placebo_cagrs is not None:
        pc = np.asarray(placebo_cagrs)
        cagr = st["strategy"]["cagr"]
        out["placebo"] = {"n": len(pc), "median": float(np.median(pc)), "p95": float(np.quantile(pc, PLACEBO_TOP)),
                          "pct": float(np.mean(pc < cagr) * 100.0), "p": float((1 + np.sum(pc >= cagr)) / (len(pc) + 1))}
        if st["strategy"]["sharpe"] is not None and st["spy"]["sharpe"] is not None:
            out["grade"] = verdict.grade({"candidate": key, "dev": {**out, "random": {
                "n": len(pc), "percentile": out["placebo"]["pct"]}}})
    return out


def _exit_kinds(book) -> dict:
    counts = book.trades["kind"].value_counts().to_dict()
    return {KIND_NAMES[int(k)]: int(v) for k, v in counts.items()}


def placebo_returns(d, rules: list, sizing, first: int, seed: int, draws: int) -> list[np.ndarray]:
    makers = [Placebo(d, r) for r in rules]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        c = concat([m.draw(rng) for m in makers])
        out.append(_book(d, c, sizing, first, _B.cost_bps).returns())
    return out


def head_to_head(d, ca: Cands, cb: Cands, first: int, seed: int, resamples: int) -> dict:
    """Mean net R per signal, H1a minus H1b, with a two-sided 5-session moving-block bootstrap over entry sessions."""
    n = d.last + 1
    parts = np.zeros((N_PARTS, 1, n))
    for slot, c in ((0, ca), (2, cb)):
        r, ok = _signal_r(c, first, _B.cost_bps / BPS)
        rows = c.fill[ok]
        parts[slot, 0] = np.bincount(rows, minlength=n)
        parts[slot + 1, 0] = np.bincount(rows, weights=r, minlength=n)
    parts = parts[:, :, first:]
    if parts[0].sum() == 0 or parts[2].sum() == 0:
        return {"status": "insufficient", "p": 1.0}
    pt = es.point(parts)
    boot = es.bootstrap(parts, seed, resamples=resamples)
    return {"n_h1a": int(parts[0].sum()), "n_h1b": int(parts[2].sum()), "mean_r_h1a": pt["rate_event"],
            "mean_r_h1b": pt["rate_base"], "difference": pt["effect"], "p": es.p_two_sided(boot.effect),
            "interval": es.interval(boot.effect, _E.alpha), "undefined_resamples": boot.undefined}


def checks(res: dict, adj_p: float, with_r: bool = True) -> dict:
    s, spy, ew = res["strategy"], res["spy"], res["equal_weight"]
    if s["cagr"] is None or (with_r and res["trades"].get("n", 0) == 0):
        return {"status": "insufficient", "why": "no trades", "adj_p": adj_p}
    best = max(spy["cagr"], ew["cagr"])
    c = {"1 Holm placebo p < 0.05": adj_p < _E.alpha, "2 CAGR above SPY and equal weight": s["cagr"] > best,
         "3 beats both in >= 60% of years": res["pct_years_beating_both"] >= PASS_YEARS,
         "4 stress CAGR above both": res["stress"]["cagr"] > best}
    if with_r:
        c["5 mean net R > 0"] = res["trades"]["mean_r"] > 0
    failed = [k for k, ok in c.items() if not ok]
    return {"status": "fail" if failed else "development criteria met (sealed window not opened)", "checks": c,
            "adj_p": adj_p, "why": "; ".join(failed) or "criteria 1 to 5 met"}


def h5_status(db_path: str, first_session: str) -> dict:
    conn = connect_ro(db_path)
    try:
        row = conn.execute("SELECT MIN(date), COUNT(*) FROM price_daily WHERE ticker = 'IWM'").fetchone()
    finally:
        conn.close()
    if row[0] is not None and row[0] <= first_session:
        raise LookupError("IWM now covers the window; H5 must be run, not skipped (registration says skipped)")
    return {"status": "skipped", "iwm_first_date": row[0], "iwm_rows": row[1],
            "why": "IWM has no rows in the development window and the database starts after 2008"}


def build(d) -> tuple[dict, dict]:
    earn, earn_info = rules_mod.earnings_rules(d)
    h2, h2_counts = rules_mod.h2_rule(d, earn_info)
    h3, h3_counts, h3_setups = rules_mod.h3_rule(d)
    h4 = rules_mod.h4_rule(d)
    rules = {**earn, "H2": h2, "H3": h3, "H4": h4}
    h7_parts, h7_counts = rules_mod.h7_rule(rules, h3_setups)
    plan = {k: [rules[k]] for k in SLEEVES} | {"H7": h7_parts}
    up = lambda r: r.where(d.spy_up[r.sig], f"{r.key}_spy50")
    var = {"H1a_since2010": ([rules["H1a_since2010"]], SIZING["H1a"], FIRST_ENTRY),
           "H1a_code33": ([rules["H1a_code33"]], SIZING["H1a"], FIRST_ENTRY),
           "H1a_flat": ([rules["H1a"]], rules_mod.FLAT_H1A, FIRST_ENTRY),
           "H1a_trail20": ([rules["H1a_trail20"]], SIZING["H1a"], FIRST_ENTRY),
           "H3_from2011": ([h3], SIZING["H3"], FROM_2011),
           "H4_day3": ([h4.where(np.ones(len(h4.sig), bool), "H4_day3", rules_mod.H4_DAY3)], SIZING["H4"], FIRST_ENTRY),
           "H4_from2011": ([h4], SIZING["H4"], FROM_2011)}
    for k in ("H1a", "H1b", "H2", "H3", "H4"):
        var[f"{k}_spy50"] = ([up(rules[k])], SIZING[k], FIRST_ENTRY)
    counts = {**earn_info["counts"], "H2": h2_counts, "H3": h3_counts, "H7": h7_counts,
              "signals": {k: int(len(r.sig)) for k, r in rules.items()}}
    return {"plan": plan, "variants": var}, counts


def run_v9(db_path: str | Path, ledger_path: str | Path, market: Market | None = None, first_entry: str = FIRST_ENTRY,
           draws: int = DRAWS, resamples: int = _E.bootstrap_resamples) -> dict:
    boundary = sealed_boundary(ledger_path)
    db_path = str(db_path)
    market = market or load_market(db_path)
    d = rules_mod.load(market, db_path, boundary)
    first = max(int(np.searchsorted(d.dates, first_entry, side="left")), 1)
    config = {"family": FAMILY, "boundary": boundary, "first_entry_session": d.dates[first],
              "last_session": d.dates[d.last], "cost_bps": _B.cost_bps, "stress_bps": _B.stress_bps,
              "placebo_draws": draws, "block": _E.bootstrap_block, "resamples": resamples, "universe": _B.universe,
              "registration": REGISTRATION.name, "registration_sha256": registration_sha256()}
    seed = ledger.seed_from_config(config)
    plan, counts = build(d)
    bench = benchmarks(d, first)
    results, draws_by = {}, {}
    for key, rules in plan["plan"].items():
        sizing = SIZING[key]
        c = concat([candidates(d, r) for r in rules])
        draws_by[key] = placebo_returns(d, rules, sizing, first, ledger.seed_from_config({**config, "spec": key}), draws)
        results[key] = portfolio_block(key, d, c, sizing, first, bench, [stats(r)["cagr"] for r in draws_by[key]])
    dates = d.dates[first:d.last + 1]
    sleeve_r = np.column_stack([results[k]["returns"] for k in SLEEVES])
    sleeve_stress = np.column_stack([results[k]["stress_returns"] for k in SLEEVES])
    blend_draws = [stats(blend(np.column_stack([draws_by[k][i] for k in SLEEVES]), dates))["cagr"] for i in range(draws)]
    results["H6"] = portfolio_block("H6", d, None, None, first, bench, blend_draws, blend(sleeve_stress, dates),
                                    blend(sleeve_r, dates))
    results["H6"]["correlation"] = np.corrcoef(sleeve_r.T).tolist()
    ca, cb = (candidates(d, plan["plan"][k][0]) for k in ("H1a", "H1b"))
    h2h = head_to_head(d, ca, cb, first, ledger.seed_from_config({**config, "spec": "H1a_vs_H1b"}), resamples)
    variants = {}
    for key, (rules, sizing, start) in plan["variants"].items():
        f = max(int(np.searchsorted(d.dates, start, side="left")), 1)  # returns start from the session after a close
        variants[key] = portfolio_block(key, d, concat([candidates(d, r) for r in rules]), sizing, f, benchmarks(d, f))
    ps = [results[k]["placebo"]["p"] if k != "H1a_vs_H1b" else h2h["p"] for k in PRIMARY]
    adj, _, _ = es.holm(ps)
    holm = dict(zip(PRIMARY, adj))
    verdicts = {k: checks(results[k], holm[k], with_r=k != "H6") for k in PRIMARY if k != "H1a_vs_H1b"}
    verdicts["H1a_vs_H1b"] = {"status": "differ" if holm["H1a_vs_H1b"] < _E.alpha else "no difference shown",
                              "adj_p": holm["H1a_vs_H1b"]}
    out = {"family": FAMILY, "seed": seed, "config": config, "boundary": boundary, "results": results,
           "variants": variants, "head_to_head": h2h, "holm": holm, "verdicts": verdicts, "counts": counts,
           "h5": h5_status(db_path, d.dates[0]), "sleeves": list(SLEEVES),
           "data": {"load_counts": d.counts, "substitutions": market.load_diag.as_dict()}}
    out["markdown"] = v9_report.markdown(out)
    summary = v9_report.summary(out)
    base = {"candidate": CANDIDATE, "config": config, "seed": seed, "code_version": ledger.code_version(),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    ledger.append(ledger_path, {**base, "event": "portfolio_study_run", "trial_id": ledger.short_id(base), "family": FAMILY,
                                "sealed_opened": False, "sealed_boundary": boundary,
                                "date_range": [config["first_entry_session"], config["last_session"]],
                                "n_sessions": d.last + 1, "params_hash": ledger.sha(config), "summary": summary})
    out["trial_id"], out["summary"] = ledger.short_id(base), summary
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trader_koo.research.bench.v9_run")
    ap.add_argument("--db", required=True, help="a copy of the research database (opened read-only)")
    ap.add_argument("--ledger")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)
    path = Path(args.ledger) if args.ledger else Path(args.db).with_name("bench_trial_ledger.jsonl")
    try:
        out = run_v9(args.db, path)
    except LookupError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(out["markdown"])
    print(f"ledger trial_id (v9): {out['trial_id']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(v9_report.jsonable(out), indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run hypotheses v7 (exposure rules on SPY) on the development window, score them, append to the ledger.

Daily return on session t, with e_t the exposure decided at the close of t:
    r_t = e_{t-2} * overnight_t + e_{t-1} * intraday_t + (1 - e_{t-1}) * cash_t - cost * |e_{t-1} - e_{t-2}|
overnight_t = open_t / close_{t-1} - 1 and intraday_t = close_t / open_t - 1 on total-return adjusted SPY. This
is the simple-return sum (the cross term of the two legs is dropped); SPY is computed the same way with e = 1.

Usage: python -m trader_koo.research.bench.v7_run --db <research db> [--ledger <path>] [--json-out <file>]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import ledger
from .data import load_market
from .v3_run import sealed_boundary
from .v7_rules import RULES, TRADING_DAYS, Inputs, _known_daily, h11_vote

FAMILY = "v7"
DEV_START = "2011-01-03"
COST_BPS = 10.0
STRESS_BPS = 20.0
DIGITS = 4  # rounding of stored summary numbers
BPS = 1e4
N_CONTROLS = 1000
MIN_SHIFT = 252
PERCENT = 100.0
# Pass rule and score (hypotheses-v7-regime.md, "Pass rule" and "Score").
DRAWDOWN_MARGIN = 0.05
CAGR_TOLERANCE = 0.01
CONTROL_PCT = 99.55
FAIL_CAP = 5.0
SCORE_ROWS = (  # key, label, zero_at, full_at, points
    ("sharpe_diff", "Sharpe minus SPY Sharpe", -0.10, 0.20, 3.0),
    ("dd_saved", "SPY max drawdown minus rule max drawdown", 0.0, 0.15, 2.0),
    ("cagr_diff", "CAGR minus SPY CAGR", -0.03, 0.0, 2.0),
    ("control_pct", "Sharpe percentile vs random timing", 50.0, CONTROL_PCT, 2.0),
)


def load_inputs(db: str, boundary: str) -> Inputs:
    market = load_market(db)
    conn = market.conn
    dates = [d for d in market.sessions if d < boundary]
    close = market.adj_close["SPY"].reindex(dates).to_numpy(float)
    open_ = market.adj_open["SPY"].reindex(dates).to_numpy(float)

    def raw(ticker: str, field: str) -> np.ndarray:
        rows = dict(conn.execute(f"SELECT date, {field} FROM price_daily WHERE ticker = ? AND date < ?",
                                 (ticker, boundary)).fetchall())
        return np.array([rows.get(d, np.nan) for d in dates], dtype=float)

    # Cash: FRED DTB3 as known at the previous close (amendment 1; ^IRX mixes units).
    bill = _known_daily(conn, "DTB3", dates)
    rf = np.roll(bill, 1) / PERCENT / TRADING_DAYS
    rf[0] = 0.0
    if np.isnan(rf[1:]).any():
        raise LookupError("DTB3 has no known value for some sessions; load it with trader_koo.research.macro")
    return Inputs(dates, close, open_, raw("SPY", "close"), raw("SPY", "volume"), raw("^VIX", "close"), rf, conn)


def returns(inp: Inputs, e: np.ndarray, cost_bps: float) -> np.ndarray:
    ovn = inp.spy_open / np.roll(inp.spy_close, 1) - 1.0
    intra = inp.spy_close / inp.spy_open - 1.0
    e1, e2 = np.roll(e, 1), np.roll(e, 2)
    r = e2 * ovn + e1 * intra + (1.0 - e1) * inp.rf_daily - cost_bps / BPS * np.abs(e1 - e2)
    r[:2] = 0.0
    return np.nan_to_num(r)


def stats(r: np.ndarray, rf: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + r)
    years = len(r) / TRADING_DAYS
    ex = r - rf
    sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std(ddof=1) > 0 else 0.0
    return {"cagr": float(eq[-1] ** (1 / years) - 1), "sharpe": sharpe,
            "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min()), "total_return": float(eq[-1] - 1)}


def _sharpe_many(inp: Inputs, exposures: np.ndarray, cost_bps: float) -> np.ndarray:
    return np.array([stats(returns(inp, e, cost_bps), inp.rf_daily)["sharpe"] for e in exposures])


def evaluate(inp: Inputs, key: str, e: np.ndarray, spy: dict, seed: int, lo: int) -> dict:
    rf = inp.rf_daily[lo:]
    s = stats(returns(inp, e, COST_BPS)[lo:], rf)
    st = stats(returns(inp, e, STRESS_BPS)[lo:], rf)
    rng = np.random.default_rng(seed)
    n = len(e) - lo
    shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=N_CONTROLS)
    window = e[lo:]
    controls = np.array([np.concatenate([e[:lo], np.roll(window, k)]) for k in shifts])
    ctrl = _sharpe_many(inp, controls, COST_BPS) if controls.size else np.array([])
    m = {"sharpe_diff": s["sharpe"] - spy["sharpe"], "dd_saved": s["max_drawdown"] - spy["max_drawdown"],
         "cagr_diff": s["cagr"] - spy["cagr"],
         "control_pct": float(PERCENT * np.mean(ctrl <= s["sharpe"])) if ctrl.size else 0.0}

    def core(x: dict) -> bool:
        return (x["sharpe"] > spy["sharpe"] and x["max_drawdown"] - spy["max_drawdown"] >= DRAWDOWN_MARGIN
                and x["cagr"] >= spy["cagr"] - CAGR_TOLERANCE)
    gates = {"sharpe_above_spy": s["sharpe"] > spy["sharpe"],
             "drawdown_5pts_smaller": m["dd_saved"] >= DRAWDOWN_MARGIN,
             "cagr_within_1pt": m["cagr_diff"] >= -CAGR_TOLERANCE,
             "above_99_55_controls": m["control_pct"] > CONTROL_PCT, "holds_at_20bps": core(st)}
    rows = []
    for k, label, zero, full, pts in SCORE_ROWS:
        earned = round(pts * min(max((m[k] - zero) / (full - zero), 0.0), 1.0), 2)
        rows.append({"criterion": label, "value": round(m[k], DIGITS), "earned": earned, "points": pts})
    rows.append({"criterion": "Criteria 1 to 3 at 20 bps", "value": gates["holds_at_20bps"],
                 "earned": 1.0 if gates["holds_at_20bps"] else 0.0, "points": 1.0})
    raw = round(sum(r["earned"] for r in rows), 2)
    passed = all(gates.values())
    return {"key": key, "strategy": s, "stress": st, "measures": m, "gates": gates, "rows": rows,
            "time_in_market": float(np.mean(window)), "switches": int(np.sum(np.abs(np.diff(window)) > 0)),
            "raw_score": raw, "score": raw if passed else min(raw, FAIL_CAP),
            "verdict": "development criteria met (sealed window not opened)" if passed else "fail",
            "pros": [f"{r['criterion']}: {r['value']} ({r['earned']:g}/{r['points']:g})" for r in rows
                     if r["earned"] >= r["points"] / 2],
            "cons": [f"{r['criterion']}: {r['value']} ({r['earned']:g}/{r['points']:g})" for r in rows
                     if r["earned"] < r["points"] / 2]}


def run_v7(db: str, ledger_path: str | Path) -> dict:
    boundary = sealed_boundary(ledger_path)
    inp = load_inputs(db, boundary)
    lo = int(np.searchsorted(inp.dates, DEV_START))
    raw = {k: fn(inp) for k, fn in RULES.items()}
    missing = {k: int(np.isnan(v[lo:]).sum()) for k, v in raw.items()}
    exp = {k: np.nan_to_num(v, nan=1.0) for k, v in raw.items()}
    exp["H11"] = h11_vote(raw)
    config = {"family": FAMILY, "boundary": boundary, "first_session": inp.dates[lo], "last_session": inp.dates[-1],
              "cost_bps": COST_BPS, "stress_bps": STRESS_BPS, "n_controls": N_CONTROLS, "min_shift": MIN_SHIFT}
    seed = ledger.seed_from_config(config)
    spy = stats(returns(inp, np.ones(len(inp.dates)), COST_BPS)[lo:], inp.rf_daily[lo:])
    results = {k: evaluate(inp, k, e, spy, ledger.seed_from_config({**config, "rule": k}), lo)
               for k, e in exp.items()}
    for k, v in missing.items():
        results[k]["days_without_value_held_at_1"] = v
    summary = {k: {"cagr": round(r["strategy"]["cagr"], DIGITS), "sharpe": round(r["strategy"]["sharpe"], DIGITS),
                   "max_drawdown": round(r["strategy"]["max_drawdown"], DIGITS), "control_pct": r["measures"]["control_pct"],
                   "time_in_market": round(r["time_in_market"], DIGITS), "score": r["score"], "verdict": r["verdict"]}
               for k, r in results.items()}
    base = {"candidate": "v7_regime_exposure", "config": config, "seed": seed, "code_version": ledger.code_version(),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    trial = ledger.short_id(base)
    ledger.append(ledger_path, {**base, "event": "exposure_run", "trial_id": trial, "family": FAMILY,
                                "sealed_opened": False, "sealed_boundary": boundary, "spy": spy,
                                "params_hash": ledger.sha(config), "summary": summary})
    return {"trial_id": trial, "config": config, "spy": spy, "results": results}


def render(out: dict) -> str:
    spy = out["spy"]
    lines = [f"v7 trial {out['trial_id']}, {out['config']['first_session']} to {out['config']['last_session']}. "
             f"SPY: CAGR {spy['cagr']:.2%}, Sharpe {spy['sharpe']:.2f}, max drawdown {spy['max_drawdown']:.1%}.", ""]
    for k, r in out["results"].items():
        s = r["strategy"]
        lines += [f"### {k}: CAGR {s['cagr']:.2%}, Sharpe {s['sharpe']:.2f}, max drawdown {s['max_drawdown']:.1%}, "
                  f"in market {r['time_in_market']:.0%}, {r['switches']} switches, "
                  f"{r['measures']['control_pct']:.1f}th pct vs random timing",
                  "Pros: " + ("; ".join(r["pros"]) or "none"), "Cons: " + ("; ".join(r["cons"]) or "none"),
                  "Gates: " + ", ".join(f"{g} {'pass' if ok else 'FAIL'}" for g, ok in r["gates"].items()),
                  f"**Score {r['score']:g}/10 (raw {r['raw_score']:g}), {r['verdict']}**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trader_koo.research.bench.v7_run")
    p.add_argument("--db", required=True)
    p.add_argument("--ledger")
    p.add_argument("--json-out")
    args = p.parse_args(argv)
    path = Path(args.ledger) if args.ledger else Path(args.db).with_name("bench_trial_ledger.jsonl")
    out = run_v7(args.db, path)
    print(render(out))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

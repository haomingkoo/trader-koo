"""Run hypotheses v10 (docs/research/hypotheses-v10-exposure.md) on the development window and append to the ledger.

H1 puts the ladder on SPY, H2 on the equal-weight universe. Both are judged as a safety layer: against buy-and-hold
on drawdown, against a constant mix of the same average exposure on return per unit of volatility and CAGR/maxDD,
and against circular shifts of their own exposure path. The two shift tests share one Holm family.

Usage: python -m trader_koo.research.bench.v10_run --db <copy of the research db> [--ledger <path>] [--json-out <file>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import eventstats as es
from . import ledger, v10_report
from . import v10_rules as rules
from .data import connect_ro, load_market
from .v2_run import load_prices
from .v3_run import sealed_boundary
from .v7_rules import TRADING_DAYS, _known_daily

FAMILY = "v10"
CANDIDATE = "v10_exposure"
REGISTRATION = Path(__file__).resolve().parents[3] / "docs" / "research" / "hypotheses-v10-exposure.md"
DEV_START = "2011-01-03"
COST_BPS = 10.0
STRESS_BPS = 20.0
N_CONTROLS = 1000
MIN_SHIFT = 252
SHIFT_QUANTILES = (0.05, 0.5, 0.95)
DD_SHARE = 0.7             # (a) and (f): at most 70% of buy-and-hold's drawdown or loss
YEAR_SHARE = 0.60          # (d)
FLOOR = 0.01               # (d): yearly volatility and drawdown floor
ALPHA = 0.05
CRASH = ("2020-02-20", "2020-03-23")   # interval start dates: open 2020-02-20 to open 2020-03-24
PERCENT = 100.0
YEAR_CHARS = 4
EXIT_REFUSED = 3
PRIMARY = ("H1", "H2")


def registration_sha256() -> str:
    return hashlib.sha256(REGISTRATION.read_bytes()).hexdigest()


def metrics(r: np.ndarray, cash: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    sd = r.std(ddof=1)
    ex = r - cash
    out = {"cagr": float(eq[-1] ** (TRADING_DAYS / len(r)) - 1.0), "max_drawdown": float((eq / peak - 1.0).min()),
           "vol": float(sd * np.sqrt(TRADING_DAYS)),
           "rv": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
           "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std(ddof=1) > 0 else 0.0}
    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] < 0 else float("inf")
    return out


def constant_mix(level: float, R: np.ndarray, cash: np.ndarray) -> np.ndarray:
    return level * R + (1.0 - level) * cash


def _max_dd(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + r)
    return float((eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1.0).min())


def shift_test(hold: np.ndarray, R: np.ndarray, cash: np.ndarray, cost_bps: float, seed: int,
               draws: int = N_CONTROLS) -> dict:
    """Circular shifts of the held path by at least MIN_SHIFT sessions; p counts shifts as shallow or shallower."""
    rng = np.random.default_rng(seed)
    own = _max_dd(rules.hold_returns(hold, R, cash, cost_bps))
    ks = rng.integers(MIN_SHIFT, len(hold) - MIN_SHIFT, size=draws)
    dd = np.array([_max_dd(rules.hold_returns(np.roll(hold, k), R, cash, cost_bps)) for k in ks])
    return {"p": float((1 + np.sum(dd >= own)) / (draws + 1)), "own": own, "dd": dd.tolist(),
            "pct_shallower_than": float(PERCENT * np.mean(dd < own)),
            "means": [float(np.roll(hold, k).mean()) for k in ks]}


def _ratio(num: float, den: float) -> float:
    return num / max(den, FLOOR)


def year_rows(dates: list[str], rule: np.ndarray, mix: np.ndarray) -> list[dict]:
    years = np.array([d[:YEAR_CHARS] for d in dates])
    rows = []
    for y in sorted(set(years)):
        m = years == y
        row = {"year": int(y), "sessions": int(m.sum())}
        for name, r in (("rule", rule[m]), ("mix", mix[m])):
            ann = r.mean() * TRADING_DAYS
            vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(r) > 1 else 0.0
            ret = float(np.prod(1.0 + r) - 1.0)
            row[f"{name}_return"] = ret
            row[f"{name}_rv"] = float(_ratio(ann, vol))
            row[f"{name}_calmar"] = float(_ratio(ret, abs(_max_dd(r))))
        row["ok"] = row["rule_rv"] >= row["mix_rv"] and row["rule_calmar"] >= row["mix_calmar"]
        rows.append(row)
    return rows


def gates(s: dict, bh: dict, mix: dict, adj_p: float, year_share: float, stress: dict,
          crash: tuple[float, float]) -> dict:
    def a(x):
        return abs(x["max_drawdown"]) <= DD_SHARE * abs(bh["max_drawdown"])

    def b(x):
        return x["rv"] >= mix["rv"] and x["calmar"] >= mix["calmar"]
    return {"a drawdown 30% shallower": a(s), "b rv and calmar >= constant mix": b(s),
            "c timing controls (Holm)": adj_p < ALPHA, "d years >= 60%": year_share >= YEAR_SHARE,
            "e holds at 20 bps": a(stress) and b(stress),
            "f March 2020 crash": abs(min(crash[0], 0.0)) <= DD_SHARE * abs(min(crash[1], 0.0))}


def load(db: str, boundary: str) -> dict:
    market = load_market(db)
    p, last, counts = load_prices(market, boundary)
    dates = list(p.close.index)
    o, c, member = p.open.to_numpy(), p.close.to_numpy(), p.member.to_numpy()
    conn = connect_ro(db)
    try:
        vol = {t: dict(conn.execute("SELECT date, volume FROM price_daily WHERE ticker = ? AND date < ?",
                                    (t, boundary)).fetchall()) for t in ("SPY", "QQQ")}
        bill = _known_daily(conn, "DTB3", dates)
    finally:
        conn.close()
    cols = list(p.close.columns)
    bars = {t: {"o": o[:, cols.index(t)], "c": c[:, cols.index(t)], "lo": p.low[t].to_numpy(),
                "v": np.array([vol[t].get(d, np.nan) for d in dates], float)} for t in ("SPY", "QQQ")}
    nxt = np.vstack([o[1:], np.full((1, o.shape[1]), np.nan)])
    with np.errstate(invalid="ignore", divide="ignore"):
        oo, oc = nxt / o - 1.0, c / o - 1.0
    ok_oo, ok_oc = member & np.isfinite(oo), member & np.isfinite(oc)
    ew_R = np.where(ok_oo, oo, 0.0).sum(1) / np.maximum(ok_oo.sum(1), 1)
    ew_intra = np.where(ok_oc, oc, 0.0).sum(1) / np.maximum(ok_oc.sum(1), 1)
    spy = bars["SPY"]
    cash = np.concatenate([[0.0], bill[:-1] / PERCENT / TRADING_DAYS])
    if np.isnan(cash[1:]).any():
        raise LookupError("DTB3 has no known value for some sessions; load it with trader_koo.research.macro")
    for t, b in bars.items():
        if not (np.isfinite(b["c"]).all() and np.isfinite(b["lo"]).all() and np.isfinite(b["v"]).all()):
            raise LookupError(f"{t} has missing bars in the window; v10 needs complete index bars")
    books = {"SPY": (np.append(spy["o"][1:] / spy["o"][:-1] - 1.0, np.nan), spy["c"] / spy["o"] - 1.0),
             "EW": (ew_R, ew_intra)}
    return {"dates": dates, "bars": bars, "books": books, "cash": cash, "counts": counts,
            "ew_members_mean": float(ok_oo.sum(1).mean()), "substitutions": market.load_diag.as_dict()}


def evaluate(key: str, data: dict, target, ftd, book: str, lo: int, hi: int, seed: int, drawdown: bool = True,
             pace: bool = True) -> dict:
    R, intra = data["books"][book]
    cash = data["cash"]
    path = rules.run_book(target, ftd, np.nan_to_num(R), intra, cash, COST_BPS, drawdown, pace)
    stress_path = rules.run_book(target, ftd, np.nan_to_num(R), intra, cash, STRESS_BPS, drawdown, pace)
    w = slice(lo, hi + 1)
    hold, Rw, cw = path.hold[w], R[w], cash[w]
    dates = data["dates"][lo:hi + 1]
    level = float(hold.mean())
    mix_r = constant_mix(level, Rw, cw)
    s, st = metrics(path.returns[w], cw), metrics(stress_path.returns[w], cw)
    bh, mix = metrics(Rw, cw), metrics(mix_r, cw)
    full = [y for y in year_rows(dates, path.returns[w], mix_r) if y["year"] < int(dates[-1][:YEAR_CHARS])]
    shifts = shift_test(hold, Rw, cw, COST_BPS, seed)
    crash = np.array([CRASH[0] <= d <= CRASH[1] for d in dates])
    return {"key": key, "book": book, "strategy": s, "stress": st, "buy_and_hold": bh, "constant_mix": mix,
            "mean_exposure": level, "share_invested": float(np.mean(hold > 0)),
            "changes": int(np.sum(np.abs(np.diff(hold)) > 0)), "years": full,
            "year_share": float(np.mean([y["ok"] for y in full])),
            "shift": {k: shifts[k] for k in ("p", "own", "pct_shallower_than")},
            "shift_dd_quantiles": np.quantile(shifts["dd"], SHIFT_QUANTILES).tolist(),
            "crash": (float(np.prod(1.0 + path.returns[w][crash]) - 1.0), float(np.prod(1.0 + Rw[crash]) - 1.0)),
            "exposure_path": hold, "dates": dates}


def run_v10(db: str, ledger_path: str | Path) -> dict:
    boundary = sealed_boundary(ledger_path)
    data = load(db, boundary)
    dates = data["dates"]
    lo = int(np.searchsorted(dates, DEV_START))
    hi = len(dates) - 2        # the last interval ends at the open of the last session before the seal
    config = {"family": FAMILY, "boundary": boundary, "first_session": dates[lo], "last_interval": dates[hi],
              "cost_bps": COST_BPS, "stress_bps": STRESS_BPS, "n_controls": N_CONTROLS, "min_shift": MIN_SHIFT,
              "registration": REGISTRATION.name, "registration_sha256": registration_sha256()}
    seed = ledger.seed_from_config(config)
    spy, qqq = data["bars"]["SPY"], data["bars"]["QQQ"]
    target, ftd, warm = rules.ladder_from_bars(dates, spy["c"], spy["lo"], spy["v"])
    q_target, q_ftd, _ = rules.ladder_from_bars(dates, qqq["c"], qqq["lo"], qqq["v"])
    above50 = (spy["c"] > rules.sma(spy["c"], rules.SMA_SLOW)).astype(float)
    sd = lambda k: ledger.seed_from_config({**config, "spec": k})  # noqa: E731
    results = {"H1": evaluate("H1", data, target, ftd, "SPY", lo, hi, sd("H1")),
               "H2": evaluate("H2", data, target, ftd, "EW", lo, hi, sd("H2"))}
    variants = {"H1_no_drawdown_ladder": evaluate("H1_no_dd", data, target, ftd, "SPY", lo, hi, sd("V1"), False),
                "H1_qqq_signals": evaluate("H1_qqq", data, q_target, q_ftd, "SPY", lo, hi, sd("V2")),
                "spy_50day_rule": evaluate("50d", data, above50, np.zeros(len(dates), bool), "SPY", lo, hi, sd("V3"),
                                           False, False)}
    adj, _, _ = es.holm([results[k]["shift"]["p"] for k in PRIMARY])
    for k, a in zip(PRIMARY, adj):
        res = results[k]
        res["adj_p"] = float(a)
        res["gates"] = gates(res["strategy"], res["buy_and_hold"], res["constant_mix"], a, res["year_share"],
                             res["stress"], res["crash"])
        res["verdict"] = ("development criteria met (sealed window not opened)" if all(res["gates"].values())
                          else "fail")
    out = {"family": FAMILY, "seed": seed, "config": config, "warm_session": dates[warm], "results": results,
           "variants": variants, "ftd_count_in_window": int(ftd[lo:hi + 1].sum()),
           "data": {"load_counts": data["counts"], "substitutions": data["substitutions"],
                    "ew_members_mean": data["ew_members_mean"]}}
    out["markdown"] = v10_report.markdown(out)
    summary = v10_report.summary(out)
    base = {"candidate": CANDIDATE, "config": config, "seed": seed, "code_version": ledger.code_version(),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    out["trial_id"], out["summary"] = ledger.short_id(base), summary
    ledger.append(ledger_path, {**base, "event": "exposure_run", "trial_id": out["trial_id"], "family": FAMILY,
                                "sealed_opened": False, "sealed_boundary": boundary,
                                "date_range": [config["first_session"], config["last_interval"]],
                                "params_hash": ledger.sha(config), "summary": summary})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trader_koo.research.bench.v10_run")
    ap.add_argument("--db", required=True, help="a copy of the research database (opened read-only)")
    ap.add_argument("--ledger")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)
    path = Path(args.ledger) if args.ledger else Path(args.db).with_name("bench_trial_ledger.jsonl")
    try:
        out = run_v10(args.db, path)
    except LookupError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(out["markdown"])
    print(f"ledger trial_id (v10): {out['trial_id']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(v10_report.jsonable(out), indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

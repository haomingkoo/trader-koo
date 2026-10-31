"""Run hypotheses v8 (docs/research/hypotheses-v8-swing.md) on the development window and append the run to the ledger.

H1 and H2 are per-trade event studies on the v2 machinery (state baseline, placebo, years, stress cost), plus
the per-trade excess over the equal-weight universe (the primary test) and a calendar-time portfolio scored with
scoring-rubric-v1. H3 is a portfolio run of the bench (`bench run --candidate v8_h3_52_week_high`); this module
reads its ledger entry so the three primary tests share one Holm family.

Usage: python -m trader_koo.research.bench.v8_run --db <copy of the research db> --h3-trial <trial id>
       [--ledger <path>] [--json-out <file>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trader_koo.research.settings import get as _settings

from . import eventstats as es
from . import ledger, v2_run, v8_report, verdict
from . import v8_signals as sig
from .data import Market, connect_ro, load_market
from .report import by_year, stats
from .v2_trades import BPS, Grid, build_grid, net
from .v3_run import sealed_boundary

_E = _settings().event_study
_B = _settings().bench
FAMILY = "v8"
CANDIDATE = "v8_swing"
H3_CANDIDATE = "v8_h3_52_week_high"
FIRST_ENTRY = "2015-01-01"  # registration: first entry 2015-01-02, the first full year of earnings data
REGISTRATION = Path(__file__).resolve().parents[3] / "docs" / "research" / "hypotheses-v8-swing.md"
YEAR_CHARS = 4
PRIMARY = ("H1", "H2", "H3")
EXIT_REFUSED = 3


def registration_sha256() -> str:
    return hashlib.sha256(REGISTRATION.read_bytes()).hexdigest()


def stored_matrix(db_path: str, column: str, boundary: str, like: pd.DataFrame) -> np.ndarray:
    """A stored ``price_daily`` column (close or volume, as stored) before the seal, on the grid of ``like``."""
    conn = connect_ro(db_path)
    try:
        raw = pd.read_sql_query(f"SELECT ticker, date, {column} FROM price_daily WHERE {sig.PRICE_BASIS} "
                                "AND ticker NOT LIKE '^%' AND date < ?", conn, params=(boundary,))
    finally:
        conn.close()
    return raw.pivot(index="date", columns="ticker", values=column).reindex(index=like.index, columns=like.columns).to_numpy()


def calendar_returns(o: np.ndarray, trades: list[tuple[int, int, int, float]], cost_bps: float) -> tuple[np.ndarray, np.ndarray]:
    """Equal weight across open trades per open-to-open interval d (open d to open d+1); 0 with no open trade.

    ``o`` is the forward-filled adjusted open; a trade (entry e, exit x, column, gross) is worth o[d]/o[e] at
    open d < x and gross at x. The fee is charged on its first and last interval; an exit at the entry open is
    one interval of fees only.
    """
    n = o.shape[0] - 1
    c = cost_bps / BPS
    total, count = np.zeros(n), np.zeros(n, dtype=int)
    for e, x, col, gross in trades:
        if x <= e:
            if e < n:
                total[e] += gross * (1.0 - c) / (1.0 + c) - 1.0
                count[e] += 1
            continue
        v = np.append(o[e:x, col] / o[e, col], gross)
        seg = v[1:] / v[:-1]
        seg[0] /= 1.0 + c
        seg[-1] *= 1.0 - c
        total[e:x] += seg - 1.0
        count[e:x] += 1
    return np.where(count > 0, total / np.maximum(count, 1), 0.0), count


def _trades(w: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    ok = (w > 0) & np.isfinite(grid.gross)
    rows, cols = np.nonzero(ok)
    rep = w[rows, cols]
    return np.repeat(rows, rep), np.repeat(cols, rep)


def ew_excess(p, grid: Grid, w: np.ndarray, seed: int) -> dict:
    """Per-trade net return minus the equal-weight universe over the same window: mean, block bootstrap, years."""
    rows, cols = _trades(w, grid)
    if rows.size == 0:
        return {"n": 0, "mean": None, "p": 1.0, "boot": None}
    _, ew = v2_run.benchmark_windows(p, p.member.to_numpy(), rows, cols, grid)
    out = {"n": int(rows.size)}
    for tag, bps in (("", _B.cost_bps), ("_stress", _B.stress_bps)):
        out[f"mean{tag}"] = float(np.mean(net(grid.gross, bps)[rows, cols] - ew))
    x = net(grid.gross, _B.cost_bps)[rows, cols] - ew
    n = grid.gross.shape[0]
    # One stratum and a zero baseline: the bootstrap effect is the mean excess per trade.
    parts = np.stack([np.bincount(rows, minlength=n), np.bincount(rows, weights=x, minlength=n),
                      np.ones(n), np.zeros(n)]).astype(float)[:, None, :]
    boot = es.bootstrap(parts, seed)
    years = np.array([d[:YEAR_CHARS] for d in p.close.index])[rows]
    by = [{"year": int(y), "trades": int((years == y).sum()), "excess": float(x[years == y].mean())}
          for y in sorted(set(years))]
    out.update(p=es.p_two_sided(boot.effect), boot=boot, undefined_resamples=boot.undefined, years=by,
               pct_years_positive=float(np.mean([y["excess"] > 0 for y in by])))
    return out


def calendar_block(name: str, p, grid: Grid, w: np.ndarray, first_row: int, last: int, placebo: dict) -> dict:
    """Calendar-time portfolio of the trades, against SPY and the equal-weight universe, graded with the rubric."""
    o = p.open.ffill().to_numpy()
    rows, cols = _trades(w, grid)
    trades = [(int(r), int(grid.exit_row[r, c]), int(c), float(grid.gross[r, c])) for r, c in zip(rows, cols)]
    spy, cum = v2_run.benchmark_levels(p, p.member.to_numpy())
    lo, hi = first_row, last  # intervals lo .. hi-1, open lo to open hi
    series = {"strategy": calendar_returns(o, trades, _B.cost_bps)[0][lo:hi],
              "stress": calendar_returns(o, trades, _B.stress_bps)[0][lo:hi],
              "spy": spy[lo + 1:hi + 1] / spy[lo:hi] - 1.0, "equal_weight": cum[lo + 1:hi + 1] / cum[lo:hi] - 1.0}
    invested = calendar_returns(o, trades, _B.cost_bps)[1][lo:hi]
    st = {k: stats(v) for k, v in series.items()}
    years = by_year(list(p.close.index[lo:hi]), series)
    dev = {"start": p.close.index[lo], "end": p.close.index[hi], "strategy": st["strategy"], "spy": st["spy"],
           "equal_weight": st["equal_weight"], "stress": st["stress"],
           "excess_vs_equal_weight": st["strategy"]["cagr"] - st["equal_weight"]["cagr"],
           "random": {"n": placebo["draws"], "percentile": placebo["pct"]},
           "pct_years_beating_both": float(np.mean([y["strategy"] > y["spy"] and y["strategy"] > y["equal_weight"]
                                                    for y in years])),
           "share_of_intervals_invested": float(np.mean(invested > 0)), "mean_open_trades": float(invested.mean()),
           "years": years}
    return {"dev": dev, "grade": verdict.grade({"candidate": name, "dev": dev})}


def analyse(key: str, p, grid: Grid, w_h: np.ndarray, w_b: np.ndarray, first_row: int, last: int, seed: int,
            calendar: bool) -> dict:
    res = v2_run.study(key, p, grid, w_h, w_b, first_row, seed)
    w = w_h.copy()
    w[:first_row] = 0
    res["ew"] = ew_excess(p, grid, w, seed)
    if calendar and res.get("placebo"):
        res["calendar"] = calendar_block(key, p, grid, w, first_row, last, res["placebo"])
    return res


def verdict_v8(res: dict, adj_p: float, level: float) -> dict:
    """Criteria 1 to 5 of the H1/H2 pass rule (criterion 6, the sealed window, is not run)."""
    ew = res["ew"]
    if res.get("p") is None or ew["mean"] is None:
        return {"status": "insufficient", "why": "no scored trades or no baseline trades", "adj_p": adj_p}
    checks = {
        "1 excess over equal weight > 0, Holm p < 0.05": ew["mean"] > 0 and adj_p < _E.alpha,
        "2 net mean above the placebo 95th percentile": res["placebo"]["pct"] > _E.placebo_pct_high,
        "3 excess over equal weight > 0 in at least 60% of years": ew["pct_years_positive"] >= _E.min_positive_year_share,
        "4 excess over equal weight > 0 at 20 bps": ew["mean_stress"] > 0,
        "5 net mean above the state baseline": res["excess"] > 0,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return {"status": "fail" if failed else "development criteria met (sealed window not opened)", "checks": checks,
            "adj_p": adj_p, "interval": es.interval(ew["boot"].effect, level), "why": "; ".join(failed) or "criteria 1 to 5 met"}


def h3_from_ledger(path: str | Path, trial: str) -> dict:
    """The H3 portfolio run's summary, its benchmarks and its one-sided p against the random portfolios."""
    entry = next((e for e in ledger.read(path) if e.get("event") == "run" and e.get("trial_id") == trial), None)
    if entry is None or entry["candidate"] != H3_CANDIDATE or entry["sealed_opened"]:
        raise LookupError(f"{trial} is not a development-window {H3_CANDIDATE} run in {path}")
    s, n = entry["summary"], entry["config"]["n_random"]
    beaten_by = round(n * (1.0 - s["random_percentile"] / 100.0))
    ew_cagr, spy_cagr = s["cagr"] - s["excess_vs_equal_weight"], s["cagr"] - s["excess_vs_spy"]
    return {"trial_id": trial, "summary": s, "n_random": n, "p": (1 + beaten_by) / (n + 1), "equal_weight_cagr": ew_cagr,
            "spy_cagr": spy_cagr, "config": entry["config"], "date_range": entry["date_range"]}


def verdict_h3(h3: dict, adj_p: float) -> dict:
    s = h3["summary"]
    checks = {"years beating SPY and equal weight >= 60%": s["pct_years_beating_both"] >= verdict.PASS_YEARS,
              "above the 95th percentile of random portfolios": s["random_percentile"] > verdict.PASS_PERCENTILE,
              "stress CAGR above both benchmarks": s["stress_cagr"] > max(h3["equal_weight_cagr"], h3["spy_cagr"]),
              "Holm p < 0.05 against the random portfolios": adj_p < _E.alpha}
    failed = [k for k, ok in checks.items() if not ok]
    return {"status": "fail" if failed else "development criteria met (sealed window not opened)", "checks": checks,
            "adj_p": adj_p, "why": "; ".join(failed) or "all development criteria met"}


def _cells(rows: np.ndarray, cols: np.ndarray, keep: np.ndarray, shape: tuple) -> np.ndarray:
    w = np.zeros(shape, dtype=int)
    np.add.at(w, (rows[keep], cols[keep]), 1)
    return w


def events_on_grid(db_path: str, boundary: str, p, last: int) -> tuple[pd.DataFrame, dict]:
    conn = connect_ro(db_path)
    try:
        ev = sig.load_events(conn, boundary)
    finally:
        conn.close()
    sessions = list(p.close.index)
    col = {t: i for i, t in enumerate(p.close.columns)}
    ev["col"] = ev["ticker"].map(col).fillna(-1).astype(int)
    ev["row"] = sig.entry_rows(ev["report_date"], ev["session"], sessions)
    counts = {"events_before_the_seal": len(ev), "events_ticker_without_prices": int((ev["col"] < 0).sum()),
              "events_without_a_later_development_session": int(((ev["row"] < 0) | (ev["row"] > last)).sum()),
              "events_unknown_session_entered_after_the_report_date": int(ev["session"].isna().sum())}
    ev = ev[(ev["col"] >= 0) & (ev["row"] >= 1) & (ev["row"] <= last)].reset_index(drop=True)
    ev["member"] = p.member.to_numpy()[ev["row"], ev["col"]]
    counts["events_not_a_member_at_entry"] = int((~ev["member"]).sum())
    return ev, counts


def run_v8(db_path: str | Path, ledger_path: str | Path, h3_trial: str, market: Market | None = None,
           first_entry: str = FIRST_ENTRY) -> dict:
    boundary = sealed_boundary(ledger_path)
    h3 = h3_from_ledger(ledger_path, h3_trial)
    market = market or load_market(db_path)
    db_path = str(db_path)
    p, last, load_counts = v2_run.load_prices(market, boundary)
    first_row = int(np.searchsorted(list(p.close.index), first_entry, side="left"))
    shape = p.close.shape
    config = {"family": FAMILY, "boundary": boundary, "first_entry_session": p.close.index[first_row],
              "last_session": p.close.index[last], "sessions": last + 1, "cost_bps": _B.cost_bps,
              "stress_bps": _B.stress_bps, "block": _E.bootstrap_block, "resamples": _E.bootstrap_resamples,
              "placebo_draws": _E.placebo_draws, "universe": _B.universe, "h3_trial": h3_trial,
              "registration": REGISTRATION.name, "registration_sha256": registration_sha256()}
    seed = ledger.seed_from_config(config)
    stored_close = stored_matrix(db_path, "close", boundary, p.close)
    volume = stored_matrix(db_path, "volume", boundary, p.close)
    ev, ev_counts = events_on_grid(db_path, boundary, p, last)
    rows, cols, member = ev["row"].to_numpy(), ev["col"].to_numpy(), ev["member"].to_numpy()
    o, h, lo, c = (getattr(p, k).to_numpy() for k in ("open", "high", "low", "close"))
    # H1: price-scaled surprise, cut on the previous quarter's eligible events.
    sue = sig.surprise(ev["eps_actual"].to_numpy(float), ev["eps_estimate"].to_numpy(float), rows, cols, stored_close)
    eligible = member & np.isfinite(sue) & np.isfinite(o[rows, cols])
    cut = sig.quarter_cutoffs([p.close.index[r] for r in rows], sue, eligible)
    top = eligible & np.isfinite(cut) & (sue >= cut)
    rest = eligible & np.isfinite(cut) & (sue < cut)
    amc = ev["session"].to_numpy() != sig.BMO
    h1 = {"events_without_eps_or_close_before_entry": int((member & ~np.isfinite(sue)).sum()),
          "events_without_an_entry_open": int((member & np.isfinite(sue) & ~np.isfinite(o[rows, cols])).sum()),
          "eligible_events": int(eligible.sum()), "eligible_without_a_previous_quarter": int((eligible & ~np.isfinite(cut)).sum()),
          "qualifying": int(top.sum()), "baseline": int(rest.sum())}
    bmo_top = top & ~amc & (rows + 1 <= last)
    h1["bmo_trades_with_higher_volume_next_session"] = int(
        (volume[np.minimum(rows + 1, last), cols] > volume[rows, cols])[bmo_top & (rows >= first_row)].sum())
    # H2: the reaction session g is the entry row of H1's timing; the trade enters at g + 1.
    g = rows
    gap_ok = sig.gap_qualifies(o, h, lo, c, volume, g, cols)
    e2 = g + 1
    in2 = (e2 <= last) & p.member.to_numpy()[np.minimum(e2, last), cols]
    h2 = {"reaction_sessions_with_member_entry": int(in2.sum()), "qualifying": int((in2 & gap_ok).sum()),
          "baseline": int((in2 & ~gap_ok).sum())}
    grids = {"H1": build_grid(market, p.open, p.close, None, sig.H1_HOLD),
             "H1_hold20": build_grid(market, p.open, p.close, None, sig.H1_VARIANT_HOLD),
             "H2": sig.stop_trail_grid(market, p, sig.H2_TRAIL, sig.H2_MAX_HOLD),
             "H2_trail20": sig.stop_trail_grid(market, p, sig.H2_VARIANT_TRAIL, sig.H2_MAX_HOLD)}
    cells = {"H1": (_cells(rows, cols, top, shape), _cells(rows, cols, rest, shape)),
             "H1_amc_only": (_cells(rows, cols, top & amc, shape), _cells(rows, cols, rest & amc, shape)),
             "H2": (_cells(e2, cols, in2 & gap_ok, shape), _cells(e2, cols, in2 & ~gap_ok, shape))}
    plan = {"H1": ("H1", "H1", True), "H2": ("H2", "H2", True), "H1_hold20": ("H1", "H1_hold20", False),
            "H1_amc_only": ("H1_amc_only", "H1", False), "H2_trail20": ("H2", "H2_trail20", False)}
    results = {k: analyse(k, p, grids[gk], *cells[ck], first_row, last, ledger.seed_from_config({**config, "spec": k}), cal)
               for k, (ck, gk, cal) in plan.items()}
    ps = [results["H1"]["ew"]["p"], results["H2"]["ew"]["p"], h3["p"]]
    adj, levels, _ = es.holm(ps)
    verdicts = {"H1": verdict_v8(results["H1"], adj[0], levels[0]), "H2": verdict_v8(results["H2"], adj[1], levels[1]),
                "H3": verdict_h3(h3, adj[2])}
    out = {"family": FAMILY, "seed": seed, "config": config, "boundary": boundary, "results": results, "h3": h3,
           "verdicts": verdicts, "holm": dict(zip(PRIMARY, adj)),
           "counts": {"events": ev_counts, "H1": h1, "H2": h2},
           "data": {"load_counts": load_counts, "substitutions": market.load_diag.as_dict(),
                    "member_quarters": member_quarter_coverage(ev, p, first_row, last)}}
    out["markdown"] = v8_report.markdown(out)
    summary = v8_report.summary(out)
    base = {"candidate": CANDIDATE, "config": config, "seed": seed, "code_version": ledger.code_version(),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    ledger.append(ledger_path, {**base, "event": "event_study_run", "trial_id": ledger.short_id(base), "family": FAMILY,
                                "sealed_opened": False, "sealed_boundary": boundary,
                                "date_range": [config["first_entry_session"], config["last_session"]],
                                "n_sessions": last + 1, "params_hash": ledger.sha(config), "summary": summary})
    out["trial_id"], out["summary"] = ledger.short_id(base), summary
    return out


def member_quarter_coverage(ev: pd.DataFrame, p, first_row: int, last: int) -> dict:
    """Index members with a price on a quarter's first session, and how many have no earnings row entering that quarter."""
    dates = list(p.close.index[first_row:last + 1])
    quarter = np.array([sig._quarter(d) for d in dates])
    starts = [first_row + int(np.argmax(quarter == q)) for q in np.unique(quarter)]
    seen = {(sig._quarter(p.close.index[r]), cc) for r, cc in zip(ev["row"], ev["col"])}
    member = p.member.to_numpy()
    total = missing = 0
    for s in starts:
        qs = sig._quarter(p.close.index[s])
        for cc in np.flatnonzero(member[s]):
            total += 1
            missing += (qs, cc) not in seen
    return {"member_quarters": total, "without_an_earnings_row": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trader_koo.research.bench.v8_run")
    ap.add_argument("--db", required=True, help="a copy of the research database (opened read-only)")
    ap.add_argument("--h3-trial", required=True, help="ledger trial id of the v8_h3_52_week_high portfolio run")
    ap.add_argument("--ledger")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)
    path = Path(args.ledger) if args.ledger else Path(args.db).with_name("bench_trial_ledger.jsonl")
    try:
        out = run_v8(args.db, path, args.h3_trial)
    except LookupError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(out["markdown"])
    print(f"ledger trial_id (v8): {out['trial_id']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(v8_report.jsonable(out), indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run hypotheses v2 (S1 to S3) as per-trade event studies on the development window and append the run to the ledger.

Every qualifying entry is a trade of equal dollar size, overlapping trades allowed (S1: one open trade per
stock). Each hypothesis is compared with its own state baseline (every cell that satisfies the state, same
exit rule), with 1,000 seeded placebo draws on the same stocks, with SPY and the equal-weight universe over
the same holding windows, by calendar year, and at the stress cost. The sealed window is never read into a result.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trader_koo.research.settings import get as _settings

from . import eventstats as es
from . import ledger, v2_report
from .data import BENCHMARK_TICKER, Market, connect_ro, load_market
from .v2_signals import (S1_MAX_HOLD, S2_MAX_HOLD, S3_HOLD, Prices, s1, s2, s3)
from .v2_trades import Grid, build_grid, entry_counts, net, one_at_a_time
from .v3_run import sealed_boundary

_E = _settings().event_study
_B = _settings().bench
FAMILY = "v2"
YEAR_CHARS = 4  # ISO date prefix width
KEYS = ("S1", "S2", "S3")
PRICE_BASIS = "COALESCE(adjustment_basis,'split_adjusted_price_only')='split_adjusted_price_only'"


def load_prices(market: Market, boundary: str) -> tuple[Prices, int, dict]:
    """Prices cut before the seal, with adjusted high and low on the same basis as the bench's adjusted open."""
    last = int(np.searchsorted(market.sessions, boundary, side="left")) - 1
    idx, cols = market.adj_close.index[:last + 1], market.adj_close.columns
    close, open_ = market.adj_close.iloc[:last + 1], market.adj_open.iloc[:last + 1]
    conn = connect_ro(market.conn.execute("PRAGMA database_list").fetchone()[2])
    try:
        raw = pd.read_sql_query(f"SELECT ticker, date, high, low, close FROM price_daily WHERE {PRICE_BASIS} "
                                "AND ticker NOT LIKE '^%' AND date < ?", conn, params=(boundary,))
    finally:
        conn.close()
    piv = {k: raw.pivot(index="date", columns="ticker", values=k).reindex(index=idx, columns=cols)
           for k in ("high", "low", "close")}
    counts = {}
    bad = (piv["high"] <= 0) | (piv["low"] <= 0) | (piv["high"] < piv["low"])
    counts["high_low_rows_nonpositive_or_crossed_set_to_missing"] = int(bad.sum().sum())
    factor = close / piv["close"]
    high, low = (piv["high"] * factor).where(~bad), (piv["low"] * factor).where(~bad)
    counts["adjusted_close_rows_without_a_high_or_low"] = int((close.notna() & (high.isna() | low.isna())).sum().sum())
    member = pd.DataFrame(False, index=idx, columns=cols)
    pos = {t: i for i, t in enumerate(cols)}
    arr = member.to_numpy()
    for i in range(last + 1):
        arr[i, [pos[t] for t in market.universe(i)]] = True
    return Prices(open_, high, low, close, pd.DataFrame(arr, index=idx, columns=cols)), last, counts


def _sums(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (w * np.where(w > 0, r, 0.0)).sum(axis=1)


def _benchmarks(p: Prices, member: np.ndarray, rows: np.ndarray, cols: np.ndarray, grid: Grid) -> tuple[float, float]:
    """Mean SPY and equal-weight-universe open-to-open return over the same holding windows."""
    o = p.open.to_numpy()
    spy = o[:, list(p.open.columns).index(BENCHMARK_TICKER)]
    daily = o[1:] / o[:-1] - 1.0
    ok = member[:-1] & np.isfinite(daily)
    ew = np.concatenate([[0.0], np.where(ok.sum(1) > 0, np.where(ok, daily, 0.0).sum(1) / np.maximum(ok.sum(1), 1), 0.0)])
    cum = np.cumprod(1.0 + ew)
    out = np.minimum(grid.exit_row[rows, cols] + grid.spinoff_exit[rows, cols], len(spy) - 1)
    return float(np.nanmean(spy[out] / spy[rows] - 1.0)), float(np.mean(cum[out] / cum[rows] - 1.0))


def _placebo(rng, r, valid, cols_used, counts) -> np.ndarray:
    """Mean net return of random entries on the same stocks with the same number of entries (draws x 1)."""
    total = np.zeros(_E.placebo_draws)
    for col, k in zip(cols_used, counts):
        days = np.flatnonzero(valid[:, col])
        pick = days[rng.integers(0, len(days), size=(_E.placebo_draws, int(k)))]
        total += r[pick, col].sum(axis=1)
    return total / counts.sum()


def study(key: str, p: Prices, grid: Grid, w_h: np.ndarray, w_b: np.ndarray, first_row: int, seed: int) -> dict:
    """All per-trade statistics of one hypothesis. ``w_h`` and ``w_b`` count trades per (entry session, ticker) cell."""
    n = w_h.shape[0]
    w_h, w_b = w_h.copy(), w_b.copy()
    w_h[:first_row], w_b[:first_row] = 0, 0
    scored = np.isfinite(grid.gross)
    counts = {"signals_entries": int(w_h.sum()), "baseline_entries": int(w_b.sum()),
              "dropped_entry_too_late_for_window": int((w_h * grid.late).sum()),
              "dropped_no_open_at_entry": int((w_h * grid.no_entry_open).sum()),
              "dropped_no_exit_price": int((w_h * grid.unscorable).sum()),
              "exit_open_missing_next_open_used": int((w_h * grid.fallback_next_open).sum()),
              "exit_open_missing_last_close_used": int((w_h * grid.fallback_last_close).sum()),
              "spinoff_exits": int((w_h * grid.spinoff_exit).sum()),
              "baseline_dropped_not_scorable": int((w_b * ~scored).sum()),
              "baseline_exit_open_missing_next_open_used": int((w_b * grid.fallback_next_open).sum()),
              "baseline_exit_open_missing_last_close_used": int((w_b * grid.fallback_last_close).sum()),
              "baseline_spinoff_exits": int((w_b * grid.spinoff_exit).sum())}
    w_h, w_b = w_h * scored, w_b * scored
    res = {"key": key, "counts": counts, "n_trades": int(w_h.sum()), "n_baseline": int(w_b.sum())}
    r = {c: net(grid.gross, c) for c in (_B.cost_bps, _B.stress_bps)}
    for tag, c in (("", _B.cost_bps), ("_stress", _B.stress_bps)):
        mh = _sums(w_h, r[c]).sum() / max(w_h.sum(), 1)
        mb = _sums(w_b, r[c]).sum() / max(w_b.sum(), 1)
        res[f"mean_net{tag}"], res[f"baseline_net{tag}"], res[f"excess{tag}"] = float(mh), float(mb), float(mh - mb)
    res["mean_gross"] = float(_sums(w_h, grid.gross).sum() / max(w_h.sum(), 1))
    if w_h.sum() == 0 or w_b.sum() == 0:
        return {**res, "p": None, "interval": None, "status": "insufficient"}
    parts = np.stack([w_h.sum(1), _sums(w_h, r[_B.cost_bps]), w_b.sum(1), _sums(w_b, r[_B.cost_bps])]).astype(float)
    boot = es.bootstrap(parts[:, None, :], seed)
    res.update(p=es.p_two_sided(boot.effect), undefined_resamples=boot.undefined, boot=boot)
    years = np.array([d[:YEAR_CHARS] for d in p.close.index])
    rows_y = []
    for y in sorted(set(years[first_row:])):
        m = years == y
        nh, nb = w_h[m].sum(), w_b[m].sum()
        mh = _sums(w_h[m], r[_B.cost_bps][m]).sum() / nh if nh else np.nan
        mb = _sums(w_b[m], r[_B.cost_bps][m]).sum() / nb if nb else np.nan
        rows_y.append({"year": int(y), "trades": int(nh), "mean_net": mh, "baseline_net": mb, "excess": mh - mb})
    res["years"] = rows_y
    pos = [y["mean_net"] > 0 for y in rows_y if np.isfinite(y["mean_net"])]
    pos_x = [y["excess"] > 0 for y in rows_y if np.isfinite(y["excess"])]
    res["pct_years_positive"], res["pct_years_excess_positive"] = float(np.mean(pos)), float(np.mean(pos_x))
    rows, cols = np.nonzero(w_h)
    rep = w_h[rows, cols]
    rows, cols = np.repeat(rows, rep), np.repeat(cols, rep)
    res["spy_mean"], res["equal_weight_mean"] = _benchmarks(p, p.member.to_numpy(), rows, cols, grid)
    entry_member = np.vstack([np.zeros((1, w_h.shape[1]), bool), p.member.to_numpy()[:-1]])
    valid = scored & entry_member
    valid[:first_row] = False
    used, k = np.unique(cols, return_counts=True)
    plac = _placebo(np.random.default_rng(seed), r[_B.cost_bps], valid, used, k)
    res["placebo"] = {"draws": len(plac), "pct": float(100.0 * np.mean(plac < res["mean_net"])),
                      "p95": float(np.percentile(plac, _E.placebo_pct_high)), "p50": float(np.median(plac))}
    return res


def verdict(res: dict, level: float, adj_p: float) -> dict:
    """The pass rule of hypotheses-v2 on the development window (criterion 5, the sealed window, is not run)."""
    if res.get("p") is None:
        return {"status": "insufficient", "why": "no scored trades or no baseline trades"}
    ci = es.interval(res["boot"].effect, level)
    checks = {
        "excess over the state baseline > 0 with Holm-corrected p < 0.05": res["excess"] > 0 and adj_p < _E.alpha,
        "net mean above the placebo 95th percentile": res["placebo"]["pct"] > _E.placebo_pct_high,
        f"positive in at least {_E.min_positive_year_share:.0%} of years (the lower of net and excess)":
            min(res["pct_years_positive"], res["pct_years_excess_positive"]) >= _E.min_positive_year_share,
        "still positive at the stress cost (net mean and excess)": res["mean_net_stress"] > 0 and res["excess_stress"] > 0,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return {"status": "fail" if failed else "development criteria met (sealed window not opened)",
            "interval": ci, "adj_p": adj_p, "checks": checks, "why": "; ".join(failed) or "criteria 1 to 4 met"}


def _by_rule(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        d = out.setdefault(r["rule"], {"tickers": 0, "member_days": 0})
        d["tickers"] += 1
        d["member_days"] += int(r["member_days"])
    return out


def run_v2(db_path: str | Path, ledger_path: str | Path, market: Market | None = None) -> dict:
    boundary = sealed_boundary(ledger_path)
    market = market or load_market(db_path)
    p, last, load_counts = load_prices(market, boundary)
    first_row = int(np.searchsorted(list(p.close.index), f"{_E.first_walk_forward_year}-01-01", side="left"))
    config = {"family": FAMILY, "boundary": boundary, "first_entry_session": p.close.index[first_row],
              "last_session": p.close.index[last], "sessions": last + 1, "cost_bps": _B.cost_bps,
              "stress_bps": _B.stress_bps, "block": _E.bootstrap_block, "resamples": _E.bootstrap_resamples,
              "placebo_draws": _E.placebo_draws, "universe": _B.universe}
    seed = ledger.seed_from_config(config)
    shape = p.close.shape
    a1, a2, a3 = s1(p), s2(p), s3(p)
    grids = {"S1": build_grid(market, p.open, p.close, a1["exit_ema"], S1_MAX_HOLD),
             "S2": build_grid(market, p.open, p.close, a2["exit_ema"], S2_MAX_HOLD),
             "S3": build_grid(market, p.open, p.close, None, S3_HOLD)}
    member = p.member.to_numpy()
    # S1: one open trade per stock; the baseline is every stack day without the pullback.
    cells = entry_counts(*np.nonzero(a1["signal"].to_numpy()), shape) > 0
    w1 = one_at_a_time(cells, grids["S1"]).astype(int)
    skipped_s1 = int(cells[first_row:].sum() - w1[first_row:].sum())
    wb1 = entry_counts(*np.nonzero(a1["state"].to_numpy()), shape)
    # S2: first retest of the 13 EMA within 10 sessions of the cross; the baseline enters after every cross.
    cr, cc = np.nonzero(a2["cross"].to_numpy())
    t_ = a2["retest_day"][cr, cc]
    has = t_ >= 0
    traded = has & member[np.maximum(t_, 0), cc]
    w2 = entry_counts(t_[traded], cc[traded], shape)
    wb2 = entry_counts(cr, cc, shape)
    w3 = entry_counts(*np.nonzero(a3["signal"].to_numpy()), shape)
    wb3 = entry_counts(*np.nonzero(a3["state"].to_numpy()), shape)
    pairs = {"S1": (w1, wb1), "S2": (w2, wb2), "S3": (w3, wb3)}
    results = {k: study(k, p, grids[k], *pairs[k], first_row, ledger.seed_from_config({**config, "spec": k})) for k in KEYS}
    results["S1"]["counts"]["signals_skipped_a_trade_was_open"] = skipped_s1
    results["S2"]["counts"].update(
        crosses=int(((cr >= first_row)).sum()), crosses_without_a_retest_in_10_sessions=int(((cr >= first_row) & ~has).sum()),
        retests_not_an_index_member=int(((cr >= first_row) & has & ~traded).sum()))
    ps = [r["p"] if r.get("p") is not None else 1.0 for r in results.values()]
    adj, levels, _ = es.holm(ps)
    verdicts = {k: verdict(results[k], levels[i], adj[i]) for i, k in enumerate(KEYS)}
    out = {"family": FAMILY, "seed": seed, "config": config, "boundary": boundary, "results": results,
           "verdicts": verdicts, "data": {"load_counts": load_counts, "substitutions": {**market.load_diag.as_dict()},
                                          "exclusion_by_rule": _by_rule(market.exclusion_summary(0, last))}}
    out["markdown"] = v2_report.markdown(out)
    summary = {k: {"trades": results[k]["n_trades"], "mean_net": round(results[k]["mean_net"], _E.report_digits),
                   "excess": round(results[k]["excess"], _E.report_digits), "adjusted_p": round(adj[i], _E.report_digits),
                   "status": verdicts[k]["status"]} for i, k in enumerate(KEYS)}
    base = {"candidate": "v2_saty_setups", "config": config, "seed": seed, "code_version": ledger.code_version(),
            "timestamp": datetime.now(timezone.utc).isoformat()}
    ledger.append(ledger_path, {**base, "event": "event_study_run", "trial_id": ledger.short_id(base), "family": FAMILY,
                                "sealed_opened": False, "sealed_boundary": boundary,
                                "date_range": [config["first_entry_session"], config["last_session"]],
                                "n_sessions": last + 1, "params_hash": ledger.sha(config), "summary": summary})
    out["trial_id"], out["summary"] = ledger.short_id(base), summary
    return out

"""Markdown report and ledger summary of a v9 run. Research output on historical data; not a claim of alpha."""
from __future__ import annotations

import json

import numpy as np

from . import verdict

PORTFOLIOS = ("H1a", "H1b", "H2", "H3", "H4", "H6", "H7")
ARRAYS = ("returns", "stress_returns")


def _p(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def _r(x) -> str:
    return "n/a" if x is None else f"{x:+.2f}R"


def _s(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _row_portfolio(k: str, r: dict, v: dict | None, holm: float | None) -> str:
    pl = r.get("placebo") or {}
    s = r["strategy"]
    holm_txt = f"{holm:.4f}" if holm is not None else "not in family"
    status = f"**{v['status']}**: {v['why']}" if v else "variant, reported only"
    sharpe = "n/a" if s["sharpe"] is None else f"{s['sharpe']:.2f}"
    pl_txt = f"{pl['pct']:.1f} (median {_p(pl['median'])}, p95 {_p(pl['p95'])}), p {pl['p']:.4f}" if pl else "not run"
    return (f"| {k} | {r.get('trades', {}).get('n', 'n/a')} | {_p(s['cagr'])} | {_p(s['max_drawdown'])} | "
            f"{sharpe} | "
            f"{_p(r['stress']['cagr'])} | {_p(r['spy']['cagr'])} | {_p(r['equal_weight']['cagr'])} | "
            f"{_s(r['pct_years_beating_both'])} | {_s(r['share_of_sessions_invested'])} | {pl_txt} | {holm_txt} | {status} |")


def _row_trades(k: str, t: dict) -> str:
    if not t or t.get("n", 0) == 0:
        return f"| {k} | 0 | | | | | | | |"
    return (f"| {k} | {t['n']} | {_s(t['win_rate'])} | {_r(t['mean_win_r'])} | {_r(t['mean_loss_r'])} | "
            f"{_r(t['mean_r'])} | {_s(t['top5_share_of_net'])} | {_s(t['top5_share_of_gross_profit'])} | "
            f"{t['median_hold']:.0f} |")


PORT_HEADER = ("| test | trades | CAGR | max drawdown | Sharpe | CAGR at 20 bps | SPY CAGR | equal-weight CAGR | "
               "years beating both | sessions invested | placebo percentile | Holm p | verdict |")
TRADE_HEADER = ("| test | trades | win rate | mean win | mean loss | mean R | top 5% share of net profit | "
                "top 5% share of gross profit | median hold (sessions) |")


def _sep(header: str) -> str:
    return "|" + "---|" * (header.count("|") - 1)


def markdown(out: dict) -> str:
    res, var, v, holm = out["results"], out["variants"], out["verdicts"], out["holm"]
    cfg = out["config"]
    lines = [f"# v9 run ({cfg['first_entry_session']} to {cfg['last_session']}, seal {out['boundary']})", "",
             "Research output on historical data; not a claim of alpha.", "",
             f"Registration `{cfg['registration']}` sha256 `{cfg['registration_sha256']}`. Seed {out['seed']}, "
             f"{cfg['placebo_draws']} placebo draws, costs {cfg['cost_bps']:g} bps per side, stress {cfg['stress_bps']:g}.",
             "", "## Portfolios", "", PORT_HEADER, _sep(PORT_HEADER)]
    lines += [_row_portfolio(k, res[k], v[k], holm[k]) for k in PORTFOLIOS]
    lines += ["", "## Trades taken by each portfolio (net R at 10 bps)", "", TRADE_HEADER, _sep(TRADE_HEADER)]
    lines += [_row_trades(k, res[k].get("trades")) for k in PORTFOLIOS if k != "H6"]
    lines += ["", "## Every signal traded once, no portfolio limits", "", TRADE_HEADER, _sep(TRADE_HEADER)]
    lines += [_row_trades(k, res[k].get("all_signals")) for k in PORTFOLIOS if k != "H6"]
    h = out["head_to_head"]
    lines += ["", "## H1a against H1b (mean net R per signal)", ""]
    if h.get("status") == "insufficient":
        lines.append("insufficient: one side has no signals")
    else:
        iv = h["interval"]
        lines.append(f"H1a {h['n_h1a']} signals, {_r(h['mean_r_h1a'])}; H1b {h['n_h1b']} signals, "
                     f"{_r(h['mean_r_h1b'])}; difference {_r(h['difference'])}, raw p {h['p']:.4f}, Holm "
                     f"{holm['H1a_vs_H1b']:.4f}, interval {'n/a' if iv is None else f'[{iv[0]:+.2f}R, {iv[1]:+.2f}R]'}: "
                     f"**{v['H1a_vs_H1b']['status']}**")
    corr = np.asarray(res["H6"]["correlation"])
    lines += ["", "## Correlation of the sleeves' daily returns", "", "| | " + " | ".join(out["sleeves"]) + " |",
              "|---|" + "---|" * len(out["sleeves"])]
    lines += [f"| {a} | " + " | ".join(f"{x:.2f}" for x in corr[i]) + " |" for i, a in enumerate(out["sleeves"])]
    lines += ["", "## Variants (reported only, no placebo)", "", PORT_HEADER, _sep(PORT_HEADER)]
    lines += [_row_portfolio(k, r, None, None) for k, r in var.items()]
    lines += ["", TRADE_HEADER, _sep(TRADE_HEADER)] + [_row_trades(k, r.get("trades")) for k, r in var.items()]
    lines += ["", "## Calendar years (strategy / SPY / equal weight)", ""]
    for k in PORTFOLIOS:
        lines.append(f"- {k}: " + ", ".join(f"{y['year']} {_p(y['strategy'])} / {_p(y['spy'])} / {_p(y['equal_weight'])}"
                                            for y in res[k]["years"]))
    lines += ["", "## Skips and exits", ""]
    for k in PORTFOLIOS:
        if "skipped" in res[k]:
            lines.append(f"- {k}: skipped {res[k]['skipped']}; exits {res[k]['exits']}; "
                         f"mean open positions {res[k]['mean_open']:.2f}")
    lines += ["", "## Counts", "", "```", json.dumps(out["counts"], indent=1, default=str), "```", "",
              f"H5: {out['h5']['status']} ({out['h5']['why']}; IWM first date {out['h5']['iwm_first_date']})", "",
              "## Rubric grades", ""]
    lines += [verdict.render(res[k]["grade"]) + "\n" for k in PORTFOLIOS if "grade" in res[k]]
    return "\n".join(lines)


def _brief(r: dict) -> dict:
    t = r.get("trades") or {}
    pl = r.get("placebo") or {}
    return {"trades": t.get("n"), "cagr": r["strategy"]["cagr"], "max_drawdown": r["strategy"]["max_drawdown"],
            "sharpe": r["strategy"]["sharpe"], "stress_cagr": r["stress"]["cagr"], "spy_cagr": r["spy"]["cagr"],
            "equal_weight_cagr": r["equal_weight"]["cagr"], "pct_years_beating_both": r["pct_years_beating_both"],
            "placebo_pct": pl.get("pct"), "p": pl.get("p"), "win_rate": t.get("win_rate"), "mean_r": t.get("mean_r"),
            "mean_win_r": t.get("mean_win_r"), "mean_loss_r": t.get("mean_loss_r"),
            "top5_share_of_net": t.get("top5_share_of_net"), "all_signals": (r.get("all_signals") or {}).get("n"),
            "all_signals_mean_r": (r.get("all_signals") or {}).get("mean_r")}


def summary(out: dict) -> dict:
    s = {k: {**_brief(out["results"][k]), "holm_p": out["holm"][k], "status": out["verdicts"][k]["status"]}
         for k in PORTFOLIOS}
    h = out["head_to_head"]
    s["H1a_vs_H1b"] = {"difference_r": h.get("difference"), "p": h["p"], "holm_p": out["holm"]["H1a_vs_H1b"],
                       "status": out["verdicts"]["H1a_vs_H1b"]["status"]}
    s["H6"]["correlation"] = out["results"]["H6"]["correlation"]
    s["H5"] = {"status": out["h5"]["status"]}
    s["variants"] = {k: _brief(r) for k, r in out["variants"].items()}
    return jsonable(s)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if k not in ARRAYS}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x

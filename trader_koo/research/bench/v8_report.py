"""Markdown report and ledger summary of a v8 run. Research output on historical data; not a claim of alpha."""
from __future__ import annotations

import json

from trader_koo.research.settings import get as _settings

from . import verdict

_E = _settings().event_study
_B = _settings().bench
EVENT_KEYS = ("H1", "H2")
VARIANTS = ("H1_hold20", "H1_amc_only", "H2_trail20")
HEADER = ("| test | trades | mean net | state baseline net | excess vs baseline | excess vs equal weight | Holm p | "
          "Holm interval | placebo pct (p95) | years excess > 0 | excess at 20 bps | SPY window | equal-weight window | verdict |")


def _p(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.3f}%"


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def _row(k: str, r: dict, v: dict | None) -> str:
    ew = r["ew"]
    if r.get("p") is None or ew["mean"] is None:
        why = v["why"] if v else "insufficient"
        return f"| {k} | {r['n_trades']} | | | | | | | | | | | | **insufficient**: {why} |"
    iv = v.get("interval") if v else None
    holm = f"{v['adj_p']:.4f}" if v else "not in family"
    interval = "n/a" if iv is None else f"[{iv[0] * 100:+.3f}%, {iv[1] * 100:+.3f}%]"
    status = f"**{v['status']}**: {v['why']}" if v else "variant, reported only"
    return (f"| {k} | {r['n_trades']} | {_p(r['mean_net'])} | {_p(r['baseline_net'])} | {_p(r['excess'])} | "
            f"{_p(ew['mean'])} (raw p {ew['p']:.4f}) | {holm} | {interval} | {r['placebo']['pct']:.1f} "
            f"({_p(r['placebo']['p95'])}) | {ew['pct_years_positive']:.0%} | {_p(ew['mean_stress'])} | "
            f"{_p(r['spy_mean'])} | {_p(r['equal_weight_mean'])} | {status} |")


def markdown(out: dict) -> str:
    res, ver, cfg, h3 = out["results"], out["verdicts"], out["config"], out["h3"]
    lines = ["# Hypotheses v8 (swing), development window", "",
             "Research output on historical data; not a claim of alpha or profitability. "
             f"{_B.cost_bps:g} bps per side (stress {_B.stress_bps:g}). Departed index members without price or earnings "
             "history are not in the universe, which flatters long-only results.", "",
             f"Entries {cfg['first_entry_session']} to {cfg['last_session']}; sealed window starts {out['boundary']} and "
             f"was not opened. Seed {out['seed']}; {_E.bootstrap_resamples} resamples of {_E.bootstrap_block}-session "
             f"blocks of entry sessions; Holm over H1, H2, H3. Registration sha256 {cfg['registration_sha256']}.", "",
             HEADER, "|" + "---|" * (HEADER.count("|") - 1)]
    lines += [_row(k, res[k], ver[k]) for k in EVENT_KEYS]
    lines += [_row(k, res[k], None) for k in VARIANTS]
    s = h3["summary"]
    lines += ["", f"H3 (portfolio run {h3['trial_id']}, {h3['date_range'][0]} to {h3['date_range'][1]}): CAGR "
              f"{_pct(s['cagr'])}, equal weight {_pct(h3['equal_weight_cagr'])}, SPY {_pct(h3['spy_cagr'])}, "
              f"random percentile {s['random_percentile']:.1f} of {h3['n_random']} (p {h3['p']:.4f}, Holm "
              f"{ver['H3']['adj_p']:.4f}), years beating both {_pct(s['pct_years_beating_both'])}, stress CAGR "
              f"{_pct(s['stress_cagr'])}. **{ver['H3']['status']}**: {ver['H3']['why']}", ""]
    for k in EVENT_KEYS:
        r = res[k]
        if "calendar" in r:
            d = r["calendar"]["dev"]
            lines += [verdict.render(r["calendar"]["grade"]), "",
                      f"Calendar portfolio {k}: invested in {d['share_of_intervals_invested']:.0%} of sessions, "
                      f"{d['mean_open_trades']:.1f} open trades on average; cash earns 0%. Max drawdown "
                      f"{_pct(d['strategy']['max_drawdown'])} (SPY {_pct(d['spy']['max_drawdown'])}).", "",
                      "| year | strategy | equal weight | SPY |", "|---|---|---|---|"]
            lines += [f"| {y['year']} | {_pct(y['strategy'])} | {_pct(y['equal_weight'])} | {_pct(y['spy'])} |"
                      for y in d["years"]]
            lines.append("")
    for k in (*EVENT_KEYS, *VARIANTS):
        r = res[k]
        lines += [f"## {k} by entry year (net of {_B.cost_bps:g} bps)", "",
                  "| year | trades | mean net | baseline net | excess vs baseline | excess vs equal weight |",
                  "|---|---|---|---|---|---|"]
        by_ew = {y["year"]: y["excess"] for y in r["ew"].get("years", [])}
        lines += [f"| {y['year']} | {y['trades']} | {_p(y['mean_net'])} | {_p(y['baseline_net'])} | {_p(y['excess'])} | "
                  f"{_p(by_ew.get(y['year']))} |" for y in r.get("years", [])]
        lines += ["", f"Counts for {k}:", "", "```", json.dumps(r["counts"], indent=1), "```", ""]
    lines += ["## Events and data", "", "```", json.dumps({**out["counts"], **out["data"]}, indent=1, default=str), "```", ""]
    return "\n".join(lines)


def _round(x):
    return None if x is None else round(float(x), _E.report_digits + 2)


def summary(out: dict) -> dict:
    res, ver = out["results"], out["verdicts"]
    s = {}
    for k in (*EVENT_KEYS, *VARIANTS):
        r = res[k]
        s[k] = {"trades": r["n_trades"], "mean_net": _round(r["mean_net"]), "excess_vs_baseline": _round(r["excess"]),
                "excess_vs_equal_weight": _round(r["ew"]["mean"]), "raw_p": _round(r["ew"]["p"])}
        if k in ver:
            s[k].update(adjusted_p=_round(ver[k]["adj_p"]), status=ver[k]["status"])
        if "calendar" in r:
            g = r["calendar"]["grade"]
            s[k].update(score=g["score"], calendar_cagr=_round(r["calendar"]["dev"]["strategy"]["cagr"]))
    h3 = out["h3"]
    s["H3"] = {"trial_id": h3["trial_id"], "random_percentile": h3["summary"]["random_percentile"], "p": _round(h3["p"]),
               "adjusted_p": _round(ver["H3"]["adj_p"]), "status": ver["H3"]["status"]}
    return s


def jsonable(out: dict) -> dict:
    """The run without bootstrap distributions (kept out of the JSON file to keep it small)."""
    def strip(r: dict) -> dict:
        r = {k: v for k, v in r.items() if k != "boot"}
        r["ew"] = {k: v for k, v in r["ew"].items() if k != "boot"}
        return r
    keep = {k: out[k] for k in ("family", "seed", "config", "trial_id", "summary", "verdicts", "holm", "counts", "data", "h3")}
    keep["results"] = {k: strip(v) for k, v in out["results"].items()}
    return keep

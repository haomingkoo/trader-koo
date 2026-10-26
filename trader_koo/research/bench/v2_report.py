"""Markdown report of a v2 run. Research output on historical data; not a claim of alpha or profitability."""
from __future__ import annotations

import json

from trader_koo.research.settings import get as _settings

_E = _settings().event_study
_B = _settings().bench


def _p(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.3f}%"


def markdown(out: dict) -> str:
    res, ver, cfg = out["results"], out["verdicts"], out["config"]
    lines = [f"# Hypotheses v2 event study ({out['family']}), development window", "",
             "Research output on historical data; not a claim of alpha or profitability. Every qualifying entry is a trade "
             f"of equal dollar size at the next open, {_B.cost_bps:g} bps per side (stress {_B.stress_bps:g}), overlapping "
             "trades allowed (S1: one open trade per stock). Departed index members without price history are not in the "
             "universe, which flatters long-only results.", "",
             f"Entries {cfg['first_entry_session']} to {cfg['last_session']}; sealed window starts {out['boundary']} and was "
             f"not opened. Seed {out['seed']}; {_E.bootstrap_resamples} resamples of {_E.bootstrap_block}-session blocks of "
             f"entry dates; Holm over {len(res)} hypotheses.", "",
             "| hypothesis | trades | mean net | mean gross | state baseline net | excess | Holm p | Holm interval | "
             "placebo percentile (p95) | years net > 0 | years excess > 0 | net at stress | excess at stress | SPY | "
             "equal weight | verdict |", "|" + "---|" * 16]
    for k, r in res.items():
        v = ver[k]
        if r.get("p") is None:
            lines.append(f"| {k} | {r['n_trades']} | | | | | | | | | | | | | | **{v['status']}**: {v['why']} |")
            continue
        iv = v["interval"]
        lines.append(
            f"| {k} | {r['n_trades']} | {_p(r['mean_net'])} | {_p(r['mean_gross'] - 1.0)} | {_p(r['baseline_net'])} | "
            f"{_p(r['excess'])} | {v['adj_p']:.4f} | {'n/a' if iv is None else f'[{iv[0] * 100:+.3f}%, {iv[1] * 100:+.3f}%]'} | "
            f"{r['placebo']['pct']:.1f} ({_p(r['placebo']['p95'])}) | {r['pct_years_positive']:.0%} | "
            f"{r['pct_years_excess_positive']:.0%} | {_p(r['mean_net_stress'])} | {_p(r['excess_stress'])} | "
            f"{_p(r['spy_mean'])} | {_p(r['equal_weight_mean'])} | **{v['status']}**: {v['why']} |")
    for k, r in res.items():
        lines += ["", f"## {k} by calendar year (entry year; net of {_B.cost_bps:g} bps)", "",
                  "| year | trades | mean net | baseline net | excess |", "|---|---|---|---|---|"]
        lines += [f"| {y['year']} | {y['trades']} | {_p(y['mean_net'])} | {_p(y['baseline_net'])} | {_p(y['excess'])} |"
                  for y in r.get("years", [])]
        lines += ["", f"Counts for {k}:", "", "```", json.dumps(r["counts"], indent=1), "```"]
    d = out["data"]
    lines += ["", "## Data", "", "Substitutions and skips while loading the price matrix:", "", "```",
              json.dumps({**d["load_counts"], **d["substitutions"]}, indent=1), "```", "",
              "A counter that is zero is omitted. Index-member days removed by the data-quality filter "
              "(`quality.py`; the same rules and the same stored series as hypotheses v1) inside the window, by rule:", "",
              "```", json.dumps(d["exclusion_by_rule"], indent=1), "```", ""]
    return "\n".join(lines)

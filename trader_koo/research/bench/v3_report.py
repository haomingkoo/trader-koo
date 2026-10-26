"""Markdown report of a v3 run. Research output on historical data; not a claim of alpha or profitability."""
from __future__ import annotations

import json

from trader_koo.research.settings import get as _settings

from . import eventstats as es

_E = _settings().event_study
D = _E.report_digits


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def pts(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}"


def ci(x) -> str:
    return "n/a" if not x else f"[{x[0] * 100:+.2f}, {x[1] * 100:+.2f}]"


def _claim(v: dict) -> str:
    c = v.get("claim")
    if not c:
        return ""
    return f"{c['value']:.0%}: rate interval [{c['interval'][0]:.3f}, {c['interval'][1]:.3f}]" + (
        " FALSIFIED" if c["falsified"] else "")


def markdown(out: dict) -> str:
    ev, ver = out["evaluations"], out["verdicts"]
    lines = [f"# Hypotheses v3 event study ({out['family']}), development window", "",
             "Research output on historical data; not a claim of alpha or profitability. A matched-baseline "
             "frequency difference is not a trading result: there are no costs, positions or sizing here.", "",
             f"Window {out['config']['first_session']} to {out['config']['last_session']} ({out['config']['sessions']} SPY "
             f"sessions). Sealed window starts {out['boundary']} and was not opened. Seed {out['seed']}; "
             f"{_E.bootstrap_resamples} resamples of {_E.bootstrap_block}-session blocks. Holm family m = {out['holm']['m']} "
             "primary tests (a blocked or insufficient test stays in the family).", "",
             "## Primary tests", "",
             "| test | signals | with outcome | independent (strict) | event rate | all events | matched | unconditional | "
             "effect (pts) | Holm p | Holm interval (pts) | stated claim | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in out["holm"]["adjusted_p"]:
        e, v = ev.get(k), ver[k]
        if e is None:
            lines.append(f"| {k} | | | | | | | | | | | | {v['status']}: {v['why']} |")
            continue
        lines.append(f"| {k} | {e.n_signals} | {e.n_all} | {e.n_independent} ({e.n_independent_strict}) | {pct(e.rate_event)} | "
                     f"{pct(e.rate_event_all)} | {pct(e.rate_matched)} | {pct(e.rate_unconditional)} | {pts(e.effect)} | "
                     f"{out['holm']['adjusted_p'][k]:.4f} | {ci(v.get('interval'))} | {_claim(v)} | **{v['status']}**: {v['why']} |")
    lines += ["", "## Secondary checks and variants (plain 95% interval, no verdict)", "",
              "| test | independent | event rate | matched | unconditional | effect (pts) | 95% interval (pts) | note |",
              "|---|---|---|---|---|---|---|---|"]
    for k, e in ev.items():
        if e.spec.role == "primary":
            continue
        lines.append(f"| {k} ({e.spec.role}) | {e.n_independent} | {pct(e.rate_event)} | {pct(e.rate_matched)} | "
                     f"{pct(e.rate_unconditional)} | {pts(e.effect)} | "
                     f"{ci(es.interval(e.boot.effect, _E.alpha)) if e.boot is not None and e.effect is not None else 'n/a'} | "
                     f"{e.spec.note} |")
    d = out["data"]
    lines += ["", "## Data", "", f"Price basis: `{json.dumps(d['basis'])}`. SPY, QQQ and ^VIX are the stored daily bars of the "
              "research database (price only, so ex-dividend gaps are in the closes).", "",
              "Substitutions, rejections and skips:", "", "```", json.dumps(d["substitutions"], indent=1), "```"]
    dropped = {k: (e.n_no_outcome, e.extra) for k, e in ev.items() if e.n_no_outcome or e.extra}
    lines += ["", "Signals dropped because their outcome window leaves the development window, and test-specific counts:", "",
              "```", json.dumps({k: {"signals_without_outcome": a, **b} for k, (a, b) in dropped.items()}, indent=1), "```"]
    return "\n".join(lines) + "\n"

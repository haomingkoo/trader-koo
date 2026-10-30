"""Score a bench run report with docs/research/scoring-rubric-v1.md: reasoning, pros, cons, gates, score.

Usage: python -m trader_koo.research.bench.verdict report.json [...]
(the JSON printed by ``bench run --json``).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

RUBRIC_VERSION = "scoring-rubric-v1"
FAIL_CAP = 5.0
COVERAGE_HALF = 0.5
PASS_YEARS = 0.60      # gate 1
PASS_PERCENTILE = 95.0  # gate 2


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    zero_at: float
    full_at: float
    points: float
    unit: str  # how the measured value is printed


# Rubric table, row for row. Changing a row is a new rubric version.
RUBRIC = (
    Criterion("edge", "Return edge vs equal weight", -0.02, 0.02, 3.0, "pct_per_year"),
    Criterion("luck", "Percentile vs random portfolios", 50.0, 95.0, 2.0, "percentile"),
    Criterion("consistency", "Years beating SPY and equal weight", 0.40, 0.60, 2.0, "share"),
    Criterion("costs", "Stress-run CAGR vs the higher benchmark", -0.01, 0.0, 1.0, "pct_per_year"),
    Criterion("drawdown", "Max drawdown vs SPY", -0.15, 0.0, 1.0, "pct_points"),
    Criterion("sharpe", "Sharpe vs SPY", -0.30, 0.0, 1.0, "ratio"),
)


def _earned(c: Criterion, value: float) -> float:
    frac = (value - c.zero_at) / (c.full_at - c.zero_at)
    return round(c.points * min(max(frac, 0.0), 1.0), 2)


def _fmt(unit: str, v: float) -> str:
    if unit == "pct_per_year":
        return f"{v * 100:+.1f}%/yr"
    if unit == "pct_points":
        return f"{v * 100:+.1f} pts"
    if unit == "share":
        return f"{v * 100:.0f}% of years"
    if unit == "percentile":
        return f"{v:.1f}th percentile"
    return f"{v:+.2f}"


def measures(dev: dict) -> dict[str, float]:
    s, spy, ew = dev["strategy"], dev["spy"], dev["equal_weight"]
    return {
        "edge": dev["excess_vs_equal_weight"],
        "luck": dev["random"]["percentile"],
        "consistency": dev["pct_years_beating_both"],
        "costs": dev["stress"]["cagr"] - max(spy["cagr"], ew["cagr"]),
        "drawdown": s["max_drawdown"] - spy["max_drawdown"],  # drawdowns are negative: below 0 is worse
        "sharpe": s["sharpe"] - spy["sharpe"],
    }


def grade(report: dict) -> dict:
    """Rubric result for one run report. Order of keys is the order the result is read in."""
    dev = report["dev"]
    m = measures(dev)
    rows = [{"criterion": c.label, "measure": _fmt(c.unit, m[c.key]), "earned": _earned(c, m[c.key]),
             "points": c.points} for c in RUBRIC]
    pros = [f"{r['criterion']}: {r['measure']} ({r['earned']:g}/{r['points']:g})" for r in rows
            if r["earned"] >= r["points"] / 2]
    cons = [f"{r['criterion']}: {r['measure']} ({r['earned']:g}/{r['points']:g})" for r in rows
            if r["earned"] < r["points"] / 2]
    cov = report.get("coverage") or {}
    if cov.get("mean_universe") and cov["mean_scored"] / cov["mean_universe"] < COVERAGE_HALF:
        cons.append(f"Coverage: scores {cov['mean_scored']:.0f} of {cov['mean_universe']:.0f} names on average, "
                    "so the result speaks for a subset")
    gates = {
        "years_beating_both": dev["pct_years_beating_both"] >= PASS_YEARS,
        "above_95th_random": dev["random"]["percentile"] > PASS_PERCENTILE,
        "stress_beats_both": m["costs"] > 0,
    }
    raw = round(sum(r["earned"] for r in rows), 2)
    passed = all(gates.values())
    score = raw if passed else min(raw, FAIL_CAP)
    verdict = "candidate" if passed else "fail"
    edge = m["edge"]
    reasoning = (f"{report['candidate']}: CAGR {dev['strategy']['cagr'] * 100:.1f}% against SPY "
                 f"{dev['spy']['cagr'] * 100:.1f}% and equal weight {dev['equal_weight']['cagr'] * 100:.1f}% "
                 f"({edge * 100:+.1f}%/yr), {dev['random']['percentile']:.1f}th percentile of "
                 f"{dev['random']['n']} random portfolios, beat both benchmarks in "
                 f"{dev['pct_years_beating_both'] * 100:.0f}% of years.")
    failed = [k for k, ok in gates.items() if not ok]
    if failed:
        reasoning += f" Fails gate(s): {', '.join(failed)}."
    return {"rubric": RUBRIC_VERSION, "candidate": report["candidate"], "window": f"{dev['start']} to {dev['end']}",
            "reasoning": reasoning, "pros": pros, "cons": cons, "rows": rows, "gates": gates,
            "raw_score": raw, "score": score, "verdict": verdict}


def render(g: dict) -> str:
    lines = [f"### {g['candidate']} ({g['window']}, {g['rubric']})", "", g["reasoning"], "", "Pros:"]
    lines += [f"- {p}" for p in g["pros"]] or ["- none"]
    lines += ["", "Cons:"] + ([f"- {c}" for c in g["cons"]] or ["- none"])
    lines += ["", "| Criterion | Measure | Points |", "|---|---|---|"]
    lines += [f"| {r['criterion']} | {r['measure']} | {r['earned']:g} / {r['points']:g} |" for r in g["rows"]]
    lines += ["", "Gates: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in g["gates"].items()),
              f"**Score {g['score']:g}/10 (raw {g['raw_score']:g}), verdict: {g['verdict']}**"]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    for path in argv:
        print(render(grade(json.loads(Path(path).read_text(encoding="utf-8")))), end="\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

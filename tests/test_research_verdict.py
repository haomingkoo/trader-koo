"""Scoring rubric v1: points per criterion, the fail cap, and pros/cons ordering."""
from __future__ import annotations

import pytest

from trader_koo.research.bench import verdict


def _report(**dev_overrides) -> dict:
    dev = {
        "start": "2011-01-04", "end": "2023-06-09",
        "strategy": {"cagr": 0.14, "sharpe": 0.80, "max_drawdown": -0.30},
        "spy": {"cagr": 0.12, "sharpe": 0.75, "max_drawdown": -0.34},
        "equal_weight": {"cagr": 0.12, "sharpe": 0.72, "max_drawdown": -0.40},
        "stress": {"cagr": 0.135}, "excess_vs_equal_weight": 0.02,
        "pct_years_beating_both": 0.70, "random": {"n": 200, "percentile": 99.0},
    }
    dev.update(dev_overrides)
    return {"candidate": "x", "dev": dev, "coverage": {"mean_universe": 400.0, "mean_scored": 300.0}}


def test_full_marks_and_candidate_when_every_gate_passes():
    g = verdict.grade(_report())
    assert g["score"] == 10.0 and g["verdict"] == "candidate" and not g["cons"]
    assert list(g)[:6] == ["rubric", "candidate", "window", "reasoning", "pros", "cons"]


def test_points_scale_linearly_between_anchors():
    g = verdict.grade(_report(excess_vs_equal_weight=0.0, random={"n": 200, "percentile": 72.5}))
    rows = {r["criterion"]: r["earned"] for r in g["rows"]}
    assert rows["Return edge vs equal weight"] == 1.5
    assert rows["Percentile vs random portfolios"] == 1.0


def test_a_failed_gate_caps_the_score_at_five():
    g = verdict.grade(_report(pct_years_beating_both=0.5))
    assert g["raw_score"] == pytest.approx(9.0) and g["score"] == 5.0 and g["verdict"] == "fail"
    assert "years_beating_both" in g["reasoning"]


def test_low_coverage_is_always_a_con():
    r = _report()
    r["coverage"] = {"mean_universe": 400.0, "mean_scored": 40.0}
    assert any(c.startswith("Coverage") for c in verdict.grade(r)["cons"])

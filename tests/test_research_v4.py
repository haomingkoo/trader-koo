"""Pre-registered hypotheses v4 on the research bench: inputs against filed rows, look-ahead guard, rules.

Uses the real-data fixture of tests/test_research_hypotheses.py (16 S&P 500 members, their filed SEC rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_research_hypotheses import AS_OF_IDX, _build
from trader_koo.research.bench import hypotheses as hyp
from trader_koo.research.bench import v4_signals as v4
from trader_koo.research.bench.data import load_market
from trader_koo.research.bench.signals import assert_no_lookahead, make_signal

V4 = ("v4_h1_magic_formula", "v4_h2_piotroski_value", "v4_h3_trend_template")


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    return load_market(_build(str(tmp_path_factory.mktemp("v4") / "r.db")))


@pytest.fixture(scope="module")
def view(market):
    return market.view(market.sessions[AS_OF_IDX])


def test_magic_formula_inputs_match_hand_computation_on_aapl(market, view):
    # Filed rows as of 2026-08-12, billions: operating income FY25 133.050 + YTD Q3 FY26 122.432
    # - YTD Q3 FY25 100.623 = 154.859; long-term debt 82.3, cash 39.544, assets 383.266.
    mcap = hyp.market_caps(view, ["AAPL"])["AAPL"]
    ey, roc = v4.magic_formula_inputs(view, "AAPL", mcap)
    assert ey == pytest.approx(154.859e9 / (mcap + 82.3e9 - 39.544e9), rel=1e-9)
    assert roc == pytest.approx(154.859 / (383.266 - 39.544), rel=1e-9)


def test_magic_formula_score_is_minus_the_rank_sum(market, view):
    uni = market.universe(AS_OF_IDX)
    scores = make_signal("v4_h1_magic_formula", {})(view.as_of, uni, view).dropna()
    mcap = hyp.market_caps(view, uni)
    x = pd.DataFrame([v4.magic_formula_inputs(view, t, mcap[t]) for t in scores.index],
                     index=scores.index, columns=["ey", "roc"])
    rank_sum = x["ey"].rank(ascending=False) + x["roc"].rank(ascending=False)
    gap = scores + rank_sum  # the earnings-yield tie-break adds less than one rank step
    assert ((gap > 0) & (gap < 1)).all()


def test_year_ago_instant_uses_the_period_one_year_earlier(view):
    assets = view.fundamentals("AAPL", "assets", "instant")
    prior = v4.year_ago_instant(view, "AAPL", "assets", assets)
    assert (assets.period_end, prior.period_end, prior.value) == ("2026-06-27", "2025-06-28", 331.495e9)


def test_f_score_on_aapl_matches_hand_count(view):
    # ROA 128.93 / 383.266 > 0; OCF 146.724 > 0; ROA up from 99.28 / 331.495; OCF > NI;
    # LTD / assets 82.3 / 383.266 below 91.8 / 331.495; shares 14.594bn below 14.840bn a year earlier.
    # Asset turnover 466.823 / 383.266 = 1.218 is below 408.625 / 331.495 = 1.233: no point. F = 6.
    assert v4.f_score(view, "AAPL") == 6.0


def test_piotroski_scores_only_the_top_book_to_market_quintile(market, view):
    uni = market.universe(AS_OF_IDX)
    scores = make_signal("v4_h2_piotroski_value", {})(view.as_of, uni, view).dropna()
    mcap = hyp.market_caps(view, uni)
    equity = pd.Series([hyp._val(view.fundamentals(t, "equity", "instant")) for t in uni], index=uni)
    bm = (equity / mcap).where((equity > 0) & (mcap > 0)).dropna()
    assert set(scores.index) <= set(bm[bm >= bm.quantile(v4.VALUE_QUANTILE)].index)
    for t in scores.index:  # F dominates; book-to-market only breaks ties
        assert np.floor(scores[t]) == v4.f_score(view, t)


def test_trend_template_rejects_each_failed_rule():
    days = 300
    up = pd.DataFrame({"UP": np.linspace(50, 100, days)})
    assert v4.trend_template_pass(up)["UP"]
    down = pd.DataFrame({"DN": np.linspace(100, 50, days)})
    assert not v4.trend_template_pass(down)["DN"]
    crashed = up.copy()
    crashed.iloc[-1] = 60.0  # below every average and more than 25% under the high
    assert not v4.trend_template_pass(crashed)["UP"]
    short = pd.DataFrame({"NEW": np.linspace(50, 100, days)})
    short.iloc[: days - 200] = np.nan  # not enough history for the 252-session window
    assert not v4.trend_template_pass(short)["NEW"]


def test_trend_template_needs_eps_growth_of_25_percent(market, view):
    uni = market.universe(AS_OF_IDX)
    scores = make_signal("v4_h3_trend_template", {})(view.as_of, uni, view).dropna()
    for t in scores.index:
        assert hyp.ttm_growth(view, t, "eps_diluted") >= v4.EPS_GROWTH_MIN


@pytest.mark.parametrize("name", V4)
def test_v4_signals_pass_the_lookahead_guard(market, name):
    d = market.sessions[AS_OF_IDX]
    assert_no_lookahead(make_signal(name, {}), market, d, market.universe(AS_OF_IDX))

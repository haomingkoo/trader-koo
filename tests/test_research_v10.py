"""v10 exposure ladder: the market state machine, the drawdown ladder, pacing, execution timing and the gates.

These test the software's rules and arithmetic on hand-built arrays; the market results come from the research database.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader_koo.research.bench import v10_rules as r
from trader_koo.research.bench import v10_run as run


def _flat(n, ema21=100.0, sma50=100.0, ema10=100.0):
    """Indicators that hold still, so only the price path drives the state machine."""
    return {"ema10": np.full(n, ema10), "ema21": np.full(n, ema21), "sma50": np.full(n, sma50),
            "lt": np.zeros(n, bool)}


def _states(c, lo, v, ind, warm=0):
    return r.market_states(np.asarray(c, float), np.asarray(lo, float), np.asarray(v, float), ind["ema10"],
                           ind["ema21"], ind["sma50"], ind["lt"], warm)


def _rally(day4_gain=0.02, day4_vol_up=True, undercut_on=None):
    """Correction start, then a low at session 1, day 1 at session 2, and a candidate FTD on day 4 (session 5)."""
    c = [100.0, 95.0, 96.0, 96.5, 97.0]
    lo = [99.0, 94.0, 95.5, 96.0, 96.5]
    v = [1e6, 1e6, 1e6, 1e6, 1e6]
    if undercut_on is not None:
        lo[undercut_on] = 93.0
    c.append(c[-1] * (1 + day4_gain))
    lo.append(c[-2])
    v.append(2e6 if day4_vol_up else 5e5)
    return c, lo, v


def test_distribution_day_is_a_drop_of_two_tenths_on_higher_volume():
    c = np.array([100.0, 99.8, 99.7, 99.5])
    v = np.array([1.0, 2.0, 3.0, 2.0])
    got = r.distribution_days(c, v)
    assert got.tolist() == [False, True, False, False]   # -0.2% on more volume; -0.1%; -0.2% on less volume


def test_follow_through_day_needs_day_four_one_point_seven_percent_and_higher_volume():
    c, lo, v = _rally()
    target, ftd = _states(c, lo, v, _flat(len(c), ema21=200.0, sma50=200.0, ema10=200.0))
    assert ftd.tolist() == [False] * 5 + [True]
    assert target[-1] == pytest.approx(0.25)              # FTD step only: price is under every average
    for kw in ({"day4_gain": 0.016}, {"day4_vol_up": False}):
        c, lo, v = _rally(**kw)
        _, ftd = _states(c, lo, v, _flat(len(c)))
        assert not ftd.any()


def test_follow_through_on_day_three_does_not_count():
    c = [100.0, 95.0, 96.0, 96.5, 98.5]
    lo = [99.0, 94.0, 95.5, 96.0, 96.4]
    v = [1e6, 1e6, 1e6, 1e6, 2e6]
    _, ftd = _states(c, lo, v, _flat(len(c)))
    assert not ftd.any()


def test_an_undercut_restarts_the_day_count():
    c, lo, v = _rally(undercut_on=3)     # session 3 undercuts and closes up: it is the new day 1
    _, ftd = _states(c, lo, v, _flat(len(c)))
    assert not ftd.any()                 # session 5 is only day 3 now


def _uptrend_then(extra_c, extra_lo, extra_v, ind_n=None):
    c, lo, v = _rally()
    c, lo, v = c + extra_c, lo + extra_lo, v + extra_v
    return c, lo, v, _flat(ind_n or len(c), ema21=90.0, sma50=90.0, ema10=95.0)


def test_close_below_the_follow_through_low_ends_the_uptrend():
    ftd_low = 97.0
    c, lo, v, ind = _uptrend_then([99.5, 99.0, 99.2, 99.4, ftd_low - 0.5], [99.0] * 4 + [96.0], [1e6] * 5)
    target, _ = _states(c, lo, v, ind)
    assert target[6] == pytest.approx(1.0)  # every step on while price is above all the averages
    assert target[-1] == 0.0


def test_distribution_within_three_sessions_of_the_ftd_fails_it_and_a_later_ftd_restarts():
    c, lo, v = _rally()
    c += [c[-1] * 0.997, c[-1] * 0.997 * 1.02]   # day 1 after the FTD: distribution; then a second FTD
    lo += [c[-3], c[-2]]
    v += [3e6, 4e6]
    target, ftd = _states(c, lo, v, _flat(len(c)))
    assert target[6] == 0.0
    assert ftd[7] and target[7] > 0.0


def test_ema21_crossing_below_sma50_ends_the_uptrend():
    c, lo, v = _rally()
    c, lo, v = c + [99.5, 99.6], lo + [99.0, 99.1], v + [1e6, 1e6]
    ind = _flat(len(c), ema21=95.0, sma50=90.0)
    ind["ema21"][-1] = 89.0
    target, _ = _states(c, lo, v, ind)
    assert target[-2] > 0.0 and target[-1] == 0.0


def test_long_term_downtrend_caps_at_one_step_and_its_start_ends_the_uptrend():
    c, lo, v = _rally()
    c, lo, v = c + [99.5, 99.6], lo + [99.0, 99.1], v + [1e6, 1e6]
    ind = _flat(len(c), ema21=90.0, sma50=90.0, ema10=95.0)
    ind["lt"][-1] = True
    target, _ = _states(c, lo, v, ind)
    assert target[-2] == pytest.approx(1.0) and target[-1] == 0.0
    ind["lt"][:] = True                       # already in a long-term downtrend when the FTD comes
    target, _ = _states(c, lo, v, ind)
    assert target[-1] == pytest.approx(0.25)


def test_weekly_flag_needs_two_consecutive_weekly_closes_below_the_30_week_average():
    dates = [d.date().isoformat() for d in pd.bdate_range("2020-01-06", periods=5 * 34)]
    c = np.full(len(dates), 100.0)
    c[-10:-5] = 90.0                         # week 33 closes below
    flag = r.long_term_downtrend(dates, c)
    assert not flag.any()                    # one weekly close below is not enough
    c[-5:] = 90.0                            # week 34 closes below too
    flag = r.long_term_downtrend(dates + ["2020-08-31"], np.append(c, 90.0))
    assert flag[len(dates) - 1] and not flag[len(dates) - 6]


def test_pace_up_is_one_step_a_session_and_down_is_immediate():
    n = 8
    target = np.array([0, 1, 1, 1, 1, 0.25, 1, 1], float)
    book = r.run_book(target, np.zeros(n, bool), np.zeros(n), np.zeros(n), np.zeros(n), 0.0, drawdown=False)
    assert book.exposure.tolist() == [0, 0.25, 0.5, 0.75, 1.0, 0.25, 0.5, 0.75]


def test_drawdown_ladder_halves_quarters_and_exits_and_resets_on_a_new_ftd():
    n = 8
    target = np.ones(n)
    ftd = np.zeros(n, bool)
    ftd[0] = ftd[6] = True
    R = np.array([0, -0.051, -0.0204, -0.07, 0, 0, 0, 0])   # the book falls through 5%, 7%, then 10%
    flat = r.run_book(target, ftd, R, np.zeros(n), np.zeros(n), 0.0, drawdown=False, pace=False)
    assert flat.exposure.tolist() == [1.0] * n
    held = r.run_book(target, ftd, R, np.zeros(n), np.zeros(n), 0.0, pace=False)
    assert held.exposure[:6].tolist() == [1.0, 1.0, 0.5, 0.25, 0.0, 0.0]
    assert held.exposure[6] == 1.0            # the new FTD resets the peak


def test_drawdown_is_marked_at_the_close_with_the_held_exposure():
    n = 4
    intra = np.array([0, 0, -0.06, 0])
    held = r.run_book(np.ones(n), np.array([True, False, False, False]), np.zeros(n), intra, np.zeros(n), 0.0,
                      pace=False)
    assert held.exposure[2] == 0.5            # the -6% session is seen at its own close


def test_exposure_decided_at_close_t_earns_the_interval_from_the_next_open():
    n = 6
    target = np.zeros(n)
    target[2:] = 1.0
    R = np.full(n, 0.01)
    cash = np.full(n, 0.001)
    book = r.run_book(target, np.zeros(n, bool), R, np.zeros(n), cash, 10.0, drawdown=False, pace=False)
    assert book.returns[2] == pytest.approx(0.001)                 # decided at close 2, still cash open 2 to 3
    assert book.returns[3] == pytest.approx(0.01 - 10 / 1e4)       # held from the open of 3, cost charged once
    assert book.returns[4] == pytest.approx(0.01)


def test_hold_returns_match_the_book_and_shifts_keep_mean_exposure():
    rng = np.random.default_rng(0)
    n = 600
    R = rng.normal(0, 0.01, n)
    cash = np.zeros(n)
    hold = np.repeat([0.0, 1.0, 0.5], n // 3)
    got = r.hold_returns(hold, R, cash, 10.0)
    assert got[n // 3] == pytest.approx(R[n // 3] - 10 / 1e4)
    out = run.shift_test(hold, R, cash, 10.0, seed=1, draws=50)
    assert 0 < out["p"] <= 1 and np.allclose(out["means"], hold.mean())
    assert len(out["dd"]) == 50


def test_metrics_and_constant_mix():
    r_ = np.array([0.01, -0.02, 0.01, 0.0])
    m = run.metrics(r_, np.zeros(4))
    assert m["max_drawdown"] == pytest.approx(-0.02)
    assert m["calmar"] == pytest.approx(m["cagr"] / abs(m["max_drawdown"]))
    mix = run.constant_mix(0.4, np.array([0.01, 0.02]), np.array([0.001, 0.001]))
    assert mix.tolist() == pytest.approx([0.4 * 0.01 + 0.6 * 0.001, 0.4 * 0.02 + 0.6 * 0.001])


def test_year_check_floors_an_all_cash_year():
    dates = [d.date().isoformat() for d in pd.bdate_range("2015-01-01", "2016-12-31")]
    n = len(dates)
    rng = np.random.default_rng(3)
    mix = rng.normal(0.0004, 0.006, n)
    rule = np.where(np.array([d < "2016" for d in dates]), 0.0001, mix * 1.1)   # cash in 2015
    rows = run.year_rows(dates, rule, mix)
    assert [row["year"] for row in rows] == [2015, 2016]
    assert all(np.isfinite(row["rule_rv"]) for row in rows)


def test_holm_and_gate_assembly():
    g = run.gates({"max_drawdown": -0.10, "rv": 1.0, "calmar": 1.0}, {"max_drawdown": -0.30},
                  {"rv": 0.9, "calmar": 0.8}, adj_p=0.01, year_share=0.7,
                  stress={"max_drawdown": -0.11, "rv": 0.95, "calmar": 0.9}, crash=(-0.05, -0.30))
    assert all(g.values())
    g = run.gates({"max_drawdown": -0.25, "rv": 1.0, "calmar": 1.0}, {"max_drawdown": -0.30},
                  {"rv": 0.9, "calmar": 0.8}, adj_p=0.01, year_share=0.7,
                  stress={"max_drawdown": -0.11, "rv": 0.95, "calmar": 0.9}, crash=(-0.25, -0.30))
    assert not g["a drawdown 30% shallower"] and not g["f March 2020 crash"]

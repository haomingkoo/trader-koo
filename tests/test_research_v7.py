"""v7 exposure rules: execution timing, the first-known rule for macro data, and rule arithmetic.

These test the software's timing and arithmetic; the market results come from the research database.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from trader_koo.research import macro
from trader_koo.research.bench import v7_rules as r
from trader_koo.research.bench import v7_run as run


def _inputs(n=300, conn=None):
    dates = [d.date().isoformat() for d in pd.bdate_range("2020-01-01", periods=n)]
    close = np.linspace(100, 130, n)
    return r.Inputs(dates, close, close.copy(), close.copy(), np.full(n, 1e6), np.full(n, 15.0),
                    np.zeros(n), conn or sqlite3.connect(":memory:"))


def test_exposure_decided_at_close_t_applies_from_the_next_open():
    inp = _inputs(10)
    inp.spy_open = inp.spy_close * 1.01   # every overnight gap is +1%, every intraday move about -1%
    e = np.zeros(10)
    e[4] = 1.0                            # decided at the close of session 4
    ret = run.returns(inp, e, cost_bps=0.0)
    assert ret[4] == 0.0                  # no effect on the day it is decided
    intraday = inp.spy_close[5] / inp.spy_open[5] - 1
    assert ret[5] == pytest.approx(intraday)       # in the market from the open of session 5
    overnight = inp.spy_open[6] / inp.spy_close[5] - 1
    assert ret[6] == pytest.approx(overnight)      # the overnight into 6 is still held, intraday 6 is cash


def test_cost_is_charged_on_each_change_of_exposure():
    inp = _inputs(10)
    e = np.ones(10)
    e[5:] = 0.5
    ret = run.returns(inp, e, cost_bps=10.0)
    base = run.returns(inp, e, cost_bps=0.0)
    assert ret[6] - base[6] == pytest.approx(-0.5 * 10 / 1e4)
    assert np.count_nonzero(ret - base) == 1


def test_macro_value_is_used_only_from_the_day_it_was_published():
    conn = sqlite3.connect(":memory:")
    macro.ensure_schema(conn)
    macro.store(conn, "DTB3", [("DTB3", "2020-01-01", 1.5, "2020-01-03", "9999-12-31", "fred_latest"),
                               ("DTB3", "2020-01-02", 1.6, "2020-01-06", "9999-12-31", "fred_latest")])
    got = r._known_daily(conn, "DTB3", ["2020-01-02", "2020-01-03", "2020-01-06"])
    assert np.isnan(got[0]) and got[1] == 1.5 and got[2] == 1.6


def test_distribution_days_need_five_in_twenty_five_sessions():
    inp = _inputs(40)
    close = np.full(40, 100.0)
    vol = np.full(40, 1e6)
    for i in (30, 32, 34, 36):             # four down 0.3% days on rising volume
        close[i:] *= 0.997
        vol[i] = vol[i - 1] + 1
    inp.spy_raw_close, inp.spy_volume = close, vol
    assert r.h4_distribution_days(inp)[-1] == 1.0
    close[38:] *= 0.997
    vol[38] = vol[37] + 1
    assert r.h4_distribution_days(inp)[-1] == 0.0


def test_monthly_rule_holds_its_decision_until_the_next_month_end():
    inp = _inputs(300)
    e = r.h5_ten_month_trend(inp)
    ends = np.flatnonzero(r._month_ends(inp.dates))
    first = ends[r.FABER_MONTHS - 1]
    assert np.isnan(e[:first]).all() and (e[first:] == 1.0).all()  # a steady rise stays in the market


def test_vote_counts_a_missing_rule_as_in_the_market():
    raw = {k: np.zeros(3) for k in r.RULES}
    raw["H1"] = np.array([np.nan, 0.0, 1.0])
    assert r.h11_vote(raw).tolist() == pytest.approx([0.1, 0.0, 0.1])

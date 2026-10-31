"""Hypotheses v9 (docs/research/hypotheses-v9-champions.md): signals, daily-bar fills and exits, the risk-sized
portfolio, trade statistics, the sleeve blend, the agreement rule, Code 33 and the sealed window.

Small hand-built arrays test arithmetic and timing only. The integration tests use real daily bars of eight
S&P 500 stocks and SPY (2017-01-03 to 2019-06-28) and their real yfinance earnings rows.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tests.test_research_v8 import _db_with_earnings
from trader_koo.research.bench import ledger, v9_book, v9_fund, v9_run, v9_signals, v9_trades
from trader_koo.research.bench.v9_trades import Exits

NAN = np.nan
N = 80


def _flat(n=N, t=1, px=100.0):
    o = np.full((n, t), px)
    return o.copy(), o + 1.0, o - 1.0, o.copy(), np.full((n, t), 1e6)


def _bars(o, h, lo, c, v=None, reaction=None, spin=None):
    v = np.full(o.shape, 1e6) if v is None else v
    return v9_trades.make_bars(o, h, lo, c, v, reaction=reaction, spin=spin)


def _sim(bars, ex, e, stop, entry=None, col=0):
    entry = bars.o[e, col] if entry is None else entry
    return v9_trades.simulate(bars, ex, np.array([e]), np.array([col]), np.array([entry]), np.array([stop]))


H1 = Exits(max_hold=120, partial_day=5, trail=10, earnings_exit=True)
H4 = Exits(max_hold=5, fixed_close_day=5)


# ---- fills and stops on daily bars ------------------------------------------------------------------------------------

def test_a_gap_through_the_stop_fills_at_the_open():
    o, h, lo, c, _ = _flat()
    o[42, 0], h[42, 0], lo[42, 0], c[42, 0] = 90.0, 91.0, 88.0, 89.0
    g = _sim(_bars(o, h, lo, c), H4, 40, stop=95.0)
    assert g.n2[0] == 42 and g.p2[0] == 90.0 and g.when2[0] == v9_trades.AT_OPEN
    assert g.kind[0] == v9_trades.STOP_GAP


def test_a_stop_touched_inside_the_bar_fills_at_the_stop():
    o, h, lo, c, _ = _flat()
    lo[41, 0] = 94.0
    g = _sim(_bars(o, h, lo, c), H4, 40, stop=95.0)
    assert g.n2[0] == 41 and g.p2[0] == 95.0 and g.when2[0] == v9_trades.INTRADAY and g.kind[0] == v9_trades.STOP


def test_the_entry_session_low_counts_against_the_stop():
    o, h, lo, c, _ = _flat()
    lo[40, 0] = 98.5
    g = _sim(_bars(o, h, lo, c), H4, 40, stop=99.0)
    assert g.n2[0] == 40 and g.p2[0] == 99.0


def test_stockbee_burst_exits_at_the_close_of_the_fifth_session():
    o, h, lo, c, _ = _flat()
    c[44, 0] = 107.0
    g = _sim(_bars(o, h, lo, c), H4, 40, stop=90.0)
    assert g.n2[0] == 44 and g.p2[0] == 107.0 and g.when2[0] == v9_trades.AT_CLOSE and g.f1[0] == 0.0


def test_h1_sells_half_on_day_five_then_the_rest_sits_on_a_breakeven_stop():
    o = np.linspace(100, 140, N)[:, None]
    h, lo, c = o + 1.0, o - 1.0, o + 0.5
    e = 40
    entry = o[e, 0]
    lo[46, 0] = entry - 0.1          # day 7: the low breaks the entry; the stop is now the entry
    g = _sim(_bars(o, h, lo, c), H1, e, stop=entry - 10.0)
    assert g.n1[0] == e + 4 and g.f1[0] == 0.5 and g.p1[0] == pytest.approx(c[e + 4, 0])
    assert g.n2[0] == 46 and g.p2[0] == pytest.approx(entry) and g.kind[0] == v9_trades.STOP


def test_h1_trail_exits_at_the_next_open_once_the_average_is_above_the_stop():
    o = np.linspace(100, 140, N)[:, None]
    h, lo, c = o + 1.0, o - 1.0, o + 0.5
    c[50, 0] = 110.0                 # far below the 10-session average, which is above the stop
    g = _sim(_bars(o, h, lo, c), H1, 40, stop=95.0)
    assert g.n2[0] == 51 and g.p2[0] == pytest.approx(o[51, 0]) and g.when2[0] == v9_trades.AT_OPEN
    assert g.kind[0] == v9_trades.NEXT_OPEN


def test_trail_on_a_flat_stock_fires_on_the_first_close_under_its_average():
    o, h, lo, c, _ = _flat()
    lo[:] = 99.4
    c[41, 0] = 99.5                  # under its 10-session average (99.95), which is above the stop
    g = _sim(_bars(o, h, lo, c), Exits(max_hold=5, trail=10), 40, stop=99.0, entry=100.0)
    assert g.n2[0] == 42 and g.p2[0] == pytest.approx(o[42, 0])


def test_maximum_hold_exits_at_the_open_of_entry_plus_hold():
    o, h, lo, c, _ = _flat()
    g = _sim(_bars(o, h, lo, c), Exits(max_hold=10), 40, stop=90.0)
    assert g.n2[0] == 50 and g.when2[0] == v9_trades.AT_OPEN


def test_no_holding_through_earnings_exits_at_the_close_before_the_reaction_session():
    o, h, lo, c, _ = _flat()
    reaction = np.zeros(o.shape, bool)
    reaction[45, 0] = True
    g = _sim(_bars(o, h, lo, c, reaction=reaction), Exits(max_hold=20, earnings_exit=True), 40, stop=90.0)
    assert g.n2[0] == 44 and g.when2[0] == v9_trades.AT_CLOSE and g.kind[0] == v9_trades.EARNINGS


def test_a_listed_spinoff_exits_at_the_close_before_the_ex_date():
    o, h, lo, c, _ = _flat()
    spin = np.zeros(o.shape, bool)
    spin[43, 0] = True
    g = _sim(_bars(o, h, lo, c, spin=spin), Exits(max_hold=20), 40, stop=90.0)
    assert g.n2[0] == 43 and g.when2[0] == v9_trades.AT_CLOSE and g.kind[0] == v9_trades.SPINOFF


def test_an_open_trade_on_the_last_session_closes_at_that_close():
    o, h, lo, c, _ = _flat()
    g = _sim(_bars(o, h, lo, c), Exits(max_hold=120), N - 3, stop=90.0)
    assert g.n2[0] == N - 1 and g.when2[0] == v9_trades.AT_CLOSE and g.kind[0] == v9_trades.WINDOW_END


def test_j_law_sells_58_8_percent_at_plus_70_and_a_gap_above_fills_at_the_open():
    o, h, lo, c, _ = _flat()
    o[42:, 0], h[42:, 0], lo[42:, 0], c[42:, 0] = 180.0, 181.0, 179.0, 180.0
    ex = Exits(max_hold=6, target=1.70, target_frac=0.588)
    g = _sim(_bars(o, h, lo, c), ex, 40, stop=95.0)
    assert g.n1[0] == 42 and g.f1[0] == pytest.approx(0.588) and g.p1[0] == 180.0
    assert g.when1[0] == v9_trades.AT_OPEN and g.n2[0] == 46


def test_band_exit_needs_a_close_two_percent_below_the_lower_average():
    o, h, lo, c, _ = _flat()
    c[45, 0] = 98.5                  # less than 2% under: no exit
    c[47, 0] = 97.0                  # more than 2% under the lower of SMA10 and SMA20
    g = _sim(_bars(o, h, lo, c), Exits(max_hold=20, band=True), 40, stop=90.0)
    assert g.n2[0] == 48


# ---- Sean Ryan: buy-stop, day one, breakeven, divergence -----------------------------------------------------------------

def test_buy_stop_fills_at_the_trigger_or_a_higher_open_and_is_cancelled_after_five_sessions():
    o, h, lo, c, _ = _flat()
    h[43, 0] = 102.0
    bars = _bars(o, h, lo, c)
    row, px = v9_trades.buy_stop(bars, np.array([40]), np.array([0]), np.array([101.5]), window=5)
    assert row[0] == 43 and px[0] == 101.5
    o[42, 0], h[42, 0] = 103.0, 104.0
    row, px = v9_trades.buy_stop(_bars(o, h, lo, c), np.array([40]), np.array([0]), np.array([101.5]), window=5)
    assert row[0] == 42 and px[0] == 103.0
    o2, h2, lo2, c2, _ = _flat()
    h2[46, 0] = 110.0                # sixth session after the setup: too late
    row, _ = v9_trades.buy_stop(_bars(o2, h2, lo2, c2), np.array([40]), np.array([0]), np.array([101.5]), window=5)
    assert row[0] == -1


def test_day_one_red_close_exits_at_that_close():
    o, h, lo, c, _ = _flat()
    c[40, 0] = 100.5
    ex = Exits(max_hold=60, day_one_red=True)
    g = _sim(_bars(o, h, lo, c), ex, 40, stop=95.0, entry=101.0)
    assert g.n2[0] == 40 and g.p2[0] == 100.5 and g.kind[0] == v9_trades.DAY_ONE


def test_breakeven_stop_from_the_third_session():
    o, h, lo, c, _ = _flat()
    c[40, 0] = 101.0
    lo[42, 0] = 99.0                 # third session: the stop is now the entry, 100
    ex = Exits(max_hold=60, day_one_red=True, breakeven_day=3, breakeven_gain=1.10)
    g = _sim(_bars(o, h, lo, c), ex, 40, stop=95.0, entry=100.0)
    assert g.n2[0] == 42 and g.p2[0] == 100.0
    g = _sim(_bars(o, h, lo, c), Exits(max_hold=60), 40, stop=95.0, entry=100.0)
    assert g.n2[0] == N - 1                        # without the rule the stop stays at 95: open at the window end


def _divergence_by_loop(c, rsi):
    n = len(c)
    high = [s >= 19 and np.isfinite(c[s - 19:s + 1]).all() and c[s] > c[s - 19:s].max() for s in range(n)]
    out = np.zeros(n, bool)
    for s in range(n):
        prior = [p for p in range(max(0, s - 60), s - 4) if high[p]]
        out[s] = bool(high[s] and prior and rsi[s] < rsi[prior[-1]])
    return out


def test_rsi_divergence_matches_a_plain_loop_over_the_registered_definition():
    up = np.concatenate([np.linspace(100, 130, 30), np.linspace(130, 120, 10), np.linspace(120, 131, 25),
                         np.linspace(131, 125, 8), np.linspace(125, 133, 30)])
    c = (up + np.sin(np.arange(len(up))) * 0.8)[:, None]
    div = v9_trades.rsi_divergence(c)
    rsi = v9_trades.wilder_rsi(c)
    want = _divergence_by_loop(c[:, 0], rsi[:, 0])
    assert want.any() and (div[:, 0] == want).all()


def test_wilder_rsi_of_a_steady_rise_is_100_and_of_alternating_equal_moves_is_near_50():
    assert v9_trades.wilder_rsi(np.linspace(1, 50, 50)[:, None])[-1, 0] == pytest.approx(100.0)
    alt = (100 + np.tile([0.0, 1.0], 100))[:, None]
    assert v9_trades.wilder_rsi(alt)[-1, 0] == pytest.approx(50.0, abs=3.0)


def test_volume_reversal_needs_volume_a_new_high_and_a_down_close():
    o, h, lo, c, v = _flat()
    h[60, 0], o[60, 0], c[60, 0], v[60, 0] = 105.0, 104.0, 99.0, 3e6
    assert v9_trades.volume_reversal(o, h, c, v)[60, 0]
    v[60, 0] = 1.5e6
    assert not v9_trades.volume_reversal(o, h, c, v)[60, 0]


# ---- signals ----------------------------------------------------------------------------------------------------------------

def _gap_event(n=300):
    o, h, lo, c, v = _flat(n)
    c[:] = np.linspace(50, 100, n)[:, None]
    o[:], h[:], lo[:] = c - 0.5, c + 1.0, c - 1.0
    g = n - 10
    o[g, 0], h[g, 0], lo[g, 0], c[g, 0], v[g, 0] = c[g - 1, 0] * 1.12, c[g - 1, 0] * 1.2, c[g - 1, 0] * 1.1, c[g - 1, 0] * 1.15, 4e6
    return o, h, lo, c, v, g


def test_gap_rule_needs_ten_percent_and_three_times_the_prior_fifty_session_volume():
    o, h, lo, c, v, g = _gap_event()
    rows, cols = np.array([g]), np.array([0])
    assert v9_signals.gap_volume(o, c, v, rows, cols).tolist() == [True]
    v[g, 0] = 2.9e6
    assert v9_signals.gap_volume(o, c, v, rows, cols).tolist() == [False]
    v[g, 0], o[g, 0] = 4e6, c[g - 1, 0] * 1.09
    assert v9_signals.gap_volume(o, c, v, rows, cols).tolist() == [False]


def test_new_closing_high_compares_with_the_251_closes_before_and_needs_all_252():
    _, _, _, c, _, g = _gap_event()
    assert v9_signals.new_high(c, np.array([g]), np.array([0]), window=252).tolist() == [True]
    c2 = c.copy()
    c2[g - 100, 0] = 1e4
    assert v9_signals.new_high(c2, np.array([g]), np.array([0]), window=252).tolist() == [False]
    c2 = c.copy()
    c2[g - 251, 0] = NAN
    assert v9_signals.new_high(c2, np.array([g]), np.array([0]), window=252).tolist() == [False]
    c2[g - 251, 0] = 1.0
    c2[g - 252, 0] = 1e4                                  # 252 sessions before: outside the window
    assert v9_signals.new_high(c2, np.array([g]), np.array([0]), window=252).tolist() == [True]


def test_quiet_period_caps_the_63_session_run_up_at_30_percent():
    c = np.full((100, 1), 100.0)
    c[98, 0] = 129.9
    assert v9_signals.quiet(c, np.array([99]), np.array([0])).tolist() == [True]
    c[98, 0] = 130.5
    assert v9_signals.quiet(c, np.array([99]), np.array([0])).tolist() == [False]


def test_rs_percentile_ranks_members_only():
    n = 300
    c = np.ones((n, 4)) * 100
    c[:, 0] *= np.linspace(1, 3, n)
    c[:, 1] *= np.linspace(1, 2, n)
    c[:, 2] *= np.linspace(1, 0.5, n)
    c[:, 3] *= np.linspace(1, 9, n)
    member = np.ones((n, 4), bool)
    member[:, 3] = False
    pct = v9_signals.rs_percentile(c, member)
    assert pct[-1, 0] == pytest.approx(100.0) and pct[-1, 2] < pct[-1, 1] < pct[-1, 0] and np.isnan(pct[-1, 3])


def test_stockbee_burst_rules():
    o, h, lo, c, v = _flat(30)
    c[:, 0] = 100.0
    c[20, 0], v[20, 0] = 104.0, 2e6
    c[19, 0] = 99.5                                       # a down day before: rule 4
    assert v9_signals.burst(h, lo, c, v)[20, 0]
    v[20, 0] = 0.9e6
    assert not v9_signals.burst(h, lo, c, v)[20, 0]       # volume not above the day before
    v[20, 0] = 2e6
    c[16:20, 0] = [100.0, 101.0, 102.0, 103.0]            # up three days in a row into t
    assert not v9_signals.burst(h, lo, c, v)[20, 0]
    c[16:21, 0] = [100.0, 100.0, 109.0, 108.5, 108.5 * 1.05]  # a down day before, but up 8.5% from t-4
    assert not v9_signals.burst(h, lo, c, v)[20, 0]
    c[:, 0] = 100.0
    c[19, 0], c[20, 0] = 99.5, 99.5 * 1.039                 # only rule 1 fails
    assert not v9_signals.burst(h, lo, c, v)[20, 0]
    c[20, 0] = 99.5 * 1.041
    assert v9_signals.burst(h, lo, c, v)[20, 0]


def test_second_move_setup_base_burst_and_rest():
    n = 120
    c = np.full((n, 1), 100.0)
    c[60:75, 0] = np.linspace(102, 140, 15)               # burst over t-20..t-6 for t = 80
    c[75:81, 0] = 138.0                                   # rest
    o, h, lo = c.copy(), c * 1.01, c * 0.99
    v = np.full((n, 1), 1e6)
    v[60:75, 0] = 2e6
    t = 80
    assert v9_signals.second_move(h, lo, c, v)[t, 0]
    v[60:75, 0] = 1.2e6
    assert not v9_signals.second_move(h, lo, c, v)[t, 0]  # burst volume below 1.5x the base
    v[60:75, 0] = 2e6
    c[76, 0] = 115.0                                      # did not hold 85% of the burst high
    assert not v9_signals.second_move(c * 1.01, c * 0.99, c, v)[t, 0]


def test_pullback_needs_a_spy_down_day_with_the_stock_holding_up():
    n = 120
    c = np.full((n, 2), 100.0)
    e = 60
    c[e:, 0] = 130.0
    c[e + 1:e + 4, 0] = [133.0, 136.0, 138.0]             # new highs after the catalyst
    c[e + 4:e + 9, 0] = [136.0, 134.0, 132.0, 131.0, 130.0]
    c[e + 9:, 0] = 130.0
    spy = np.full(n, 400.0)
    spy[e + 5] = 396.0                                    # SPY down 1%; the stock fell 1.5%: fails
    o, h, lo = c.copy(), c + 0.5, c - 0.5
    v = np.full((n, 2), 1e6)
    v[e:e + 4] = 5e6
    got = v9_signals.pullbacks(o, h, lo, c, v, spy, np.array([e]), np.array([0]))
    assert got[0][0] == -1
    spy[e + 5] = 390.0                                    # SPY down 2.5%: the stock held up better
    got = v9_signals.pullbacks(o, h, lo, c, v, spy, np.array([e]), np.array([0]))
    t = got[0][0]
    assert e + 5 <= t <= e + 30 and got[1][0] == pytest.approx(lo[e + 4:t + 1, 0].min() * 0.99)


# ---- agreement (H7) -----------------------------------------------------------------------------------------------------

def test_agreement_needs_two_families_within_ten_sessions_and_the_later_one_triggers():
    fam = {"F1": [(10, 0)], "F2": [(19, 0), (30, 1)], "F3": [(29, 0)], "F4": [(30, 1)]}
    got = v9_signals.agreement(fam, window=10)
    assert (19, 0, "F2") in got                           # F1 at 10 is 9 sessions before
    assert (29, 0, "F3") not in got                       # F2 at 19 is ten sessions before: outside 20..29
    assert (30, 1, "F2") in got and (30, 1, "F4") not in got  # same session: F2 before F4
    assert all(r != 10 for r, _, _ in got)


# ---- portfolio ----------------------------------------------------------------------------------------------------------

def _cands(rows, cols, entry, stop, n1, p1, f1, n2, p2, when2=None, when1=None, prio=None):
    k = len(rows)
    a = lambda x: np.asarray(x, dtype=float)
    return v9_book.Cands(sig=np.asarray(rows) - 1, fill=np.asarray(rows), col=np.asarray(cols), entry=a(entry), stop=a(stop),
                         n1=np.asarray(n1), when1=np.asarray(when1 if when1 is not None else [v9_trades.AT_CLOSE] * k),
                         p1=a(p1), f1=a(f1), n2=np.asarray(n2),
                         when2=np.asarray(when2 if when2 is not None else [v9_trades.AT_CLOSE] * k), p2=a(p2),
                         kind=np.zeros(k, int), prio=a(prio if prio is not None else np.ones(k)))


def test_risk_sizing_caps_the_name_and_charges_both_sides():
    close = np.full((10, 2), 100.0)
    close[5:, 0] = 110.0
    c = _cands([2], [0], [100.0], [95.0], [-1], [NAN], [0.0], [5], [110.0])
    book = v9_book.run_book(c, close, first=1, last=9, sizing=v9_book.Sizing(risk=0.01, cap=0.30), cost_bps=10.0)
    fee = 10.0 / 1e4
    # 1% risk over a 5% stop is 20% of equity (under the 30% cap)
    assert book.trades["notional"].iloc[0] == pytest.approx(0.20)
    expect = 1.0 - 0.20 * fee + 0.20 * (1.1 * (1 - fee) - 1.0)
    assert book.equity[-1] == pytest.approx(expect)
    assert book.trades["r_net"].iloc[0] == pytest.approx((110 * (1 - fee) - 100 * (1 + fee)) / 5.0)
    capped = v9_book.run_book(c, close, first=1, last=9, sizing=v9_book.Sizing(risk=0.05, cap=0.30), cost_bps=0.0)
    assert capped.trades["notional"].iloc[0] == pytest.approx(0.30)


def test_gross_cap_one_position_per_stock_and_dollar_volume_priority():
    close = np.full((10, 3), 100.0)
    c = _cands([2, 2, 2, 3], [0, 1, 2, 0], [100.0] * 4, [99.0] * 4, [-1] * 4, [NAN] * 4, [0.0] * 4, [8] * 4,
               [100.0] * 4, prio=[1.0, 3.0, 2.0, 9.0])
    book = v9_book.run_book(c, close, first=1, last=9, sizing=v9_book.Sizing(risk=0.01, cap=0.6), cost_bps=0.0)
    t = book.trades.sort_values("fill")
    assert t["col"].tolist() == [1, 2]                     # col 1 first (60%), col 2 gets the 40% left, col 0 none
    assert t["notional"].tolist() == pytest.approx([0.6, 0.4])
    assert book.skipped["held"] == 0 and book.skipped["no_room"] == 2


def test_exits_at_the_open_free_room_for_that_sessions_entries():
    close = np.full((10, 2), 100.0)
    c = _cands([2, 4], [0, 1], [100.0, 100.0], [99.0, 99.0], [-1, -1], [NAN, NAN], [0.0, 0.0], [4, 8],
               [100.0, 100.0], when2=[v9_trades.AT_OPEN, v9_trades.AT_CLOSE])
    book = v9_book.run_book(c, close, first=1, last=9, sizing=v9_book.Sizing(risk=0.05, cap=1.0), cost_bps=0.0)
    assert len(book.trades) == 2


def test_flanders_tiers_and_the_monthly_minus_5r_brake():
    days = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10",
            "2020-01-13", "2020-02-03", "2020-02-04"]
    close = np.full((10, 9), 100.0)
    rows = [1, 2, 3, 4, 5, 6, 7, 8]            # each gaps to half its entry: far more than 1R lost
    c = _cands(rows, list(range(8)), [100.0] * 8, [90.0] * 8, [-1] * 8, [NAN] * 8, [0.0] * 8, rows, [50.0] * 8,
               when2=[v9_trades.INTRADAY] * 8)
    s = v9_book.Sizing(risk=0.005, cap=0.2, tiers=True)
    book = v9_book.run_book(c, close, first=1, last=9, sizing=s, cost_bps=0.0, dates=days)
    jan = book.trades[book.trades["fill"] <= 7]
    assert book.skipped["brake"] >= 1 and len(jan) < 7
    assert book.trades["risk_pct"].iloc[1] == pytest.approx(0.0025)  # month-to-date below 0: quarter size
    assert book.trades[book.trades["fill"] == 8]["risk_pct"].iloc[0] == pytest.approx(0.005)  # new month


def test_equal_weight_sizing_ignores_the_stop():
    close = np.full((10, 1), 100.0)
    c = _cands([2], [0], [100.0], [50.0], [-1], [NAN], [0.0], [6], [100.0])
    book = v9_book.run_book(c, close, first=1, last=9, sizing=v9_book.Sizing(equal_weight=0.10), cost_bps=0.0)
    assert book.trades["notional"].iloc[0] == pytest.approx(0.10)


def test_partial_and_final_legs_both_count_in_profit_and_r():
    close = np.full((10, 1), 100.0)
    c = _cands([2], [0], [100.0], [90.0], [4], [120.0], [0.5], [6], [100.0])
    book = v9_book.run_book(c, close, first=1, last=9, sizing=v9_book.Sizing(risk=0.01, cap=0.5), cost_bps=0.0)
    assert book.trades["r_net"].iloc[0] == pytest.approx(0.5 * 20 / 10)


def test_trade_stats_and_the_top_five_percent_share():
    r = np.array([5.0] + [-1.0] * 19)
    pnl = np.array([10.0] + [-0.5] * 19)
    s = v9_book.trade_stats(r, pnl, np.ones(20))
    assert s["n"] == 20 and s["win_rate"] == pytest.approx(0.05)
    assert s["mean_win_r"] == 5.0 and s["mean_loss_r"] == -1.0
    assert s["top5_share_of_net"] == pytest.approx(10.0 / 0.5)       # one trade of 20 is the top 5%
    assert s["top5_share_of_gross_profit"] == pytest.approx(1.0)


def test_blend_resets_to_equal_weights_on_the_first_session_of_each_month():
    dates = ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"]
    r = np.array([[0.10, 0.0], [0.10, 0.0], [0.0, 0.10], [0.0, 0.0]])
    got = v9_book.blend(r, dates)
    w1 = 0.5 * 1.1 / (0.5 * 1.1 + 0.5)
    assert got[0] == pytest.approx(0.05) and got[1] == pytest.approx(w1 * 0.10)
    assert got[2] == pytest.approx(0.05)                               # reset in February


# ---- Code 33 ------------------------------------------------------------------------------------------------------------

def _facts_db(path: Path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sec_cik_map (ticker TEXT, cik INTEGER, title TEXT, mapped_ts TEXT)")
    conn.execute("CREATE TABLE sec_fundamentals (cik INTEGER, taxonomy TEXT, tag TEXT, unit TEXT, period_start TEXT,"
                 " period_end TEXT, fiscal_year INTEGER, fiscal_period TEXT, value REAL, form TEXT, accession TEXT,"
                 " filed_date TEXT, frame TEXT)")
    conn.execute("INSERT INTO sec_cik_map VALUES ('ACME', 1, 'Acme', 't')")
    conn.executemany("INSERT INTO sec_fundamentals VALUES (1,'us-gaap',?,?,?,?,NULL,NULL,?,'10-Q',?,?,NULL)", rows)
    conn.commit()
    return conn


QUARTERS = [("2018-01-01", "2018-03-31"), ("2018-04-01", "2018-06-30"), ("2018-07-01", "2018-09-30"),
            ("2018-10-01", "2018-12-31"), ("2019-01-01", "2019-03-31"), ("2019-04-01", "2019-06-30"),
            ("2019-07-01", "2019-09-30"), ("2019-10-01", "2019-12-31")]


def _accelerating(q4_as_annual=False):
    eps = [1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.6, 2.0]       # YoY growth 10%, 30%, 60%, 100%
    rev = [10.0, 10.0, 10.0, 10.0, 11.0, 13.0, 16.0, 20.0]
    ni = [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 2.0, 2.8]          # margins rising
    rows = []
    for k, (s, e) in enumerate(QUARTERS):
        filed = f"{e[:4]}-{int(e[5:7]) % 12 + 1:02d}-15" if e[5:7] != "12" else f"{int(e[:4]) + 1}-02-15"
        for tag, unit, vals in (("EarningsPerShareDiluted", "USD/shares", eps), ("Revenues", "USD", rev),
                                ("NetIncomeLoss", "USD", ni)):
            if q4_as_annual and e.endswith("12-31"):
                y = e[:4]
                total = sum(vals[k - 3:k + 1])
                rows.append((tag, unit, f"{y}-01-01", e, total, f"a{k}{tag}", filed))
                nine = sum(vals[k - 3:k])
                rows.append((tag, unit, f"{y}-01-01", f"{y}-09-30", nine, f"n{k}{tag}", f"{y}-10-15"))
            else:
                rows.append((tag, unit, s, e, vals[k], f"q{k}{tag}", filed))
    return rows


def test_code_33_needs_three_straight_accelerations_in_eps_sales_and_margin(tmp_path):
    conn = _facts_db(tmp_path / "f.db", _accelerating())
    assert v9_fund.code33(conn, "ACME", "2020-03-01")
    assert not v9_fund.code33(conn, "ACME", "2020-02-14")   # Q4 2019 not yet filed: only seven quarters known


def test_code_33_derives_q4_from_the_year_minus_nine_months(tmp_path):
    conn = _facts_db(tmp_path / "f.db", _accelerating(q4_as_annual=True))
    q = v9_fund.quarters(conn, "ACME", "EarningsPerShareDiluted", "2020-03-01")
    assert q["2019-12-31"] == pytest.approx(2.0) and q["2018-12-31"] == pytest.approx(1.0)
    assert v9_fund.code33(conn, "ACME", "2020-03-01")


def test_code_33_is_no_when_growth_stalls(tmp_path):
    rows = [r if not (r[0] == "Revenues" and r[3] == "2019-12-31") else (*r[:4], 15.0, *r[5:]) for r in _accelerating()]
    assert not v9_fund.code33(_facts_db(tmp_path / "f.db", rows), "ACME", "2020-03-01")


def test_code_33_is_no_when_the_latest_quarter_has_no_revenue(tmp_path):
    rows = _accelerating()
    later = [("EarningsPerShareDiluted", "USD/shares", "2020-01-01", "2020-03-31", 2.5, "q8eps", "2020-04-15")]
    assert not v9_fund.code33(_facts_db(tmp_path / "f.db", rows + later), "ACME", "2020-05-01")


# ---- integration: real bars, real earnings rows, the sealed window --------------------------------------------------------

def _run(tmp_path: Path, name: str, scale_after: str | None = None) -> dict:
    db = _db_with_earnings(tmp_path / f"{name}.db", scale_after)
    conn = sqlite3.connect(db)
    _facts_db_schema(conn)
    conn.close()
    path = tmp_path / f"{name}.jsonl"
    ledger.append(path, {"event": "seal", "boundary_date": "2019-01-02"})
    return v9_run.run_v9(db, path, first_entry="2018-01-02", draws=3, resamples=200)


def _facts_db_schema(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS sec_cik_map (ticker TEXT, cik INTEGER, title TEXT, mapped_ts TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS sec_fundamentals (cik INTEGER, taxonomy TEXT, tag TEXT, unit TEXT,"
                 " period_start TEXT, period_end TEXT, fiscal_year INTEGER, fiscal_period TEXT, value REAL, form TEXT,"
                 " accession TEXT, filed_date TEXT, frame TEXT)")
    conn.commit()


def test_run_appends_one_entry_with_the_registration_hash_and_eight_tests(tmp_path):
    out = _run(tmp_path, "a")
    entries = [e for e in ledger.read(tmp_path / "a.jsonl") if e.get("event") == "portfolio_study_run"]
    assert len(entries) == 1 and entries[0]["family"] == "v9" and entries[0]["sealed_opened"] is False
    assert entries[0]["config"]["registration_sha256"] == v9_run.registration_sha256()
    assert entries[0]["config"]["last_session"] < "2019-01-02"
    assert list(out["holm"]) == list(v9_run.PRIMARY) and len(v9_run.PRIMARY) == 8
    assert out["h5"]["status"] == "skipped"
    for k in ("H1a", "H1b", "H2", "H3", "H4", "H6", "H7"):
        assert k in out["results"]


def test_sealed_rows_do_not_change_any_result(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b", scale_after="2019-01-02")
    for k in ("H1a", "H1b", "H2", "H3", "H4", "H6", "H7"):
        assert a["summary"][k] == b["summary"][k]


LOOSE = {"GAP": 0.01, "VOLUME_MULTIPLE": 1.1, "RS_CUT": 0.0, "QUIET_MAX": 1.0, "SURGE": 0.02, "MIN_DOLLAR_VOLUME": 0.0,
         "BASE_RANGE": 0.6, "BURST_MIN": 0.03, "BURST_VOLUME": 0.5, "REST_RANGE": 0.5, "REST_HOLD": 0.5}


@pytest.fixture
def loose(monkeypatch):
    """Thresholds loosened so every rule trades on the eight megacaps; the code paths are the registered ones."""
    for k, val in LOOSE.items():
        monkeypatch.setattr(v9_signals, k, val)
    monkeypatch.setattr(v9_signals, "HIGH_WINDOW", 60)
    monkeypatch.setattr(v9_signals.new_high, "__defaults__", (60,))
    from trader_koo.research.bench import v9_rules
    monkeypatch.setattr(v9_rules, "H2_MAX_STOP", 0.5)
    monkeypatch.setattr(v9_rules, "FIVE_YEARS", 100)


def test_every_rule_trades_with_loose_thresholds_and_sealed_rows_change_nothing(tmp_path, loose):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b", scale_after="2019-01-02")
    for k in ("H1a", "H1b", "H2", "H3", "H4", "H7"):
        assert a["summary"][k]["all_signals"] > 0, k
    assert a["summary"]["H1a_vs_H1b"]["p"] is not None
    for k in ("H1a", "H1b", "H2", "H3", "H4", "H6", "H7", "H1a_vs_H1b"):
        assert a["summary"][k] == b["summary"][k], k
    assert a["summary"]["variants"] == b["summary"]["variants"]

"""Hypotheses v8 (docs/research/hypotheses-v8-swing.md): entry timing, surprise cut-offs, the gap filter, the
stop-and-trail exit, the calendar-time portfolio, the 52-week-high score and the sealed window.

Small hand-built arrays test arithmetic and timing only. The integration tests use real daily bars of eight
S&P 500 stocks and SPY (2017-01-03 to 2019-06-28) and their real yfinance earnings rows.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.test_research_bench import _build_db, _fixture_frames
from tests.test_research_v2 import _db as _v2_db
from trader_koo.collection.earnings_history import ensure_earnings_events_schema
from trader_koo.research.bench import ledger, v8_run, v8_signals
from trader_koo.research.bench.data import load_market
from trader_koo.research.bench.run import BenchConfig, run_bench
from trader_koo.research.bench.signals import assert_no_lookahead, make_signal
from trader_koo.research.bench.v2_signals import Prices

FIX = Path(__file__).parent / "fixtures" / "research_bench"
NAN = np.nan


# ---- entry timing -------------------------------------------------------------------------------------------------------

SESSIONS = ["2020-01-02", "2020-01-03", "2020-01-06"]


@pytest.mark.parametrize("date, session, row", [
    ("2020-01-03", "BMO", 1),   # before the open: that session's open
    ("2020-01-03", "AMC", 2),   # after the close: the next session
    ("2020-01-03", None, 2),    # unknown time: treated as after the close, never earlier
    ("2020-01-04", "BMO", 2),   # a Saturday report: the next session
    ("2020-01-04", "AMC", 2),
    ("2020-01-06", "AMC", -1),  # no later session in the data
])
def test_entry_is_the_first_open_after_the_report_is_public(date, session, row):
    assert v8_signals.entry_rows([date], [session], SESSIONS).tolist() == [row]


# ---- H1 surprise and cut-offs ---------------------------------------------------------------------------------------------

def test_cutoff_comes_from_the_previous_calendar_quarter_only():
    dates = ["2020-02-03"] * 10 + ["2020-05-01", "2020-05-04", "2020-08-03"]
    sue = np.array(list(range(10)) + [8.5, 100.0, 0.0], dtype=float)
    cut = v8_signals.quarter_cutoffs(dates, sue, np.ones(len(sue), bool))
    assert np.isnan(cut[:10]).all()                          # Q1 has no previous quarter
    assert cut[10] == cut[11] == pytest.approx(np.quantile(np.arange(10.0), 0.9))
    assert cut[12] == pytest.approx(np.quantile([8.5, 100.0], 0.9))  # Q3 uses Q2 only


def test_ineligible_events_do_not_move_the_cutoff():
    dates = ["2020-02-03"] * 4 + ["2020-05-01"]
    sue = np.array([1.0, 2.0, 3.0, 1e9, 0.0])
    eligible = np.array([True, True, True, False, True])
    assert v8_signals.quarter_cutoffs(dates, sue, eligible)[4] == pytest.approx(np.quantile([1.0, 2.0, 3.0], 0.9))


def test_surprise_is_scaled_by_the_last_close_before_entry():
    stored_close = np.array([[50.0], [80.0], [200.0]])
    sue = v8_signals.surprise(np.array([1.5]), np.array([1.0]), np.array([2]), np.array([0]), stored_close)
    assert sue.tolist() == [pytest.approx(0.5 / 80.0)]       # row 1 is the close before the entry at row 2


# ---- H2 gap filter --------------------------------------------------------------------------------------------------------

def _bars(n=60):
    o = np.full((n, 1), 100.0)
    return o.copy(), o + 1.0, o - 1.0, o.copy(), np.full((n, 1), 1e6)


def test_gap_day_needs_gap_volume_and_a_close_in_the_upper_half():
    o, h, lo, c, v = _bars()
    g = 55
    o[g], h[g], lo[g], c[g], v[g] = 111.0, 120.0, 110.0, 118.0, 3.1e6
    ok = v8_signals.gap_qualifies(o, h, lo, c, v, np.array([g]), np.array([0]))
    assert ok.tolist() == [True]
    for change in ({"o": 109.0}, {"v": 2.9e6}, {"c": 114.0}):  # gap 9%, volume 2.9x, close in the lower half
        o2, c2, v2 = o.copy(), c.copy(), v.copy()
        o2[g], c2[g], v2[g] = change.get("o", o[g, 0]), change.get("c", c[g, 0]), change.get("v", v[g, 0])
        assert v8_signals.gap_qualifies(o2, h, lo, c2, v2, np.array([g]), np.array([0])).tolist() == [False], change


def test_volume_average_is_the_fifty_sessions_before_the_gap_day():
    o, h, lo, c, v = _bars()
    g = 55
    o[g], h[g], lo[g], c[g] = 111.0, 120.0, 110.0, 118.0
    v[g] = 3.1e6
    v[:g - 50] = 1e9                       # older than 50 sessions: must not count
    assert v8_signals.gap_qualifies(o, h, lo, c, v, np.array([g]), np.array([0])).tolist() == [True]
    v[g - 1] = 1e8                         # inside the 50: raises the average above a third of 3.1e6
    assert v8_signals.gap_qualifies(o, h, lo, c, v, np.array([g]), np.array([0])).tolist() == [False]


def test_volume_floor_is_three_hundred_thousand_shares():
    o, h, lo, c, v = _bars()
    v[:] = 1e4
    g = 55
    o[g], h[g], lo[g], c[g], v[g] = 111.0, 120.0, 110.0, 118.0, 2.9e5
    assert v8_signals.gap_qualifies(o, h, lo, c, v, np.array([g]), np.array([0])).tolist() == [False]


# ---- H2 stop and trailing exit --------------------------------------------------------------------------------------------

def _grid(o, h, lo, c, sma_n=10, max_hold=60):
    idx = [f"d{i:03d}" for i in range(len(o))]
    df = lambda a: pd.DataFrame(a, index=idx, columns=["X"])  # noqa: E731
    p = Prices(df(o), df(h), df(lo), df(c), df(np.ones_like(o, bool)))
    return v8_signals.stop_trail_grid(SimpleNamespace(exits={}), p, sma_n, max_hold)


def _rising(n=100):
    c = (100.0 + np.arange(n, dtype=float))[:, None]
    return c - 0.5, c + 1.0, c - 1.0, c.copy()


def test_open_below_the_stop_exits_at_that_open():
    o, h, lo, c = _rising()
    e = 30                                   # stop = low of row 29 = 128
    o[33], lo[33] = 120.0, 119.0
    g = _grid(o, h, lo, c)
    assert g.exit_row[e, 0] == 33 and g.gross[e, 0] == pytest.approx(120.0 / o[e, 0])


def test_low_through_the_stop_exits_at_the_stop():
    o, h, lo, c = _rising()
    e = 30
    lo[34] = 120.0
    g = _grid(o, h, lo, c)
    assert g.exit_row[e, 0] == 35 and g.gross[e, 0] == pytest.approx(lo[29, 0] / o[e, 0])


def test_close_below_the_average_exits_next_open_once_the_average_is_above_the_stop():
    o, h, lo, c = _rising()
    e = 30
    c[40] = c[40] - 8.0                      # below its 10-session average, still above the stop (128)
    g = _grid(o, h, lo, c)
    assert g.exit_row[e, 0] == 41 and g.gross[e, 0] == pytest.approx(o[41, 0] / o[e, 0])


def test_average_below_the_stop_does_not_trigger_the_trail():
    o, h, lo, c = _rising()
    e = 30
    lo[29] = 128.5                           # stop above the 10-session average (125.5 at entry) on the first sessions
    c[31] = 124.0                            # close below its average (125.8), which is below the stop: no trail
    lo[31] = 128.6                           # the low stays above the stop
    g = _grid(o, h, lo, c, max_hold=5)
    assert g.exit_row[e, 0] == e + 5


def test_no_trigger_exits_at_the_maximum_hold():
    o, h, lo, c = _rising()
    g = _grid(o, h, lo, c, max_hold=20)
    assert g.exit_row[30, 0] == 50 and g.gross[30, 0] == pytest.approx(o[50, 0] / o[30, 0])
    assert g.late[80, 0] and np.isnan(g.gross[80, 0])


# ---- calendar-time portfolio ----------------------------------------------------------------------------------------------

def test_calendar_portfolio_equal_weights_open_trades_and_charges_both_sides():
    o = np.array([[100.0, 50.0], [110.0, 50.0], [121.0, 55.0], [121.0, 55.0]])
    trades = [(0, 2, 0, 121.0 / 100.0), (1, 3, 1, 55.0 / 50.0)]
    r, invested = v8_run.calendar_returns(o, trades, cost_bps=0.0)
    assert r.tolist() == pytest.approx([0.10, (0.10 + 0.10) / 2, 0.0])
    assert invested.tolist() == [1, 2, 1]
    r_cost, _ = v8_run.calendar_returns(o, trades, cost_bps=10.0)
    c = 10.0 / 1e4
    assert r_cost[0] == pytest.approx(1.1 / (1 + c) - 1)
    assert r_cost[2] == pytest.approx(1.0 * (1 - c) - 1)


def test_calendar_portfolio_counts_a_same_open_exit_as_its_costs_only():
    o = np.array([[100.0], [90.0], [80.0]])
    r, invested = v8_run.calendar_returns(o, [(1, 1, 0, 1.0)], cost_bps=10.0)
    c = 10.0 / 1e4
    assert r[1] == pytest.approx((1 - c) / (1 + c) - 1) and invested.tolist() == [0, 1]


# ---- H3 52-week high ------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bench_market(tmp_path_factory):
    return load_market(_build_db(str(tmp_path_factory.mktemp("v8h3") / "r.db"), *_fixture_frames()))


def test_52_week_high_score_is_close_over_the_highest_close_and_has_no_lookahead(bench_market):
    s = bench_market.sessions
    as_of = s[300]
    uni = bench_market.universe(300)
    got = make_signal("v8_h3_52_week_high", {})(as_of, uni, bench_market.view(as_of))
    conn = bench_market.conn
    for t in uni:
        rows = conn.execute("SELECT close FROM price_daily WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 252",
                            (t, as_of)).fetchall()
        closes = [r[0] for r in rows]
        if len(closes) == 252 and bench_market.adj_close[t].iloc[49:301].notna().all():
            assert got[t] == pytest.approx(closes[0] / max(closes))
    assert got.notna().any() and (got <= 1.0).all()
    assert_no_lookahead(make_signal("v8_h3_52_week_high", {}), bench_market, as_of, uni)


# ---- integration: real bars, real earnings rows, the sealed window --------------------------------------------------------

def _db_with_earnings(path: Path, scale_after: str | None = None) -> str:
    db = _v2_db(path, scale_after)
    conn = sqlite3.connect(db)
    ensure_earnings_events_schema(conn)
    ev = pd.read_csv(FIX / "earnings_2016_2019.csv.gz")
    if scale_after:  # a later report must not change anything either
        later = ev["report_date"] >= scale_after
        ev.loc[later, ["eps_actual", "eps_estimate"]] *= 7.0
    ev.to_sql("earnings_events", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return db


def _h3_trial(db: str, path: Path) -> str:
    cfg = BenchConfig.from_settings("v8_h3_52_week_high", n_random=3, top_n=3, buffer_n=3, review_every=21, min_hold=0)
    run_bench(load_market(db), cfg, path)
    return next(e for e in ledger.read(path) if e.get("event") == "run")["trial_id"]


def _run(tmp_path: Path, name: str, scale_after: str | None = None) -> dict:
    db = _db_with_earnings(tmp_path / f"{name}.db", scale_after)
    path = tmp_path / f"{name}.jsonl"
    ledger.append(path, {"event": "seal", "boundary_date": "2019-01-02"})
    return v8_run.run_v8(db, path, _h3_trial(db, path), first_entry="2017-04-01")


def test_run_appends_one_event_study_entry_with_the_registration_hash(tmp_path):
    out = _run(tmp_path, "a")
    entries = [e for e in ledger.read(tmp_path / "a.jsonl") if e.get("event") == "event_study_run"]
    assert len(entries) == 1 and entries[0]["family"] == "v8" and entries[0]["sealed_opened"] is False
    assert entries[0]["config"]["registration_sha256"] == v8_run.registration_sha256()
    assert entries[0]["config"]["last_session"] < "2019-01-02"
    assert set(out["summary"]) == {"H1", "H2", "H3", "H1_hold20", "H1_amc_only", "H2_trail20"}
    assert out["results"]["H1"]["n_trades"] > 0      # the fixture's megacaps still have top-decile surprises


def test_sealed_rows_do_not_change_any_result(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b", scale_after="2019-01-02")
    for k in ("H1", "H2"):
        assert a["summary"][k]["trades"] == b["summary"][k]["trades"]
        assert a["summary"][k]["mean_net"] == b["summary"][k]["mean_net"]
        assert a["summary"][k]["excess_vs_equal_weight"] == b["summary"][k]["excess_vs_equal_weight"]

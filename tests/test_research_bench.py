"""Research bench contract tests, on real price rows (17 tickers, 352 sessions, Jun 2025 onward)."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader_koo.db.membership import ensure_membership_schema
from trader_koo.fundamentals.schema import ensure_sec_schema
from trader_koo.research.bench import ledger
from trader_koo.research.bench.__main__ import main as bench_main
from trader_koo.research.bench.data import load_market
from trader_koo.research.bench.portfolio import (
    Rules, Schedule, build_schedule, equal_weight_schedule, select_holdings, simulate,
)
from trader_koo.research.bench.run import BenchConfig, run_bench
from trader_koo.research.bench.signals import LookaheadError, assert_no_lookahead, make_signal

FIX = Path(__file__).parent / "fixtures" / "research_bench"


def _build_db(path: str, prices: pd.DataFrame, actions: pd.DataFrame, extra_facts=()) -> str:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, open REAL, close REAL, adjustment_basis TEXT)")
    conn.execute("CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, value REAL)")
    prices.assign(adjustment_basis=None).to_sql("price_daily", conn, if_exists="append", index=False)
    actions.to_sql("price_corporate_actions", conn, if_exists="append", index=False)
    ensure_membership_schema(conn)
    members = sorted(set(prices["ticker"]) - {"SPY"})
    dates = sorted(prices["date"].unique())
    rows = [(t, "2000-01-01", dates[200] if t == "ORCL" else None, "t", "t") for t in members]
    conn.executemany("INSERT INTO sp500_membership VALUES (?,?,?,?,?)", rows)
    ensure_sec_schema(conn)
    conn.execute("INSERT INTO sec_cik_map VALUES ('AAPL', 1, 'x', 'now')")
    for filed, value in ((dates[100], 10.0), (dates[200], 99.0), *extra_facts):
        conn.execute(
            "INSERT INTO sec_fundamentals VALUES (1,'us-gaap','Assets','USD','',?,2025,'FY',?,'10-K',?,?,NULL)",
            (filed, value, f"acc-{filed}", filed))
    conn.commit()
    conn.close()
    return path


def _fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(FIX / "prices.csv.gz"), pd.read_csv(FIX / "actions.csv.gz")


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> str:
    return _build_db(str(tmp_path_factory.mktemp("bench") / "research.db"), *_fixture_frames())


@pytest.fixture(scope="module")
def market(db_path):
    return load_market(db_path)


def _cfg(**kw) -> BenchConfig:
    return BenchConfig(**{"candidate": "equal_weight", "n_random": 5, **kw})


def test_universe_is_point_in_time_and_needs_a_price(market):
    early, late = 150, 250
    assert "ORCL" in market.universe(early)
    assert "ORCL" not in market.universe(late)  # membership ended at session 200
    assert "SPY" not in market.universe(early)  # not an index member


def test_lookahead_guard_passes_clean_signals(market):
    as_of = market.sessions[300]
    uni = market.universe(300)
    assert_no_lookahead(make_signal("momentum_12_1", {}), market, as_of, uni)

    def fundamentals_signal(a, u, data):
        fact = data.fundamentals("AAPL", "Assets")
        return pd.Series(fact.value if fact else np.nan, index=u)

    assert_no_lookahead(fundamentals_signal, market, market.sessions[150], market.universe(150))


def test_lookahead_guard_catches_price_and_sec_leaks(market):
    def leaky_price(a, u, data):
        rows = data.conn.execute("SELECT ticker, close FROM price_daily ORDER BY date DESC LIMIT 17").fetchall()
        return pd.Series({t: c for t, c in rows}).reindex(u)

    with pytest.raises(LookaheadError):
        assert_no_lookahead(leaky_price, market, market.sessions[150], market.universe(150))

    def leaky_sec(a, u, data):
        row = data.conn.execute("SELECT value FROM sec_fundamentals ORDER BY filed_date DESC LIMIT 1").fetchone()
        return pd.Series(row[0], index=u)

    with pytest.raises(LookaheadError):
        assert_no_lookahead(leaky_sec, market, market.sessions[150], market.universe(150))


def test_buffer_and_min_hold_rules():
    scores = pd.Series({f"T{i:02d}": float(100 - i) for i in range(60)})  # T00 is rank 1
    rules = Rules(top_n=5, buffer_n=10, min_hold=20)
    held = {"T08": 0, "T09": 0, "T20": 0, "T30": 95}
    out = select_holdings(scores, held, exec_idx=100, rules=rules)
    assert "T08" in out and "T09" in out  # fell out of the top 5 but inside the top 10
    assert "T20" not in out  # outside the buffer and past the minimum hold
    assert "T30" in out  # outside the buffer but bought 5 sessions ago
    assert len(out) == 5 and "T00" in out and "T01" in out
    assert "T08" not in select_holdings(scores.drop("T08"), held, 100, rules)  # unscored holdings leave


def test_schedule_respects_min_hold_and_top_n(market):
    rules = Rules()
    sig = make_signal("momentum_12_1", {"lookback": 100, "skip": 10})
    sched = build_schedule(market, list(range(110, 340, 10)), rules,
                           lambda s, u: sig(market.sessions[s], u, market.view(market.sessions[s])))
    entry: dict[str, int] = {}
    for e, targets in zip(sched.exec_idx, sched.targets):
        assert len(targets) <= rules.top_n
        for t in list(entry):
            if t not in targets and t in market.universe(e - 1):
                assert e - entry[t] >= rules.min_hold
                entry.pop(t)
            elif t not in targets:
                entry.pop(t)
        for t in targets:
            entry.setdefault(t, e)


def test_costs_apply_to_every_fill(market):
    names = market.universe(10)[:4]
    sched = Schedule([11, 20], [names, []])
    free, paid = simulate(market, sched, 40, 0.0), simulate(market, sched, 40, 50.0)
    c = 0.005
    assert paid.equity[0] / free.equity[0] == pytest.approx(1 / (1 + c))  # buy cost
    assert paid.equity[-1] / free.equity[-1] == pytest.approx((1 - c) / (1 + c))  # buy and sell cost
    assert len(paid.trade_idx) == 8 and (paid.trade_notional > 0).all()


def test_equal_weight_candidate_tracks_the_equal_weight_benchmark(market, tmp_path):
    out = run_bench(market, _cfg(), tmp_path / "l.jsonl")
    d = out["dev"]
    assert d["strategy"]["total_return"] == pytest.approx(d["equal_weight"]["total_return"], abs=1e-6)
    assert d["strategy"]["cagr"] <= d["spy"]["cagr"] + 1.0  # sanity: finite and in range


def test_sealed_window_refuses_without_flag_and_ledger_records_opening(market, tmp_path):
    path = tmp_path / "l.jsonl"
    boundary = ledger.seal_boundary(len(market.sessions))
    with pytest.raises(ledger.SealedWindowError):
        run_bench(market, _cfg(end=market.sessions[boundary + 5]), path)
    with pytest.raises(ledger.SealedWindowError):
        run_bench(market, _cfg(start=market.sessions[boundary]), path)
    assert [e["event"] for e in ledger.read(path)] == ["seal"]  # refusals write only the pinned boundary
    out = run_bench(market, _cfg(open_sealed=True), path)
    assert out["sealed"] is not None
    events = [e["event"] for e in ledger.read(path)]
    assert events == ["seal", "sealed_open", "run"]
    assert ledger.read(path)[-1]["sealed_opened"] is True
    dev_only = run_bench(market, _cfg(), path)
    assert dev_only["sealed"] is None and dev_only["dev"]["end"] < market.sessions[boundary]
    assert ledger.sealed_openings(path, "equal_weight") == 1


def test_ledger_entry_is_complete_and_replayable(db_path, market, tmp_path, capsys):
    path = tmp_path / "l.jsonl"
    a = run_bench(market, _cfg(candidate="momentum_12_1", start=market.sessions[240]), path)
    b = run_bench(market, _cfg(candidate="momentum_12_1", start=market.sessions[240]), path)
    assert a["summary"] == b["summary"] and a["seed"] == b["seed"]
    entry = next(e for e in ledger.read(path) if e["event"] == "run")
    for key in ("candidate", "params_hash", "code_version", "date_range", "costs_bps", "universe",
                "sealed_opened", "timestamp", "summary", "config", "trial_id"):
        assert key in entry
    code = bench_main(["replay", "--db", db_path, "--trial", entry["trial_id"], "--ledger", str(path)])
    assert code == 0 and "MATCHES" in capsys.readouterr().out
    assert len(ledger.read(path)) == 4  # seal, two runs, and the replay is itself recorded


def test_report_has_required_fields(market, tmp_path):
    out = run_bench(market, _cfg(candidate="momentum_12_1", start=market.sessions[240]), tmp_path / "l.jsonl")
    d = out["dev"]
    for key in ("strategy", "spy", "equal_weight", "excess_vs_spy", "excess_vs_equal_weight", "turnover",
                "hit_rate", "pct_years_beating_spy", "random", "stress", "years"):
        assert key in d
    assert 0 <= d["random"]["percentile"] <= 100
    assert any("Survivorship" in c for c in out["caveats"])
    assert "Names lacking data" in out["markdown"]


# ---- leakage regressions -------------------------------------------------------------------

def _market_with(tmp_path, name, prices, actions, extra_facts=()):
    return load_market(_build_db(str(tmp_path / name), prices, actions, extra_facts))


def test_later_dividend_and_split_cannot_change_the_view_at_t(market, tmp_path):
    prices, actions = _fixture_frames()
    t_idx, event_idx = 300, 320
    dates = sorted(prices["date"].unique())
    event = dates[event_idx]
    # What a later re-fetch would store after a 2:1 split on `event` and a new dividend there:
    # every row before the split date halves, the earlier dividends halve with it.
    pre = (prices["ticker"] == "AAPL") & (prices["date"] < event)
    p2 = prices.copy()
    p2.loc[pre, ["open", "close"]] = p2.loc[pre, ["open", "close"]] / 2.0
    a2 = actions.copy()
    old = (a2["ticker"] == "AAPL") & (a2["action_type"] == "dividend") & (a2["action_date"] < event)
    a2.loc[old, "value"] = a2.loc[old, "value"] / 2.0
    a2 = pd.concat([a2, pd.DataFrame([
        {"ticker": "AAPL", "action_date": event, "action_type": "split", "value": 2.0},
        {"ticker": "AAPL", "action_date": dates[t_idx + 5], "action_type": "dividend", "value": 1.5}])])
    later = _market_with(tmp_path, "later.db", p2, a2)

    as_of = market.sessions[t_idx]
    a, b = market.view(as_of), later.view(as_of)
    pd.testing.assert_frame_equal(a.close, b.close, check_exact=False, rtol=1e-12)
    pd.testing.assert_frame_equal(a.open, b.open, check_exact=False, rtol=1e-12)
    mom = make_signal("momentum_12_1", {"lookback": 200, "skip": 20})
    uni = market.universe(t_idx)
    pd.testing.assert_series_equal(mom(as_of, uni, a), mom(as_of, uni, b), check_exact=False, rtol=1e-12)
    # The P&L path does see the new dividend after t: AAPL's chain differs from the dividend day on.
    d_row = market.sessions.index(dates[t_idx + 5])
    ratio = (later.adj_close["AAPL"] / market.adj_close["AAPL"]).to_numpy()
    assert ratio[d_row - 1] == pytest.approx(ratio[0], rel=1e-9)
    assert ratio[d_row] > ratio[d_row - 1] * 1.005


def test_sec_fact_is_usable_from_the_next_session_only(market, tmp_path):
    sessions = market.sessions
    fri = next(i for i in range(120, 190) if dt.date.fromisoformat(sessions[i]).weekday() == 4)
    assert dt.date.fromisoformat(sessions[fri + 1]).weekday() == 0
    saturday = (dt.date.fromisoformat(sessions[fri]) + dt.timedelta(days=1)).isoformat()
    mkt = _market_with(tmp_path, "sec.db", *_fixture_frames(),
                       extra_facts=((sessions[fri], 55.0), (saturday, 66.0)))
    assert mkt.view(sessions[100]).fundamentals("AAPL", "Assets") is None  # filed that day: not yet
    assert mkt.view(sessions[101]).fundamentals("AAPL", "Assets").value == 10.0
    assert mkt.view(sessions[fri]).fundamentals("AAPL", "Assets").value == 10.0  # Friday filing waits
    monday = mkt.view(sessions[fri + 1]).fundamentals("AAPL", "Assets")
    assert monday.value == 66.0  # Friday and Saturday filings are both usable Monday


def test_guard_catches_a_same_day_sec_read(market):
    def same_day(a, u, data):
        row = data.conn.execute("SELECT value FROM sec_fundamentals WHERE filed_date = ?", (a,)).fetchone()
        return pd.Series(row[0] if row else np.nan, index=u)

    with pytest.raises(LookaheadError):
        assert_no_lookahead(same_day, market, market.sessions[100], market.universe(100))


def test_guard_runs_on_a_file_without_copying_it(market):
    from trader_koo.research.bench.signals import _perturbed_conn

    as_of = market.sessions[150]
    mem = _perturbed_conn(market.conn, as_of)
    q = "SELECT close FROM price_daily WHERE ticker='AAPL' AND date=?"
    try:
        assert not any(r[1] == "main" and r[2] for r in mem.execute("PRAGMA database_list"))  # main is memory
        assert mem.execute(q, (as_of,)).fetchone() == market.conn.execute(q, (as_of,)).fetchone()
        nxt = market.sessions[151]
        assert mem.execute(q, (nxt,)).fetchone()[0] == pytest.approx(market.conn.execute(q, (nxt,)).fetchone()[0] * 3.0)
        assert mem.execute("SELECT count(*) FROM sp500_membership").fetchone()[0] == 16
        with pytest.raises(sqlite3.OperationalError):
            mem.execute("INSERT INTO src.price_daily VALUES ('X','2030-01-01',1,1,NULL)")
    finally:
        mem.close()


# ---- universe, seal, seeds, flags ----------------------------------------------------------

def test_membership_interval_is_half_open_on_session_dates(market):
    assert "ORCL" in market.universe(199)
    assert "ORCL" not in market.universe(200)  # end_date is the first day it is out


def test_sealed_boundary_is_pinned_and_does_not_slide_with_new_data(market, tmp_path):
    path = tmp_path / "l.jsonl"
    first = ledger.pinned_boundary(path, market.sessions)
    longer = market.sessions + [f"2030-01-{d:02d}" for d in range(1, 29)]
    assert ledger.pinned_boundary(path, longer) == first
    assert first != longer[ledger.seal_boundary(len(longer))]  # recomputing would have moved it
    assert [e["event"] for e in ledger.read(path)] == ["seal"]
    with pytest.raises(ValueError):
        ledger.pinned_boundary(path, market.sessions[:50])


def test_random_control_is_reproducible_and_seed_follows_config(market, tmp_path):
    a = run_bench(market, _cfg(n_random=8), tmp_path / "a.jsonl")
    b = run_bench(market, _cfg(n_random=8), tmp_path / "b.jsonl")
    c = run_bench(market, _cfg(n_random=8, top_n=5), tmp_path / "c.jsonl")
    assert a["dev"]["random"] == b["dev"]["random"] and a["seed"] == b["seed"]
    assert c["seed"] != a["seed"]


def test_flagged_moves_keep_real_losses_except_on_split_dates(tmp_path):
    prices, actions = _fixture_frames()
    dates = sorted(prices["date"].unique())
    crash, split_day = dates[120], dates[220]
    p2 = prices.copy()
    for tkr, day, f in (("JNJ", crash, 0.55), ("KO", split_day, 0.5)):
        m = (p2["ticker"] == tkr) & (p2["date"] >= day)
        p2.loc[m, ["open", "close"]] = p2.loc[m, ["open", "close"]] * f
    a2 = pd.concat([actions, pd.DataFrame([
        {"ticker": "KO", "action_date": split_day, "action_type": "split", "value": 2.0}])])
    mkt = _market_with(tmp_path, "flag.db", p2, a2)
    i, j = mkt.sessions.index(crash), mkt.sessions.index(split_day)
    jnj, ko = mkt.adj_close["JNJ"], mkt.adj_close["KO"]
    assert jnj.iloc[i] / jnj.iloc[i - 1] - 1 < -0.4  # a real -45% day is not hidden as 0%
    assert ko.iloc[j] / ko.iloc[j - 1] - 1 == pytest.approx(0.0)  # unadjusted split is a 0% day
    rows = mkt.flagged_in(0, len(mkt.sessions) - 1).set_index("ticker")
    assert rows.loc["JNJ", "treatment"] == "real move kept"
    assert rows.loc["KO", "treatment"] == "0% day or NaN"
    assert len(mkt.flagged_in(0, i - 1)) == 0  # rows after the window are not counted


# ---- spin-offs -----------------------------------------------------------------------------

def _spin_market(tmp_path, ex_idx=150, effect="raw_drop"):
    prices, actions = _fixture_frames()
    dates = sorted(prices["date"].unique())
    ex = dates[ex_idx]
    p2 = prices.copy()
    m = (p2["ticker"] == "AAPL") & (p2["date"] >= ex)
    p2.loc[m, ["open", "close"]] = p2.loc[m, ["open", "close"]] * 0.2  # holders keep the spun-off shares
    csv = tmp_path / "spin.csv"
    csv.write_text(
        "parent_ticker,ex_date,spun_off,kind,date_basis,price_effect,source_url\n"
        f"AAPL,{ex},TestCo,spinoff,regular_way_start,{effect},https://example.com/8k\n")
    db = _build_db(str(tmp_path / "spin.db"), p2, actions)
    return load_market(db, csv), ex_idx


def test_spinoffs_csv_is_complete_and_sourced():
    from trader_koo.research.bench.spinoffs import load_spinoffs

    frame = load_spinoffs()
    assert len(frame) >= 20
    assert frame["source_url"].str.startswith("https://").all()
    assert {"CTVA"} <= set(frame.loc[frame["price_effect"] == "raw_drop", "parent_ticker"])
    assert frame.loc[frame["kind"] == "spinoff", "ex_date"].min() >= "2010-01-01"


def test_holding_is_sold_before_a_spinoff_ex_date_and_no_loss_is_booked(tmp_path):
    mkt, x = _spin_market(tmp_path)
    col = mkt.columns["AAPL"]
    assert mkt.exits[col] == [x - 1]
    ret = mkt.adj_close["AAPL"].iloc[x] / mkt.adj_close["AAPL"].iloc[x - 1] - 1
    assert ret == pytest.approx(0.0)  # raw -80% drop is neutralised in the chain
    sched = Schedule([x - 20], [["AAPL", "MSFT"]])
    res = simulate(mkt, sched, x + 10, 0.0)
    assert res.exits == [(x - 1, col)]
    assert res.trade_idx.tolist().count(x - 1) == 1


def test_spinoff_leaves_equity_flat_for_the_exited_half(tmp_path):
    mkt, x = _spin_market(tmp_path)
    start = x - 20
    both = simulate(mkt, Schedule([start], [["AAPL", "MSFT"]]), x + 10, 0.0)
    only_msft = simulate(mkt, Schedule([start], [["MSFT"]]), x + 10, 0.0)
    aapl_leg = both.equity - 0.5 * only_msft.equity  # MSFT half is half of a MSFT-only book
    i = x - 1 - start
    assert aapl_leg[i:] == pytest.approx(np.full(len(aapl_leg) - i, aapl_leg[i]))  # idle cash after the exit
    assert aapl_leg[i] > 0.3


def test_buy_is_skipped_when_the_ex_date_is_inside_the_holding_period(tmp_path):
    mkt, x = _spin_market(tmp_path)
    rules = Rules(top_n=20, buffer_n=40, review_every=10, min_hold=0)
    score = lambda s, uni: pd.Series(1.0, index=uni)  # noqa: E731
    # review at x-5 fills at x-4 and the next fill is x+6: the ex-date x is inside and AAPL is not held
    sched = build_schedule(mkt, [x - 5, x + 5], rules, score, end_idx=x + 20)
    assert "AAPL" not in sched.targets[0]
    assert "AAPL" in sched.targets[1]  # after the ex-date it is buyable again
    assert sched.blocked_spinoff == 1
    ew = equal_weight_schedule(mkt, [x - 5, x + 5], x + 20)
    assert "AAPL" not in ew.targets[0] and "AAPL" in ew.targets[1]
    # already held ahead of the exit: kept in the target, sold mid-period by simulate
    held = build_schedule(mkt, [x - 30, x - 5], rules, score, end_idx=x + 20)
    assert "AAPL" in held.targets[0] and "AAPL" in held.targets[1] and held.blocked_spinoff == 0

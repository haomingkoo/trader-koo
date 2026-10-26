"""Research bench contract tests, on real price rows (17 tickers, 352 sessions, Jun 2025 onward)."""
from __future__ import annotations

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


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("bench") / "research.db")
    prices = pd.read_csv(FIX / "prices.csv.gz")
    actions = pd.read_csv(FIX / "actions.csv.gz")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, open REAL, close REAL, adjustment_basis TEXT)")
    conn.execute("CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, value REAL)")
    prices.assign(adjustment_basis=None).to_sql("price_daily", conn, if_exists="append", index=False)
    actions.to_sql("price_corporate_actions", conn, if_exists="append", index=False)
    ensure_membership_schema(conn)
    members = sorted(set(prices["ticker"]) - {"SPY"})
    last = sorted(prices["date"].unique())[200]
    rows = [(t, "2000-01-01", last if t == "ORCL" else None, "t", "t") for t in members]
    conn.executemany("INSERT INTO sp500_membership VALUES (?,?,?,?,?)", rows)
    ensure_sec_schema(conn)
    conn.execute("INSERT INTO sec_cik_map VALUES ('AAPL', 1, 'x', 'now')")
    first, mid = sorted(prices["date"].unique())[100], sorted(prices["date"].unique())[200]
    for filed, value in ((first, 10.0), (mid, 99.0)):
        conn.execute(
            "INSERT INTO sec_fundamentals VALUES (1,'us-gaap','Assets','USD','',?,2025,'FY',?,'10-K',?,?,NULL)",
            (filed, value, f"acc-{filed}", filed))
    conn.commit()
    conn.close()
    return path


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
    assert d["strategy"]["total_return"] == pytest.approx(d["equal_weight"]["total_return"], abs=0.02)
    assert d["strategy"]["cagr"] <= d["spy"]["cagr"] + 1.0  # sanity: finite and in range


def test_sealed_window_refuses_without_flag_and_ledger_records_opening(market, tmp_path):
    path = tmp_path / "l.jsonl"
    boundary = ledger.seal_boundary(len(market.sessions))
    with pytest.raises(ledger.SealedWindowError):
        run_bench(market, _cfg(end=market.sessions[boundary + 5]), path)
    with pytest.raises(ledger.SealedWindowError):
        run_bench(market, _cfg(start=market.sessions[boundary]), path)
    assert ledger.read(path) == []
    out = run_bench(market, _cfg(open_sealed=True), path)
    assert out["sealed"] is not None
    events = [e["event"] for e in ledger.read(path)]
    assert events == ["sealed_open", "run"]
    assert ledger.read(path)[-1]["sealed_opened"] is True
    dev_only = run_bench(market, _cfg(), path)
    assert dev_only["sealed"] is None and dev_only["dev"]["end"] < market.sessions[boundary]
    assert ledger.sealed_openings(path, "equal_weight") == 1


def test_ledger_entry_is_complete_and_replayable(db_path, market, tmp_path, capsys):
    path = tmp_path / "l.jsonl"
    a = run_bench(market, _cfg(candidate="momentum_12_1", start=market.sessions[240]), path)
    b = run_bench(market, _cfg(candidate="momentum_12_1", start=market.sessions[240]), path)
    assert a["summary"] == b["summary"] and a["seed"] == b["seed"]
    entry = ledger.read(path)[0]
    for key in ("candidate", "params_hash", "code_version", "date_range", "costs_bps", "universe",
                "sealed_opened", "timestamp", "summary", "config", "trial_id"):
        assert key in entry
    code = bench_main(["replay", "--db", db_path, "--trial", entry["trial_id"], "--ledger", str(path)])
    assert code == 0 and "MATCHES" in capsys.readouterr().out
    assert len(ledger.read(path)) == 3  # replay is itself recorded


def test_report_has_required_fields(market, tmp_path):
    out = run_bench(market, _cfg(candidate="momentum_12_1", start=market.sessions[240]), tmp_path / "l.jsonl")
    d = out["dev"]
    for key in ("strategy", "spy", "equal_weight", "excess_vs_spy", "excess_vs_equal_weight", "turnover",
                "hit_rate", "pct_years_beating_spy", "random", "stress", "years"):
        assert key in d
    assert 0 <= d["random"]["percentile"] <= 100
    assert any("Survivorship" in c for c in out["caveats"])
    assert "Names lacking data" in out["markdown"]

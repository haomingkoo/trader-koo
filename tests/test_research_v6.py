"""Pre-registered hypotheses v6 (insider buying) on the real-data bench fixture.

Insider rows are the SEC's 2026 Q1 Form 4 open-market trades for the fixture's 16 S&P 500 members,
exported from ``sec_insider_trades`` as loaded by trader_koo/research/insider.py.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from tests.test_research_hypotheses import _build
from trader_koo.research import insider
from trader_koo.research.bench import v6_signals as v6
from trader_koo.research.bench.data import load_market
from trader_koo.research.bench.signals import LookaheadError, assert_no_lookahead, make_signal

FIX = Path(__file__).parent / "fixtures" / "research_bench" / "insider_trades_2026q1.csv.gz"
AS_OF = "2026-03-31"


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    path = _build(str(tmp_path_factory.mktemp("v6") / "r.db"))
    conn = sqlite3.connect(path)
    insider.ensure_schema(conn)
    pd.read_csv(FIX).to_sql("sec_insider_trades", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return load_market(path)


def _scores(market, name):
    idx = list(market.sessions).index(AS_OF)
    d = market.sessions[idx]
    uni = market.universe(idx)
    return make_signal(name, {})(d, uni, market.view(d)), d, uni


def test_abt_qualifies_with_two_buyers_and_net_buying(market):
    scores, d, _ = _scores(market, "v6_h1_insider_clusters")
    # ABT, Q1 2026: buyers 1612571 (18,800 at 107.13) and 902132 (5,033 at 108.33, 4,967 at 109.14);
    # six officers sold 4,199 shares at 115.58. Net +2.62M. MSFT had one buyer only.
    assert d == AS_OF
    assert scores.dropna().index.tolist() == ["ABT"]
    assert 2.0 < scores["ABT"] < 3.0  # two buyers, tie-break below one


def test_trades_window_uses_filings_before_the_signal_date(market):
    view = market.view(AS_OF)
    df = v6.insider_trades(view, ["ABT"], v6._since(AS_OF))
    assert df["filing_date"].max() < AS_OF and df["filing_date"].min() > v6._since(AS_OF)
    net = (df["value"].where(df["code"] == "P", -df["value"])).sum()
    assert net == pytest.approx(18800 * 107.13 + 5033 * 108.33 + 4967 * 109.14 - 4199 * 115.58)


def test_opportunistic_keeps_buyers_without_three_years_of_history(market):
    scores, _, _ = _scores(market, "v6_h2_insider_opportunistic")
    assert scores.dropna().index.tolist() == ["ABT"]  # the fixture holds one quarter, so no one is routine


@pytest.mark.parametrize("name", ["v6_h1_insider_clusters", "v6_h2_insider_opportunistic"])
def test_v6_signals_pass_the_lookahead_guard(market, name):
    _, d, uni = _scores(market, name)
    assert_no_lookahead(make_signal(name, {}), market, d, uni)


def test_guard_catches_a_signal_that_reads_later_insider_filings(market):
    def leaky(as_of, universe, data):
        n = data.conn.execute("SELECT COUNT(*) FROM sec_insider_trades WHERE filing_date >= ?", (as_of,)).fetchone()[0]
        return pd.Series(float(n), index=universe)

    _, d, uni = _scores(market, "v6_h1_insider_clusters")
    with pytest.raises(LookaheadError):
        assert_no_lookahead(leaky, market, "2026-02-02", uni)

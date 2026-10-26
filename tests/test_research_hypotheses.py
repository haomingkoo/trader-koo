"""Pre-registered hypotheses v1 on the research bench: score formulas, look-ahead guard, schedule, filter.

Fixture rows are real: 16 S&P 500 members' daily prices from June 2025, their price_corporate_actions,
and their SEC companyfacts rows filed since mid 2022 (tests/fixtures/research_bench).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader_koo.db.membership import ensure_membership_schema
from trader_koo.fundamentals.schema import ensure_sec_schema
from trader_koo.research.bench import hypotheses as hyp
from trader_koo.research.bench import quality
from trader_koo.research.bench.data import AsOfView, load_market
from trader_koo.research.bench.hypotheses import accept_share_count
from trader_koo.research.bench.portfolio import Rules, annual_reviews, build_schedule
from trader_koo.research.bench.run import BenchConfig, run_bench
from trader_koo.research.bench.signals import assert_no_lookahead, make_signal

FIX = Path(__file__).parent / "fixtures" / "research_bench"
AS_OF_IDX = 300


def _build(path: str) -> str:
    prices = pd.read_csv(FIX / "prices.csv.gz")
    actions = pd.read_csv(FIX / "actions.csv.gz")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, open REAL, close REAL, volume REAL, "
                 "adjustment_basis TEXT)")
    conn.execute("CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, value REAL)")
    prices.assign(volume=1e6, adjustment_basis=None).to_sql("price_daily", conn, if_exists="append", index=False)
    actions.to_sql("price_corporate_actions", conn, if_exists="append", index=False)
    ensure_membership_schema(conn)
    conn.executemany("INSERT INTO sp500_membership VALUES (?,?,?,?,?)",
                     [(t, "2000-01-01", None, "t", "t") for t in sorted(set(prices["ticker"]) - {"SPY"})])
    ensure_sec_schema(conn)
    pd.read_csv(FIX / "sec_cik_map.csv.gz").assign(title="x", mapped_ts="now").to_sql(
        "sec_cik_map", conn, if_exists="append", index=False)
    facts = pd.read_csv(FIX / "sec_facts.csv.gz", keep_default_na=False,
                        dtype={"period_start": str, "frame": str, "form": str, "fiscal_period": str})
    facts["fiscal_year"] = pd.to_numeric(facts["fiscal_year"], errors="coerce")
    facts.to_sql("sec_fundamentals", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    return load_market(_build(str(tmp_path_factory.mktemp("hyp") / "r.db")))


def _scores(market, name, idx=AS_OF_IDX, params=None):
    d = market.sessions[idx]
    uni = market.universe(idx)
    return make_signal(name, params or {})(d, uni, market.view(d)), uni


# ---- score formulas, checked against figures read off the filed rows -----------------------------

def test_h1_inputs_match_hand_computation_on_aapl(market):
    # Filed rows as of 2026-08-12, in billions:
    # GrossProfit FY25 195.201 + YTD Q3 FY26 178.782 - YTD Q3 FY25 146.860 = 227.123; Assets 383.266.
    view = market.view(market.sessions[AS_OF_IDX])
    assert view.as_of == "2026-08-12"
    assert hyp.gross_profit_to_assets(view, "AAPL") == pytest.approx(227.123 / 383.266, rel=1e-9)
    assert hyp.liabilities_to_assets(view, "AAPL") == pytest.approx(275.746 / 383.266, rel=1e-9)
    # OCF 111.482 + 116.996 - 81.754 = 146.724; capex 12.715 + 6.799 - 9.473 = 10.041.
    assert hyp.free_cash_flow(view, "AAPL") == pytest.approx((146.724 - 10.041) * 1e9, rel=1e-9)


def test_zscores_are_population_scores_clipped_at_three():
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0], "b": [5.0, 5.0, 5.0, 5.0, 5.0]})
    mean = 22.0
    sd = np.sqrt(((np.array([1, 2, 3, 4, 100]) - mean) ** 2).mean())
    z = hyp.zscores(x, hyp.WINSOR)
    assert z["a"].iloc[0] == pytest.approx((1 - mean) / sd)
    assert z["a"].iloc[4] == pytest.approx(min((100 - mean) / sd, 3.0))
    big = pd.DataFrame({"a": [0.0] * 20 + [1000.0]})
    assert hyp.zscores(big, 3.0)["a"].iloc[20] == 3.0 and hyp.zscores(big)["a"].iloc[20] > 3.0  # winsorised only if asked
    assert z["b"].isna().all()  # no dispersion, no score


def test_h1_scores_only_names_with_every_input(market):
    scores, uni = _scores(market, "h1_quality_fair_price")
    view = market.view(market.sessions[AS_OF_IDX])
    complete = {t for t in uni if not np.isnan([hyp.gross_profit_to_assets(view, t), hyp.liabilities_to_assets(view, t),
                                                hyp.free_cash_flow(view, t)]).any()}
    assert set(scores.dropna().index) <= complete and len(scores.dropna()) > 0
    assert "JPM" not in scores.dropna().index  # banks report no gross profit
    assert scores.dropna().mean() == pytest.approx(0.0, abs=1e-9)  # mean of z-scores of the complete names


def test_h2_inputs_match_hand_computation_on_aapl(market):
    view = market.view(market.sessions[AS_OF_IDX])
    # Revenue TTM, billions: now 416.161 + 364.357 - 313.695 = 466.823; a year ago 391.035 + 313.695 - 296.105 = 408.625.
    # One quarter earlier 416.161 + 254.940 - 219.659 = 451.442; a year before that 391.035 + 219.659 - 210.328 = 400.366.
    g_now, g_q = 466.823 / 408.625 - 1, 451.442 / 400.366 - 1
    assert hyp.ttm_growth(view, "AAPL", "revenue") == pytest.approx(g_now, rel=1e-9)
    assert hyp.ttm_growth(view, "AAPL", "revenue", quarters_back=1) == pytest.approx(g_q, rel=1e-9)
    assert hyp.revenue_acceleration(view, "AAPL") == pytest.approx(g_now - g_q, rel=1e-9)
    # Diluted EPS TTM: 7.46 + 6.88 - 5.62 = 8.72 against 6.08 + 5.62 - 5.11 = 6.59.
    assert hyp.ttm_growth(view, "AAPL", "eps_diluted") == pytest.approx((8.72 - 6.59) / 6.59, rel=1e-9)


def test_h2_is_unscored_without_two_years_of_filings(market):
    thin = AsOfView("2023-06-01", market.adj_close.loc[:"2023-06-01"], market.adj_open.loc[:"2023-06-01"], market.conn)
    assert np.isnan(hyp.revenue_acceleration(thin, "AAPL"))  # the fixture holds about a year of filings by then
    assert not np.isnan(hyp.revenue_acceleration(market.view(market.sessions[AS_OF_IDX]), "AAPL"))


def test_h2_score_is_the_mean_of_two_zscores(market):
    scores, uni = _scores(market, "h2_accelerating_fundamentals")
    view = market.view(market.sessions[AS_OF_IDX])
    x = pd.DataFrame({"a": [hyp.revenue_acceleration(view, t) for t in uni],
                      "g": [hyp.ttm_growth(view, t, "eps_diluted") for t in uni]}, index=uni).dropna()
    expected = ((x - x.mean()) / x.std(ddof=0)).mean(axis=1)
    pd.testing.assert_series_equal(scores.dropna().sort_index(), expected.sort_index(), check_names=False)


def test_h3_ranks_by_momentum_only_among_positive_eps_growth(market):
    scores, uni = _scores(market, "h3_momentum_confirmed")
    view = market.view(market.sessions[AS_OF_IDX])
    mom = make_signal("momentum_12_1", {})(view.as_of, uni, view)
    assert hyp.ttm_growth(view, "JNJ", "eps_diluted") < 0 and np.isnan(scores["JNJ"])
    assert hyp.ttm_growth(view, "AAPL", "eps_diluted") > 0 and scores["AAPL"] == mom["AAPL"]
    assert all(hyp.ttm_growth(view, t, "eps_diluted") > 0 for t in scores.dropna().index)


def test_h4_market_cap_is_raw_close_times_latest_filed_shares(market):
    view = market.view(market.sessions[AS_OF_IDX])
    close = market.conn.execute("SELECT close FROM price_daily WHERE ticker='AAPL' AND date=?", (view.as_of,)).fetchone()[0]
    assert hyp.latest_filed_shares(view, "AAPL") == (14594180000.0, "2026-07-31")
    assert hyp.market_caps(view, ["AAPL"]).iloc[0] == pytest.approx(close * 14594180000.0)
    on_filing_day = market.view("2026-07-31")  # filed that day: usable from the next session only
    assert hyp.latest_filed_shares(on_filing_day, "AAPL") == (14687356000.0, "2026-05-01")


def test_market_cap_is_continuous_through_a_split_stored_after_the_date(market, tmp_path):
    ex = market.sessions[310]  # after the 2026-07-31 share-count filing
    path = _build(str(tmp_path / "s.db"))
    conn = sqlite3.connect(path)
    # What a later re-fetch stores after a 2:1 split on `ex`: every earlier close halved, the split in the actions.
    conn.execute("UPDATE price_daily SET open = open / 2.0, close = close / 2.0 WHERE ticker = 'AAPL'")
    conn.execute("INSERT INTO price_corporate_actions VALUES ('AAPL', ?, 'split', 2.0)", (ex,))
    conn.commit()
    conn.close()
    later = load_market(path)
    for idx in (200, 309, 310, 340):  # before, the day before, on and after the ex-date
        d = market.sessions[idx]
        a, b = hyp.market_caps(market.view(d), ["AAPL"]).iloc[0], hyp.market_caps(later.view(d), ["AAPL"]).iloc[0]
        assert a == pytest.approx(b, rel=1e-12), d


def test_low_volatility_matches_numpy(market):
    scores, _ = _scores(market, "low_volatility", params={"window": 100})
    px = market.adj_close.iloc[AS_OF_IDX - 100:AS_OF_IDX + 1]["KO"].to_numpy()
    assert scores["KO"] == pytest.approx(-np.std(px[1:] / px[:-1] - 1, ddof=1), rel=1e-12)


# ---- look-ahead guard ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["h1_quality_fair_price", "h2_accelerating_fundamentals", "h3_momentum_confirmed",
                                  "h4_market_cap_leaders", "low_volatility"])
@pytest.mark.parametrize("idx", [150, AS_OF_IDX])
def test_every_candidate_passes_the_lookahead_guard(market, name, idx):
    assert_no_lookahead(make_signal(name, {}), market, market.sessions[idx], market.universe(idx))


# ---- H4 schedule -----------------------------------------------------------------------------------

def test_annual_schedule_uses_the_last_session_of_each_year():
    s = ["2010-12-30", "2010-12-31", "2011-01-03", "2011-06-01", "2011-12-30", "2012-01-03", "2012-06-01"]
    assert annual_reviews(s, start=2, end=6) == [1, 4]  # signal 2010-12-31 (start - 1), then 2011-12-30
    assert annual_reviews(s, start=2, end=4) == [1]


def test_h4_buys_the_five_largest_at_the_next_open_and_holds_the_year(market):
    reviews = annual_reviews(market.sessions, 0, len(market.sessions))
    assert [market.sessions[i] for i in reviews] == ["2025-12-31"]
    sig = make_signal("h4_market_cap_leaders", {})
    sched = build_schedule(market, reviews, Rules(5, 5, 20, 0),
                           lambda s, u: sig(market.sessions[s], u, market.view(market.sessions[s])))
    assert [market.sessions[e] for e in sched.exec_idx] == ["2026-01-02"]
    caps = hyp.market_caps(market.view("2025-12-31"), market.universe(reviews[0]))
    assert sched.targets[0] == sorted(caps.sort_values(ascending=False).index[:5])


def test_h4_is_registered_with_five_names_and_no_buffer(market, tmp_path):
    with pytest.raises(ValueError, match="registered"):
        run_bench(market, BenchConfig("h4_market_cap_leaders", n_random=2), tmp_path / "l.jsonl")


def test_run_reports_coverage_exclusions_and_years_beating_both(market, tmp_path):
    out = run_bench(market, BenchConfig("h1_quality_fair_price", n_random=3), tmp_path / "l.jsonl")
    assert 0 < out["coverage"]["mean_scored"] < out["coverage"]["mean_universe"]
    assert "pct_years_beating_both" in out["dev"] and out["exclusions"] == []


def test_h5_drops_the_top_decile_of_short_float_and_keeps_unknowns():
    universe = [f"T{i:02d}" for i in range(20)]
    sf = pd.Series({t: (i + 1) / 100.0 for i, t in enumerate(universe) if t != "T05"})
    kept = hyp.h5_fade_the_crowd(universe[:], sf, universe)
    assert "T19" not in kept and "T18" not in kept  # 90th percentile of the 19 values is 0.182
    assert "T17" in kept and "T05" in kept and len(kept) == 18


# ---- data-quality filter -----------------------------------------------------------------------------

def _rows(ticker, closes, volumes=None, start="2012-01-02"):
    dates = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": closes,
                         "volume": 1e6 if volumes is None else volumes})


def test_five_zero_volume_sessions_in_a_row_are_invalid_and_four_are_not():
    five = _rows("A", [10.0] * 12, [1e6, 0, 0, 0, 0, 0, 1e6, 1e6, 0, 0, 0, 0])
    out = quality.invalid_rows(five, None)
    assert out["rule"].eq("zero_volume_run").all() and list(out["date"]) == list(five["date"].iloc[1:6])


def test_seam_between_backfill_and_stored_series_invalidates_the_backfilled_spell():
    a = _rows("A", [10.4, 10.5, 10.6, 70.9, 71.2, 71.0])  # CTVA-like: backfill near 10.6, stored series from 70.9
    out = quality.invalid_rows(a, a["date"].iloc[3])
    assert list(out["date"]) == list(a["date"].iloc[:3]) and out["rule"].eq("seam").all()
    ok = _rows("B", [10.0, 10.5, 11.0, 13.7, 13.8, 13.9])  # 24.5% across the seam: valid
    assert quality.invalid_rows(ok, ok["date"].iloc[3]).empty
    assert quality.invalid_rows(_rows("C", [10.0, 10.1]), None).empty


def test_seven_big_moves_in_one_calendar_year_mark_that_year_only():
    erratic = [10.0, 14.0] * 4 + [14.0] * 10  # seven moves above 25% (10 -> 14 is 40%, 14 -> 10 is 29%)
    out = quality.invalid_rows(_rows("A", erratic), None)
    assert len(out) == len(erratic) and out["rule"].eq("erratic_year").all()
    assert quality.invalid_rows(_rows("B", [10.0, 14.0] * 3 + [14.0] * 10), None).empty  # six moves
    assert quality.invalid_rows(_rows("C", erratic, start="2012-12-26"), None).empty  # split across two years


def test_filtered_days_leave_the_universe_for_everyone(market, tmp_path):
    path = _build(str(tmp_path / "q.db"))
    conn = sqlite3.connect(path)
    dates = market.sessions[100:106]
    conn.execute(f"UPDATE price_daily SET volume = 0 WHERE ticker = 'KO' AND date IN ({','.join('?' * 6)})", dates)
    conn.commit()
    conn.close()
    filtered = load_market(path)
    assert "KO" in market.universe(99) and "KO" in filtered.universe(99)
    assert all("KO" not in filtered.universe(i) and "KO" in market.universe(i) for i in range(100, 106))
    assert "KO" in filtered.universe(106)
    (row,) = filtered.exclusion_summary(0, 200)
    assert (row["ticker"], row["rule"], row["member_days"], row["first"], row["last"]) == (
        "KO", "zero_volume_run", 6, dates[0], dates[-1])


# ---- share-count scale check and split ledger -----------------------------------------------------

def test_share_count_with_a_wrong_scale_is_rejected_and_the_next_older_is_used():
    history = [("2012-09-24", 4.8e15), ("2012-06-26", 4.88e9), ("2012-03-30", 4.9e9), ("2011-12-20", 4.95e9),
               ("2011-09-26", 4.96e9)]
    rejected: list = []
    assert accept_share_count(history, [], rejected) == (4.88e9, "2012-06-26")
    assert rejected == [("2012-09-24", 4.8e15)]
    assert accept_share_count(history[1:], [], None) == (4.88e9, "2012-06-26")  # a clean series is untouched


def test_a_split_between_two_filings_is_not_a_scale_error():
    history = [("2012-10-25", 1.125e10), ("2012-07-26", 2.25e9), ("2012-04-26", 2.26e9), ("2012-01-26", 2.27e9)]
    assert accept_share_count(history, [("2012-08-13", 5.0)]) == (1.125e10, "2012-10-25")
    assert accept_share_count(history, []) == (2.25e9, "2012-07-26")  # without the split on record it is rejected


def test_a_count_with_too_little_history_is_accepted_unchecked():
    assert accept_share_count([("2012-09-24", 4.8e15), ("2012-06-26", 4.88e9)], []) == (4.8e15, "2012-09-24")


def test_split_ledger_adds_splits_missing_from_the_action_table(tmp_path):
    path = _build(str(tmp_path / "led.db"))
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE price_series_revisions (ticker TEXT PRIMARY KEY, evidence_json TEXT)")
    conn.execute("INSERT INTO price_series_revisions VALUES ('KO', ?)", (
        '{"vendor_action_ledger":[{"action_date":"2020-04-23","action_type":"reverse_split","value":0.02},'
        '{"action_date":"2020-05-01","action_type":"dividend","value":0.4}]}',))
    conn.commit()
    conn.close()
    splits = load_market(path).splits
    assert ("2020-04-23", 0.02) in splits["KO"] and not any(d == "2020-05-01" for d, _ in splits["KO"])
    assert splits["KO"] == sorted(splits["KO"])

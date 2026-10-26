"""Every substitution, rejection and skip in the research bench is counted in the run report or fails closed."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from tests.test_research_bench import _build_db, _fixture_frames, _market_with
from trader_koo.fundamentals import config as fcfg
from trader_koo.fundamentals.schema import ensure_sec_schema
from trader_koo.research.bench import hypotheses as hyp
from trader_koo.research.bench import ledger, report
from trader_koo.research.bench.__main__ import main as bench_main
from trader_koo.research.bench.data import AsOfView, Market, load_market, load_splits
from trader_koo.research.bench.diagnostics import Diagnostics
from trader_koo.research.bench.portfolio import Result, Rules, Schedule, build_schedule, equal_weight_schedule, simulate
from trader_koo.research.bench.run import BenchConfig, run_bench
from trader_koo.research.settings import get as settings


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> str:
    return _build_db(str(tmp_path_factory.mktemp("fallbacks") / "research.db"), *_fixture_frames())


@pytest.fixture(scope="module")
def market(db_path):
    return load_market(db_path)


def _cfg(**kw) -> BenchConfig:
    return BenchConfig.from_settings("equal_weight", n_random=3, **kw)


# ---- the run report ---------------------------------------------------------------------------

def test_run_report_lists_every_substitution_with_a_count(market, tmp_path):
    out = run_bench(market, _cfg(), tmp_path / "l.jsonl")
    sub = out["data"]["substitutions"]
    for key in ("leading_reviews_with_no_selection_skipped", "reviews_with_no_target_held_as_cash",
                "held_names_sold_for_having_no_score", "held_name_days_valued_flat_at_last_close",
                "held_names_filled_at_last_known_open", "candidate_cash_periods",
                "equal_weight_entries_skipped_no_open", "equal_weight_held_name_days_valued_flat"):
        assert isinstance(sub[key], int)
    assert "Substitutions, rejections and skips" in out["markdown"]
    assert all("hit_rate_periods_without_benchmark" in out[w] for w in ("dev",))


def test_ledger_entries_of_a_run_carry_the_settings_hash(market, tmp_path):
    path = tmp_path / "l.jsonl"
    run_bench(market, _cfg(), path)
    entries = ledger.read(path)
    assert [e["event"] for e in entries] == ["seal", "run"]
    assert {e["settings_sha256"] for e in entries} == {settings().sha256}


def test_replay_refuses_a_ledger_entry_run_under_other_settings(db_path, market, tmp_path, capsys):
    path = tmp_path / "l.jsonl"
    run_bench(market, _cfg(), path)
    entries = ledger.read(path)
    trial = entries[-1]["trial_id"]
    entries[-1]["settings_sha256"] = "0" * 64
    path.write_text("".join(ledger.canonical(e) + "\n" for e in entries))
    assert bench_main(["replay", "--db", db_path, "--trial", trial, "--ledger", str(path)]) == 2
    assert "refused" in capsys.readouterr().err
    del entries[-1]["settings_sha256"]  # a v1 entry written before hashing: replays, with a note
    path.write_text("".join(ledger.canonical(e) + "\n" for e in entries))
    assert bench_main(["replay", "--db", db_path, "--trial", trial, "--ledger", str(path)]) == 0
    assert "predates settings hashing" in capsys.readouterr().err


# ---- loading ------------------------------------------------------------------------------------

def test_market_inputs_are_required_not_defaulted(market):
    with pytest.raises(TypeError):
        Market(market.adj_open, market.adj_close, market.conn)


def test_missing_split_tables_fail_closed(tmp_path):
    conn = sqlite3.connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, value REAL)")
    with pytest.raises(LookupError, match="price_series_revisions"):
        load_splits(conn)


def test_flagged_returns_filled_with_zero_are_counted(tmp_path):
    prices, actions = _fixture_frames()
    p = prices.copy()
    day = sorted(p["date"].unique())[120]
    p.loc[(p["ticker"] == "KO") & (p["date"] == day), "close"] = np.nan
    mkt = _market_with(tmp_path, "nan.db", p, actions)
    d = mkt.load_diag.as_dict()
    assert d["nan_returns_filled_as_0pct_on_flagged_rows"] == 1  # the bad close
    assert d["nan_returns_filled_as_0pct_after_invalid_close"] == 1  # the day after it has no prior close


def test_spinoffs_that_cannot_be_applied_are_listed(tmp_path):
    prices, actions = _fixture_frames()
    dates = sorted(prices["date"].unique())
    csv = tmp_path / "spin.csv"
    csv.write_text(
        "parent_ticker,ex_date,spun_off,kind,date_basis,price_effect,source_url\n"
        f"ZZZZ,{dates[150]},A,spinoff,regular_way_start,raw_drop,https://example.com/a\n"
        "AAPL,1999-01-04,B,spinoff,regular_way_start,raw_drop,https://example.com/b\n")
    mkt = load_market(_build_db(str(tmp_path / "s.db"), prices, actions), csv)
    listed = mkt.load_diag.as_dict()["spinoffs_not_applied"]
    assert listed["distinct"] == 2
    assert {tuple(i)[2] for i in listed["items"]} == {"parent has no prices", "ex-date outside the loaded sessions"}


def test_a_skipped_quality_rule_is_reported(tmp_path):
    db = _build_db(str(tmp_path / "q.db"), *_fixture_frames())
    conn = sqlite3.connect(db)
    conn.execute("UPDATE price_daily SET adjustment_basis='split_adjusted_price_only'")  # no unversioned rows
    conn.commit()
    conn.close()
    notes = load_market(db).load_diag.as_dict()["quality_rules_not_applied"]
    assert notes["distinct"] == 1 and "R2 seam" in notes["items"][0]


# ---- portfolio ----------------------------------------------------------------------------------

def test_held_names_with_a_missing_close_or_open_are_counted(tmp_path):
    mkt = _market_with(tmp_path, "gap.db", *_fixture_frames())
    names = mkt.universe(10)[:3]
    col = mkt.columns[names[0]]
    mkt.adj_close.iloc[14:18, col] = np.nan  # four held sessions without a close
    mkt.adj_open.iloc[11, col] = np.nan  # the fill session has no open
    mkt.close_ff, mkt.open_ff = mkt.adj_close.ffill(), mkt.adj_open.ffill()
    res = simulate(mkt, Schedule([11, 20], [names, names]), 40, 0.0)
    assert res.flat_valued_name_days == 4 and res.stale_open_fills == 1 and res.cash_periods == 0
    assert simulate(mkt, Schedule([11, 20], [names, []]), 40, 0.0).cash_periods == 1


def test_unscored_holdings_and_missing_opens_are_counted(tmp_path):
    mkt = _market_with(tmp_path, "unscored.db", *_fixture_frames())
    rules = Rules(top_n=3, buffer_n=5, review_every=10, min_hold=100)
    first_pick: list[str] = []

    def score(s, uni):
        scores = pd.Series(np.arange(len(uni), dtype=float), index=uni)
        if not first_pick:
            first_pick.extend(scores.sort_values(ascending=False).index[:3])
            return scores
        return scores.where(scores.index != first_pick[0])  # one holding loses its score

    sched = build_schedule(mkt, [110, 120], rules, score, 200)
    assert sched.dropped_unscored_held == 1  # sold although inside its minimum hold, and counted
    uni = mkt.universe(110)
    mkt.adj_open.iloc[111, mkt.columns[uni[0]]] = np.nan
    assert equal_weight_schedule(mkt, [110], 200).dropped_no_open == 1


def test_hit_rate_counts_periods_it_cannot_compare():
    cand = Result(0, np.array([]), np.array([], int), np.array([]), [np.array([0.1]), np.array([0.2])], [5, 6])
    ew = Result(0, np.array([]), np.array([], int), np.array([]), [np.array([0.0])], [5])
    assert report.hit_rate(cand, ew, 0, 10) == (1.0, 1)


# ---- signals ------------------------------------------------------------------------------------

def _share_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_sec_schema(conn)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, close REAL, adjustment_basis TEXT)")
    tag = fcfg.CONCEPTS["shares_outstanding"][0][1]
    for ticker, cik in (("AAA", 1), ("BBB", 2)):
        conn.execute("INSERT INTO sec_cik_map VALUES (?, ?, 'x', 'now')", (ticker, cik))
    rows = [(1, "2024-01-10", 1.0e9), (1, "2024-04-10", 1.0e9), (1, "2024-07-10", 1.0e9),
            (1, "2024-10-10", 1.0e12),   # off-scale
            (1, "2024-11-10", 0.0),      # not a share count
            (2, "2024-06-01", 5.0e8)]    # one filing: cannot be checked
    for i, (cik, filed, value) in enumerate(rows):
        conn.execute("INSERT INTO sec_fundamentals VALUES (?,?,?,?,?,?,2024,'Q',?,'10-Q',?,?,NULL)",
                     (cik, "us-gaap", tag, "shares", "", filed, value, f"a{i}", filed))
    conn.execute("INSERT INTO price_daily VALUES ('AAA', '2025-01-02', 10.0, NULL)")
    return conn


def _view(conn, diag) -> AsOfView:
    empty = pd.DataFrame()
    return AsOfView("2025-01-02", empty, empty, conn, {}, diag)


def test_rejected_and_ignored_share_counts_are_listed():
    diag = Diagnostics()
    assert hyp.latest_filed_shares(_view(_share_db(), diag), "AAA") == (1.0e9, "2024-07-10")
    d = diag.as_dict()
    assert d["share_counts_rejected_off_scale"]["items"] == [["AAA", "2024-10-10", 1.0e12]]
    assert d["share_counts_nonpositive_ignored"]["items"] == [["AAA", "2024-11-10", 0.0]]
    assert hyp.latest_filed_shares(_view(_share_db(), diag), "BBB") == (5.0e8, "2024-06-01")
    assert diag.as_dict()["share_counts_accepted_unchecked"] == 1


def test_market_cap_nan_has_two_counted_reasons():
    diag = Diagnostics()
    caps = hyp.market_caps(_view(_share_db(), diag), ["AAA", "BBB", "CCC"])
    assert caps["AAA"] == pytest.approx(10.0 * 1.0e9) and caps[["BBB", "CCC"]].isna().all()
    d = diag.as_dict()
    assert d["market_cap_nan_no_close_on_date"] == 1  # BBB has shares, no close
    assert d["market_cap_nan_no_usable_share_count"] == 1  # CCC has neither


def test_names_missing_a_signal_input_are_counted_by_input():
    diag = Diagnostics()
    x = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.inf], "b": [1.0, 2.0, np.nan, 4.0]}, index=list("wxyz"))
    kept = hyp.complete_rows(x, diag, "h1")
    assert list(kept.index) == ["w"]
    d = diag.as_dict()
    assert d["h1_input_missing:a"] == 2 and d["h1_input_missing:b"] == 1  # the inf became missing
    assert d["h1_infinite_input_set_to_missing"] == 1 and d["h1_unscored_missing_input"] == 3


def test_diagnostics_export_is_json_serialisable():
    diag = Diagnostics()
    diag.count("a", 2)
    diag.note("b", ("T", "2024-01-01", 1.5))
    diag.note("b", ("T", "2024-01-01", 1.5))
    assert json.loads(json.dumps(diag.as_dict())) == {"a": 2, "b": {"distinct": 1, "items": [["T", "2024-01-01", 1.5]]}}

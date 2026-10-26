"""Hypotheses v3: look-ahead guard per hypothesis, hand-checked events on real rows, independent reference signals.

The fixture is real SPY, QQQ and ^VIX bars from the research database, 2010-01-04 to 2012-12-31.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trader_koo.research.bench import ledger, v3_data, v3_gap, v3_price, v3_run, v3_vol
from trader_koo.research.bench.eventstudy import evaluate
from trader_koo.research.bench.v3_data import Frame, load_frame

FIX = Path(__file__).parent / "fixtures" / "research_bench"
BEYOND = "2013-01-01"


def _db(path: Path, extra_vix: bool = False, drop_qqq: str | None = None) -> str:
    bars = pd.read_csv(FIX / "index_bars.csv.gz")
    if extra_vix:
        bars = pd.concat([bars, bars[(bars.ticker == "^VIX") & (bars.date == "2010-01-08")].assign(date="2010-01-09")])
    if drop_qqq:
        bars = bars[~((bars.ticker == "QQQ") & (bars.date == drop_qqq))]
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,"
                 " adjustment_basis TEXT, basis_status TEXT)")
    conn.execute("CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, value REAL)")
    bars.assign(adjustment_basis="split_adjusted_price_only", basis_status="verified").to_sql(
        "price_daily", conn, if_exists="append", index=False)
    pd.read_csv(FIX / "index_dividends.csv.gz").to_sql("price_corporate_actions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture(scope="module")
def frame(tmp_path_factory) -> Frame:
    return load_frame(_db(tmp_path_factory.mktemp("v3") / "r.db"), BEYOND)


@pytest.fixture(scope="module")
def specs(frame):
    out = {s.key: s for s in v3_run.build_specs(frame)}
    assert "H10" in out  # the sourced FOMC list covers 2010-2023, so H10 is not blocked
    return out


def at(frame: Frame, date: str) -> int:
    return frame.dates.index(date)


def event_outcome(spec, i: int) -> float:
    return float(spec.outcome[spec.stratum[i], i])


# ---- data ----------------------------------------------------------------------------------------------------------

def test_frame_stops_before_the_boundary_and_never_reads_later_rows(tmp_path, frame):
    cut = load_frame(_db(tmp_path / "a.db"), "2011-06-01")
    assert cut.dates[-1] < "2011-06-01" and cut.dates == frame.dates[:cut.n]
    for t in ("SPY", "QQQ", "^VIX"):
        for f, a in cut.bars[t].items():
            assert np.array_equal(a, frame.bars[t][f][:cut.n], equal_nan=True)
    assert frame.basis["adjustment_basis"] == {"split_adjusted_price_only": 3 * 754}


def test_rows_that_are_dropped_or_empty_are_counted(tmp_path):
    f = load_frame(_db(tmp_path / "b.db", extra_vix=True, drop_qqq="2010-03-02"), BEYOND)
    d = f.diag.as_dict()
    assert d["^VIX_rows_on_dates_without_a_SPY_session_dropped"] == 1
    assert d["QQQ_sessions_without_a_row"] == 1 and np.isnan(f.bars["QQQ"]["close"][at(f, "2010-03-02")])


def test_fomc_list_is_complete_and_sourced():
    dates, missing = v3_data.load_fomc()
    assert missing == [] and len(dates) == 111
    rows = pd.read_csv(v3_data.FOMC_FILE)
    assert rows["source_url"].str.startswith("https://www.federalreserve.gov/monetarypolicy/").all()
    assert set(rows[rows.kind != "scheduled"].kind) >= {"unscheduled_conference_call", "unscheduled_meeting", "cancelled"}


def test_incomplete_fomc_list_blocks_h10(tmp_path):
    short = tmp_path / "fomc.csv"
    rows = pd.read_csv(v3_data.FOMC_FILE)
    rows[rows.year != 2015].to_csv(short, index=False)
    f = load_frame(_db(tmp_path / "c.db"), BEYOND, short)
    assert f.fomc is None and f.fomc_missing == [2015] and v3_price.h10(f) is None
    out = v3_run._status("H10", {}, 1.0, 0.05, {})
    assert out["status"] == "blocked"


# ---- hand-checked events on real rows --------------------------------------------------------------------------------

def test_h1_vix_drop_2010_05_10(frame, specs):
    # VIX closed 40.95 on 2010-05-07 and 28.84 on 2010-05-10 (-29.6%); SPY 116.16 on 05-10 and 117.45 on 05-12 (not lower)
    i = at(frame, "2010-05-10")
    assert specs["H1"].signal[i] and event_outcome(specs["H1"], i) == 0.0
    assert not specs["H1"].signal[at(frame, "2010-05-11")]  # -1.8%: inside no band, not a signal


def test_h2_vix_jump_2010_05_06(frame, specs):
    # VIX 32.80 on 05-06 (+31.7%); SPY 112.94 -> 116.16 on 05-10 (t+2), VIX 32.80 -> 28.84; t+1 (05-07): SPY 111.26, VIX 40.95
    i = at(frame, "2010-05-06")
    assert specs["H2a"].signal[i] and event_outcome(specs["H2a"], i) == 1.0 and event_outcome(specs["H2b"], i) == 1.0
    assert event_outcome(specs["H2a-sec"], i) == 0.0 and event_outcome(specs["H2b-sec"], i) == 0.0
    assert specs["H2a"].claim == 0.85 and specs["H2b"].claim is None


def test_h7_divergence_2010_01_28(frame, specs):
    # SPY 108.57 is the lowest close of the last 10 sessions; VIX 23.73 is below its 10-session high 27.31 (01-22);
    # SPY closes at 110.38 on 02-02 (t+3)
    i = at(frame, "2010-01-28")
    assert specs["H7"].signal[i] and event_outcome(specs["H7"], i) == 1.0 and not specs["H7"].baseline[0, i]


def test_h10_down_fomc_day_2010_08_10(frame, specs):
    # decision day 2010-08-10: SPY 112.99 -> 112.38 (-0.54%); next close 109.30 is lower, so the flip did not happen
    i = at(frame, "2010-08-10")
    assert specs["H10"].signal[i] and event_outcome(specs["H10"], i) == 0.0 and specs["H10"].claim == 0.83
    assert not specs["H10"].baseline[0, i]
    assert not specs["H10"].signal[at(frame, "2010-08-09")]  # not a decision day


def test_h11_friday(frame, specs):
    i = at(frame, "2010-01-08")  # VIX 19.06 -> 18.13 on a Friday
    assert specs["H11"].signal[i] and event_outcome(specs["H11"], i) == 1.0
    assert not specs["H11"].signal[i - 1] and specs["H11"].baseline[0, i - 1] and not specs["H11"].baseline[0, i]


def test_h12_small_up_gap_filled_the_same_day(frame):
    # 2010-01-11 opens 115.08 above the prior high 114.62 (prior close 114.57, gap +0.45%); the low 114.24 is under 114.57
    g = v3_gap.gap_arrays(frame)
    i = at(frame, "2010-01-11")
    assert g["sig_up"][i] and not g["sig_dn"][i] and g["out_up"][i] == 1.0
    assert g["dist_up"][i] == pytest.approx((115.080002 - 114.57) / 115.080002, rel=1e-4)


def test_h12_excludes_ex_dividend_sessions_and_counts_them(frame):
    g = v3_gap.gap_arrays(frame)
    ex = frame.ex_dividend["SPY"]
    assert ex.sum() > 0 and not (g["sig_up"] & ex).any() and not (g["sig_dn"] & ex).any() and not (g["nongap"] & ex).any()
    assert g["excluded_ex_dividend"] >= 0


def test_h13_quiet_days_2010_12_22(frame, specs):
    # ranges 0.48% (12-21) and 0.33% (12-22) of the previous close, both under 0.60%; pivot = high of sessions 4-5 = 126.20
    # (12-29); closes on sessions 8 and 9 are 126.98 and 127.64, above the trigger close 125.78; lowest low in sessions
    # 8-19 is 126.15, nowhere near 126.20 x (1 - 0.0236) = 123.22
    i = at(frame, "2010-12-22")
    assert specs["H13-SPY-a"].signal[i] and event_outcome(specs["H13-SPY-a"], i) == 0.0
    assert event_outcome(specs["H13-SPY-b"], i) == 0.0 and event_outcome(specs["H13-SPY-a-both"], i) == 0.0
    assert specs["H13-SPY-a"].stratum[i] == 11  # December
    assert not specs["H13-SPY-a"].baseline[11, i]  # a trigger session is not in its own baseline


def test_h6_topping_tail_2010_03_31(frame, specs):
    # open 116.95 high 117.52 low 116.61 close 117.00: upper wick 0.52 >= 2 x body (0.05) and >= half the range (0.91);
    # close in the lower half; 117.00 is within 2% of the 60-session high 117.41; close two sessions later 118.76 is higher
    i = at(frame, "2010-03-31")
    s = specs["H6-SPY-top"]
    assert s.signal[i] and event_outcome(s, i) == 0.0 and not s.baseline[0, i]


# ---- independent reference implementations ---------------------------------------------------------------------------

def _ema(x, n):
    out, prev = [np.nan] * len(x), None
    for i, v in enumerate(x):
        prev = v if prev is None else prev + 2.0 / (n + 1) * (v - prev)
        out[i] = prev if i >= n - 1 else np.nan
    return out


def _sma(x, n):
    return [np.mean(x[i - n + 1:i + 1]) if i >= n - 1 else np.nan for i in range(len(x))]


def _bars(frame, t):
    return {k: list(v) for k, v in frame.bars[t].items()}


def test_vix_change_signals_match_a_plain_loop(frame, specs):
    v = _bars(frame, "^VIX")["close"]
    r = [np.nan] + [v[i] / v[i - 1] - 1 for i in range(1, len(v))]
    assert [i for i, x in enumerate(r) if x <= -0.15] == list(np.flatnonzero(specs["H1"].signal))
    assert [i for i, x in enumerate(r) if x >= 0.20] == list(np.flatnonzero(specs["H2a"].signal))
    assert [i for i, x in enumerate(r) if -0.15 <= x <= -0.05] == list(np.flatnonzero(specs["H1"].baseline[0]))
    assert [i for i, x in enumerate(r) if 0.08 <= x < 0.20] == list(np.flatnonzero(specs["H2a"].baseline[0]))


def test_h7_matches_a_plain_loop(frame, specs):
    c, v = _bars(frame, "SPY")["close"], _bars(frame, "^VIX")["close"]
    sig, base = [], []
    for i in range(9, len(c)):
        if c[i] <= min(c[i - 9:i + 1]):
            (sig if v[i] < max(v[i - 9:i + 1]) else base).append(i)
    assert sig == list(np.flatnonzero(specs["H7"].signal)) and base == list(np.flatnonzero(specs["H7"].baseline[0]))


def test_h8_and_h9_match_a_plain_loop(frame, specs):
    b = _bars(frame, "SPY")
    c, vol = b["close"], b["volume"]
    avgs = [_sma(c, 5), _ema(c, 8), _sma(c, 10), _sma(c, 20), _ema(c, 21), _sma(c, 50)]
    above = [i for i in range(len(c)) if all(a[i] == a[i] for a in avgs) and all(c[i] > a[i] for a in avgs)]
    below4 = [i for i in range(len(c)) if all(a[i] == a[i] for a in avgs) and sum(c[i] < a[i] for a in avgs) >= 4]
    assert above == list(np.flatnonzero(specs["H8-SPY"].signal)) and below4 == list(np.flatnonzero(specs["H8-SPY"].baseline[0]))
    e8, s5, va = _ema(c, 8), _sma(c, 5), _sma(vol, 20)
    prev = set(above)
    sig = [i for i in range(1, len(c)) if c[i] < e8[i] and c[i] < s5[i] and vol[i] > va[i] and (i - 1) in prev]
    base = [i for i in range(1, len(c)) if c[i] < e8[i] and c[i] < s5[i] and (i - 1) not in prev]
    assert sig == list(np.flatnonzero(specs["H9"].signal)) and base == list(np.flatnonzero(specs["H9"].baseline[0]))
    assert len(sig) > 0 and len(above) > 0


def test_h3_compression_matches_a_plain_loop(frame, specs):
    v = _bars(frame, "^VIX")["close"]
    width = [4.0 * np.std(v[i - 19:i + 1], ddof=1) / np.mean(v[i - 19:i + 1]) if i >= 19 else np.nan for i in range(len(v))]
    comp = [i for i in range(len(v)) if i >= 19 + 251 and width[i] <= np.quantile(width[i - 251:i + 1], 0.10)]
    assert comp == list(np.flatnonzero(specs["H3"].signal))
    tier = {i: int(v[i] > np.quantile(v[i - 251:i + 1], 1 / 3)) + int(v[i] > np.quantile(v[i - 251:i + 1], 2 / 3)) for i in comp}
    assert tier == {i: int(specs["H3"].stratum[i]) for i in comp}


def test_h5_close_break_matches_a_plain_loop(frame, specs):
    b = _bars(frame, "SPY")
    c = b["close"]

    def qual(i):  # range of closes over i-20..i-1 within 5% of their mean
        w = c[i - 20:i]
        return (max(w) - min(w)) / np.mean(w) <= 0.05

    longs = [i for i in range(31, len(c)) if all(qual(j) for j in range(i - 10, i + 1)) and c[i] > max(c[i - 20:i])]
    shorts = [i for i in range(31, len(c)) if all(qual(j) for j in range(i - 10, i + 1)) and c[i] < min(c[i - 20:i])]
    s = specs["H5-A"]
    assert sorted(longs + shorts) == list(np.flatnonzero(s.signal))
    assert all(s.stratum[i] == 0 for i in longs) and all(s.stratum[i] == 1 for i in shorts)
    assert len(longs) + len(shorts) > 0


def test_h5_wick_break_closes_back_inside(frame, specs):
    b = _bars(frame, "SPY")
    c, h = b["close"], b["high"]
    sa, sb = specs["H5-A"], specs["H5-B"]
    for i in np.flatnonzero(sb.signal):
        top = max(c[i - 20:i])
        bottom = min(c[i - 20:i])
        assert (h[i] > top and c[i] <= top) or (b["low"][i] < bottom and c[i] >= bottom)
        assert not sa.signal[i]


def test_h13_quiet_signal_matches_a_plain_loop(frame, specs):
    for t in ("SPY", "QQQ"):
        b = _bars(frame, t)
        q = [False] + [(b["high"][i] - b["low"][i]) / b["close"][i - 1] < 0.006 for i in range(1, len(b["close"]))]
        sig = [i for i in range(2, len(q)) if q[i] and q[i - 1]]
        assert sig == list(np.flatnonzero(specs[f"H13-{t}-a"].signal)) and len(sig) > 0
        ended = [i for i in sig if i + 1 < len(q) and not q[i + 1]]
        assert ended == list(np.flatnonzero(specs[f"H13-{t}-a-lastday"].signal))


def test_h6_tails_match_a_plain_loop(frame, specs):
    for t in ("SPY", "QQQ"):
        b = _bars(frame, t)
        o, h, l, c = b["open"], b["high"], b["low"], b["close"]
        top, bottom = [], []
        for i in range(59, len(c)):
            rng, body = h[i] - l[i], abs(c[i] - o[i])
            if rng <= 0:
                continue
            up, lo = h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i]
            if up >= 2 * body and up >= 0.5 * rng and c[i] - l[i] <= 0.5 * rng and c[i] >= 0.98 * max(c[i - 59:i + 1]):
                top.append(i)
            if lo >= 2 * body and lo >= 0.5 * rng and c[i] - l[i] >= 0.5 * rng and c[i] <= 1.02 * min(c[i - 59:i + 1]):
                bottom.append(i)
        assert top == list(np.flatnonzero(specs[f"H6-{t}-top"].signal))
        assert bottom == list(np.flatnonzero(specs[f"H6-{t}-bottom"].signal))


def test_h4_signal_requires_both_compressions(frame, specs):
    s = specs["H4"]
    both = v3_vol.compression(frame.bars["^VIX"]["close"], 0.10) & v3_vol.compression(frame.bars["SPY"]["close"], 0.10)
    assert np.array_equal(s.signal & (s.stratum >= 0) & both, s.signal & both)
    assert not (s.signal & ~both).any()


# ---- look-ahead guard --------------------------------------------------------------------------------------------------

def _corrupt(frame: Frame, t: int, seed: int = 0) -> Frame:
    rng = np.random.default_rng(seed)
    bars = {tk: {f: np.where(np.arange(len(a)) > t, a * rng.uniform(0.2, 5.0, len(a)), a) for f, a in d.items()}
            for tk, d in frame.bars.items()}
    return dataclasses.replace(frame, bars=bars)


def _visible(spec, t: int):
    sig = spec.signal[:t + 1]
    return sig, np.where(sig, spec.stratum[:t + 1], -2), spec.baseline[:, :t + 1]


@pytest.mark.parametrize("t", [300, 480, 640])
def test_no_signal_or_baseline_membership_uses_data_after_t(frame, specs, t):
    pert = {s.key: s for s in v3_run.build_specs(_corrupt(frame, t))}
    assert pert.keys() == specs.keys()
    leaky = []
    for key, s in specs.items():
        a, b = _visible(s, t), _visible(pert[key], t)
        if not all(np.array_equal(x, y) for x, y in zip(a, b)):
            leaky.append(key)
    # the published last-session trigger of H13 needs the next session by construction; it is labelled a variant
    assert set(leaky) <= {"H13-SPY-a-lastday", "H13-SPY-b-lastday", "H13-QQQ-a-lastday", "H13-QQQ-b-lastday"}


def test_guard_sees_the_published_convention_look_ahead(frame, specs):
    """Negative control: the last-session trigger of H13 is only known once the next session is, and the guard finds it."""
    t = at(frame, "2010-12-22")  # a quiet pair ends here only if the next session is not quiet
    seen = False
    for seed in range(5):
        pert = {s.key: s for s in v3_run.build_specs(_corrupt(frame, t, seed))}
        seen |= not np.array_equal(specs["H13-SPY-a-lastday"].signal[:t + 1], pert["H13-SPY-a-lastday"].signal[:t + 1])
        assert np.array_equal(specs["H13-SPY-a"].signal[:t + 1], pert["H13-SPY-a"].signal[:t + 1])
    assert seen


@pytest.mark.parametrize("t", [300, 480, 640])
def test_gap_signals_and_baseline_use_data_through_t_only(frame, t):
    a, b = v3_gap.gap_arrays(frame), v3_gap.gap_arrays(_corrupt(frame, t))
    for k in ("sig_up", "sig_dn", "nongap"):
        assert np.array_equal(a[k][:t + 1], b[k][:t + 1])


def test_outcomes_of_early_sessions_do_not_depend_on_data_far_beyond_their_window(frame, specs):
    pert = {s.key: s for s in v3_run.build_specs(_corrupt(frame, 400))}
    # H1's outcome reaches t+2: sessions up to 397 cannot see session 401+
    assert np.array_equal(specs["H1"].outcome[:, :398], pert["H1"].outcome[:, :398], equal_nan=True)


# ---- end to end ----------------------------------------------------------------------------------------------------------

def test_run_appends_one_ledger_entry_and_never_creates_a_seal(tmp_path, frame):
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(LookupError, match="no seal entry"):
        v3_run.run_v3("unused.db", path, frame)
    ledger.append(path, {"event": "seal", "boundary_date": "2012-10-01"})
    cut = load_frame(_db(tmp_path / "d.db"), "2012-10-01")
    out = v3_run.run_v3("unused.db", path, cut)
    entries = ledger.read(path)
    assert [e["event"] for e in entries] == ["seal", "event_study_run"]
    run = entries[1]
    assert run["trial_id"] == out["trial_id"] and run["sealed_opened"] is False and run["date_range"][1] < "2012-10-01"
    assert set(run["summary"]) == set(v3_run.PRIMARY) and out["holm"]["m"] == 21
    assert "Substitutions, rejections and skips" in out["markdown"] and "not a claim of alpha" in out["markdown"]
    # a second run appends, never rewrites, and a re-run reproduces the numbers
    again = v3_run.run_v3("unused.db", path, cut)
    assert len(ledger.read(path)) == 3 and again["summary"] == out["summary"]


def test_h12_evaluation_matches_a_direct_count(frame):
    ev = v3_gap.h12(frame, seed=11)
    g = v3_gap.gap_arrays(frame)
    sig = g["sig_up"] | g["sig_dn"]
    assert ev.n_signals == int(sig.sum()) and ev.n_independent <= ev.n_all
    assert ev.effect == pytest.approx(ev.rate_event - ev.rate_matched) and ev.boot.effect.size > 0
    # the matched baseline is a share of sessions, between 0 and 1
    assert 0.0 <= ev.rate_matched <= 1.0 and 0.0 <= ev.rate_unconditional <= 1.0


def test_evaluate_every_spec_runs_and_reports_counts(specs):
    for key in ("H1", "H3", "H5-A", "H8-SPY", "H11", "H13-QQQ-b"):
        ev = evaluate(specs[key], seed=3)
        assert ev.n_signals >= ev.n_all >= ev.n_independent >= 0
        if ev.effect is not None:
            assert -1.0 <= ev.effect <= 1.0

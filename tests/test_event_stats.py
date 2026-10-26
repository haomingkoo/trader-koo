"""Event-study statistics on small cases whose answers are known by hand."""
from __future__ import annotations

import numpy as np
import pytest

from trader_koo.research.bench import eventstats as es
from trader_koo.research.bench.eventstudy import Spec, evaluate
from trader_koo.research.settings import get as settings

E = settings().event_study


def _signal(n: int, at: list[int]) -> np.ndarray:
    s = np.zeros(n, dtype=bool)
    s[at] = True
    return s


def test_independence_drops_signals_with_an_earlier_signal_closer_than_the_spacing():
    # spacing 5: a signal is kept only if the previous signal (kept or not) is at least 5 sessions earlier
    assert np.flatnonzero(es.independent(_signal(30, [0, 1, 3, 5, 9, 10, 11]))).tolist() == [0]
    assert np.flatnonzero(es.independent(_signal(30, [0, 5, 10, 14, 20]))).tolist() == [0, 5, 10, 20]
    assert np.flatnonzero(es.independent(_signal(30, [3, 9, 15]))).tolist() == [3, 9, 15]


def test_strict_reading_also_drops_a_signal_exactly_five_sessions_after_the_previous_one():
    sig = _signal(30, [0, 5, 10, 16])
    assert np.flatnonzero(es.independent(sig)).tolist() == [0, 5, 10, 16]
    assert np.flatnonzero(es.independent(sig, strict=True)).tolist() == [0, 16]
    assert E.spacing_sessions == 5


def test_holm_matches_the_textbook_step_down():
    adj, level, rejected = es.holm([0.01, 0.04, 0.03, 0.5], alpha=0.05)
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert rejected == [True, False, False, False]
    # rank 0 uses alpha/4; rank 1 fails at alpha/3 and every later rank keeps that failing step's level
    assert level == pytest.approx([0.05 / 4, 0.05 / 3, 0.05 / 3, 0.05 / 3])


def test_holm_counts_a_test_without_a_p_value_in_the_family():
    adj, _, rejected = es.holm([0.004, 1.0, 1.0], alpha=0.05)
    assert adj[0] == pytest.approx(0.012) and rejected == [True, False, False]


def test_p_value_and_interval_of_a_known_distribution():
    dist = np.arange(1, 101) / 100.0  # all positive
    assert es.p_two_sided(dist) == pytest.approx(2.0 * 1 / 101)
    assert es.p_two_sided(np.array([-1.0, 1.0] * 50)) == 1.0
    lo, hi = es.interval(dist, 0.05)
    assert lo == pytest.approx(np.quantile(dist, 0.025)) and hi == pytest.approx(np.quantile(dist, 0.975))
    assert es.p_two_sided(np.array([])) == 1.0 and es.interval(np.array([]), 0.05) is None


def _parts(ev_hit, base_hit, ev_mask, base_mask):
    n = len(ev_hit)
    ev, base = np.asarray(ev_mask, float), np.asarray(base_mask, float)
    return np.stack([ev, ev * np.asarray(ev_hit, float), base, base * np.asarray(base_hit, float)])[:, None, :]


def test_point_estimate_is_event_rate_minus_baseline_rate():
    # events at sessions 0..3 hit 3 of 4; baseline sessions 4..9 hit 3 of 6
    hits = [1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    got = es.point(_parts(hits, hits, [1] * 4 + [0] * 6, [0] * 4 + [1] * 6))
    assert got["n_events"] == 4 and got["rate_event"] == 0.75 and got["rate_base"] == 0.5 and got["effect"] == 0.25


def test_one_block_as_long_as_the_series_reproduces_the_series():
    hits = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0], float)
    parts = _parts(hits, hits, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    boot = es.bootstrap(parts, seed=1, block=10, resamples=40)
    assert boot.undefined == 0 and np.allclose(boot.effect, es.point(parts)["effect"])


def test_bootstrap_of_a_constant_gap_has_a_degenerate_interval_and_counts_undefined_resamples():
    n = 40
    hits = np.r_[np.ones(20), np.zeros(20)]  # events (first half) always hit, baseline (second half) never
    parts = _parts(hits, hits, [1] * 20 + [0] * 20, [0] * 20 + [1] * 20)
    boot = es.bootstrap(parts, seed=3, block=5, resamples=2000)
    assert boot.effect.size + boot.undefined == 2000
    assert np.allclose(boot.effect, 1.0) and es.interval(boot.effect, 0.05) == (1.0, 1.0)
    one = np.zeros(10)  # a single event session: a resample that never draws it has no events
    p1 = _parts(np.ones(10), np.ones(10), np.r_[1.0, one[1:]], np.r_[0.0, np.ones(9)])
    b1 = es.bootstrap(p1, seed=3, block=1, resamples=2000)
    assert b1.undefined > 0 and b1.effect.size + b1.undefined == 2000
    assert abs(b1.undefined / 2000 - 0.9 ** 10) < 0.04  # chance of never drawing session 0 in 10 draws


def test_bootstrap_spread_matches_the_binomial_standard_error_for_independent_sessions():
    rng = np.random.default_rng(7)
    n = 2000
    hits = (rng.random(n) < 0.5).astype(float)
    ev = np.ones(n)
    base_hits = np.zeros(n)
    parts = np.stack([ev, hits, ev, base_hits])[:, None, :]  # the baseline rate is exactly 0, so effect = event rate
    boot = es.bootstrap(parts, seed=5, block=1, resamples=3000)
    assert boot.effect.mean() == pytest.approx(hits.mean(), abs=0.005)
    assert boot.effect.std() == pytest.approx(np.sqrt(0.25 / n), rel=0.15)


def test_matched_baseline_weights_strata_by_their_share_of_events():
    n = 12
    # stratum 0: 3 events (rate 1/3), baseline rate 1/2; stratum 1: 1 event (rate 1), baseline rate 0
    ev = np.zeros((2, n)); ev[0, :3] = 1; ev[1, 3] = 1
    ev_hit = np.zeros((2, n)); ev_hit[0, 0] = 1; ev_hit[1, 3] = 1
    base = np.zeros((2, n)); base[0, 4:8] = 1; base[1, 8:12] = 1
    base_hit = np.zeros((2, n)); base_hit[0, 4:6] = 1
    got = es.point(np.stack([ev, ev_hit, base, base_hit]))
    assert got["rate_event"] == pytest.approx(0.5)
    assert got["rate_base"] == pytest.approx(0.75 * 0.5 + 0.25 * 0.0)
    assert got["effect"] == pytest.approx(0.5 - 0.375)


def test_a_stratum_with_events_but_no_baseline_session_has_no_effect():
    ev = np.zeros((2, 4)); ev[0, 0] = 1; ev[1, 1] = 1
    base = np.zeros((2, 4)); base[0, 2] = 1
    assert np.isnan(es.point(np.stack([ev, ev, base, base]))["effect"])


def test_evaluate_counts_independent_events_and_signals_without_an_outcome():
    n = 60
    sig = _signal(n, [2, 3, 10, 20, 26, 58])
    out = np.full((1, n), 0.0)
    out[0, [2, 10, 20, 26]] = 1.0  # event outcomes: 3 events hit... (3 is dropped as non-independent)
    out[0, 3] = 0.0
    out[0, 56:] = np.nan  # the window ends: signal 58 has no outcome
    base = np.zeros((1, n), bool); base[0, 30:50] = True
    ev = evaluate(Spec("X", "t", "primary", sig, np.zeros(n, int), out, base), seed=1)
    assert (ev.n_signals, ev.n_no_outcome, ev.n_all, ev.n_independent) == (6, 1, 5, 4)
    assert ev.rate_event == 1.0 and ev.rate_event_all == 0.8 and ev.rate_matched == 0.0 and ev.effect == 1.0
    assert not ev.sufficient  # 4 < 30
    assert ev.rate_unconditional == pytest.approx(np.nanmean(out))

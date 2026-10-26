from __future__ import annotations

import datetime as dt
import math

import pytest

from trader_koo.collection.options_metrics import (
    atm_iv,
    compute_daily_metrics,
    constant_maturity_iv,
    put_delta,
)

ASOF = dt.date(2026, 10, 26)
SPOT = 100.0
RATE = 0.04
NAN = float("nan")


def _legs(kind: str, base: float, volume: float = 5.0):
    out = []
    for k in range(80, 121):
        iv = base + 0.005 if kind == "call" else base - 0.005 + 0.002 * max(0.0, SPOT - k)
        out.append({"strike": float(k), "impliedVolatility": iv, "openInterest": 10.0, "volume": volume})
    return out


def _chain(days: int, base: float):
    return (ASOF + dt.timedelta(days=days), _legs("call", base), _legs("put", base, 7.0))


CHAINS = [_chain(28, 0.20), _chain(63, 0.30), _chain(91, 0.25)]


def test_put_delta_known_value():
    # S=K, sigma=0.2, t=1, r=0 -> d1=0.1 -> N(0.1)-1 = -0.4602
    assert put_delta(100, 100, 0.2, 1.0, 0.0) == pytest.approx(-0.46017, abs=1e-4)


def test_atm_iv_is_mean_of_call_and_put_at_closest_strike():
    assert atm_iv(100.4, CHAINS[0][1], CHAINS[0][2]) == pytest.approx(0.20)
    assert atm_iv(100, [{"strike": 100, "impliedVolatility": 0.2}], [{"strike": 90, "impliedVolatility": 0.2}]) is None


def test_dead_quotes_are_not_used():
    calls = [{"strike": 100, "impliedVolatility": 1e-5}]
    puts = [{"strike": 100, "impliedVolatility": 0.3}]
    assert atm_iv(100, calls, puts) is None


def test_constant_maturity_interpolates_total_variance():
    def w(iv, d):
        return iv * iv * d

    expected = math.sqrt((w(0.20, 28) + (w(0.30, 63) - w(0.20, 28)) * (30 - 28) / (63 - 28)) / 30)
    assert constant_maturity_iv(SPOT, CHAINS, ASOF, 30) == pytest.approx(expected)
    assert constant_maturity_iv(SPOT, CHAINS, ASOF, 63) == pytest.approx(0.30)


def test_constant_maturity_is_none_without_bracketing_expiries():
    assert constant_maturity_iv(SPOT, CHAINS[:1], ASOF, 60) is None
    assert constant_maturity_iv(SPOT, CHAINS[1:], ASOF, 30) is None


def test_full_row_skew_spread_and_totals():
    row = compute_daily_metrics(SPOT, RATE, CHAINS, ASOF, (30, 60, 90))
    assert row["quality"] == "ok" and row["n_expiries"] == 3 and row["n_strikes"] == 41
    t = 28 / 365
    puts = [p for p in CHAINS[0][2] if p["strike"] < SPOT]
    best = min(puts, key=lambda p: abs(put_delta(SPOT, p["strike"], p["impliedVolatility"], t, RATE) + 0.25))
    delta = put_delta(SPOT, best["strike"], best["impliedVolatility"], t, RATE)
    assert abs(delta + 0.25) < 0.02
    assert row["put_skew_25d"] == pytest.approx(best["impliedVolatility"] - 0.20)
    assert row["put_skew_25d"] > 0
    diffs = [
        c["impliedVolatility"] - p["impliedVolatility"]
        for c, p in zip(CHAINS[0][1], CHAINS[0][2])
        if abs(c["strike"] / SPOT - 1) <= 0.05
    ]
    assert row["cp_iv_spread"] == pytest.approx(sum(diffs) / len(diffs))
    assert row["call_volume"] == 5.0 * 41 * 3 and row["put_volume"] == 7.0 * 41 * 3
    assert row["call_oi"] == 10.0 * 41 * 3 and row["pc_oi_ratio"] == pytest.approx(1.0)
    assert row["pc_volume_ratio"] == pytest.approx(1.4)


def test_missing_inputs_yield_none_not_invented_values():
    leg = [{"strike": 100, "impliedVolatility": NAN, "volume": NAN, "openInterest": NAN}]
    row = compute_daily_metrics(SPOT, None, [(CHAINS[0][0], leg, leg)], ASOF, (30, 60, 90))
    assert row["iv_30d"] is None and row["put_skew_25d"] is None and row["cp_iv_spread"] is None
    assert row["call_volume"] is None and row["pc_volume_ratio"] is None and row["pc_oi_ratio"] is None
    assert row["n_strikes"] == 0 and row["quality"] == "missing"


def test_no_spot_means_no_iv_metrics():
    row = compute_daily_metrics(None, RATE, CHAINS, ASOF, (30, 60, 90))
    assert row["spot"] is None and row["iv_30d"] is None and row["quality"] == "missing"
    assert row["call_volume"] is not None


def test_zero_call_volume_gives_null_ratio():
    chains = [(CHAINS[0][0], [{"strike": 100, "volume": 0.0}], [{"strike": 100, "volume": 3.0}])]
    assert compute_daily_metrics(SPOT, RATE, chains, ASOF, (30,))["pc_volume_ratio"] is None

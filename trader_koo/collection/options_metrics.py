"""Derived daily option metrics from an in-memory chain. Pure math, no I/O.

Conventions (documented because they are choices, not facts):

* Time to expiry is calendar days / 365 (``DAYS_PER_YEAR``).
* ATM IV of one expiry is the mean of the call and put IV at the strike closest
  to spot that has a valid IV on both sides. Valid means within
  ``MIN_VALID_IV``..``MAX_VALID_IV``.
* Constant-maturity IV at T days interpolates total variance (iv^2 * t)
  linearly in time between the expiries bracketing T. If either bracketing
  expiry is missing or has no ATM IV the value is None; there is no
  extrapolation.
* 25-delta put skew uses the expiry closest to ``SKEW_MATURITY_DAYS``. Put
  delta is Black-Scholes N(d1) - 1 with the put's own IV, the day's spot, the
  ^IRX rate and zero dividend yield (dividends are ignored). Skew is the IV of
  the OTM put whose delta is closest to -0.25 minus that expiry's ATM IV; None
  when the rate is missing or no put is within ``SKEW_DELTA_TOLERANCE``.
* ``cp_iv_spread`` (Cremers-Weinbaum style) is call IV minus put IV at matched
  strikes within ``CP_SPREAD_MONEYNESS_BAND`` of spot on that same expiry,
  averaged. Negate it for the paper's put-minus-call sign.
* Volume and open-interest totals sum the expiries that were fetched
  (``n_expiries``), not the whole chain. Ratios are put / call, None when the
  call total is missing or zero.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Iterable, Mapping

from trader_koo.collection.config import (
    CP_SPREAD_MONEYNESS_BAND,
    DAYS_PER_YEAR,
    MAX_VALID_IV,
    MIN_VALID_IV,
    SKEW_DELTA_TOLERANCE,
    SKEW_MATURITY_DAYS,
    SKEW_TARGET_DELTA,
)

Leg = Mapping[str, Any]  # yfinance columns: strike, impliedVolatility, openInterest, volume
Chain = tuple[dt.date, list[Leg], list[Leg]]  # (expiry, calls, puts)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _iv(leg: Leg) -> float | None:
    iv = _num(leg.get("impliedVolatility"))
    return iv if iv is not None and MIN_VALID_IV <= iv <= MAX_VALID_IV else None


def _by_strike(legs: Iterable[Leg]) -> dict[float, float]:
    out: dict[float, float] = {}
    for leg in legs:
        strike, iv = _num(leg.get("strike")), _iv(leg)
        if strike is not None and strike > 0 and iv is not None:
            out[strike] = iv
    return out


def _norm_cdf(x: float) -> float:
    return math.erfc(-x / math.sqrt(2)) / 2


def put_delta(spot: float, strike: float, iv: float, t_years: float, rate: float) -> float:
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2) * t_years) / (iv * math.sqrt(t_years))
    return _norm_cdf(d1) - 1


def atm_iv(spot: float, calls: list[Leg], puts: list[Leg]) -> float | None:
    c, p = _by_strike(calls), _by_strike(puts)
    common = set(c) & set(p)
    if not common:
        return None
    k = min(common, key=lambda x: (abs(x - spot), x))
    return (c[k] + p[k]) / 2


def constant_maturity_iv(
    spot: float, chains: list[Chain], asof: dt.date, target_days: int
) -> float | None:
    points: list[tuple[int, float]] = []
    for expiry, calls, puts in chains:
        days = (expiry - asof).days
        iv = atm_iv(spot, calls, puts) if days >= 1 else None
        if iv is not None:
            points.append((days, iv))
    below = [p for p in points if p[0] <= target_days]
    above = [p for p in points if p[0] >= target_days]
    if not below or not above:
        return None
    d1, iv1 = max(below)
    d2, iv2 = min(above)
    if d1 == d2:
        return iv1
    w1, w2 = iv1 * iv1 * d1, iv2 * iv2 * d2  # total variance (day units)
    w = w1 + (w2 - w1) * (target_days - d1) / (d2 - d1)
    return math.sqrt(w / target_days) if w > 0 else None


def _skew_chain(chains: list[Chain], asof: dt.date) -> Chain | None:
    live = [c for c in chains if (c[0] - asof).days >= 1]
    return min(live, key=lambda c: abs((c[0] - asof).days - SKEW_MATURITY_DAYS), default=None)


def put_skew_25d(spot: float, rate: float | None, chains: list[Chain], asof: dt.date) -> float | None:
    chain = _skew_chain(chains, asof)
    if chain is None or rate is None:
        return None
    expiry, calls, puts = chain
    base = atm_iv(spot, calls, puts)
    if base is None:
        return None
    t = (expiry - asof).days / DAYS_PER_YEAR
    best: tuple[float, float] | None = None  # (|delta - target|, iv)
    for strike, iv in _by_strike(puts).items():
        if strike >= spot:
            continue
        gap = abs(put_delta(spot, strike, iv, t, rate) - SKEW_TARGET_DELTA)
        if gap <= SKEW_DELTA_TOLERANCE and (best is None or gap < best[0]):
            best = (gap, iv)
    return None if best is None else best[1] - base


def cp_iv_spread(spot: float, chains: list[Chain], asof: dt.date) -> float | None:
    chain = _skew_chain(chains, asof)
    if chain is None:
        return None
    c, p = _by_strike(chain[1]), _by_strike(chain[2])
    diffs = [
        c[k] - p[k]
        for k in set(c) & set(p)
        if abs(k / spot - 1) <= CP_SPREAD_MONEYNESS_BAND
    ]
    return sum(diffs) / len(diffs) if diffs else None


def _total(legs: Iterable[Leg], key: str) -> float | None:
    values = [v for v in (_num(leg.get(key)) for leg in legs) if v is not None]
    return sum(values) if values else None


def _ratio(put: float | None, call: float | None) -> float | None:
    return put / call if put is not None and call else None


def compute_daily_metrics(
    spot: float | None,
    rate: float | None,
    chains: list[Chain],
    asof: dt.date,
    target_days: tuple[int, ...],
) -> dict[str, Any]:
    """One row of metrics; every value that cannot be computed is None."""
    cm = {d: None for d in target_days}
    out: dict[str, Any] = {
        "spot": spot if spot and spot > 0 else None,
        "rate": rate,
        "put_skew_25d": None,
        "cp_iv_spread": None,
        "n_expiries": len(chains),
        "n_strikes": 0,
    }
    calls = [leg for _, cs, _ in chains for leg in cs]
    puts = [leg for _, _, ps in chains for leg in ps]
    for name, legs in (("call", calls), ("put", puts)):
        out[f"{name}_volume"] = _total(legs, "volume")
        out[f"{name}_oi"] = _total(legs, "openInterest")
    out["pc_volume_ratio"] = _ratio(out["put_volume"], out["call_volume"])
    out["pc_oi_ratio"] = _ratio(out["put_oi"], out["call_oi"])
    out["n_strikes"] = len(set(_by_strike(calls)) | set(_by_strike(puts)))
    if out["spot"] is not None:
        for d in target_days:
            cm[d] = constant_maturity_iv(out["spot"], chains, asof, d)
        out["put_skew_25d"] = put_skew_25d(out["spot"], rate, chains, asof)
        out["cp_iv_spread"] = cp_iv_spread(out["spot"], chains, asof)
    for d in target_days:
        out[f"iv_{d}d"] = cm[d]
    ivs = [cm[d] for d in target_days]
    if all(v is not None for v in ivs) and out["put_skew_25d"] is not None:
        out["quality"] = "ok"
    elif any(v is not None for v in ivs):
        out["quality"] = "partial"
    else:
        out["quality"] = "missing"
    return out

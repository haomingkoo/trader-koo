"""Frozen, fail-closed preregistration for the non-TA challenger tournament.

The sealed holdout is deliberately unreachable until the copied research
database proves a consistent total-return basis. Signal helpers are pure so
their formulas can be verified without consuming any research window.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import statistics
from typing import Any, Iterable

from trader_koo.db.price_contract import research_price_contract
from trader_koo.research.next_open_baseline import canonical_json_bytes

SCHEMA_VERSION = "challenger-tournament-v1"
MAX_HOLDING_SESSIONS = 21

CHALLENGERS: dict[str, dict[str, Any]] = {
    "C1": {
        "name": "long_only_12_1_cross_sectional_momentum",
        "signal": "adjusted_close_t_minus_21 / adjusted_close_t_minus_252 - 1",
        "schedule": "month_end_signal_next_open_rebalance",
        "selection": "top_quintile_max_20",
        "weighting": "inverse_63_session_volatility_10_pct_name_cap",
        "long_only": True,
        "leverage": False,
        "one_way_cost_bps": 10,
        "stress_cost_bps": 20,
    },
    "C2": {
        "name": "liquid_large_cap_five_session_residual_reversal",
        "signal": "five_session_return_minus_prior_126_session_spy_beta_times_spy_return",
        "schedule": "week_end_signal_next_open_entry_five_session_hold",
        "selection": "bottom_decile_max_20",
        "weighting": "equal_weight_10_pct_name_cap",
        "minimum_median_20_session_dollar_volume": 50_000_000,
        "maximum_adv_pct": 1,
        "one_way_cost_scenarios_bps": [10, 25, 50],
        "edge_must_survive_bps": 25,
        "long_only": True,
        "leverage": False,
    },
    "C3": {
        "name": "volatility_managed_spy_core",
        "signal": "min(1, 10_pct_annual_volatility / prior_20_session_realized_volatility)",
        "trend_gate": "prior_126_session_spy_return_positive_else_cash",
        "schedule": "month_end_decision_next_open_rebalance",
        "long_only": True,
        "leverage": False,
        "one_way_cost_bps": 1,
        "stress_cost_bps": 2,
    },
}


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def frozen_preregistration(dataset_hash: str) -> dict[str, Any]:
    """Return the complete selection contract frozen before validation."""
    config_hashes = {name: _sha256(spec) for name, spec in CHALLENGERS.items()}
    body = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "challengers": json.loads(canonical_json_bytes(CHALLENGERS)),
        "config_hashes": config_hashes,
        "selection": {
            "candidates": ["C1", "C2", "C3"],
            "development_pct": 60,
            "validation_pct": 20,
            "sealed_heldout_pct": 20,
            "purge_sessions": MAX_HOLDING_SESSIONS,
            "embargo_sessions": MAX_HOLDING_SESSIONS,
            "validation": "expanding_walk_forward",
            "multiple_testing": "holm_three_challengers",
            "winner_count_max": 1,
            "heldout_reuse": False,
            "automatic_promotion": False,
        },
        "historical_shadow_gate": {
            "minimum_years": 5,
            "minimum_volatility_regimes": 3,
            "positive_net_active_return_fold_pct": 70,
            "double_cost_net_return_minimum": 0,
            "maximum_profit_concentration_pct": 50,
            "maximum_adv_pct": 1,
            "risk_rule_required": True,
        },
        "prohibited": [
            "technical_or_candlestick_weights", "llm_weights", "deep_ml",
            "covariance_optimization", "broad_parameter_grids", "equity_shorts",
        ],
    }
    return {**body, "preregistration_sha256": _sha256(body)}


def chronological_split(sessions: Iterable[str]) -> dict[str, Any]:
    """Create a 60/20/20 split with overlap purged and embargoed."""
    dates = tuple(sorted(set(sessions)))
    first, second = int(len(dates) * .6), int(len(dates) * .8)
    purge = MAX_HOLDING_SESSIONS
    windows = {
        "development": dates[:max(0, first - purge)],
        "validation": dates[min(len(dates), first + purge):max(first + purge, second - purge)],
        "heldout": dates[min(len(dates), second + purge):],
    }
    return {
        name: {
            "start": values[0] if values else None,
            "end": values[-1] if values else None,
            "session_count": len(values),
            "session_sha256": _sha256(values),
        }
        for name, values in windows.items()
    }


def _returns(values: list[float]) -> list[float]:
    return [after / before - 1 for before, after in zip(values, values[1:]) if before > 0]


def _capped_inverse_volatility(volatility: dict[str, float]) -> dict[str, float]:
    remaining = {name: 1 / value for name, value in volatility.items() if value > 0}
    weights: dict[str, float] = {}
    budget = 1.0
    while remaining and budget > 1e-12:
        total = sum(remaining.values())
        proposed = {name: budget * value / total for name, value in remaining.items()}
        capped = {name for name, value in proposed.items() if value > .10}
        if not capped:
            weights.update(proposed)
            break
        for name in capped:
            weights[name] = .10
            budget -= .10
            remaining.pop(name)
    return {name: round(value * 100, 10) for name, value in sorted(weights.items())}


def c1_signal(history: dict[str, list[float]]) -> dict[str, Any]:
    """Compute frozen C1 ranks and inverse-volatility portfolio weights."""
    scored: list[tuple[str, float, float]] = []
    for ticker, closes in history.items():
        if len(closes) < 253 or min(closes[-253:]) <= 0:
            continue
        score = closes[-22] / closes[-253] - 1
        daily = _returns(closes[-64:])
        volatility = statistics.stdev(daily) * math.sqrt(252) if len(daily) > 1 else 0
        if volatility > 0:
            scored.append((ticker, score, volatility))
    selected_count = min(20, math.ceil(len(scored) * .2))
    selected = sorted(scored, key=lambda row: (-row[1], row[0]))[:selected_count]
    return {
        "scores": {ticker: score for ticker, score, _vol in sorted(scored)},
        "weights_pct": _capped_inverse_volatility({ticker: vol for ticker, _score, vol in selected}),
    }


def c2_signal(
    history: dict[str, list[tuple[float, float]]],
    spy_closes: list[float],
) -> dict[str, Any]:
    """Compute frozen C2 five-session residual reversal ranks."""
    if len(spy_closes) < 127 or min(spy_closes[-127:]) <= 0:
        return {"residuals": {}, "weights_pct": {}}
    spy_daily = _returns(spy_closes[-127:])
    spy_mean = statistics.fmean(spy_daily)
    spy_variance = sum((value - spy_mean) ** 2 for value in spy_daily)
    spy_five = spy_closes[-1] / spy_closes[-6] - 1
    eligible: list[tuple[str, float]] = []
    for ticker, rows in history.items():
        if len(rows) < 127:
            continue
        closes = [float(row[0]) for row in rows]
        if min(closes[-127:]) <= 0:
            continue
        dollar_volume = [float(close) * float(volume) for close, volume in rows[-20:]]
        if statistics.median(dollar_volume) < 50_000_000:
            continue
        daily = _returns(closes[-127:])
        mean = statistics.fmean(daily)
        covariance = sum(
            (asset - mean) * (market - spy_mean)
            for asset, market in zip(daily, spy_daily)
        )
        beta = covariance / spy_variance if spy_variance > 0 else 0
        residual = closes[-1] / closes[-6] - 1 - beta * spy_five
        eligible.append((ticker, residual))
    selected_count = min(20, math.ceil(len(eligible) * .1))
    selected = sorted(eligible, key=lambda row: (row[1], row[0]))[:selected_count]
    weight = min(10.0, 100 / len(selected)) if selected else 0
    return {
        "residuals": {ticker: value for ticker, value in sorted(eligible)},
        "weights_pct": {ticker: weight for ticker, _value in selected},
    }


def c3_exposure(spy_closes: list[float]) -> float:
    """Compute frozen C3 exposure using only observations before the decision."""
    if len(spy_closes) < 127 or min(spy_closes[-127:]) <= 0:
        return 0.0
    if spy_closes[-1] / spy_closes[-127] - 1 <= 0:
        return 0.0
    daily = _returns(spy_closes[-21:])
    realized = statistics.stdev(daily) * math.sqrt(252) if len(daily) > 1 else 0
    return min(1.0, .10 / realized) if realized > 0 else 0.0


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values for exactly C1-C3."""
    if set(p_values) != set(CHALLENGERS):
        raise ValueError("Holm correction requires exactly C1, C2, and C3")
    ordered = sorted(p_values.items(), key=lambda row: (row[1], row[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def _table_exists(conn: Any, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(conn: Any, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def dataset_audit(conn: Any) -> dict[str, Any]:
    """Audit the exact copied database before any validation or holdout read."""
    if not _table_exists(conn, "price_daily"):
        return {"eligible": False, "reasons": ["price_daily_unavailable"]}
    columns = _columns(conn, "price_daily")
    required = {"ticker", "date", "open", "close", "volume"}
    if not required.issubset(columns):
        return {"eligible": False, "reasons": ["required_price_columns_unavailable"]}
    price_rows = list(conn.execute(
        "SELECT ticker,date,open,close,volume FROM price_daily"
    ))
    tickers = sorted({str(row[0]) for row in price_rows if str(row[0]).strip()})
    ticker_count = len(tickers)
    row_count = len(price_rows)
    reasons: list[str] = []
    valid_dates: list[dt.date] = []
    spy_prices: list[tuple[dt.date, float]] = []
    invalid_spy_prices = 0
    for ticker, date_value, open_value, close_value, volume_value in price_rows:
        try:
            parsed_date = dt.date.fromisoformat(str(date_value))
            valid_dates.append(parsed_date)
        except (TypeError, ValueError):
            reasons.append("invalid_price_date")
            parsed_date = None
        try:
            open_price, close_price, volume = (
                float(open_value), float(close_value), float(volume_value)
            )
            valid_values = (
                math.isfinite(open_price) and open_price > 0
                and math.isfinite(close_price) and close_price > 0
                and math.isfinite(volume) and volume >= 0
            )
        except (TypeError, ValueError):
            valid_values = False
        if not valid_values:
            reasons.append("invalid_price_value")
            if str(ticker) == "SPY":
                invalid_spy_prices += 1
        elif str(ticker) == "SPY" and parsed_date is not None:
            spy_prices.append((parsed_date, close_price))
    start = min(valid_dates).isoformat() if valid_dates else None
    end = max(valid_dates).isoformat() if valid_dates else None
    try:
        contract = research_price_contract(conn, tickers)
    except Exception as exc:
        contract = {
            "eligible": False, "basis": "unknown", "status": "unresolved",
            "reason": f"price_contract_error:{type(exc).__name__}",
        }
    if not contract.get("eligible"):
        reasons.append(str(contract.get("reason") or "price_contract_unverified"))
    if contract.get("basis") != "total_return" or not contract.get("distributions_included"):
        reasons.append("consistent_total_return_basis_required")
    years = 0.0
    if start and end:
        try:
            years = (
                dt.date.fromisoformat(str(end)) - dt.date.fromisoformat(str(start))
            ).days / 365.2425
        except ValueError:
            reasons.append("invalid_price_date")
    if years < 5:
        reasons.append("fewer_than_five_years")
    spy_closes = [close for _, close in sorted(spy_prices)]
    if invalid_spy_prices:
        reasons.append("invalid_spy_price")
    rolling_volatility = [
        statistics.stdev(_returns(spy_closes[index - 20:index + 1])) * math.sqrt(252)
        for index in range(20, len(spy_closes))
        if len(_returns(spy_closes[index - 20:index + 1])) > 1
    ]
    regime_count = 0
    if rolling_volatility:
        ordered_volatility = sorted(rolling_volatility)
        low = ordered_volatility[len(ordered_volatility) // 3]
        high = ordered_volatility[len(ordered_volatility) * 2 // 3]
        regime_count = len({
            "low" if value <= low else "high" if value >= high else "middle"
            for value in rolling_volatility
        })
    if regime_count < 3:
        reasons.append("fewer_than_three_volatility_regimes")
    membership_table = next((
        name for name in ("index_membership_history", "universe_membership_history")
        if _table_exists(conn, name)
    ), None)
    universe = "point_in_time_membership" if membership_table else "fixed_universe_survivor_study"
    snapshot = {
        "price_start": start, "price_end": end, "years": years,
        "ticker_count": ticker_count, "row_count": row_count,
        "volatility_regime_count": regime_count,
        "price_contract": contract, "universe_treatment": universe,
        "membership_table": membership_table,
    }
    return {
        **snapshot,
        "dataset_sha256": _sha256(snapshot),
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
    }


def run_challenger_tournament(conn: Any) -> dict[str, Any]:
    """Attempt exactly C1-C3, leaving sealed validation untouched on bad data."""
    audit = dataset_audit(conn)
    preregistration = frozen_preregistration(
        str(audit.get("dataset_sha256") or _sha256(audit))
    )
    if not audit["eligible"]:
        results = {
            name: {
                "status": "failed_data_gate", "reasons": audit["reasons"],
                "config_sha256": preregistration["config_hashes"][name],
                "metrics": None,
            }
            for name in CHALLENGERS
        }
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_before_validation",
            "preregistration": preregistration,
            "dataset_audit": audit,
            "split": None,
            "challenger_results": results,
            "holm_adjusted_p_values": None,
            "selected_challenger": None,
            "sealed_heldout": {"accessed": False, "access_log": []},
            "prospective_shadow_candidate": None,
            "automatic_promotion": False,
        }
        return {**body, "artifact_sha256": _sha256(body)}
    results = {
        name: {
            "status": "not_run", "reasons": ["validation_executor_not_sealed"],
            "config_sha256": preregistration["config_hashes"][name],
            "metrics": None,
        }
        for name in CHALLENGERS
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_before_validation",
        "preregistration": preregistration,
        "dataset_audit": audit,
        "split": None,
        "challenger_results": results,
        "holm_adjusted_p_values": None,
        "selected_challenger": None,
        "sealed_heldout": {"accessed": False, "access_log": []},
        "prospective_shadow_candidate": None,
        "automatic_promotion": False,
        "blocking_reasons": ["validation_executor_not_sealed"],
    }
    return {**body, "artifact_sha256": _sha256(body)}

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
from pathlib import Path
from typing import Any, Iterable

from trader_koo.db.price_contract import research_price_contract
from trader_koo.ml.features import ML_CONTEXT_TICKERS
from trader_koo.report.runs import current_code_version
from trader_koo.research.challenger_executor import (
    c1_signal,
    c2_signal,
    c3_exposure,
    daily_returns,
    execute_validation_tournament,
    holm_adjust,
)
from trader_koo.research.next_open_baseline import canonical_json_bytes

SCHEMA_VERSION = "challenger-tournament-v1"
MAX_HOLDING_SESSIONS = 21
IMPLEMENTATION_PATH = Path(__file__)
IMPLEMENTATION_PATHS = (
    IMPLEMENTATION_PATH,
    IMPLEMENTATION_PATH.with_name("challenger_executor.py"),
    IMPLEMENTATION_PATH.with_name("next_open_baseline.py"),
)

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
        "selection_cost_bps": 25,
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


def _implementation_hash() -> str:
    """Hash the complete local signal and portfolio execution closure."""
    digest = hashlib.sha256()
    for path in sorted(IMPLEMENTATION_PATHS, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
            "maximum_drawdown_pct": 25,
            "holm_adjusted_p_value_max": .05,
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
    tickers: set[str] = set()
    row_count = 0
    reasons: list[str] = []
    start_date: dt.date | None = None
    end_date: dt.date | None = None
    spy_prices: list[tuple[dt.date, float]] = []
    invalid_spy_prices = 0
    market_rows_hasher = hashlib.sha256()
    excluded_context_rows = 0
    for ticker, date_value, open_value, close_value, volume_value in conn.execute(
        """SELECT ticker,date,open,close,volume FROM price_daily
           ORDER BY ticker,date,open,close,volume"""
    ):
        ticker_name = str(ticker).strip() if ticker is not None else ""
        if ticker_name != "SPY" and (
            ticker_name in ML_CONTEXT_TICKERS or ticker_name.startswith("^")
        ):
            excluded_context_rows += 1
            continue
        market_rows_hasher.update(canonical_json_bytes([
            ticker, date_value, open_value, close_value, volume_value,
        ]))
        market_rows_hasher.update(b"\n")
        row_count += 1
        if ticker_name:
            tickers.add(ticker_name)
        else:
            reasons.append("invalid_ticker")
        try:
            parsed_date = dt.date.fromisoformat(str(date_value))
            start_date = parsed_date if start_date is None else min(start_date, parsed_date)
            end_date = parsed_date if end_date is None else max(end_date, parsed_date)
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
            if ticker_name == "SPY":
                invalid_spy_prices += 1
        elif ticker_name == "SPY" and parsed_date is not None:
            spy_prices.append((parsed_date, close_price))
    start = start_date.isoformat() if start_date else None
    end = end_date.isoformat() if end_date else None
    ticker_names = sorted(tickers)
    ticker_count = len(ticker_names)
    try:
        contract = research_price_contract(conn, ticker_names)
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
        statistics.stdev(daily_returns(spy_closes[index - 20:index + 1])) * math.sqrt(252)
        for index in range(20, len(spy_closes))
        if len(daily_returns(spy_closes[index - 20:index + 1])) > 1
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
    if membership_table is None:
        reasons.append("point_in_time_universe_membership_required")
    else:
        # Detection alone is not evidence that every historical signal was
        # filtered through the table. Keep validation sealed until the executor
        # implements and tests that date-aware membership join.
        reasons.append("point_in_time_universe_membership_enforcement_unimplemented")
    snapshot = {
        "price_start": start, "price_end": end, "years": years,
        "ticker_count": ticker_count, "row_count": row_count,
        "market_rows_sha256": market_rows_hasher.hexdigest(),
        "excluded_context_row_count": excluded_context_rows,
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


def run_challenger_tournament(
    conn: Any,
    *,
    consume_heldout: bool = False,
) -> dict[str, Any]:
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
            "code_sha": current_code_version(),
            "implementation_sha256": _implementation_hash(),
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
    execution = execute_validation_tournament(
        conn, preregistration, consume_heldout=consume_heldout
    )
    selected = execution["selected_challenger"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "code_sha": current_code_version(),
        "implementation_sha256": _implementation_hash(),
        "status": (
            "sealed_heldout_complete"
            if execution["sealed_heldout"]["accessed"]
            else "validation_complete_winner_pending_sealed_holdout"
            if selected
            else "validation_complete_no_eligible_challenger"
        ),
        "preregistration": preregistration,
        "dataset_audit": audit,
        **execution,
        "automatic_promotion": False,
    }
    return {**body, "artifact_sha256": _sha256(body)}

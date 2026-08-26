"""Canonical frozen contract for the non-TA challenger tournament."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "challenger-tournament-v2"
UNIVERSE_ID = "sp500"
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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def frozen_preregistration(
    dataset_hash: str,
    membership_sha256: str | None,
) -> dict[str, Any]:
    """Return the one accepted universe, candidate, split, and gate contract."""
    config_hashes = {name: _sha256(spec) for name, spec in CHALLENGERS.items()}
    body = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "membership_sha256": membership_sha256,
        "universe_id": UNIVERSE_ID,
        "universe_semantics": "verified_membership_snapshot_at_each_signal_date",
        "challengers": json.loads(_canonical_bytes(CHALLENGERS)),
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

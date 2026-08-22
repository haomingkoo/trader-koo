from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from trader_koo.ml.scorer import (
    _as_probability,
    _prediction_label,
    _signal_from_probability,
    score_universe,
)


def test_as_probability_keeps_lightgbm_probability_outputs():
    probs = _as_probability(np.array([0.2, 0.5, 0.8]))

    assert probs.tolist() == [0.2, 0.5, 0.8]


def test_as_probability_converts_raw_margins_only():
    probs = _as_probability(np.array([-1.38629436, 0.0, 1.38629436]))

    assert abs(probs[0] - 0.2) < 1e-6
    assert abs(probs[1] - 0.5) < 1e-9
    assert abs(probs[2] - 0.8) < 1e-6


def test_barrier_low_probability_is_not_bearish_signal():
    assert _signal_from_probability(0.3, "barrier") == "neutral"


def test_directional_targets_can_emit_bearish_signal():
    assert _signal_from_probability(0.3, "return_sign") == "bearish"
    assert _signal_from_probability(0.3, "rank") == "bearish"


def test_prediction_label_names_target_meaning():
    assert _prediction_label("barrier") == "target_hit_probability"


def test_scoring_fails_closed_for_unresolved_price_basis():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE price_daily (
        ticker TEXT, date TEXT, close REAL,
        adjustment_basis TEXT, adjustment_version TEXT, basis_status TEXT)"""
    )
    conn.execute(
        "INSERT INTO price_daily VALUES ('BAD', '2026-08-20', 100, ?, ?, 'unresolved')",
        ("split_adjusted_price_only", "test-v1"),
    )

    with pytest.raises(ValueError, match="not research eligible"):
        score_universe(conn, as_of_date="2026-08-20")

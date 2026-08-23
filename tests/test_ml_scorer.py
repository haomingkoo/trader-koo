from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from trader_koo.ml import scorer
from trader_koo.ml.scorer import (
    _as_probability,
    _prediction_label,
    _signal_from_probability,
    model_status,
    require_model_price_contract,
    score_single_ticker,
    score_universe,
)
from trader_koo.db.price_contract import record_price_series_revision


class _Model:
    def predict(self, values):
        return np.full(len(values), 0.7)


def _verified_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE price_daily (
        ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
        adjustment_basis TEXT, adjustment_version TEXT, basis_status TEXT,
        unresolved_reason TEXT)"""
    )
    for ticker in ("AAA", "SPY"):
        conn.execute(
            "INSERT INTO price_daily VALUES (?, '2026-08-20', 100, 100, 100, 100, 1, ?, ?, 'verified', NULL)",
            (ticker, "split_adjusted_price_only", "test-v1"),
        )
        record_price_series_revision(
            conn,
            ticker,
            evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
            fetch_timestamp="2026-08-20T00:00:00Z",
        )
    return conn


def _meta(**overrides):
    value = {
        "feature_columns": ["feature"],
        "target_mode": "barrier",
        "return_basis": "split_adjusted_price_only",
        "adjustment_version": "test-v1",
        "distributions_included": False,
    }
    value.update(overrides)
    return value


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


@pytest.mark.parametrize(
    "meta",
    [
        None,
        _meta(return_basis="unknown"),
        _meta(adjustment_version="legacy-v0"),
        _meta(distributions_included=True),
    ],
)
def test_model_contract_missing_or_mismatch_fails_closed(meta):
    current = {
        "basis": "split_adjusted_price_only",
        "version": "test-v1",
        "distributions_included": False,
    }
    with pytest.raises(ValueError, match="incompatible"):
        require_model_price_contract(meta, current)


def test_scoring_requires_matching_saved_model_contract(monkeypatch):
    conn = _verified_conn()
    monkeypatch.setattr(scorer, "load_model", lambda: (_Model(), _meta()))
    monkeypatch.setattr(
        scorer,
        "extract_features_for_universe",
        lambda *args, **kwargs: pd.DataFrame({"feature": [1.0]}, index=["AAA"]),
    )

    universe = score_universe(
        conn, as_of_date="2026-08-20", tickers=["AAA"], top_n=1
    )
    single = score_single_ticker(conn, ticker="AAA", as_of_date="2026-08-20")

    assert universe[0]["return_basis"] == "split_adjusted_price_only"
    assert universe[0]["adjustment_version"] == "test-v1"
    assert universe[0]["distributions_included"] is False
    assert single["return_basis"] == universe[0]["return_basis"]
    status = model_status(conn)
    assert status["basis_compatible"] is True
    assert status["basis_error"] is None
    assert status["model_price_contract"]["adjustment_version"] == "test-v1"


def test_score_and_model_status_expose_mismatch_and_do_not_predict(monkeypatch):
    conn = _verified_conn()
    bad_meta = _meta(adjustment_version="old-v0")
    monkeypatch.setattr(scorer, "load_model", lambda: (_Model(), bad_meta))
    monkeypatch.setattr(
        scorer,
        "extract_features_for_universe",
        lambda *args, **kwargs: pytest.fail("features must not run on a basis mismatch"),
    )

    with pytest.raises(ValueError, match="adjustment_version"):
        score_universe(conn, as_of_date="2026-08-20", tickers=["AAA"])
    with pytest.raises(ValueError, match="adjustment_version"):
        score_single_ticker(conn, ticker="AAA", as_of_date="2026-08-20")

    status = model_status(conn)
    assert status["loaded"] is True
    assert status["basis_compatible"] is False
    assert "adjustment_version" in status["basis_error"]
    assert status["model_price_contract"]["adjustment_version"] == "old-v0"
    assert status["current_price_contract"]["version"] == "test-v1"

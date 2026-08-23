from __future__ import annotations

import math
import sqlite3

import pytest

from trader_koo.research.challenger_tournament import (
    CHALLENGERS,
    c1_signal,
    c2_signal,
    c3_exposure,
    chronological_split,
    frozen_preregistration,
    holm_adjust,
    run_challenger_tournament,
)


def test_preregistration_freezes_exactly_three_config_hashes() -> None:
    first = frozen_preregistration("d" * 64)
    second = frozen_preregistration("d" * 64)

    assert first == second
    assert set(first["challengers"]) == {"C1", "C2", "C3"}
    assert set(first["config_hashes"]) == set(CHALLENGERS)
    assert first["selection"]["automatic_promotion"] is False
    assert first["selection"]["heldout_reuse"] is False


def test_chronological_split_purges_and_embargoes_holding_overlap() -> None:
    sessions = [f"{2020 + index // 336}-{index // 28 % 12 + 1:02d}-{index % 28 + 1:02d}" for index in range(500)]
    split = chronological_split(sessions)

    assert split["development"]["end"] < split["validation"]["start"]
    assert split["validation"]["end"] < split["heldout"]["start"]
    assert sum(item["session_count"] for item in split.values()) < len(sessions)


def test_c1_is_12_1_top_quintile_with_inverse_volatility_cap() -> None:
    history = {}
    for ticker_index in range(10):
        history[f"T{ticker_index}"] = [
            100 * (1 + (.0001 + ticker_index * .00005) * day)
            + math.sin(day) * (ticker_index + 1) * .01
            for day in range(253)
        ]
    signal = c1_signal(history)

    assert len(signal["weights_pct"]) == 2
    assert max(signal["weights_pct"].values()) <= 10
    assert set(signal["weights_pct"]) == {"T8", "T9"}


def test_c2_is_liquid_bottom_decile_residual_reversal() -> None:
    spy = [100 + index * .1 for index in range(127)]
    history = {}
    for ticker_index in range(10):
        closes = [100 + index * .1 for index in range(127)]
        closes[-1] -= ticker_index
        history[f"T{ticker_index}"] = [(close, 1_000_000) for close in closes]
    signal = c2_signal(history, spy)

    assert signal["weights_pct"] == {"T9": 10.0}
    assert signal["residuals"]["T9"] < signal["residuals"]["T0"]


def test_c3_uses_trailing_trend_and_realized_volatility_without_leverage() -> None:
    rising = [100 * (1.001 ** index) * (1 + math.sin(index) * .0005) for index in range(127)]
    falling = list(reversed(rising))

    assert 0 < c3_exposure(rising) <= 1
    assert c3_exposure(falling) == 0


def test_holm_requires_and_adjusts_exactly_three_challengers() -> None:
    assert holm_adjust({"C1": .01, "C2": .03, "C3": .04}) == {
        "C1": pytest.approx(.03), "C2": pytest.approx(.06),
        "C3": pytest.approx(.06),
    }
    with pytest.raises(ValueError, match="exactly C1, C2, and C3"):
        holm_adjust({"C1": .01})


def test_unverified_database_fails_all_challengers_before_holdout_access() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE price_daily (ticker TEXT,date TEXT,open REAL,close REAL,volume REAL)"
    )
    conn.execute(
        "INSERT INTO price_daily VALUES ('SPY','2026-01-01',100,100,1000000)"
    )
    artifact = run_challenger_tournament(conn)

    assert artifact["status"] == "blocked_before_validation"
    assert artifact["selected_challenger"] is None
    assert artifact["sealed_heldout"] == {"accessed": False, "access_log": []}
    assert set(artifact["challenger_results"]) == {"C1", "C2", "C3"}
    assert {
        result["status"] for result in artifact["challenger_results"].values()
    } == {"failed_data_gate"}

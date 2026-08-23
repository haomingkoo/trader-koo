from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
from types import SimpleNamespace

import pytest

from trader_koo.research import challenger_executor as executor
from trader_koo.research import experiment_results
from trader_koo.research import challenger_tournament as tournament
from trader_koo.research.challenger_tournament import (
    CHALLENGERS,
    c1_signal,
    c2_signal,
    c3_exposure,
    chronological_split,
    dataset_audit,
    frozen_preregistration,
    holm_adjust,
    run_challenger_tournament,
)
from trader_koo.research.challenger_executor import execute_validation_tournament
from trader_koo.research.next_open_baseline import SessionPrice


def _unverified_dataset(rows: list[tuple[object, ...]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE price_daily (ticker TEXT,date TEXT,open REAL,close REAL,volume REAL)"
    )
    conn.executemany("INSERT INTO price_daily VALUES (?,?,?,?,?)", rows)
    return conn


def _qualified_executor_fixture() -> sqlite3.Connection:
    conn = _unverified_dataset([])
    dates: list[str] = []
    day = dt.date(2019, 1, 1)
    while len(dates) < 1_500:
        if day.weekday() < 5:
            dates.append(day.isoformat())
        day += dt.timedelta(days=1)
    rows = []
    for ticker_index, ticker in enumerate(
        ["SPY", *(f"T{index:02d}" for index in range(15))]
    ):
        for index, date in enumerate(dates):
            trend = .00025 + ticker_index * .00001
            close = 100 * math.exp(trend * index) * (
                1 + .01 * math.sin(index / 17 + ticker_index)
            )
            rows.append((ticker, date, close * .9995, close, 5_000_000))
    conn.executemany("INSERT INTO price_daily VALUES (?,?,?,?,?)", rows)
    return conn


def _passing_executor_result() -> dict[str, object]:
    return {
        "metrics": {
            "net_total_return_pct": 5.0,
            "net_active_return_pct": 2.0,
            "active_return_p_value": .001,
            "profit_concentration_pct": 10.0,
            "max_drawdown_pct": 2.0,
            "trade_count": 10,
        },
        "equity_curve": [],
        "ledger": {"engine_version": "portfolio-execution-v1.0"},
        "decision_count": 10,
        "rejections": [],
    }


def test_preregistration_freezes_exactly_three_config_hashes() -> None:
    first = frozen_preregistration("d" * 64)
    second = frozen_preregistration("d" * 64)

    assert first == second
    assert set(first["challengers"]) == {"C1", "C2", "C3"}
    assert set(first["config_hashes"]) == set(CHALLENGERS)
    assert first["selection"]["automatic_promotion"] is False
    assert first["selection"]["heldout_reuse"] is False


def test_tournament_hash_covers_signal_and_execution_implementation() -> None:
    digest = hashlib.sha256()
    for path in sorted(tournament.IMPLEMENTATION_PATHS, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    assert tournament._implementation_hash() == digest.hexdigest()


def test_tournament_loader_rejects_a_different_implementation(tmp_path) -> None:
    body = {
        "schema_version": "challenger-tournament-v1",
        "implementation_sha256": "0" * 64,
        "status": "blocked_before_validation",
    }
    artifact = {**body, "artifact_sha256": tournament._sha256(body)}
    path = tmp_path / "tournament.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = experiment_results._load_tournament(path)

    assert loaded["available"] is False
    assert loaded["warnings"] == ["tournament_implementation_hash_mismatch"]


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
    dates = [(dt.date(2020, 1, 1) + dt.timedelta(days=index)).isoformat() for index in range(127)]
    spy = [
        SessionPrice("SPY", date, close, close, volume=1_000_000)
        for date, close in zip(dates, (100 + index * .1 for index in range(127)))
    ]
    history = {}
    for ticker_index in range(10):
        closes = [100 + index * .1 for index in range(127)]
        closes[-1] -= ticker_index
        ticker = f"T{ticker_index}"
        history[ticker] = [
            SessionPrice(ticker, date, close, close, volume=1_000_000)
            for date, close in zip(dates, closes)
        ]
    signal = c2_signal(history, spy)

    assert signal["weights_pct"] == {"T9": 10.0}
    assert signal["residuals"]["T9"] < signal["residuals"]["T0"]


def test_c2_aligns_asset_and_spy_returns_by_date() -> None:
    dates = [(dt.date(2020, 1, 1) + dt.timedelta(days=index)).isoformat() for index in range(128)]
    spy = [
        SessionPrice("SPY", date, 100 + index, 100 + index, volume=1_000_000)
        for index, date in enumerate(dates)
    ]
    # The asset misses one interior SPY session but still has 127 aligned rows.
    # Positional zipping would pair every return after the gap to the wrong day.
    asset = [
        SessionPrice("AAA", date, 100 + index, 100 + index, volume=1_000_000)
        for index, date in enumerate(dates)
        if index != 64
    ]

    result = c2_signal({"AAA": asset}, spy)

    assert set(result["residuals"]) == {"AAA"}
    aligned_spy = [row.close for row in spy if row.date in {item.date for item in asset}]
    asset_returns = executor.daily_returns([row.close for row in asset])
    spy_returns = executor.daily_returns(aligned_spy)
    asset_mean = sum(asset_returns) / len(asset_returns)
    spy_mean = sum(spy_returns) / len(spy_returns)
    beta = sum(
        (left - asset_mean) * (right - spy_mean)
        for left, right in zip(asset_returns, spy_returns)
    ) / sum((value - spy_mean) ** 2 for value in spy_returns)
    expected = asset[-1].close / asset[-6].close - 1 - beta * (
        aligned_spy[-1] / aligned_spy[-6] - 1
    )
    assert result["residuals"]["AAA"] == pytest.approx(expected)


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


def test_dataset_hash_binds_exact_market_rows_independent_of_insert_order() -> None:
    rows = [
        ("SPY", "2020-01-02", 100, 101, 1_000_000),
        ("AAA", "2020-01-02", 20, 21, 2_000_000),
    ]
    first = dataset_audit(_unverified_dataset(rows))
    reordered = dataset_audit(_unverified_dataset(list(reversed(rows))))
    changed = dataset_audit(_unverified_dataset([
        rows[0], ("AAA", "2020-01-02", 20, 22, 2_000_000),
    ]))

    assert first["market_rows_sha256"] == reordered["market_rows_sha256"]
    assert first["dataset_sha256"] == reordered["dataset_sha256"]
    assert changed["market_rows_sha256"] != first["market_rows_sha256"]
    assert changed["dataset_sha256"] != first["dataset_sha256"]


def test_dataset_audit_excludes_non_tradable_negative_yield_context() -> None:
    audit = dataset_audit(_unverified_dataset([
        ("SPY", "2020-03-19", 100, 101, 1_000_000),
        ("AAA", "2020-03-19", 20, 21, 2_000_000),
        ("^IRX", "2020-03-19", -0.033, -0.028, 0),
    ]))

    assert "invalid_price_value" not in audit["reasons"]
    assert audit["ticker_count"] == 2
    assert audit["row_count"] == 2
    assert audit["excluded_context_row_count"] == 1


def test_malformed_market_rows_fail_closed_without_raising() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE price_daily (ticker TEXT,date TEXT,open REAL,close REAL,volume REAL)"
    )
    conn.executemany(
        "INSERT INTO price_daily VALUES ('SPY',?,?,?,1000000)",
        [("not-a-date", 100, "not-a-price"), ("2026-01-02", 101, 101)],
    )

    artifact = run_challenger_tournament(conn)

    assert artifact["status"] == "blocked_before_validation"
    assert "invalid_price_date" in artifact["dataset_audit"]["reasons"]
    assert "invalid_spy_price" in artifact["dataset_audit"]["reasons"]
    assert artifact["sealed_heldout"]["accessed"] is False


def test_audit_checks_interior_dates_and_non_spy_values() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE price_daily (ticker TEXT,date TEXT,open REAL,close REAL,volume REAL)"
    )
    conn.executemany(
        "INSERT INTO price_daily VALUES (?,?,?,?,?)",
        [
            ("SPY", "2020-01-02", 100, 100, 1_000_000),
            ("AAA", "2023-not-a-date", 100, "bad-close", 1_000_000),
            ("SPY", "2026-01-02", 101, 101, 1_000_000),
        ],
    )

    artifact = run_challenger_tournament(conn)

    assert artifact["status"] == "blocked_before_validation"
    assert "invalid_price_date" in artifact["dataset_audit"]["reasons"]
    assert "invalid_price_value" in artifact["dataset_audit"]["reasons"]
    assert artifact["sealed_heldout"]["accessed"] is False


def test_audit_rejects_anonymous_market_rows() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE price_daily (ticker TEXT,date TEXT,open REAL,close REAL,volume REAL)"
    )
    conn.executemany(
        "INSERT INTO price_daily VALUES (?,?,?,?,?)",
        [
            ("SPY", "2020-01-02", 100, 100, 1_000_000),
            ("   ", "2023-01-02", 100, 100, 1_000_000),
            (None, "2024-01-02", 100, 100, 1_000_000),
            ("SPY", "2026-01-02", 101, 101, 1_000_000),
        ],
    )

    artifact = run_challenger_tournament(conn)

    assert artifact["status"] == "blocked_before_validation"
    assert "invalid_ticker" in artifact["dataset_audit"]["reasons"]
    assert artifact["sealed_heldout"]["accessed"] is False


def test_eligible_data_runs_sealed_validation_without_touching_holdout(monkeypatch) -> None:
    monkeypatch.setattr(tournament, "dataset_audit", lambda _conn: {
        "eligible": True, "reasons": [], "dataset_sha256": "d" * 64,
        "price_contract": {"basis": "total_return", "eligible": True},
    })
    monkeypatch.setattr(tournament, "execute_validation_tournament", lambda _conn, _prereg, **_kwargs: {
        "split": {"development": {}, "validation": {}, "heldout": {}},
        "challenger_results": {
            name: {"status": "validation_complete", "metrics": {}}
            for name in CHALLENGERS
        },
        "holm_adjusted_p_values": {name: 1.0 for name in CHALLENGERS},
        "selected_challenger": None,
        "sealed_heldout": {"accessed": False, "access_log": []},
        "prospective_shadow_candidate": None,
    })
    artifact = run_challenger_tournament(object())

    assert artifact["status"] == "validation_complete_no_eligible_challenger"
    assert artifact["sealed_heldout"]["accessed"] is False
    assert {row["status"] for row in artifact["challenger_results"].values()} == {
        "validation_complete"
    }


def test_validation_executor_runs_all_challengers_on_canonical_ledger() -> None:
    result = execute_validation_tournament(
        _qualified_executor_fixture(), frozen_preregistration("d" * 64)
    )

    assert set(result["challenger_results"]) == {"C1", "C2", "C3"}
    assert result["sealed_heldout"]["accessed"] is False
    assert result["sealed_heldout"]["access_log"] == []
    assert result["sealed_heldout"]["result"] is None
    assert result["split"]["development"]["end"] < result["split"]["validation"]["start"]
    assert result["split"]["validation"]["end"] < result["split"]["heldout"]["start"]
    for challenger in result["challenger_results"].values():
        assert challenger["status"] == "validation_complete"
        assert challenger["metrics"]["trade_count"] > 0
        assert challenger["ledger"]["engine_version"] == "portfolio-execution-v1.0"
        assert len(challenger["walk_forward_folds"]) == 5
    assert set(result["challenger_results"]["C2"]["cost_scenarios"]) == {
        "10.0", "25.0", "50.0",
    }


def test_executor_fails_closed_when_strategy_and_spy_marks_are_not_date_aligned(
    monkeypatch,
) -> None:
    sessions = ["2026-01-01", "2026-01-02", "2026-01-03"]
    spy = [
        SessionPrice("SPY", date, 100 + index, 100 + index, volume=1_000_000)
        for index, date in enumerate(sessions)
    ]
    monkeypatch.setattr(
        executor,
        "simulate_portfolio",
        lambda *_args, **_kwargs: SimpleNamespace(
            equity_curve=[
                {"date": sessions[0], "equity": 1_000_000},
                {"date": sessions[2], "equity": 1_010_000},
            ]
        ),
    )

    with pytest.raises(ValueError, match="date-aligned"):
        executor._execute("C3", sessions, {"SPY": spy}, sessions, 1.0)


def test_selected_challenger_consumes_heldout_once_and_replays_stored_result(
    monkeypatch,
) -> None:
    conn = _qualified_executor_fixture()
    conn.commit()

    def passing_run(*_args, **_kwargs):
        return _passing_executor_result()

    monkeypatch.setattr(executor, "_execute", passing_run)
    preregistration = frozen_preregistration("d" * 64)
    first = execute_validation_tournament(
        conn, preregistration, consume_heldout=True
    )
    retry = execute_validation_tournament(
        conn, preregistration, consume_heldout=True
    )

    assert first == retry
    assert first["selected_challenger"] == "C3"
    assert first["prospective_shadow_candidate"] == "C3"
    assert first["sealed_heldout"]["accessed"] is True
    assert first["sealed_heldout"]["result"]["reusable_for_policy_selection"] is False
    assert conn.execute("SELECT COUNT(*) FROM challenger_holdout_access").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM challenger_holdout_results").fetchone()[0] == 1

    with pytest.raises(ValueError, match="already consumed by different evidence"):
        execute_validation_tournament(
            conn, frozen_preregistration("e" * 64), consume_heldout=True
        )
    with pytest.raises(sqlite3.IntegrityError, match="access is immutable"):
        conn.execute("UPDATE challenger_holdout_access SET challenger='C1'")


def test_crash_after_access_log_permanently_fails_closed(monkeypatch) -> None:
    conn = _qualified_executor_fixture()
    conn.commit()
    calls = 0

    def crash_on_first_heldout(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 23:
            raise RuntimeError("simulated heldout executor crash")
        return _passing_executor_result()

    monkeypatch.setattr(executor, "_execute", crash_on_first_heldout)
    preregistration = frozen_preregistration("d" * 64)
    with pytest.raises(RuntimeError, match="simulated heldout executor crash"):
        execute_validation_tournament(
            conn, preregistration, consume_heldout=True
        )

    assert conn.execute("SELECT COUNT(*) FROM challenger_holdout_access").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM challenger_holdout_results").fetchone()[0] == 0
    with pytest.raises(ValueError, match="incomplete and cannot be repeated"):
        execute_validation_tournament(
            conn, preregistration, consume_heldout=True
        )

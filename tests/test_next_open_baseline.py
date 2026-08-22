from __future__ import annotations

import copy
import sqlite3

import pytest

from trader_koo.research.next_open_baseline import (
    BaselineConfig,
    artifact_state,
    canonical_json_bytes,
    run_next_open_baseline,
    write_artifact,
)


def _database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY(ticker, date)
        );
        CREATE TABLE setup_call_evaluations (
            id INTEGER PRIMARY KEY,
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            call_direction TEXT NOT NULL,
            score REAL,
            setup_family TEXT,
            setup_tier TEXT
        );
        """
    )
    dates = [f"2026-01-{day:02d}" for day in range(1, 32)]
    for index, date in enumerate(dates):
        spy = 100.0 + index
        asset = 50.0 + index
        conn.execute(
            "INSERT INTO price_daily VALUES ('SPY', ?, ?, ?, 1000000)",
            (date, spy, spy + 0.5),
        )
        conn.execute(
            "INSERT INTO price_daily VALUES ('AAA', ?, ?, ?, 100000)",
            (date, asset, asset + 0.5),
        )
        conn.execute(
            "INSERT INTO price_daily VALUES ('BBB', ?, ?, ?, 100000)",
            (date, asset + 5.0, asset + 5.5),
        )
    conn.commit()
    return conn


def _call(conn: sqlite3.Connection, call_id: int, date: str, ticker: str = "AAA") -> None:
    conn.execute(
        "INSERT INTO setup_call_evaluations VALUES (?, ?, ?, 'long', 90, 'bullish', 'A')",
        (call_id, date, ticker),
    )
    conn.commit()


def test_uses_immediate_next_open_and_exact_tenth_close() -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01")
    artifact = run_next_open_baseline(conn, consume_heldout=False)

    assert artifact["summary"]["closed_trades"] == 1
    trade = artifact["trades"][0]
    assert trade["entry_date"] == "2026-01-02"
    assert trade["exit_date"] == "2026-01-11"
    assert trade["entry_price"] == pytest.approx(51.0 * 1.001)
    assert trade["exit_price"] == pytest.approx(60.5 * 0.999)
    assert len(artifact["equity_curve"]) == 11
    assert artifact["summary"]["daily_observation_count"] == 11
    assert artifact["summary"]["effective_non_overlapping_block_count"] == 1.1
    assert artifact["confidence_intervals"] is None


def test_missing_immediate_open_is_excluded_not_delayed() -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01")
    conn.execute("UPDATE price_daily SET open=NULL WHERE ticker='AAA' AND date='2026-01-02'")
    conn.commit()

    artifact = run_next_open_baseline(conn, consume_heldout=False)

    assert artifact["summary"]["closed_trades"] == 0
    assert artifact["exclusions"] == [{
        "call_id": 1,
        "reason": "immediate_next_session_open_missing",
        "required_entry_date": "2026-01-02",
    }]


def test_missing_exact_exit_is_excluded_not_closed_later() -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01")
    conn.execute("UPDATE price_daily SET close=NULL WHERE ticker='AAA' AND date='2026-01-11'")
    conn.commit()

    artifact = run_next_open_baseline(conn, consume_heldout=False)

    assert artifact["summary"]["closed_trades"] == 0
    assert artifact["exclusions"][0]["reason"] == "exact_tenth_session_close_missing"
    assert artifact["exclusions"][0]["required_exit_date"] == "2026-01-11"


def test_unpriced_benchmark_financing_nulls_active_metrics() -> None:
    conn = _database()
    conn.execute(
        "INSERT INTO setup_call_evaluations VALUES (1, '2026-01-01', 'AAA', 'short', 90, 'bearish', 'A')"
    )
    conn.commit()
    artifact = run_next_open_baseline(
        conn,
        config=BaselineConfig(spy_short_borrow_bps_annual=None),
        consume_heldout=False,
    )

    assert artifact["summary"]["closed_trades"] == 1
    assert artifact["summary"]["active_metrics_available"] is False
    assert artifact["summary"]["matched_spy_net_pnl"] is None
    assert artifact["summary"]["active_net_pnl"] is None
    assert artifact["trades"][0]["spy_matched_net_pnl"] is None


def test_holdout_consumption_is_idempotent_and_immutable() -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01")
    _call(conn, 2, "2026-01-20")

    first = run_next_open_baseline(conn)
    second = run_next_open_baseline(conn)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["consumed_window"]["reusable_for_policy_selection"] is False
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM research_holdout_consumptions")
    changed = copy.deepcopy(dataclasses_asdict(BaselineConfig()))
    changed["position_pct"] = 4.0
    with pytest.raises(ValueError, match="already consumed"):
        run_next_open_baseline(conn, config=BaselineConfig(**changed))


def dataclasses_asdict(config: BaselineConfig) -> dict[str, object]:
    # Local helper keeps the test independent of the module's private helpers.
    return {field.name: getattr(config, field.name) for field in config.__dataclass_fields__.values()}


def test_current_equity_capacity_and_concentration_are_reported() -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01")
    conn.execute(
        "INSERT INTO setup_call_evaluations VALUES (2, '2026-01-01', 'BBB', 'long', 89, 'bullish', 'A')"
    )
    conn.commit()
    artifact = run_next_open_baseline(
        conn,
        config=BaselineConfig(position_pct=5.0, max_name_pct=5.0, max_adv_pct=0.01),
        consume_heldout=False,
    )

    assert artifact["summary"]["closed_trades"] == 2
    assert artifact["summary"]["max_name_weight_pct"] <= 5.0
    assert artifact["summary"]["max_gross_exposure_pct"] <= 10.0
    assert all(trade["position_weight_at_entry"] <= 0.05 for trade in artifact["trades"])


def test_market_context_symbols_are_not_strategy_trades() -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01", ticker="SPY")
    artifact = run_next_open_baseline(conn, consume_heldout=False)
    assert artifact["summary"]["closed_trades"] == 0
    assert artifact["exclusions"] == [
        {"call_id": 1, "reason": "non_tradable_context_ticker"}
    ]


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN"):
        canonical_json_bytes({"bad": float("nan")})


def test_diagnostic_horizon_is_explicit_and_cannot_consume_holdout() -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01")
    diagnostic = run_next_open_baseline(
        conn, config=BaselineConfig(holding_sessions=5), consume_heldout=False
    )
    assert diagnostic["evidence_state"] == "diagnostic_invalid"
    assert diagnostic["trades"][0]["exit_date"] == "2026-01-06"
    with pytest.raises(ValueError, match="cannot consume"):
        run_next_open_baseline(
            conn, config=BaselineConfig(holding_sessions=5), consume_heldout=True
        )


def test_artifact_loader_verifies_embedded_hash(tmp_path) -> None:
    conn = _database()
    _call(conn, 1, "2026-01-01")
    artifact = run_next_open_baseline(conn, consume_heldout=False)
    path = tmp_path / "baseline.json"
    write_artifact(path, artifact)
    loaded = artifact_state(path)
    assert loaded["available"] is True
    tampered = path.read_text().replace('"closed_trades":1', '"closed_trades":2')
    path.write_text(tampered)
    assert artifact_state(path)["error"] == "artifact_hash_mismatch"

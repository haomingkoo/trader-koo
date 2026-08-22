from __future__ import annotations

import dataclasses
import json
import sqlite3

import pytest

from trader_koo.research import next_open_baseline as baseline
from trader_koo.research.next_open_baseline import (
    BaselineConfig,
    ExecutionDecision,
    SessionPrice,
    artifact_state,
    canonical_json_bytes,
    execute_portfolio,
    run_next_open_baseline,
    write_artifact,
)


def _prices(tickers: tuple[str, ...] = ("AAA", "BBB", "SPY"), days: int = 15) -> list[SessionPrice]:
    return [
        SessionPrice(ticker, f"2026-01-{day:02d}", 100 + day, 100.5 + day)
        for ticker in tickers
        for day in range(1, days + 1)
    ]


def _decision(
    decision_id: str,
    *,
    ticker: str = "AAA",
    entry: str = "2026-01-02",
    exit_: str = "2026-01-11",
    capacity: float = 1_000_000,
    score: float = 90,
    direction: str = "long",
    signal: str = "2026-01-01",
) -> ExecutionDecision:
    return ExecutionDecision(
        decision_id, ticker, direction, signal, entry, exit_, score, capacity
    )


def test_canonical_execution_is_next_open_exact_close_and_byte_stable() -> None:
    config = BaselineConfig(initial_capital=100_000, position_pct=10, max_name_pct=10)
    args = ([_decision("1")], _prices(), [f"2026-01-{day:02d}" for day in range(1, 12)], config)
    first = execute_portfolio(*args)
    second = execute_portfolio(*args)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.trades[0]["entry_date"] == "2026-01-02"
    assert first.trades[0]["exit_date"] == "2026-01-11"
    assert first.trades[0]["entry_price"] == pytest.approx(102 * 1.001)
    assert first.trades[0]["exit_price"] == pytest.approx(111.5 * .999)


def test_execution_rejects_future_informed_chronology() -> None:
    with pytest.raises(ValueError, match="signal_date < entry_date <= exit_date"):
        execute_portfolio(
            [_decision("future", signal="2026-01-03", entry="2026-01-02", exit_="2026-01-03")],
            _prices(tickers=("AAA",), days=3),
            ["2026-01-01", "2026-01-02", "2026-01-03"],
            BaselineConfig(initial_capital=100_000),
        )


def test_open_orders_cannot_spend_same_day_close_proceeds() -> None:
    config = BaselineConfig(
        initial_capital=10_500, position_pct=100, max_name_pct=100,
        commission_bps_per_side=0, minimum_commission_per_side=0,
    )
    result = execute_portfolio(
        [
            _decision("old", signal="2025-12-31", entry="2026-01-01", exit_="2026-01-02"),
            _decision("new", ticker="BBB", entry="2026-01-02", exit_="2026-01-03"),
        ],
        _prices(days=3),
        ["2026-01-01", "2026-01-02", "2026-01-03"],
        config,
    )
    assert [trade["decision_id"] for trade in result.trades] == ["old"]
    assert {row["decision_id"]: row["reason"] for row in result.exclusions}["new"] == "insufficient_cash"


def test_same_ticker_orders_share_name_and_adv_limits() -> None:
    config = BaselineConfig(
        initial_capital=100_000, position_pct=10, max_name_pct=15, max_adv_pct=100,
        commission_bps_per_side=0, minimum_commission_per_side=0,
    )
    result = execute_portfolio(
        [_decision("1", capacity=12_000), _decision("2", capacity=12_000, score=89)],
        _prices(tickers=("AAA",), days=11),
        [f"2026-01-{day:02d}" for day in range(1, 12)],
        config,
    )
    total = sum(float(trade["entry_notional"]) for trade in result.trades)
    assert total <= 12_000
    assert result.max_name_weight_pct <= 15


def test_cash_interest_and_short_borrow_change_daily_equity_and_sizing() -> None:
    sessions = [f"2026-01-{day:02d}" for day in range(1, 4)]
    decisions = [_decision("1", direction="short", signal="2025-12-31", entry="2026-01-01", exit_="2026-01-03")]
    free = execute_portfolio(
        decisions, _prices(tickers=("AAA",), days=3), sessions,
        BaselineConfig(initial_capital=100_000, cash_rate_bps_annual=0, short_borrow_bps_annual=0),
    )
    financed = execute_portfolio(
        decisions, _prices(tickers=("AAA",), days=3), sessions,
        BaselineConfig(initial_capital=100_000, cash_rate_bps_annual=500, short_borrow_bps_annual=1000),
    )
    assert financed.trades[0]["borrow"] > 0
    assert financed.equity_curve[1]["equity"] != free.equity_curve[1]["equity"]
    unpriced = execute_portfolio(
        decisions, _prices(tickers=("AAA",), days=3), sessions,
        BaselineConfig(initial_capital=100_000, short_borrow_bps_annual=None),
    )
    assert unpriced.financing_priced is False


def test_short_borrow_uses_daily_marked_value() -> None:
    sessions = ["2026-01-01", "2026-01-02", "2026-01-03"]
    decision = _decision("1", direction="short", signal="2025-12-31", entry=sessions[0], exit_=sessions[-1])
    low = _prices(tickers=("AAA",), days=3)
    high = list(low)
    low[1] = dataclasses.replace(low[1], close=50)
    high[1] = dataclasses.replace(high[1], close=200)
    config = BaselineConfig(initial_capital=100_000, cash_rate_bps_annual=0, short_borrow_bps_annual=1000)
    low_result = execute_portfolio([decision], low, sessions, config)
    high_result = execute_portfolio([decision], high, sessions, config)
    assert high_result.trades[0]["borrow"] > low_result.trades[0]["borrow"]


def test_short_entry_respects_marked_name_cap() -> None:
    prices = [
        SessionPrice("AAA", "2026-01-01", 100, 100),
        SessionPrice("AAA", "2026-01-02", 100, 100),
        SessionPrice("AAA", "2026-01-03", 100, 100),
    ]
    result = execute_portfolio(
        [_decision("1", direction="short", signal="2026-01-01", entry="2026-01-02", exit_="2026-01-03")],
        prices,
        [row.date for row in prices],
        BaselineConfig(initial_capital=99_950, position_pct=10, max_name_pct=10),
    )
    assert result.max_name_weight_pct <= 10


def test_short_entry_respects_cap_after_slippage_and_commission() -> None:
    prices = [
        SessionPrice("AAA", "2026-01-01", 100, 100),
        SessionPrice("AAA", "2026-01-02", 100, 100),
    ]
    result = execute_portfolio(
        [_decision("1", direction="short", signal="2025-12-31", entry="2026-01-01", exit_="2026-01-02")],
        prices,
        ["2026-01-01", "2026-01-02"],
        BaselineConfig(initial_capital=100_000, position_pct=10, max_name_pct=10),
    )
    assert result.trades[0]["shares"] == 99
    assert result.max_name_weight_pct <= 10


def test_same_session_exit_still_records_opening_risk() -> None:
    result = execute_portfolio(
        [_decision("1", signal="2025-12-31", entry="2026-01-01", exit_="2026-01-01")],
        [SessionPrice("AAA", "2026-01-01", 100, 101)],
        ["2026-01-01"],
        BaselineConfig(initial_capital=100_000, position_pct=10, max_name_pct=10),
    )
    assert result.trades
    assert 0 < result.max_name_weight_pct <= 10
    assert 0 < result.max_gross_exposure_pct <= 10


def test_null_marks_are_not_valid_observations() -> None:
    prices = _prices(tickers=("AAA",), days=3)
    prices[1] = dataclasses.replace(prices[1], close=None)
    result = execute_portfolio(
        [_decision("1", signal="2025-12-31", entry="2026-01-01", exit_="2026-01-03")], prices,
        ["2026-01-01", "2026-01-02", "2026-01-03"], BaselineConfig(initial_capital=100_000),
    )
    assert result.equity_curve[1] == {
        "date": "2026-01-02", "equity": None, "status": "unpriceable_mark"
    }


def _database() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL, date TEXT NOT NULL, open REAL, close REAL, volume REAL,
            adjustment_basis TEXT, adjustment_version TEXT, basis_status TEXT,
            PRIMARY KEY(ticker,date));
        CREATE TABLE report_runs (
            run_id TEXT PRIMARY KEY, status TEXT, is_generation_canonical INTEGER,
            publication_verified INTEGER);
        CREATE TABLE setup_call_evaluations (
            id INTEGER PRIMARY KEY, asof_date TEXT, ticker TEXT, call_direction TEXT,
            score REAL, setup_family TEXT, setup_tier TEXT, report_run_id TEXT);
        INSERT INTO report_runs VALUES ('run','published',1,1);
    """)
    for ticker in ("AAA", "SPY"):
        for day in range(1, 32):
            conn.execute(
                "INSERT INTO price_daily VALUES (?,?,?,?,?,'split_adjusted','v1','verified')",
                (ticker, f"2026-01-{day:02d}", 100 + day, 100.5 + day, 100_000),
            )
    conn.commit()
    return conn


def _eligible_contract(*_args: object, **_kwargs: object) -> tuple[dict[str, object], list[str]]:
    return ({"eligible": True, "basis": "split_adjusted", "distributions_included": False}, [])


def _verified_fixture(
    conn: sqlite3.Connection, _report_dir: object
) -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    rows = conn.execute(
        "SELECT id,asof_date,ticker,call_direction,score,setup_family,setup_tier,report_run_id "
        "FROM setup_call_evaluations ORDER BY asof_date,score DESC,id"
    ).fetchall()
    return ([{
        "call_id": int(row[0]), "signal_date": str(row[1]), "ticker": str(row[2]),
        "direction": str(row[3]), "score": float(row[4]), "setup_family": row[5],
        "setup_tier": row[6], "report_run_id": row[7],
    } for row in rows], [], {"run": {"content_hash": "fixture"}})


def test_sql_adapter_fails_closed_without_accepted_contracts() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE setup_call_evaluations (id INTEGER)")
    artifact = run_next_open_baseline(conn, consume_heldout=False)
    assert artifact["summary"]["closed_trades"] == 0
    assert "report_publication_contract_unavailable" in artifact["readiness_reasons"]
    assert "price_series_revision_unavailable" in artifact["readiness_reasons"]


def test_adapter_uses_prior_session_capacity_and_one_execution_path(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _database()
    conn.execute("INSERT INTO setup_call_evaluations VALUES (1,'2026-01-01','AAA','long',90,'bullish','A','run')")
    conn.execute("UPDATE price_daily SET volume=1 WHERE ticker='AAA' AND date='2026-01-02'")
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    artifact = run_next_open_baseline(conn, consume_heldout=False)
    trade = artifact["trades"][0]
    # Signal-day capacity (101.5 * 100k) is available; entry-day volume=1 is ignored.
    assert trade["entry_notional"] > 1_000
    assert artifact["benchmark_basis"].endswith("canonical_execution_ledger")
    assert artifact["summary"]["full_investment_spy_net_return_pct"] is not None


def test_zero_trade_policy_still_reports_cash_and_spy_opportunity_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _database()
    conn.execute("INSERT INTO setup_call_evaluations VALUES (1,'2026-01-01','AAA','long',90,'bullish','A','run')")
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    artifact = run_next_open_baseline(
        conn, config=BaselineConfig(minimum_score=100), consume_heldout=False,
    )
    assert artifact["summary"]["closed_trades"] == 0
    assert artifact["summary"]["net_return_pct"] == pytest.approx(0)
    assert artifact["summary"]["full_investment_spy_net_return_pct"] is not None
    assert artifact["summary"]["opportunity_cost_vs_full_spy_pct"] is not None


def test_failed_full_spy_path_is_unpriced(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _database()
    conn.execute("INSERT INTO setup_call_evaluations VALUES (1,'2026-01-01','AAA','long',90,'bullish','A','run')")
    conn.execute("UPDATE price_daily SET open=NULL WHERE ticker='SPY' AND date='2026-01-02'")
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    artifact = run_next_open_baseline(conn, consume_heldout=False)
    assert artifact["summary"]["closed_trades"] == 1
    assert artifact["summary"]["full_investment_spy_net_return_pct"] is None
    assert artifact["summary"]["opportunity_cost_vs_full_spy_pct"] is None
    assert "full_investment_spy_unpriced" in artifact["readiness_reasons"]


def test_effective_sample_counts_return_intervals(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _database()
    conn.execute("INSERT INTO setup_call_evaluations VALUES (1,'2026-01-01','AAA','long',90,'bullish','A','run')")
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    artifact = run_next_open_baseline(conn, consume_heldout=False)
    summary = artifact["summary"]
    assert summary["equity_point_count"] == summary["daily_observation_count"] + 1
    assert summary["effective_non_overlapping_block_count"] == pytest.approx(
        summary["daily_observation_count"] / 10
    )


def test_adapter_nulls_active_metrics_when_financing_is_unpriced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _database()
    conn.execute(
        "INSERT INTO setup_call_evaluations VALUES "
        "(1,'2026-01-01','AAA','short',90,'bearish','A','run')"
    )
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    artifact = run_next_open_baseline(
        conn,
        config=BaselineConfig(short_borrow_bps_annual=None),
        consume_heldout=False,
    )
    assert artifact["summary"]["active_metrics_available"] is False
    assert artifact["summary"]["net_return_pct"] is None
    assert artifact["summary"]["matched_spy_net_pnl"] is None
    assert artifact["summary"]["active_net_pnl"] is None
    assert "financing_inputs_unpriced" in artifact["readiness_reasons"]


def test_holdout_is_globally_sealed_and_exact_rerun_only(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _database()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    # Widely spaced dates survive purge and make one observed heldout point.
    for call_id, day in enumerate((1, 2, 3, 14), 1):
        conn.execute(
            "INSERT INTO setup_call_evaluations VALUES (?,?,?,?,90,'bullish','A','run')",
            (call_id, f"2026-01-{day:02d}", "AAA", "long"),
        )
    conn.commit()
    first = run_next_open_baseline(conn)
    second = run_next_open_baseline(conn)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["consumed_window"]["consumed"] is True
    conn.execute("INSERT INTO setup_call_evaluations VALUES (9,'2026-01-15','AAA','long',89,'bullish','A','run')")
    conn.commit()
    with pytest.raises(ValueError, match="already consumed"):
        run_next_open_baseline(conn)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM research_holdout_dates")
    with pytest.raises(sqlite3.IntegrityError, match="sealed"):
        conn.execute(
            "INSERT INTO research_holdout_consumptions VALUES ('forged','x','a','b','c','d',0)"
        )
    with pytest.raises(sqlite3.IntegrityError, match="sealed"):
        conn.execute("INSERT INTO research_holdout_dates VALUES ('2099-01-01','forged')")


def test_baseline_preserves_caller_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _database()
    conn.execute("CREATE TABLE caller_work (value TEXT)")
    for call_id, day in enumerate((1, 2, 3, 14), 1):
        conn.execute(
            "INSERT INTO setup_call_evaluations VALUES (?,?,?,?,90,'bullish','A','run')",
            (call_id, f"2026-01-{day:02d}", "AAA", "long"),
        )
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    conn.execute("BEGIN")
    conn.execute("INSERT INTO caller_work VALUES ('keep-owned')")
    run_next_open_baseline(conn)
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM caller_work").fetchone()[0] == 0


def test_unlinked_setup_call_fails_lineage_closed(tmp_path) -> None:
    conn = _database()
    conn.execute(
        "INSERT INTO setup_call_evaluations VALUES (1,'2026-01-01','AAA','long',90,'bullish','A',NULL)"
    )
    calls, reasons, lineage = baseline._verified_calls(conn, tmp_path)
    assert calls == []
    assert reasons == ["setup_call_report_run_id_missing"]
    assert lineage == {}


def test_artifact_loader_rejects_content_and_implementation_tamper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    minimal = {
        "available": True,
        "provenance": {"implementation_sha256": baseline._implementation_hash()},
        "summary": {"closed_trades": 1},
    }
    minimal["provenance"]["artifact_sha256"] = baseline._sha256(minimal)
    minimal_path = tmp_path / "minimal.json"
    write_artifact(minimal_path, minimal)
    assert artifact_state(minimal_path)["error"] == "artifact_schema_invalid"

    conn = _database()
    conn.execute("INSERT INTO setup_call_evaluations VALUES (1,'2026-01-01','AAA','long',90,'bullish','A','run')")
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    artifact = run_next_open_baseline(conn, consume_heldout=False)
    path = tmp_path / "baseline.json"
    write_artifact(path, artifact)
    assert artifact_state(path)["available"] is True

    payload = json.loads(path.read_text())
    payload["summary"]["closed_trades"] = 2
    path.write_text(json.dumps(payload))
    assert artifact_state(path)["error"] == "artifact_schema_invalid"

    payload["summary"]["closed_trades"] = artifact["summary"]["closed_trades"]
    payload["provenance"]["implementation_sha256"] = "0" * 64
    check = dict(payload)
    check["provenance"] = dict(payload["provenance"])
    check["provenance"].pop("artifact_sha256")
    payload["provenance"]["artifact_sha256"] = baseline._sha256(check)
    path.write_text(json.dumps(payload))
    assert artifact_state(path)["error"] == "implementation_hash_mismatch"


def test_artifact_loader_rejects_malformed_hash_valid_payload(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _database()
    conn.execute("INSERT INTO setup_call_evaluations VALUES (1,'2026-01-01','AAA','long',90,'bullish','A','run')")
    conn.commit()
    monkeypatch.setattr(baseline, "_price_contract", _eligible_contract)
    monkeypatch.setattr(baseline, "_verified_calls", _verified_fixture)
    payload = run_next_open_baseline(conn, consume_heldout=False)
    payload.update({"evidence_state": "promotion_ready", "readiness_status": "eligible"})
    payload["summary"].update({"selected_calls": "forged", "closed_trades": -99, "initial_capital": "money"})
    payload["config"] = {}
    payload["execution_contract"] = {}
    payload["provenance"]["config_sha256"] = "z" * 64
    payload["provenance"].pop("artifact_sha256")
    payload["provenance"]["artifact_sha256"] = baseline._sha256(payload)
    path = tmp_path / "forged.json"
    write_artifact(path, payload)
    assert artifact_state(path)["error"] == "artifact_schema_invalid"


def test_canonical_json_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="NaN"):
        canonical_json_bytes({"bad": float("nan")})


def test_checked_in_browser_fixture_passes_backend_loader() -> None:
    fixture = baseline.IMPLEMENTATION_PATH.resolve().parents[2] / "tests/fixtures/next_open_baseline_schema_v2.json"
    state = artifact_state(fixture)
    assert state["available"] is True
    assert state["schema_version"] == baseline.SCHEMA_VERSION

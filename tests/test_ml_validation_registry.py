from __future__ import annotations

import sqlite3

from trader_koo.research.next_open_baseline import canonical_json_bytes

from trader_koo.ml.validation_registry import (
    champion_status,
    evaluate_champion_eligibility,
    list_validation_runs,
    model_version_card,
    record_validation_run,
)
import hashlib


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _good_result(*, include_rule_baseline: bool = True) -> dict:
    def ledger() -> dict:
        body = {
            "schema_version": "portfolio-ledger-v1",
            "engine_version": "portfolio-execution-v1.0",
            "cash": {"initial": 100_000.0, "final": 118_000.0},
            "equity": [
                {"date": "2025-01-01", "equity": 100_000.0},
                {"date": "2026-01-01", "equity": 118_000.0},
            ],
            "provenance": {"config_sha256": "a" * 64, "input_sha256": "b" * 64},
        }
        return {
            **body,
            "provenance": {
                **body["provenance"],
                "ledger_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
            },
        }

    result = {
        "ok": True,
        "config": {
            "start_date": "2025-01-01",
            "end_date": "2026-01-01",
        },
        "training": {
            "ok": True,
            "target_mode": "barrier",
            "fold_count": 8,
            "total_samples": 5000,
            "model_path": "/data/models/model.txt",
            "aggregate_metrics": {
                "avg_auc": 0.56,
                "best_auc": 0.61,
                "avg_accuracy": 0.53,
                "avg_precision": 0.52,
            },
        },
        "backtest": {
            "ok": True,
            "execution_ledger": ledger(),
            "summary": {
                "return_basis": "total_return",
                "benchmark_return_basis": "total_return",
                "distributions_included": True,
                "total_return_pct": 18.0,
                "spy_return_pct": 10.0,
                "alpha_pct": 8.0,
                "max_drawdown_pct": -8.5,
                "total_trades": 90,
                "profit_factor": 1.45,
            },
        },
    }
    if include_rule_baseline:
        result["rule_baseline"] = {
            "return_pct": 12.0,
            "execution_ledger": ledger(),
            "summary": {
                "return_pct": 12.0,
                "return_basis": "total_return",
                "benchmark_return_basis": "total_return",
                "distributions_included": True,
            },
        }
    return result


def test_missing_rule_baseline_blocks_champion_eligibility():
    evaluation = evaluate_champion_eligibility(_good_result(include_rule_baseline=False))

    assert evaluation["champion_eligible"] is False
    assert "missing_rule_baseline" in evaluation["eligibility_reasons"]


def test_good_result_with_rule_baseline_is_champion_eligible():
    evaluation = evaluate_champion_eligibility(_good_result())

    assert evaluation["champion_eligible"] is True
    assert evaluation["eligibility_reasons"] == []
    assert evaluation["metrics"]["alpha_vs_rule_baseline_pct"] == 6.0


def test_summary_only_rule_baseline_is_not_promotion_evidence():
    result = _good_result(include_rule_baseline=False)
    result["rule_baseline"] = {
        "ok": True,
        "summary": {
            "method": "current_rule_technical_proxy",
            "return_pct": 11.5,
        },
    }

    evaluation = evaluate_champion_eligibility(result)

    assert evaluation["champion_eligible"] is False
    assert "rule_canonical_execution_ledger_missing" in evaluation["eligibility_reasons"]
    assert evaluation["metrics"]["rule_baseline_return_pct"] == 11.5
    assert evaluation["metrics"]["alpha_vs_rule_baseline_pct"] == 6.5


def test_mismatched_comparison_dates_block_promotion():
    result = _good_result()
    result["rule_baseline"]["execution_ledger"]["equity"][1]["date"] = "2025-12-31"
    ledger = result["rule_baseline"]["execution_ledger"]
    body = {
        **ledger,
        "provenance": {
            key: value for key, value in ledger["provenance"].items()
            if key != "ledger_sha256"
        },
    }
    ledger["provenance"]["ledger_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()

    evaluation = evaluate_champion_eligibility(result)

    assert evaluation["champion_eligible"] is False
    assert "model_rule_equity_dates_not_identical" in evaluation["eligibility_reasons"]


def test_record_validation_run_persists_promotion_status():
    conn = _conn()

    row = record_validation_run(
        conn,
        _good_result(include_rule_baseline=False),
        source="test",
        artifact_path="/tmp/artifact.json",
        run_id="run-1",
    )
    runs = list_validation_runs(conn)
    status = champion_status(conn)

    assert row["promotion_status"] == "blocked"
    assert row["artifact_path"] == "/tmp/artifact.json"
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["champion_eligible"] is False
    assert status["latest_run"]["run_id"] == "run-1"
    assert status["latest_eligible_run"] is None
    assert status["promotion_gates"]["requires_rule_baseline"] is True


def test_champion_status_returns_latest_eligible_run():
    conn = _conn()
    record_validation_run(
        conn,
        _good_result(include_rule_baseline=False),
        source="test",
        run_id="blocked",
    )
    record_validation_run(
        conn,
        _good_result(include_rule_baseline=True),
        source="test",
        run_id="eligible",
    )

    status = champion_status(conn)

    assert status["latest_run"]["run_id"] == "eligible"
    assert status["latest_eligible_run"]["run_id"] == "eligible"
    assert status["latest_eligible_run"]["champion_eligible"] is True


def test_model_version_card_explains_no_runs():
    conn = _conn()

    card = model_version_card(conn)

    assert card["deployment_state"] == "no_validation_runs"
    assert card["candidate"] is None
    assert card["latest_eligible"] is None
    assert card["promotion_gates"]["requires_rule_baseline"] is True


def test_model_version_card_keeps_previous_eligible_when_latest_blocked():
    conn = _conn()
    record_validation_run(
        conn,
        _good_result(include_rule_baseline=True),
        source="test",
        artifact_path="/tmp/eligible.json",
        run_id="eligible",
    )
    record_validation_run(
        conn,
        _good_result(include_rule_baseline=False),
        source="test",
        artifact_path="/tmp/blocked.json",
        run_id="blocked",
    )

    card = model_version_card(conn)

    assert card["deployment_state"] == "using_previous_eligible"
    assert card["candidate"]["run_id"] == "blocked"
    assert card["candidate"]["promotion_status"] == "blocked"
    assert "missing_rule_baseline" in card["candidate"]["eligibility_reasons"]
    assert card["latest_eligible"]["run_id"] == "eligible"
    assert card["latest_eligible"]["artifact_path"] == "/tmp/eligible.json"

"""Integration tests for the paper trades router endpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import sqlite3
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers.admin import paper_campaigns
from trader_koo.backend.routers.admin import router as admin_router
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator
from trader_koo.paper_trades import create_paper_trades_from_report
from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trade.campaign import (
    canonical_hash, record_experiment_preregistration, record_promotion_experiment,
)
from trader_koo.paper_trade.config import config_snapshot
from trader_koo.paper_trades import _build_config
from trader_koo.report.runs import complete_report_run, publish_report_run


def _publish_report_fixture(
    conn: sqlite3.Connection,
    report_dir: Path,
    *,
    run_id: str,
    generated_ts: str,
) -> None:
    ensure_paper_trade_schema(conn)
    config_hash = hashlib.sha256(b"{}").hexdigest()
    conn.execute(
        """INSERT INTO report_runs
           (run_id,report_kind,status,started_ts,config_json,config_hash,code_version,
            is_generation_canonical,publication_verified)
           VALUES (?,'daily','started','2026-08-21T10:00:00Z','{}',?,?,0,0)""",
        (run_id, config_hash, "a" * 40),
    )
    conn.commit()
    decision = {
        "ticker": "REJECT", "selected_rank": 1, "decision": "accepted",
        "reason_codes": ["selected_report_cohort"], "inputs": {},
    }
    report = {
        "generated_ts": generated_ts,
        "meta": {"report_kind": "daily", "report_run": {"run_id": run_id}},
        "latest_data": {},
        "signals": {"report_decisions": [decision], "scanned_universe": ["REJECT"]},
        "counts": {}, "risk_filters": {}, "warnings": [], "ok": True,
    }
    stamp = generated_ts.replace("-", "").replace(":", "")
    artifact = report_dir / f"daily_report_{stamp}_{run_id}.json"
    markdown = report_dir / f"daily_report_{stamp}_{run_id}.md"
    artifact.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text("fixture\n", encoding="utf-8")
    complete_report_run(
        conn, run_id=run_id, report=report, artifact_path=artifact,
        markdown_path=markdown,
        content_hash=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        completed_ts="2026-08-21T12:00:00Z",
    )
    publish_report_run(conn, run_id=run_id, report_dir=report_dir)


class TestPaperTradesListEndpoint:
    def test_paper_trades_returns_200(self, test_app):
        response = test_app.get("/api/paper-trades")

        assert response.status_code == 200

    def test_paper_trades_has_trades_key(self, test_app):
        response = test_app.get("/api/paper-trades")
        data = response.json()

        assert "trades" in data
        assert isinstance(data["trades"], list)

    def test_paper_trades_has_ok_key(self, test_app):
        response = test_app.get("/api/paper-trades")
        data = response.json()

        assert "ok" in data
        assert data["ok"] is True

    def test_paper_trades_with_status_filter(self, test_app):
        response = test_app.get("/api/paper-trades?status=open")

        assert response.status_code == 200


class TestPaperTradeSummaryEndpoint:
    def test_summary_returns_200(self, test_app):
        response = test_app.get("/api/paper-trades/summary")

        assert response.status_code == 200

    def test_summary_has_overall_key(self, test_app, seeded_conn, tmp_path):
        _publish_report_fixture(
            seeded_conn, tmp_path, run_id="api-report-run-1",
            generated_ts="2026-08-21T11:00:00Z",
        )
        create_paper_trades_from_report(
            seeded_conn,
            setup_rows=[{
                "ticker": "REJECT",
                "setup_tier": "F",
                "score": 10.0,
                "actionability": "watch",
                "signal_bias": "bullish",
                "close": 100.0,
            }],
            report_date="2026-08-21",
            generated_ts="generated-1",
            report_run_id="api-report-run-1",
        )
        seeded_conn.commit()
        response = test_app.get("/api/paper-trades/summary")
        data = response.json()

        assert "overall" in data
        assert "policy" in data
        assert "feedback" in data
        evidence = data["strategy_evidence"]
        assert evidence["readiness_status"] == "insufficient_history"
        assert evidence["observation_count"] == 20
        assert evidence["traded_signal_date_count"] == 4
        assert evidence["effective_non_overlapping_block_count"] == 2.0
        assert evidence["consumed_window"]["reusable_for_policy_selection"] is False
        assert evidence["causal_validity"]["valid"] is False
        assert evidence["return_basis"] == "split_adjusted_price_return_only_dividends_omitted"
        assert data["campaign_health"]["campaign_id"] == "paper-v2"
        assert data["campaign_health"]["policy_version"] == "paper-campaign-v2.0"
        assert data["campaign_health"]["campaigns"][0]["campaign_id"] == "paper-v1"
        assert data["campaign_health"]["latest_report"]["report_run_id"] == "api-report-run-1"
        assert data["campaign_health"]["latest_report"]["ranked"] == 1
        assert data["campaign_health"]["latest_report"]["rejections_by_gate"] == [{
            "gate": "eligibility.tier",
            "reason_code": "tier_below_minimum",
            "count": 1,
        }]

    def test_summary_ok_is_true(self, test_app):
        response = test_app.get("/api/paper-trades/summary")
        data = response.json()

        assert data["ok"] is True

    def test_exact_evidence_provenance_route(self, test_app):
        summary = test_app.get("/api/paper-trades/summary").json()
        provenance = summary["strategy_evidence"]["provenance"]

        response = test_app.get(provenance["href"])

        assert response.status_code == 200
        state = response.json()["strategy_evidence"]
        assert state["provenance"]["artifact_sha256"] == provenance["artifact_sha256"]
        assert state["provenance"]["input_hash_sha256"] == provenance["input_hash_sha256"]
        assert state["provenance"]["verified"] is True

    def test_missing_packaged_evidence_fails_closed_through_real_api(
        self, test_app, monkeypatch, tmp_path
    ):
        manifest = tmp_path / "strategy_evidence_20260822.json"
        manifest.write_text(
            '{"artifact_file":"missing.json","input_manifest_file":"inputs.json"}',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "trader_koo.research.strategy_evidence._SNAPSHOT_PATH", manifest
        )

        response = test_app.get("/api/paper-trades/summary")

        assert response.status_code == 200
        evidence = response.json()["strategy_evidence"]
        assert evidence["readiness_status"] == "evidence_unavailable"
        assert evidence["decision_eligible"] is False
        assert evidence["provenance"]["verified"] is False
        assert evidence["provenance"]["href"] is None

    def test_wrong_evidence_hash_fails_closed(self, test_app):
        response = test_app.get(
            f"/api/research/strategy-evidence/{'0' * 64}/inputs/{'1' * 64}"
        )

        assert response.status_code == 404

    def test_missing_next_open_artifact_is_visible_and_fails_closed(
        self, test_app, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(
            "TRADER_KOO_NEXT_OPEN_BASELINE_ARTIFACT",
            str(tmp_path / "missing.json"),
        )
        response = test_app.get("/api/research/next-open-baseline")
        assert response.status_code == 200
        baseline = response.json()["baseline"]
        assert baseline["available"] is False
        assert baseline["causal_valid"] is False
        assert baseline["decision_eligible"] is False

    def test_experiment_results_keep_failed_tournament_visible(self, test_app):
        response = test_app.get("/api/research/experiments")

        assert response.status_code == 200
        experiments = response.json()["experiments"]
        assert {item["experiment_id"] for item in experiments} == {
            "next-open-baseline", "challenger-tournament",
        }
        tournament = next(
            item for item in experiments
            if item["experiment_id"] == "challenger-tournament"
        )
        assert tournament["evidence_label"] == "invalid"
        assert tournament["status"] == "blocked_before_validation"
        assert tournament["heldout"]["accessed"] is False
        assert set(tournament["challengers"]) == {"C1", "C2", "C3"}

    def test_experiment_manifest_is_downloadable_but_missing_ledger_is_not(
        self, test_app
    ):
        manifest = test_app.get(
            "/api/research/experiments/challenger-tournament/download/manifest"
        )
        missing = test_app.get(
            "/api/research/experiments/challenger-tournament/download/ledger"
        )

        assert manifest.status_code == 200
        assert manifest.json()["artifact_sha256"]
        assert missing.status_code == 404

    def test_unknown_experiment_fails_closed(self, test_app):
        response = test_app.get("/api/research/experiments/unknown")
        assert response.status_code == 404

    def test_sealed_decisions_api_preserves_exact_rank_gate_and_hashes(
        self, test_app, seeded_conn, tmp_path
    ):
        _publish_report_fixture(
            seeded_conn, tmp_path, run_id="api-decision-run",
            generated_ts="2026-08-21T11:00:00Z",
        )
        create_paper_trades_from_report(
            seeded_conn,
            setup_rows=[{
                "ticker": "REJECT", "setup_tier": "F", "score": 10.0,
                "actionability": "watch", "signal_bias": "bullish", "close": 100.0,
            }],
            report_date="2026-08-21", generated_ts="api-decision-ts",
            report_run_id="api-decision-run",
        )
        seeded_conn.commit()

        response = test_app.get(
            "/api/paper-trades/decisions?report_run_id=api-decision-run"
        )
        assert response.status_code == 200
        decision = response.json()["decisions"][0]
        assert (decision["candidate_rank"], decision["ticker"]) == (1, "REJECT")
        assert (decision["final_gate"], decision["reason_code"]) == (
            "eligibility.tier", "tier_below_minimum"
        )
        assert decision["tradeability"] == "not_actionable"
        assert len(decision["inputs_hash"]) == 64
        assert len(decision["candidates_hash"]) == 64


def test_campaign_transition_route_requires_identity_and_audits_actor(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "campaign-admin.db"
    conn = sqlite3.connect(db_path)
    ensure_paper_trade_schema(conn)
    conn.close()

    def open_db():
        return sqlite3.connect(db_path)

    monkeypatch.setattr(paper_campaigns, "get_conn", open_db)
    app = FastAPI()
    app.state.admin_authenticator = AdminAuthenticator(
        AdminAuthConfig(api_key="x" * 32, username="campaign-owner")
    )
    app.include_router(admin_router)
    unauthenticated = TestClient(app).post(
        "/api/admin/paper-campaigns/paper-v2/transition",
        json={
            "action": "activate", "reason": "paper validation approved",
            "idempotency_key": "admin-activation-001",
        },
    )
    assert unauthenticated.status_code == 401

    authenticated_app = FastAPI()
    authenticated_app.state.admin_authenticator = AdminAuthenticator(
        AdminAuthConfig(api_key="x" * 32, username="campaign-owner")
    )
    authenticated_app.include_router(admin_router)
    response = TestClient(authenticated_app).post(
        "/api/admin/paper-campaigns/paper-v2/transition",
        headers={"X-API-Key": "x" * 32},
        json={
            "action": "activate", "reason": "paper validation approved",
            "idempotency_key": "admin-activation-001",
        },
    )
    assert response.status_code == 409
    approval_conn = sqlite3.connect(db_path)
    config = _build_config()
    record_experiment_preregistration(
        approval_conn, preregistration_id="admin-prereg", campaign_id="paper-v2",
        policy_version=config.decision_version,
        policy_hash=canonical_hash(config_snapshot(config)), dataset_hash="d" * 64,
        gates={
            "risk_gates": {"max_drawdown_pct": 10.0},
            "active_return_gate": {"minimum_pct": 0.0},
        },
    )
    experiment = record_promotion_experiment(
        approval_conn, experiment_id="admin-exp", preregistration_id="admin-prereg",
        campaign_id="paper-v2",
        policy_version=config.decision_version,
        policy_hash=canonical_hash(config_snapshot(config)), dataset_hash="d" * 64,
        metrics={
            "engine_version": "portfolio-execution-v1.0", "closed_trades": 2,
            "conversion_rate_pct": 50.0, "average_exposure_pct": 10.0,
            "turnover_pct": 20.0, "portfolio_return_pct": 2.0,
            "matched_spy_return_pct": 1.0,
            "matched_spy_active_return_pct": 1.0,
            "max_drawdown_pct": 1.0, "profit_factor": 1.5,
            "mean_trade_return_pct_ci95": [-1.0, 2.0],
            "walk_forward": {"folds": [{"metrics": {"closed_trades": 1}}]},
            "held_out": {"metrics": {"closed_trades": 1}, "trades": []},
        },
        parity_status="matched",
    )
    approval_conn.commit()
    approval_conn.close()
    approval_response = TestClient(authenticated_app).post(
        "/api/admin/paper-campaigns/paper-v2/approvals",
        headers={"X-API-Key": "x" * 32},
        json={
            "approval_id": "admin-approval", "experiment_id": "admin-exp",
            "experiment_evidence_hash": experiment["evidence_hash"],
            "reason": "approved evidence",
            "artifact": {"approved": True, "signed": True},
        },
    )
    assert approval_response.status_code == 200
    response = TestClient(authenticated_app).post(
        "/api/admin/paper-campaigns/paper-v2/transition",
        headers={"X-API-Key": "x" * 32},
        json={
            "action": "activate", "reason": "paper validation approved",
            "idempotency_key": "admin-activation-002",
        },
    )
    assert response.status_code == 200
    verify = sqlite3.connect(db_path)
    assert verify.execute(
        "SELECT actor,reason FROM paper_campaign_audit"
    ).fetchone() == ("campaign-owner", "paper validation approved")
    verify.close()


class TestPaperTradeDetailEndpoint:
    def test_nonexistent_trade_returns_404(self, test_app):
        response = test_app.get("/api/paper-trades/99999")

        assert response.status_code == 404

    def test_nonexistent_trade_error_message(self, test_app):
        response = test_app.get("/api/paper-trades/99999")
        data = response.json()

        assert "detail" in data
        assert "99999" in data["detail"]

    def test_inserted_trade_returns_200(self, test_app, seeded_conn):
        seeded_conn.execute(
            """INSERT INTO paper_trades
               (report_date, ticker, direction, entry_price, entry_date, status,
                current_price, generated_ts)
               VALUES ('2026-03-14', 'SPY', 'long', 580.0, '2026-03-14', 'open',
                       580.0, '2026-03-14T22:00:00Z')"""
        )
        seeded_conn.commit()
        row = seeded_conn.execute("SELECT id FROM paper_trades WHERE ticker='SPY'").fetchone()
        trade_id = row[0]

        response = test_app.get(f"/api/paper-trades/{trade_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["trade"]["ticker"] == "SPY"
        assert "decision_state" in data["trade"]

"""Integration tests for the paper trades router endpoints."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


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

    def test_summary_has_overall_key(self, test_app):
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

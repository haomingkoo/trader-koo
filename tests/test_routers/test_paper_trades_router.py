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

    def test_summary_ok_is_true(self, test_app):
        response = test_app.get("/api/paper-trades/summary")
        data = response.json()

        assert data["ok"] is True


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

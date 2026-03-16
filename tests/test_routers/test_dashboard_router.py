"""Integration tests for the dashboard router endpoints."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class TestTickersEndpoint:
    def test_tickers_returns_200(self, test_app):
        response = test_app.get("/api/tickers")

        assert response.status_code == 200

    def test_tickers_returns_list(self, test_app):
        response = test_app.get("/api/tickers")
        data = response.json()

        assert "tickers" in data
        assert isinstance(data["tickers"], list)

    def test_tickers_contains_seeded_spy(self, test_app):
        response = test_app.get("/api/tickers")
        data = response.json()

        assert "SPY" in data["tickers"]

    def test_tickers_count_matches_list_length(self, test_app):
        response = test_app.get("/api/tickers")
        data = response.json()

        assert data["count"] == len(data["tickers"])


class TestDashboardEndpoint:
    @patch("trader_koo.backend.services.chart_builder.llm_status", return_value={})
    @patch(
        "trader_koo.backend.services.chart_builder.maybe_rewrite_setup_copy",
        return_value=None,
    )
    @patch(
        "trader_koo.backend.services.chart_builder.get_ticker_earnings_markers",
        return_value=[],
    )
    @patch(
        "trader_koo.backend.services.chart_builder.latest_report_setup_for_ticker",
        return_value=None,
    )
    def test_dashboard_spy_returns_200(self, mock_setup, mock_earn, mock_llm_rw, mock_llm_st, test_app):
        response = test_app.get("/api/dashboard/SPY?months=1")

        assert response.status_code == 200

    @patch("trader_koo.backend.services.chart_builder.llm_status", return_value={})
    @patch(
        "trader_koo.backend.services.chart_builder.maybe_rewrite_setup_copy",
        return_value=None,
    )
    @patch(
        "trader_koo.backend.services.chart_builder.get_ticker_earnings_markers",
        return_value=[],
    )
    @patch(
        "trader_koo.backend.services.chart_builder.latest_report_setup_for_ticker",
        return_value=None,
    )
    def test_dashboard_spy_has_expected_keys(self, mock_setup, mock_earn, mock_llm_rw, mock_llm_st, test_app):
        response = test_app.get("/api/dashboard/SPY?months=1")
        data = response.json()

        expected_keys = {"ticker", "chart", "levels", "fundamentals"}
        assert expected_keys.issubset(set(data.keys()))
        assert data["ticker"] == "SPY"

    @patch("trader_koo.backend.services.chart_builder.llm_status", return_value={})
    @patch(
        "trader_koo.backend.services.chart_builder.maybe_rewrite_setup_copy",
        return_value=None,
    )
    @patch(
        "trader_koo.backend.services.chart_builder.get_ticker_earnings_markers",
        return_value=[],
    )
    @patch(
        "trader_koo.backend.services.chart_builder.latest_report_setup_for_ticker",
        return_value=None,
    )
    def test_dashboard_spy_chart_is_list(self, mock_setup, mock_earn, mock_llm_rw, mock_llm_st, test_app):
        response = test_app.get("/api/dashboard/SPY?months=1")
        data = response.json()

        assert isinstance(data["chart"], list)
        assert len(data["chart"]) > 0

    def test_dashboard_invalid_ticker_returns_404(self, test_app):
        response = test_app.get("/api/dashboard/INVALID_TICKER_XYZ_999?months=1")

        assert response.status_code == 404

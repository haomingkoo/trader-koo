"""Integration tests for the report router endpoints."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class TestDailyReportEndpoint:
    @patch(
        "trader_koo.backend.services.report_loader.latest_daily_report_json",
        return_value=(None, None),
    )
    @patch(
        "trader_koo.backend.services.pipeline.pipeline_status_snapshot",
        return_value={"active": False, "stage": "idle", "latest_run": None, "run_log_path": "/tmp/log"},
    )
    @patch(
        "trader_koo.backend.routers.report.pipeline_status_snapshot",
        return_value={"active": False, "stage": "idle", "latest_run": None, "run_log_path": "/tmp/log"},
    )
    def test_daily_report_returns_200(self, mock_pipe_r, mock_pipe_s, mock_latest, test_app):
        response = test_app.get("/api/daily-report")

        assert response.status_code == 200

    @patch(
        "trader_koo.backend.services.report_loader.latest_daily_report_json",
        return_value=(None, None),
    )
    @patch(
        "trader_koo.backend.services.pipeline.pipeline_status_snapshot",
        return_value={"active": False, "stage": "idle", "latest_run": None, "run_log_path": "/tmp/log"},
    )
    @patch(
        "trader_koo.backend.routers.report.pipeline_status_snapshot",
        return_value={"active": False, "stage": "idle", "latest_run": None, "run_log_path": "/tmp/log"},
    )
    def test_daily_report_has_ok_and_latest_keys(self, mock_pipe_r, mock_pipe_s, mock_latest, test_app):
        response = test_app.get("/api/daily-report")
        data = response.json()

        assert "ok" in data
        assert "latest" in data

    @patch(
        "trader_koo.backend.services.report_loader.latest_daily_report_json",
        return_value=(None, None),
    )
    @patch(
        "trader_koo.backend.services.pipeline.pipeline_status_snapshot",
        return_value={"active": False, "stage": "idle", "latest_run": None, "run_log_path": "/tmp/log"},
    )
    @patch(
        "trader_koo.backend.routers.report.pipeline_status_snapshot",
        return_value={"active": False, "stage": "idle", "latest_run": None, "run_log_path": "/tmp/log"},
    )
    def test_daily_report_ok_false_when_no_report_files(self, mock_pipe_r, mock_pipe_s, mock_latest, test_app):
        response = test_app.get("/api/daily-report")
        data = response.json()

        assert data["ok"] is False


class TestMarketSummaryEndpoint:
    def test_market_summary_returns_200(self, test_app):
        response = test_app.get("/api/market-summary?days=30")

        assert response.status_code == 200

    def test_market_summary_has_tickers_key(self, test_app):
        response = test_app.get("/api/market-summary?days=30")
        data = response.json()

        assert "tickers" in data
        assert isinstance(data["tickers"], dict)

    def test_market_summary_contains_spy(self, test_app):
        response = test_app.get("/api/market-summary?days=30")
        data = response.json()

        spy = data["tickers"].get("SPY")
        assert spy is not None
        assert "price" in spy
        assert "history" in spy
        assert isinstance(spy["history"], list)
        assert len(spy["history"]) > 0


class TestMarketSentimentEndpoint:
    @patch(
        "trader_koo.structure.fear_greed.get_external_news_sentiment",
        return_value={
            "provider": "alpha_vantage",
            "source_type": "news",
            "available": False,
            "score": None,
            "raw_score": None,
            "label": None,
            "article_count": 0,
            "updated_at": "2026-03-17T12:00:00Z",
            "lookback_hours": 72,
            "tickers": ["SPY", "QQQ", "DIA", "IWM"],
            "topics": ["financial_markets", "economy_macro"],
            "note": "Configure TRADER_KOO_ALPHA_VANTAGE_KEY to enable external news sentiment.",
            "headlines": [],
        },
    )
    def test_market_sentiment_exposes_methodology_metadata(self, _mock_news, test_app):
        response = test_app.get("/api/fear-greed")

        assert response.status_code == 200
        data = response.json()

        assert data["ok"] is True
        assert data["methodology"] == "internal_market_composite"
        assert data["uses_social_sentiment"] is False
        assert isinstance(data["summary"], str)
        assert "No social or news scraping".lower() in data["summary"].lower()
        assert isinstance(data["basis"], list)
        assert "SPY vs 125-day moving average" in data["basis"]
        assert "VIX level" in data["basis"]
        assert "external_news" in data
        assert data["external_news"]["provider"] == "alpha_vantage"
        assert data["external_news"]["source_type"] == "news"
        assert data["blended_score"] is None
        assert isinstance(data["components"], list)
        assert len(data["components"]) == 5

    @patch(
        "trader_koo.structure.fear_greed.get_external_news_sentiment",
        return_value={
            "provider": "alpha_vantage",
            "source_type": "news",
            "available": False,
            "score": None,
            "raw_score": None,
            "label": None,
            "article_count": 0,
            "updated_at": "2026-03-17T12:00:00Z",
            "lookback_hours": 72,
            "tickers": ["SPY", "QQQ", "DIA", "IWM"],
            "topics": ["financial_markets", "economy_macro"],
            "note": "Configure TRADER_KOO_ALPHA_VANTAGE_KEY to enable external news sentiment.",
            "headlines": [],
        },
    )
    def test_market_sentiment_alias_returns_200(self, _mock_news, test_app):
        response = test_app.get("/api/market-sentiment")

        assert response.status_code == 200

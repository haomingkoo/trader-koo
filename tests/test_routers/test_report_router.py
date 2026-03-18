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

    @patch(
        "trader_koo.backend.services.report_loader.latest_daily_report_json",
        return_value=(
            None,
            {
                "generated_ts": "2026-03-16T05:32:19Z",
                "counts": {"tracked_tickers": 510, "price_rows": 1028349},
                "latest_data": {"price_date": "2026-03-13"},
                "latest_ingest_run": {"status": "ok"},
                "signals": {"setup_quality_top": [], "setup_evaluation": {}, "tonight_key_changes": [], "regime_context": None},
                "risk_filters": {"trade_mode": "normal", "hard_blocks": 0, "soft_flags": 0, "conditions": []},
                "yolo": {"summary": {}, "timeframes": []},
            },
        ),
    )
    @patch(
        "trader_koo.backend.services.pipeline.pipeline_status_snapshot",
        return_value={
            "active": False,
            "stage": "idle",
            "latest_run": {
                "finished_ts": "2026-03-17T22:12:02Z",
                "status": "failed",
            },
            "run_log_path": "/tmp/log",
        },
    )
    @patch(
        "trader_koo.backend.routers.report.pipeline_status_snapshot",
        return_value={
            "active": False,
            "stage": "idle",
            "latest_run": {
                "finished_ts": "2026-03-17T22:12:02Z",
                "status": "failed",
            },
            "run_log_path": "/tmp/log",
        },
    )
    def test_daily_report_surfaces_stale_report_detail(self, _mock_pipe_r, _mock_pipe_s, _mock_latest, test_app):
        response = test_app.get("/api/daily-report")
        data = response.json()

        assert response.status_code == 200
        assert "detail" in data
        assert "Report output is stale" in str(data["detail"])


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
    @patch(
        "trader_koo.structure.fear_greed.get_social_sentiment",
        return_value={
            "provider": "reddit_public_json",
            "source_type": "social",
            "available": False,
            "score": None,
            "raw_score": None,
            "label": None,
            "post_count": 0,
            "subreddit_count": 3,
            "updated_at": "2026-03-17T12:00:00Z",
            "lookback_hours": 24,
            "subreddits": ["stocks", "investing", "wallstreetbets"],
            "note": "No Reddit posts passed the engagement and keyword filters for the current window.",
            "bullish_terms_total": 0,
            "bearish_terms_total": 0,
            "posts": [],
            "source_breakdown": [],
        },
    )
    def test_market_sentiment_exposes_methodology_metadata(self, _mock_social, _mock_news, test_app):
        response = test_app.get("/api/fear-greed")

        assert response.status_code == 200
        data = response.json()

        assert data["ok"] is True
        assert data["methodology"] == "internal_market_composite"
        assert data["uses_social_sentiment"] is False
        assert isinstance(data["summary"], str)
        assert "External news and social pulses".lower() in data["summary"].lower()
        assert isinstance(data["basis"], list)
        assert "SPY vs 125-day moving average" in data["basis"]
        assert "VIX level" in data["basis"]
        assert "external_news" in data
        assert "social_sentiment" in data
        assert data["external_news"]["provider"] == "alpha_vantage"
        assert data["external_news"]["source_type"] == "news"
        assert data["social_sentiment"]["provider"] == "reddit_public_json"
        assert "methodology_meta" in data
        assert data["methodology_meta"]["version"] == "2026-03-17.market-sentiment-v2"
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
    @patch(
        "trader_koo.structure.fear_greed.get_social_sentiment",
        return_value={
            "provider": "reddit_public_json",
            "source_type": "social",
            "available": False,
            "score": None,
            "raw_score": None,
            "label": None,
            "post_count": 0,
            "subreddit_count": 3,
            "updated_at": "2026-03-17T12:00:00Z",
            "lookback_hours": 24,
            "subreddits": ["stocks", "investing", "wallstreetbets"],
            "note": "No Reddit posts passed the engagement and keyword filters for the current window.",
            "bullish_terms_total": 0,
            "bearish_terms_total": 0,
            "posts": [],
            "source_breakdown": [],
        },
    )
    def test_market_sentiment_alias_returns_200(self, _mock_social, _mock_news, test_app):
        response = test_app.get("/api/market-sentiment")

        assert response.status_code == 200

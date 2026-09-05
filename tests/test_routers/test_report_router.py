"""Integration tests for the report router endpoints."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class TestDailyReportEndpoint:
    def test_daily_report_ui_view_skips_unrendered_history_verification(self, test_app):
        with patch(
            "trader_koo.backend.services.report_loader.daily_report_history"
        ) as history:
            response = test_app.get("/api/daily-report?view=ui")

        assert response.status_code == 200
        history.assert_not_called()
        assert response.json()["history"] == []

    def test_daily_report_ui_view_omits_unrendered_bulk_fields(self, test_app):
        sealed = {
            "generated_ts": "2026-08-22T12:00:00Z",
            "meta": {"bulk": [1, 2, 3]},
            "signals": {
                "setup_quality_top": [{"ticker": "AAA"}],
                "setup_quality_all": [{"ticker": "AAA"}, {"ticker": "BBB"}],
                "setup_quality_lookup": {"AAA": {"ticker": "AAA"}},
                "report_decisions": [{"ticker": "AAA"}],
                "regime_context": {
                    "ma_matrix": [{}],
                    "comparison": {"series": [{}]},
                },
            },
        }
        with (
            patch(
                "trader_koo.backend.services.report_loader.latest_daily_report_json",
                return_value=(None, sealed),
            ),
            patch(
                "trader_koo.backend.routers.report.pipeline_status_snapshot",
                return_value={"active": False, "stage": "idle", "latest_run": None},
            ),
        ):
            compact = test_app.get("/api/daily-report?view=ui").json()["latest"]
            full = test_app.get("/api/daily-report?view=full").json()["latest"]

        assert compact["signals"]["setup_quality_top"] == [{"ticker": "AAA"}]
        assert "setup_quality_all" not in compact["signals"]
        assert "setup_quality_lookup" not in compact["signals"]
        assert "report_decisions" not in compact["signals"]
        assert "meta" not in compact
        assert full["signals"]["setup_quality_all"][1]["ticker"] == "BBB"
        assert full["meta"]["bulk"] == [1, 2, 3]

    @patch(
        "trader_koo.backend.services.report_loader.latest_daily_report_json",
        return_value=(
            None,
            {
                "generated_ts": "2026-08-22T12:00:00Z",
                "report_run": {
                    "run_id": "run-230",
                    "state": "published",
                    "lineage": "linked",
                    "content_hash": "abc123",
                    "config_hash": "cfg123",
                    "code_version": "sha123",
                    "generation_key": "daily:2026-08-22T12:00:00Z",
                    "canonical_generation": True,
                },
                "signals": {"regime_context": {"ma_matrix": [{}], "comparison": {"series": [{}]}}},
            },
        ),
    )
    @patch(
        "trader_koo.backend.routers.report.pipeline_status_snapshot",
        return_value={"active": False, "stage": "idle", "latest_run": None},
    )
    def test_daily_report_exposes_exact_canonical_provenance(
        self,
        _mock_pipeline,
        _mock_latest,
        test_app,
    ):
        response = test_app.get("/api/daily-report")
        assert response.status_code == 200
        assert response.json()["latest"]["report_run"] == {
            "run_id": "run-230",
            "state": "published",
            "lineage": "linked",
            "content_hash": "abc123",
            "config_hash": "cfg123",
            "code_version": "sha123",
            "generation_key": "daily:2026-08-22T12:00:00Z",
            "canonical_generation": True,
        }
        assert (
            response.json()["latest"]["signals"]["regime_context"]["source"]
            == "not_recorded"
        )

    def test_daily_report_labels_only_a_usable_live_regime_patch(self, test_app):
        sealed = {
            "generated_ts": "2026-08-22T12:00:00Z",
            "signals": {
                "regime_context": {
                    "ma_matrix": [],
                    "comparison": {},
                }
            },
        }
        live = {
            "source": "price_daily:^VIX",
            "asof_date": "2026-08-25",
            "vix": {"ticker": "^VIX", "close": 15.45},
            "ma_matrix": [{"metric": "Close vs MA20"}],
            "comparison": {"series": [{"ticker": "^VIX"}]},
        }
        with (
            patch(
                "trader_koo.backend.services.report_loader.latest_daily_report_json",
                return_value=(None, sealed),
            ),
            patch(
                "trader_koo.backend.routers.report.pipeline_status_snapshot",
                return_value={"active": False, "stage": "idle", "latest_run": None},
            ),
            patch(
                "trader_koo.backend.routers.report._report_build_regime_context",
                return_value=live,
            ),
        ):
            response = test_app.get("/api/daily-report")

        regime = response.json()["latest"]["signals"]["regime_context"]
        assert regime["source"] == "regime_context_live_patch:price_daily:^VIX"
        assert regime["asof_date"] == "2026-08-25"

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
        assert data["detail_code"] == "report_stale"
        assert data["detail_level"] == "warning"
        assert data["detail_blocks_main_report"] is False

    @patch(
        "trader_koo.backend.services.report_loader.latest_daily_report_json",
        return_value=(
            None,
            {
                "generated_ts": "2026-03-17T22:32:19Z",
                "counts": {"tracked_tickers": 510, "price_rows": 1028349},
                "latest_data": {"price_date": "2026-03-17"},
                "latest_ingest_run": {"status": "ok"},
                "signals": {"setup_quality_top": [], "setup_evaluation": {}, "tonight_key_changes": [], "regime_context": None},
                "risk_filters": {"trade_mode": "normal", "hard_blocks": 0, "soft_flags": 0, "conditions": []},
                "yolo": {"summary": {}, "timeframes": []},
                "email": {"attempted": True, "sent": False, "error": "smtp timeout"},
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
                "status": "ok",
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
                "status": "ok",
            },
            "run_log_path": "/tmp/log",
        },
    )
    def test_daily_report_marks_email_failure_as_non_blocking(self, _mock_pipe_r, _mock_pipe_s, _mock_latest, test_app):
        response = test_app.get("/api/daily-report")
        data = response.json()

        assert response.status_code == 200
        assert "email delivery failed" in str(data["detail"]).lower()
        assert data["detail_code"] == "email_delivery_failed"
        assert data["detail_level"] == "warning"
        assert data["detail_blocks_main_report"] is False


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
        response = test_app.get("/api/market-sentiment")

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

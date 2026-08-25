"""Integration tests for the system router endpoints."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, test_app):
        response = test_app.get("/api/health")

        assert response.status_code == 200

    def test_health_response_has_ok_key(self, test_app):
        response = test_app.get("/api/health")
        data = response.json()

        assert "ok" in data

    def test_health_response_has_db_exists_key(self, test_app):
        response = test_app.get("/api/health")
        data = response.json()

        assert "db_exists" in data

    def test_health_db_exists_is_true_with_seeded_db(self, test_app):
        response = test_app.get("/api/health")
        data = response.json()

        assert data["db_exists"] is True


class TestConfigEndpoint:
    def test_config_returns_200(self, test_app):
        response = test_app.get("/api/config")

        assert response.status_code == 200

    def test_config_has_auth_block(self, test_app):
        response = test_app.get("/api/config")
        data = response.json()

        assert "auth" in data
        assert "admin_api_key_required" in data["auth"]
        assert "admin_api_key_header" in data["auth"]

    def test_config_does_not_expose_secrets(self, test_app):
        response = test_app.get("/api/config")
        text = response.text.lower()

        assert "api_key" not in text or "required" in text
        assert "password" not in text
        assert "secret" not in text


class TestVixGlossaryEndpoint:
    def test_vix_glossary_returns_200(self, test_app):
        response = test_app.get("/api/vix-glossary")

        assert response.status_code == 200

    def test_vix_glossary_returns_glossary_key(self, test_app):
        response = test_app.get("/api/vix-glossary")
        data = response.json()

        assert "glossary" in data
        assert isinstance(data["glossary"], dict)
        assert len(data["glossary"]) > 0

    def test_vix_glossary_contains_known_patterns(self, test_app):
        response = test_app.get("/api/vix-glossary")
        glossary = response.json()["glossary"]

        assert "bull_trap" in glossary
        assert "bear_trap" in glossary


class TestMethodologyStatsEndpoint:
    def test_separates_current_campaign_from_legacy_history(self, test_app, seeded_conn):
        with patch("trader_koo.backend.routers.system.get_conn", return_value=seeded_conn):
            response = test_app.get("/api/methodology-stats")

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["paper_campaign_id"] == "paper-v2"
        assert data["paper_trades_total"] == 0
        assert "legacy_paper_trades_total" in data
        assert data["paper_campaign_write_state"] in {"enabled", "paused"}
        assert data["ml_auc"] is None
        assert data["ml_enabled"] is False


class TestVixPatternMarkersEndpoint:
    def test_vix_markers_returns_200(self, test_app):
        response = test_app.get("/api/vix-pattern-markers")

        assert response.status_code == 200

    def test_vix_markers_returns_markers_key(self, test_app):
        response = test_app.get("/api/vix-pattern-markers")
        data = response.json()

        assert "markers" in data
        assert isinstance(data["markers"], dict)
        assert len(data["markers"]) > 0

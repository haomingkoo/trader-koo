"""Tests for the slim app factory in trader_koo.backend.main."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import FileResponse


class TestAppFactory:
    """Verify the app object and its router/middleware configuration.

    These tests import the ``app`` from main.py directly but do NOT
    trigger the lifespan (no TestClient), so they are safe to run
    without a database or scheduler.
    """

    @pytest.fixture(autouse=True)
    def _app(self):
        """Import the app object once per test class."""
        from trader_koo.backend.main import app

        self.app = app

    def test_app_is_fastapi_instance(self):
        assert isinstance(self.app, FastAPI)

    def test_app_title(self):
        assert self.app.title == "trader_koo API"

    def test_all_routers_mounted(self):
        """Verify that all 8 routers contribute routes to the app.

        The expected API prefixes are: system, dashboard, report,
        opportunities, paper_trades, email, usage, admin.
        """
        routes = [getattr(r, "path", "") for r in self.app.routes]
        expected_paths = [
            "/api/health",
            "/api/tickers",
            "/api/daily-report",
            "/api/opportunities",
            "/api/paper-trades",
            "/api/email/subscribe",
            "/api/vix-glossary",
            "/api/market-summary",
        ]
        for path in expected_paths:
            assert any(path in r for r in routes), f"Expected route {path} not found"

    def test_root_route_exists(self):
        routes = [getattr(r, "path", "") for r in self.app.routes]

        assert "/" in routes

    def test_v2_shell_routes_disable_caching(self):
        routes_by_path = {
            getattr(route, "path", ""): route
            for route in self.app.routes
        }

        v2_route = routes_by_path["/v2"]
        v2_response = v2_route.endpoint()

        assert isinstance(v2_response, FileResponse)
        assert v2_response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert v2_response.headers["pragma"] == "no-cache"
        assert v2_response.headers["expires"] == "0"
        assert Path(v2_response.path).name == "index.html"

        v2_fallback = routes_by_path["/v2/{rest_of_path:path}"]
        fallback_response = v2_fallback.endpoint("chart")

        assert isinstance(fallback_response, FileResponse)
        assert fallback_response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert fallback_response.headers["pragma"] == "no-cache"
        assert fallback_response.headers["expires"] == "0"
        assert Path(fallback_response.path).name == "index.html"

    def test_has_middleware(self):
        """The app should register middleware even before the first request."""
        assert len(self.app.user_middleware) > 0

    def test_minimum_route_count(self):
        """With 8 routers + root, expect at least 15 routes."""
        route_count = len(self.app.routes)

        assert route_count >= 15, f"Only {route_count} routes found; expected >= 15"

    def test_docs_disabled_by_default(self):
        assert self.app.docs_url is None
        assert self.app.redoc_url is None

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

    def test_root_shell_route_disables_caching(self):
        routes_by_path = {
            getattr(route, "path", ""): route
            for route in self.app.routes
        }

        root_route = routes_by_path["/"]
        root_response = root_route.endpoint()

        if not isinstance(root_response, FileResponse):
            pytest.skip("dist-v2 not built; root returns JSON fallback")

        assert root_response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert root_response.headers["pragma"] == "no-cache"
        assert root_response.headers["expires"] == "0"
        assert Path(root_response.path).name == "index.html"

    def test_spa_route_disables_caching(self):
        routes_by_path = {
            getattr(route, "path", ""): route
            for route in self.app.routes
        }

        spa_route = routes_by_path.get("/report")
        if spa_route is None:
            pytest.skip("SPA routes not registered (dist-v2 not built)")

        spa_response = spa_route.endpoint()

        assert isinstance(spa_response, FileResponse)
        assert spa_response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert spa_response.headers["pragma"] == "no-cache"
        assert spa_response.headers["expires"] == "0"
        assert Path(spa_response.path).name == "index.html"

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

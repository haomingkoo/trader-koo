"""Integration tests for rate limiting middleware.

Tests Requirements 17.2, 17.3, 17.8
"""

import pytest
from datetime import timedelta
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from trader_koo.ratelimit.service import RateLimiter, RateLimitConfig
from trader_koo.ratelimit.middleware import RateLimitMiddleware


@pytest.fixture
def app():
    """Create test FastAPI app with rate limiting."""
    app = FastAPI()

    # Configure with low limits for testing
    config = RateLimitConfig(
        public_limit=5,
        public_window=timedelta(seconds=60),
        authenticated_limit=10,
        authenticated_window=timedelta(seconds=60),
    )

    rate_limiter = RateLimiter(config)
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter, config=config)

    @app.get("/api/test")
    def test_endpoint():
        return {"message": "success"}

    @app.get("/health")
    def health_endpoint():
        return {"status": "healthy"}

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestRateLimitMiddleware:
    """Test rate limiting middleware integration."""

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are included in response."""
        response = client.get("/api/test")

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])

        assert limit == 5
        assert remaining == 4  # First request consumed one

    def test_rate_limit_enforcement(self, client):
        """Test that rate limits are enforced (Requirement 17.1)."""
        # Make 5 requests (the limit)
        for i in range(5):
            response = client.get("/api/test")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # 6th request should be rate limited
        response = client.get("/api/test")
        assert response.status_code == 429, "6th request should be rate limited"

    def test_http_429_response(self, client):
        """Test HTTP 429 response format (Requirement 17.2)."""
        # Use up rate limit
        for _ in range(5):
            client.get("/api/test")

        # Next request should return 429
        response = client.get("/api/test")

        assert response.status_code == 429
        assert "Retry-After" in response.headers

        data = response.json()
        assert "detail" in data
        assert "limit" in data
        assert "window_seconds" in data
        assert "reset_at" in data

        # Verify Retry-After header (Requirement 17.3)
        retry_after = int(response.headers["Retry-After"])
        assert retry_after > 0

    def test_health_endpoint_bypass(self, client):
        """Test that health check endpoints bypass rate limiting."""
        for _ in range(20):
            response = client.get("/health")
            assert response.status_code == 200, "Health endpoint should not be rate limited"

    def test_non_api_paths_bypass(self, client, app):
        """Test that non-/api/ paths (SPA routes, assets) skip rate limiting."""
        @app.get("/opportunities")
        def spa_route():
            return {"page": "spa"}

        for _ in range(20):
            response = client.get("/opportunities")
            assert response.status_code == 200, "SPA routes should not be rate limited"

    def test_different_ips_independent_limits(self, client, monkeypatch):
        """Test that different IPs have independent rate limits."""
        monkeypatch.setenv("TRADER_KOO_TRUST_PROXY_HEADERS", "1")

        # Use up limit for first IP
        for _ in range(5):
            response = client.get("/api/test", headers={"X-Forwarded-For": "192.168.1.1"})
            assert response.status_code == 200

        # First IP should be rate limited
        response = client.get("/api/test", headers={"X-Forwarded-For": "192.168.1.1"})
        assert response.status_code == 429

        # Second IP should still work
        response = client.get("/api/test", headers={"X-Forwarded-For": "192.168.1.2"})
        assert response.status_code == 200

    def test_spoofed_forwarded_headers_do_not_split_rate_limits_by_default(self, client):
        """Untrusted peers should not bypass limits by rotating X-Forwarded-For."""
        for _ in range(5):
            response = client.get("/api/test", headers={"X-Forwarded-For": "192.168.1.1"})
            assert response.status_code == 200

        response = client.get("/api/test", headers={"X-Forwarded-For": "192.168.1.2"})
        assert response.status_code == 429


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

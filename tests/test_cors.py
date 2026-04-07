"""Unit tests for CORS configuration and middleware.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7**

This module tests the CORS middleware implementation to ensure:
- Allowed origins pass through
- Disallowed origins are rejected
- Rejection logging works correctly
- Development mode localhost exception works
- Access-Control-Allow-Credentials header is set to false
"""

import os
import pytest
import anyio
from unittest.mock import Mock, patch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from trader_koo.config import Config, ConfigError
from trader_koo.middleware.cors import RestrictiveCORSMiddleware


class TestCORSConfiguration:
    """Unit tests for CORS configuration parsing and validation."""

    def test_empty_cors_origins_default(self):
        """Test that CORS origins default to empty list when not set."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            config = Config()
            assert config.cors_allowed_origins == []

    def test_single_cors_origin_parsing(self):
        """Test parsing a single CORS origin."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "https://example.com",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            config = Config()
            assert config.cors_allowed_origins == ["https://example.com"]

    def test_multiple_cors_origins_parsing(self):
        """Test parsing multiple comma-separated CORS origins."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "https://example.com,https://app.example.com,http://localhost:3000",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            config = Config()
            assert len(config.cors_allowed_origins) == 3
            assert "https://example.com" in config.cors_allowed_origins
            assert "https://app.example.com" in config.cors_allowed_origins
            assert "http://localhost:3000" in config.cors_allowed_origins

    def test_cors_origins_whitespace_handling(self):
        """Test that whitespace around origins is stripped."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": " https://example.com , https://app.example.com ",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            config = Config()
            assert config.cors_allowed_origins == ["https://example.com", "https://app.example.com"]

    def test_https_origin_validation_passes(self):
        """Test that HTTPS origins pass validation."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "https://example.com",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            # Should not raise
            config = Config()
            assert "https://example.com" in config.cors_allowed_origins

    def test_localhost_origin_validation_passes(self):
        """Test that http://localhost origins pass validation."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "http://localhost,http://localhost:3000",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            # Should not raise
            config = Config()
            assert "http://localhost" in config.cors_allowed_origins
            assert "http://localhost:3000" in config.cors_allowed_origins

    def test_http_non_localhost_origin_validation_fails(self):
        """Test that HTTP non-localhost origins fail validation."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "http://example.com",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                Config()
            assert "Invalid CORS origin" in str(exc_info.value)
            assert "http://example.com" in str(exc_info.value)

    def test_no_protocol_origin_validation_fails(self):
        """Test that origins without protocol fail validation."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "example.com",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                Config()
            assert "Invalid CORS origin" in str(exc_info.value)

    def test_ftp_protocol_origin_validation_fails(self):
        """Test that FTP protocol origins fail validation."""
        with patch.dict(os.environ, {
            "TRADER_KOO_CORS_ORIGINS": "ftp://example.com",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            with pytest.raises(ConfigError) as exc_info:
                Config()
            assert "Invalid CORS origin" in str(exc_info.value)

    def test_development_mode_detection(self):
        """Test development mode environment variable detection."""
        with patch.dict(os.environ, {
            "TRADER_KOO_DEVELOPMENT_MODE": "1",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            config = Config()
            assert config.development_mode is True

        with patch.dict(os.environ, {
            "TRADER_KOO_DEVELOPMENT_MODE": "0",
            "ADMIN_STRICT_API_KEY": "0",
        }, clear=False):
            config = Config()
            assert config.development_mode is False


class TestCORSMiddleware:
    """Unit tests for CORS middleware request filtering."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI app."""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        return app

    @pytest.fixture
    def mock_request(self):
        """Create a mock request."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/test"
        request.method = "GET"
        request.headers = {}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.mark.anyio
    async def test_allowed_origin_passes(self, app, mock_request):
        """Test that requests from allowed origins pass through."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=["https://example.com"],
            development_mode=False,
        )

        mock_request.headers = {"origin": "https://example.com"}

        # Mock call_next
        async def mock_call_next(request):
            response = JSONResponse({"message": "success"})
            return response

        response = await middleware.dispatch(mock_request, mock_call_next)

        # Should succeed and have CORS headers
        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"
        assert response.headers["Access-Control-Allow-Credentials"] == "false"

    @pytest.mark.anyio
    async def test_disallowed_origin_rejected(self, app, mock_request):
        """Test that requests from disallowed origins are rejected."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=["https://example.com"],
            development_mode=False,
        )

        mock_request.headers = {"origin": "https://evil.com"}

        # Mock call_next (should not be called)
        async def mock_call_next(request):
            pytest.fail("call_next should not be called for rejected origins")

        with patch("trader_koo.middleware.cors.LOG") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            # Should be rejected
            assert response.status_code == 403
            assert "CORS policy" in response.body.decode()
            assert response.headers["Access-Control-Allow-Credentials"] == "false"

            # Should log rejection
            mock_log.warning.assert_called_once()
            log_call = mock_log.warning.call_args[0][0]
            assert "CORS request rejected" in log_call

    @pytest.mark.anyio
    async def test_no_origin_header_passes(self, app, mock_request):
        """Test that requests without Origin header pass through."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=["https://example.com"],
            development_mode=False,
        )

        # No origin header
        mock_request.headers = {}

        # Mock call_next
        async def mock_call_next(request):
            response = JSONResponse({"message": "success"})
            return response

        response = await middleware.dispatch(mock_request, mock_call_next)

        # Should succeed (not a CORS request)
        assert response.status_code == 200
        # Should still set credentials header
        assert response.headers["Access-Control-Allow-Credentials"] == "false"

    @pytest.mark.anyio
    async def test_development_mode_localhost_allowed(self, app, mock_request):
        """Test that localhost origins are allowed in development mode."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=[],  # Empty allowed list
            development_mode=True,
        )

        # Test various localhost formats
        for origin in ["http://localhost", "http://localhost:3000", "http://localhost:8080"]:
            mock_request.headers = {"origin": origin}

            # Mock call_next
            async def mock_call_next(request):
                response = JSONResponse({"message": "success"})
                return response

            response = await middleware.dispatch(mock_request, mock_call_next)

            # Should succeed
            assert response.status_code == 200
            assert response.headers["Access-Control-Allow-Origin"] == origin

    @pytest.mark.anyio
    async def test_development_mode_non_localhost_rejected(self, app, mock_request):
        """Test that non-localhost origins are still rejected in development mode."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=[],
            development_mode=True,
        )

        mock_request.headers = {"origin": "https://example.com"}

        # Mock call_next (should not be called)
        async def mock_call_next(request):
            pytest.fail("call_next should not be called for rejected origins")

        response = await middleware.dispatch(mock_request, mock_call_next)

        # Should be rejected
        assert response.status_code == 403

    @pytest.mark.anyio
    async def test_preflight_request_handling(self, app, mock_request):
        """Test that OPTIONS preflight requests are handled correctly."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=["https://example.com"],
            development_mode=False,
        )

        mock_request.method = "OPTIONS"
        mock_request.headers = {"origin": "https://example.com"}

        # Mock call_next (should not be called for preflight)
        async def mock_call_next(request):
            pytest.fail("call_next should not be called for preflight requests")

        response = await middleware.dispatch(mock_request, mock_call_next)

        # Should return 200 with CORS headers
        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"
        assert response.headers["Access-Control-Allow-Credentials"] == "false"
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers

    @pytest.mark.anyio
    async def test_preflight_request_rejected_for_disallowed_origin(self, app, mock_request):
        """Test that preflight requests from disallowed origins are rejected."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=["https://example.com"],
            development_mode=False,
        )

        mock_request.method = "OPTIONS"
        mock_request.headers = {"origin": "https://evil.com"}

        # Mock call_next (should not be called)
        async def mock_call_next(request):
            pytest.fail("call_next should not be called")

        with patch("trader_koo.middleware.cors.LOG") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            # Should be rejected
            assert response.status_code == 403

            # Should log rejection
            mock_log.warning.assert_called_once()
            log_call = mock_log.warning.call_args[0][0]
            assert "CORS preflight rejected" in log_call

    @pytest.mark.anyio
    async def test_credentials_header_always_false(self, app, mock_request):
        """Test that Access-Control-Allow-Credentials is always false."""
        middleware = RestrictiveCORSMiddleware(
            app=app,
            allowed_origins=["https://example.com"],
            development_mode=False,
        )

        mock_request.headers = {"origin": "https://example.com"}

        # Mock call_next
        async def mock_call_next(request):
            response = JSONResponse({"message": "success"})
            return response

        response = await middleware.dispatch(mock_request, mock_call_next)

        # Credentials header should be false
        assert response.headers["Access-Control-Allow-Credentials"] == "false"

    def test_client_ip_extraction_from_xff(self, mock_request):
        """Test client IP extraction from X-Forwarded-For header."""
        mock_request.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}

        ip = RestrictiveCORSMiddleware._get_client_ip(mock_request)

        # Should extract first IP
        assert ip == "1.2.3.4"

    def test_client_ip_extraction_from_client(self, mock_request):
        """Test client IP extraction from request.client."""
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.1"

        ip = RestrictiveCORSMiddleware._get_client_ip(mock_request)

        assert ip == "192.168.1.1"

    def test_client_ip_ignores_spoofed_xff_from_untrusted_peer(self, mock_request):
        """Forwarded headers are ignored unless the immediate peer is trusted."""
        mock_request.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        mock_request.client.host = "8.8.8.8"

        ip = RestrictiveCORSMiddleware._get_client_ip(mock_request)

        assert ip == "8.8.8.8"

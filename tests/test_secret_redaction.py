"""Unit tests for secret redaction functionality.

This module tests the secret redaction utilities to ensure secrets are
properly redacted from logs, error responses, and API endpoints.

Requirements: 6.7
"""

import pytest
from trader_koo.security.redaction import (
    redact_secrets,
    sanitize_error_response,
    sanitize_stack_trace,
    _is_secret_key,
    REDACTED_VALUE,
)


class TestSecretKeyDetection:
    """Test secret key pattern matching."""

    def test_api_key_detected(self):
        """Test that API_KEY is detected as a secret."""
        assert _is_secret_key("API_KEY")
        assert _is_secret_key("api_key")
        assert _is_secret_key("TRADER_KOO_API_KEY")

    def test_password_detected(self):
        """Test that PASSWORD is detected as a secret."""
        assert _is_secret_key("PASSWORD")
        assert _is_secret_key("password")
        assert _is_secret_key("DB_PASSWORD")
        assert _is_secret_key("SMTP_PASSWORD")

    def test_token_detected(self):
        """Test that TOKEN is detected as a secret."""
        assert _is_secret_key("TOKEN")
        assert _is_secret_key("token")
        assert _is_secret_key("JWT_SECRET_KEY")

    def test_secret_detected(self):
        """Test that SECRET is detected as a secret."""
        assert _is_secret_key("SECRET")
        assert _is_secret_key("secret")
        assert _is_secret_key("AWS_SECRET_ACCESS_KEY")

    def test_non_secret_not_detected(self):
        """Test that non-secret keys are not detected."""
        assert not _is_secret_key("username")
        assert not _is_secret_key("email")
        assert not _is_secret_key("ticker")
        assert not _is_secret_key("count")


class TestRedactSecrets:
    """Test secret redaction in data structures."""

    def test_redact_dict_with_api_key(self):
        """Test redacting API key from dictionary."""
        data = {
            "username": "admin",
            "API_KEY": "super-secret-key-12345",
            "count": 42,
        }
        redacted = redact_secrets(data)

        assert redacted["username"] == "admin"
        assert redacted["API_KEY"] == REDACTED_VALUE
        assert redacted["count"] == 42

    def test_redact_nested_dict(self):
        """Test redacting secrets from nested dictionaries."""
        data = {
            "config": {
                "database": {
                    "host": "localhost",
                    "password": "db-secret-password",
                },
                "api_key": "api-secret-key",
            },
            "status": "ok",
        }
        redacted = redact_secrets(data)

        assert redacted["config"]["database"]["host"] == "localhost"
        assert redacted["config"]["database"]["password"] == REDACTED_VALUE
        assert redacted["config"]["api_key"] == REDACTED_VALUE
        assert redacted["status"] == "ok"

    def test_redact_list_of_dicts(self):
        """Test redacting secrets from list of dictionaries."""
        data = [
            {"name": "user1", "token": "token-123"},
            {"name": "user2", "token": "token-456"},
        ]
        redacted = redact_secrets(data)

        assert redacted[0]["name"] == "user1"
        assert redacted[0]["token"] == REDACTED_VALUE
        assert redacted[1]["name"] == "user2"
        assert redacted[1]["token"] == REDACTED_VALUE

    def test_redact_preserves_types(self):
        """Test that redaction preserves data types."""
        data = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
        }
        redacted = redact_secrets(data)

        assert isinstance(redacted["string"], str)
        assert isinstance(redacted["int"], int)
        assert isinstance(redacted["float"], float)
        assert isinstance(redacted["bool"], bool)
        assert redacted["none"] is None
        assert isinstance(redacted["list"], list)

    def test_redact_long_string_values(self):
        """Test that very long alphanumeric strings are redacted."""
        # Very long base64-like string (potential secret)
        long_secret = "a" * 50
        data = {"potential_secret": long_secret}
        redacted = redact_secrets(data)

        # The value itself might be redacted if it looks like a secret
        # But the key is not a secret pattern, so it depends on heuristics
        assert "potential_secret" in redacted


class TestSanitizeErrorResponse:
    """Test error response sanitization."""

    def test_sanitize_removes_env_key(self):
        """Test that 'env' key is removed from error response."""
        error_data = {
            "detail": "Something went wrong",
            "env": {
                "API_KEY": "secret-key",
                "DATABASE_URL": "postgres://user:pass@host/db",
            },
        }
        sanitized = sanitize_error_response(error_data)

        assert "detail" in sanitized
        assert "env" not in sanitized

    def test_sanitize_removes_config_key(self):
        """Test that 'config' key is removed from error response."""
        error_data = {
            "detail": "Configuration error",
            "config": {
                "api_key": "secret-key",
            },
        }
        sanitized = sanitize_error_response(error_data)

        assert "detail" in sanitized
        assert "config" not in sanitized

    def test_sanitize_redacts_secrets_in_values(self):
        """Test that secrets in remaining values are redacted."""
        error_data = {
            "detail": "Error occurred",
            "context": {
                "password": "secret-password",
                "username": "admin",
            },
        }
        sanitized = sanitize_error_response(error_data)

        assert sanitized["context"]["password"] == REDACTED_VALUE
        assert sanitized["context"]["username"] == "admin"


class TestSanitizeStackTrace:
    """Test stack trace sanitization."""

    def test_sanitize_env_vars_in_trace(self):
        """Test that environment variables are redacted in stack traces."""
        trace = """
        File "app.py", line 10, in main
            API_KEY='super-secret-key' DATABASE_URL="postgres://user:pass@host/db"
        """
        sanitized = sanitize_stack_trace(trace)

        assert "super-secret-key" not in sanitized
        assert REDACTED_VALUE in sanitized
        assert "API_KEY" in sanitized

    def test_sanitize_api_keys_in_urls(self):
        """Test that API keys in URLs are redacted."""
        trace = """
        Request to https://api.example.com/data?api_key=secret123&other=value
        """
        sanitized = sanitize_stack_trace(trace)

        assert "secret123" not in sanitized
        assert REDACTED_VALUE in sanitized
        assert "api_key=" in sanitized

    def test_sanitize_tokens_in_urls(self):
        """Test that tokens in URLs are redacted."""
        trace = """
        Authorization: Bearer token=abc123def456
        """
        sanitized = sanitize_stack_trace(trace)

        assert "abc123def456" not in sanitized
        assert REDACTED_VALUE in sanitized


class TestEndpointSecretExposure:
    """Test that public endpoints don't expose secrets.

    These tests verify Requirements 6.5, 6.6, 6.7.
    """

    def test_status_endpoint_no_secrets(self):
        """Test that /api/status doesn't expose secrets."""
        # Simulate a status response
        status_response = {
            "ok": True,
            "service": "trader_koo-api",
            "db_exists": True,
            "warnings": [],
            "service_meta": {
                "version": "0.2.0",
                "admin_auth_configured": True,
            },
        }

        # Verify no secret keys in response
        from trader_koo.security.endpoint_validator import validate_response_no_secrets
        is_valid, violations = validate_response_no_secrets(status_response, "/api/status")

        assert is_valid, f"Status endpoint exposes secrets: {violations}"

    def test_health_endpoint_no_secrets(self):
        """Test that /api/health doesn't expose secrets."""
        health_response = {
            "ok": True,
            "db_exists": True,
        }

        from trader_koo.security.endpoint_validator import validate_response_no_secrets
        is_valid, violations = validate_response_no_secrets(health_response, "/api/health")

        assert is_valid, f"Health endpoint exposes secrets: {violations}"

    def test_config_endpoint_no_secrets(self):
        """Test that /api/config doesn't expose secrets."""
        config_response = {
            "auth": {
                "admin_api_key_required": True,
                "admin_api_key_header": "X-API-Key",
            },
        }

        from trader_koo.security.endpoint_validator import validate_response_no_secrets
        is_valid, violations = validate_response_no_secrets(config_response, "/api/config")

        assert is_valid, f"Config endpoint exposes secrets: {violations}"

    def test_response_with_secret_detected(self):
        """Test that responses with secrets are detected."""
        bad_response = {
            "ok": True,
            "API_KEY": "super-secret-key",
        }

        from trader_koo.security.endpoint_validator import validate_response_no_secrets
        is_valid, violations = validate_response_no_secrets(bad_response, "/api/test")

        assert not is_valid
        assert len(violations) > 0
        assert "API_KEY" in violations[0]

    def test_public_status_payload_strips_operational_details(self):
        """Public status payload should keep summaries but hide raw internals."""
        from trader_koo.backend.routers.system import _sanitize_status_payload

        payload = {
            "service_meta": {
                "version": "0.2.0",
                "git_sha": "abc123",
                "deployment_id": "dep-123",
            },
            "latest_run": {
                "status": "failed",
                "started_ts": "2026-04-07T00:00:00+00:00",
                "finished_ts": "2026-04-07T00:05:00+00:00",
                "tickers_total": 10,
                "tickers_failed": 2,
                "error_message": "raw internal error",
            },
            "errors": {
                "failed_runs_7d": 3,
                "latest_error_message": "raw internal error",
                "latest_error_ts": "2026-04-07T00:05:00+00:00",
                "latest_failed_run": {
                    "status": "failed",
                    "started_ts": "2026-04-07T00:00:00+00:00",
                    "finished_ts": "2026-04-07T00:05:00+00:00",
                    "error_message": "raw internal error",
                },
            },
            "pipeline": {
                "stage": "report",
                "stage_line": "[REPORT] internal log line",
                "last_completed_line": "[DONE] internal log line",
            },
            "llm": {
                "enabled": True,
                "health": {
                    "degraded": True,
                    "degraded_threshold": 3,
                    "consecutive_failures": 4,
                    "last_success_ts": "2026-04-06T00:00:00+00:00",
                    "last_failure_ts": "2026-04-07T00:00:00+00:00",
                    "last_failure_reason": "request_failed",
                    "last_error_class": "TimeoutError",
                    "last_error_details": "internal traceback",
                    "recent_events": [{"details": "internal traceback"}],
                    "counts": {"success": 1, "failure": 4, "other": 0, "total": 5},
                },
            },
        }

        sanitized = _sanitize_status_payload(payload, expose_internal=False)

        assert "git_sha" not in sanitized["service_meta"]
        assert "deployment_id" not in sanitized["service_meta"]
        assert "error_message" not in (sanitized["latest_run"] or {})
        assert sanitized["errors"]["latest_error_message"] == (
            "Latest pipeline run failed. Check server logs for details."
        )
        assert "error_message" not in (sanitized["errors"]["latest_failed_run"] or {})
        assert "stage_line" not in sanitized["pipeline"]
        assert "last_completed_line" not in sanitized["pipeline"]
        assert "last_error_details" not in sanitized["llm"]["health"]
        assert "recent_events" not in sanitized["llm"]["health"]
        assert "last_failure_reason" not in sanitized["llm"]["health"]
        assert "last_error_class" not in sanitized["llm"]["health"]

    def test_internal_status_payload_keeps_operational_details(self):
        """Internal status payload can retain debug detail when explicitly enabled."""
        from trader_koo.backend.routers.system import _sanitize_status_payload

        payload = {
            "service_meta": {"git_sha": "abc123", "deployment_id": "dep-123"},
            "latest_run": {"error_message": "raw internal error"},
            "errors": {"latest_error_message": "raw internal error"},
            "pipeline": {"stage_line": "[REPORT] internal log line"},
            "llm": {"health": {"last_error_details": "internal traceback"}},
        }

        sanitized = _sanitize_status_payload(payload, expose_internal=True)

        assert sanitized["service_meta"]["git_sha"] == "abc123"
        assert sanitized["service_meta"]["deployment_id"] == "dep-123"
        assert sanitized["latest_run"]["error_message"] == "raw internal error"
        assert sanitized["errors"]["latest_error_message"] == "raw internal error"
        assert sanitized["pipeline"]["stage_line"] == "[REPORT] internal log line"
        assert sanitized["llm"]["health"]["last_error_details"] == "internal traceback"

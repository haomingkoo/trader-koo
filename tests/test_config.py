"""Tests for configuration validation module."""

import os
import pytest
from hypothesis import given, strategies as st
from trader_koo.config import Config, ConfigError, validate_config


class TestConfigValidation:
    """Test suite for configuration validation."""

    def test_strict_mode_requires_api_key(self, monkeypatch):
        """Test that strict mode requires TRADER_KOO_API_KEY to be set."""
        monkeypatch.setenv("ADMIN_STRICT_API_KEY", "1")
        monkeypatch.delenv("TRADER_KOO_API_KEY", raising=False)

        with pytest.raises(ConfigError) as exc_info:
            Config()

        assert "TRADER_KOO_API_KEY is required" in str(exc_info.value)
        assert "Setup instructions" in str(exc_info.value)

    def test_strict_mode_default_value(self, monkeypatch):
        """Test that ADMIN_STRICT_API_KEY defaults to 1."""
        monkeypatch.delenv("ADMIN_STRICT_API_KEY", raising=False)
        monkeypatch.delenv("TRADER_KOO_API_KEY", raising=False)

        with pytest.raises(ConfigError) as exc_info:
            Config()

        assert "TRADER_KOO_API_KEY is required" in str(exc_info.value)

    def test_api_key_minimum_length(self, monkeypatch):
        """Test that API key must be at least 32 characters."""
        monkeypatch.setenv("ADMIN_STRICT_API_KEY", "1")
        monkeypatch.setenv("TRADER_KOO_API_KEY", "short-key")

        with pytest.raises(ConfigError) as exc_info:
            Config()

        assert "must be at least 32 characters" in str(exc_info.value)
        assert "current length: 9" in str(exc_info.value)

    def test_valid_api_key_passes(self, monkeypatch):
        """Test that valid API key passes validation."""
        monkeypatch.setenv("ADMIN_STRICT_API_KEY", "1")
        monkeypatch.setenv("TRADER_KOO_API_KEY", "a" * 32)

        config = Config()
        assert config.admin_strict_api_key is True
        assert config.trader_koo_api_key == "a" * 32

    def test_local_dev_mode_allows_empty_key(self, monkeypatch):
        """Test that ADMIN_STRICT_API_KEY=0 allows empty API key."""
        monkeypatch.setenv("ADMIN_STRICT_API_KEY", "0")
        monkeypatch.delenv("TRADER_KOO_API_KEY", raising=False)

        config = Config()
        assert config.admin_strict_api_key is False
        assert config.trader_koo_api_key == ""

    def test_validate_config_function(self, monkeypatch):
        """Test the validate_config convenience function."""
        monkeypatch.setenv("ADMIN_STRICT_API_KEY", "1")
        monkeypatch.setenv("TRADER_KOO_API_KEY", "b" * 32)

        config = validate_config()
        assert isinstance(config, Config)
        assert config.trader_koo_api_key == "b" * 32

    def test_as_bool_conversion(self):
        """Test boolean conversion from environment variables."""
        config = Config.__new__(Config)

        assert config._as_bool("1") is True
        assert config._as_bool("true") is True
        assert config._as_bool("TRUE") is True
        assert config._as_bool("yes") is True
        assert config._as_bool("YES") is True
        assert config._as_bool("on") is True
        assert config._as_bool("ON") is True

        assert config._as_bool("0") is False
        assert config._as_bool("false") is False
        assert config._as_bool("no") is False
        assert config._as_bool("") is False
        assert config._as_bool(None) is False
        assert config._as_bool("random") is False

    def test_error_message_includes_generation_command(self, monkeypatch):
        """Test that error messages include command to generate secure key."""
        monkeypatch.setenv("ADMIN_STRICT_API_KEY", "1")
        monkeypatch.setenv("TRADER_KOO_API_KEY", "short")

        with pytest.raises(ConfigError) as exc_info:
            Config()

        assert "secrets.token_urlsafe(32)" in str(exc_info.value)


class TestPropertyBasedValidation:
    """Property-based tests for configuration validation."""

    @given(api_key=st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),  # Printable ASCII
        min_size=0,
        max_size=100
    ))
    def test_api_key_length_validation(self, api_key):
        """
        Feature: enterprise-platform-upgrade, Property 1: API Key Length Validation

        **Validates: Requirements 1.5**

        For any API key string provided as TRADER_KOO_API_KEY, if the key is non-empty,
        then its length must be at least 32 characters, otherwise the platform should
        reject it during startup validation.
        """
        # Save original environment variables
        original_strict = os.environ.get("ADMIN_STRICT_API_KEY")
        original_api_key = os.environ.get("TRADER_KOO_API_KEY")

        try:
            # Set test environment
            os.environ["ADMIN_STRICT_API_KEY"] = "1"
            os.environ["TRADER_KOO_API_KEY"] = api_key

            if not api_key:
                # Empty API key should fail in strict mode
                with pytest.raises(ConfigError) as exc_info:
                    Config()
                assert "TRADER_KOO_API_KEY is required" in str(exc_info.value)
            elif len(api_key) < 32:
                # API key shorter than 32 characters should fail
                with pytest.raises(ConfigError) as exc_info:
                    Config()
                assert "must be at least 32 characters" in str(exc_info.value)
            else:
                # API key with 32+ characters should succeed
                config = Config()
                assert config.trader_koo_api_key == api_key
                assert len(config.trader_koo_api_key) >= 32
        finally:
            # Restore original environment variables
            if original_strict is None:
                os.environ.pop("ADMIN_STRICT_API_KEY", None)
            else:
                os.environ["ADMIN_STRICT_API_KEY"] = original_strict

            if original_api_key is None:
                os.environ.pop("TRADER_KOO_API_KEY", None)
            else:
                os.environ["TRADER_KOO_API_KEY"] = original_api_key

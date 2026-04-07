"""Unit tests for email token expiration functionality.

Tests Requirements 3.1, 3.2, 3.3, 3.4 from the enterprise platform upgrade spec.
"""

import datetime as dt
from pathlib import Path
import tempfile
import pytest

from trader_koo.email_subscribers import (
    upsert_subscriber_pending,
    confirm_subscriber_token,
    _calculate_token_expiry,
    _is_token_expired,
    _utc_now,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


class TestTokenExpiryCalculation:
    """Test token expiration calculation."""

    def test_calculate_token_expiry_default_7_days(self):
        """Test that default expiry is 7 days (168 hours)."""
        created = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        expires = _calculate_token_expiry(created)
        expected = created + dt.timedelta(days=7)
        assert expires == expected

    def test_calculate_token_expiry_custom_hours(self):
        """Test custom expiry hours."""
        created = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        expires = _calculate_token_expiry(created, expiry_hours=24)
        expected = created + dt.timedelta(hours=24)
        assert expires == expected

    def test_is_token_expired_not_expired(self):
        """Test token that has not expired."""
        expires = _utc_now() + dt.timedelta(hours=1)
        assert not _is_token_expired(expires)

    def test_is_token_expired_expired(self):
        """Test token that has expired."""
        expires = _utc_now() - dt.timedelta(hours=1)
        assert _is_token_expired(expires)

    def test_is_token_expired_exactly_at_expiry(self):
        """Test token exactly at expiry time."""
        now = _utc_now()
        assert _is_token_expired(now, now)

    def test_is_token_expired_none_expires_ts(self):
        """Test that None expires_ts is not considered expired."""
        assert not _is_token_expired(None)


class TestTokenCreationWithExpiry:
    """Test token creation includes expiration timestamps."""

    def test_upsert_subscriber_sets_expiry(self, temp_db):
        """Test that creating a subscriber sets token expiry timestamps.

        Requirements: 3.1, 3.2
        """
        now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        result = upsert_subscriber_pending(
            temp_db,
            email="test@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=now,
            token_expiry_hours=168,
        )

        assert result["ok"]
        assert "confirm_token_expires_ts" in result

        # Parse the expiry timestamp
        expires_str = result["confirm_token_expires_ts"]
        expires = dt.datetime.fromisoformat(expires_str.replace("Z", "+00:00"))

        # Should be exactly 7 days (168 hours) from creation
        expected_expires = now + dt.timedelta(hours=168)
        assert expires == expected_expires

    def test_upsert_subscriber_custom_expiry(self, temp_db):
        """Test custom expiry hours."""
        now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        result = upsert_subscriber_pending(
            temp_db,
            email="test@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=now,
            token_expiry_hours=24,
        )

        assert result["ok"]
        expires_str = result["confirm_token_expires_ts"]
        expires = dt.datetime.fromisoformat(expires_str.replace("Z", "+00:00"))

        expected_expires = now + dt.timedelta(hours=24)
        assert expires == expected_expires


class TestTokenValidation:
    """Test token validation rejects expired tokens."""

    def test_confirm_valid_token_succeeds(self, temp_db):
        """Test that valid non-expired token can be confirmed.

        Requirements: 3.3
        """
        now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

        # Create subscriber with token
        result = upsert_subscriber_pending(
            temp_db,
            email="test@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=now,
            token_expiry_hours=168,
        )
        token = result["confirm_token"]

        # Confirm 1 day later (well within 7 days)
        confirm_time = now + dt.timedelta(days=1)
        confirm_result = confirm_subscriber_token(
            temp_db,
            token=token,
            now_utc=confirm_time,
        )

        assert confirm_result is not None
        assert "error" not in confirm_result
        assert confirm_result["status"] == "active"
        assert confirm_result["email"] == "test@example.com"

    def test_confirm_expired_token_fails(self, temp_db):
        """Test that expired token is rejected.

        Requirements: 3.3, 3.4
        """
        now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

        # Create subscriber with token
        result = upsert_subscriber_pending(
            temp_db,
            email="test@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=now,
            token_expiry_hours=168,
        )
        token = result["confirm_token"]

        # Try to confirm 8 days later (after 7 day expiry)
        confirm_time = now + dt.timedelta(days=8)
        confirm_result = confirm_subscriber_token(
            temp_db,
            token=token,
            now_utc=confirm_time,
        )

        assert confirm_result is not None
        assert confirm_result.get("error") == "token_expired"
        assert "expired" in confirm_result.get("detail", "").lower()
        assert confirm_result["email"] == "test@example.com"

    def test_confirm_token_exactly_at_expiry(self, temp_db):
        """Test token exactly at expiry boundary is rejected.

        Requirements: 3.3, 3.4
        """
        now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

        # Create subscriber with token
        result = upsert_subscriber_pending(
            temp_db,
            email="test@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=now,
            token_expiry_hours=168,
        )
        token = result["confirm_token"]

        # Try to confirm exactly at expiry time
        confirm_time = now + dt.timedelta(hours=168)
        confirm_result = confirm_subscriber_token(
            temp_db,
            token=token,
            now_utc=confirm_time,
        )

        assert confirm_result is not None
        assert confirm_result.get("error") == "token_expired"

    def test_confirm_token_one_second_before_expiry(self, temp_db):
        """Test token one second before expiry is still valid."""
        now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

        # Create subscriber with token
        result = upsert_subscriber_pending(
            temp_db,
            email="test@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=now,
            token_expiry_hours=168,
        )
        token = result["confirm_token"]

        # Confirm one second before expiry
        confirm_time = now + dt.timedelta(hours=168) - dt.timedelta(seconds=1)
        confirm_result = confirm_subscriber_token(
            temp_db,
            token=token,
            now_utc=confirm_time,
        )

        assert confirm_result is not None
        assert "error" not in confirm_result
        assert confirm_result["status"] == "active"


class TestErrorMessages:
    """Test that error messages are clear and actionable."""

    def test_expired_token_error_message(self, temp_db):
        """Test that expired token returns clear error message.

        Requirements: 3.4
        """
        now = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)

        result = upsert_subscriber_pending(
            temp_db,
            email="test@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=now,
            token_expiry_hours=168,
        )
        token = result["confirm_token"]

        # Expire the token
        confirm_time = now + dt.timedelta(days=8)
        confirm_result = confirm_subscriber_token(
            temp_db,
            token=token,
            now_utc=confirm_time,
        )

        assert confirm_result.get("error") == "token_expired"
        detail = confirm_result.get("detail", "")
        assert "expired" in detail.lower()
        assert "request a new" in detail.lower() or "new one" in detail.lower()

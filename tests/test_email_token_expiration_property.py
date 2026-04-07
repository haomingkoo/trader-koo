"""Property-based tests for email token expiration.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

Property 4: Email Token Expiration
For any email authentication token, the platform should set expiration to exactly 7 days
from creation, store the creation timestamp, and reject any authentication attempt where
the token age exceeds 7 days.
"""

import datetime as dt
from pathlib import Path
import tempfile
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from trader_koo.email_subscribers import (
    upsert_subscriber_pending,
    confirm_subscriber_token,
    _parse_iso,
)


# Strategy for generating valid datetimes (naive, then add UTC timezone)
datetime_strategy = st.datetimes(
    min_value=dt.datetime(2020, 1, 1),
    max_value=dt.datetime(2030, 12, 31),
).map(lambda d: d.replace(tzinfo=dt.timezone.utc))


@given(
    creation_time=datetime_strategy,
    hours_until_auth=st.integers(min_value=0, max_value=400),  # 0 to ~16 days
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_email_token_expiration_property(creation_time, hours_until_auth):
    """Property test: Tokens expire exactly after configured hours.

    Feature: enterprise-platform-upgrade, Property 4: Email Token Expiration
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

    For any creation time and authentication time:
    - Token should have expiration set to exactly 7 days (168 hours) from creation
    - Authentication should succeed if within expiration period
    - Authentication should fail if after expiration period
    - Error message should be clear when token is expired
    """
    # Create temp db for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db = Path(tmpdir) / "test.db"

        expiry_hours = 168  # 7 days

        # Create a subscriber with a token
        result = upsert_subscriber_pending(
            temp_db,
            email=f"test_{creation_time.timestamp()}@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=creation_time,
            token_expiry_hours=expiry_hours,
        )

        assert result["ok"], "Token creation should succeed"
        token = result["confirm_token"]

        # Verify expiration timestamp is set correctly (Requirement 3.1, 3.2)
        expires_str = result["confirm_token_expires_ts"]
        expires_ts = _parse_iso(expires_str)
        assert expires_ts is not None, "Expiration timestamp should be stored"

        expected_expires = creation_time + dt.timedelta(hours=expiry_hours)
        assert expires_ts == expected_expires, "Expiration should be exactly 7 days from creation"

        # Calculate authentication time
        auth_time = creation_time + dt.timedelta(hours=hours_until_auth)

        # Attempt to confirm the token
        confirm_result = confirm_subscriber_token(
            temp_db,
            token=token,
            now_utc=auth_time,
        )

        assert confirm_result is not None, "Confirm should return a result"

        # Property: Token should be valid if and only if auth_time < expires_ts (Requirement 3.3)
        if hours_until_auth < expiry_hours:
            # Token should be valid
            assert "error" not in confirm_result, \
                f"Token should be valid at {hours_until_auth} hours (< {expiry_hours} hours)"
            assert confirm_result.get("status") == "active", \
                "Valid token should activate subscription"
        else:
            # Token should be expired (Requirement 3.4)
            assert confirm_result.get("error") == "token_expired", \
                f"Token should be expired at {hours_until_auth} hours (>= {expiry_hours} hours)"
            assert "detail" in confirm_result, "Expired token should have error detail"
            assert "expired" in confirm_result["detail"].lower(), \
                "Error message should mention expiration"


@given(
    creation_time=datetime_strategy,
    expiry_hours=st.integers(min_value=1, max_value=720),  # 1 hour to 30 days
    hours_until_auth=st.integers(min_value=0, max_value=800),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_token_expiration_with_custom_expiry(creation_time, expiry_hours, hours_until_auth):
    """Property test: Token expiration works with any valid expiry duration.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

    For any expiry duration:
    - Token should expire exactly after the configured duration
    - Validation should correctly determine if token is expired
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db = Path(tmpdir) / "test.db"

        # Create a subscriber with custom expiry
        result = upsert_subscriber_pending(
            temp_db,
            email=f"test_{creation_time.timestamp()}_{expiry_hours}@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=creation_time,
            token_expiry_hours=expiry_hours,
        )

        assert result["ok"]
        token = result["confirm_token"]

        # Verify expiration is set correctly
        expires_str = result["confirm_token_expires_ts"]
        expires_ts = _parse_iso(expires_str)
        expected_expires = creation_time + dt.timedelta(hours=expiry_hours)
        assert expires_ts == expected_expires

        # Attempt authentication
        auth_time = creation_time + dt.timedelta(hours=hours_until_auth)
        confirm_result = confirm_subscriber_token(
            temp_db,
            token=token,
            now_utc=auth_time,
        )

        # Property: Token valid iff auth_time < expires_ts
        if hours_until_auth < expiry_hours:
            assert "error" not in confirm_result
            assert confirm_result.get("status") == "active"
        else:
            assert confirm_result.get("error") == "token_expired"


@given(
    creation_time=datetime_strategy,
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_token_expiration_boundary_conditions(creation_time):
    """Property test: Token expiration boundary conditions.

    **Validates: Requirements 3.3, 3.4**

    Test the exact boundary:
    - Token valid 1 second before expiry
    - Token expired exactly at expiry
    - Token expired 1 second after expiry
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_db = Path(tmpdir) / "test.db"

        expiry_hours = 168

        result = upsert_subscriber_pending(
            temp_db,
            email=f"boundary_{creation_time.timestamp()}@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=creation_time,
            token_expiry_hours=expiry_hours,
        )

        token = result["confirm_token"]
        expires_ts = _parse_iso(result["confirm_token_expires_ts"])

        # Test 1 second before expiry - should be valid
        auth_time_before = expires_ts - dt.timedelta(seconds=1)
        result_before = confirm_subscriber_token(temp_db, token=token, now_utc=auth_time_before)
        assert "error" not in result_before, "Token should be valid 1 second before expiry"

        # Need to recreate token since it was consumed
        result2 = upsert_subscriber_pending(
            temp_db,
            email=f"boundary2_{creation_time.timestamp()}@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=creation_time,
            token_expiry_hours=expiry_hours,
        )
        token2 = result2["confirm_token"]
        expires_ts2 = _parse_iso(result2["confirm_token_expires_ts"])

        # Test exactly at expiry - should be expired
        result_exact = confirm_subscriber_token(temp_db, token=token2, now_utc=expires_ts2)
        assert result_exact.get("error") == "token_expired", "Token should be expired exactly at expiry time"

        # Need to recreate token again
        result3 = upsert_subscriber_pending(
            temp_db,
            email=f"boundary3_{creation_time.timestamp()}@example.com",
            source_ip="127.0.0.1",
            source_user_agent="test",
            now_utc=creation_time,
            token_expiry_hours=expiry_hours,
        )
        token3 = result3["confirm_token"]
        expires_ts3 = _parse_iso(result3["confirm_token_expires_ts"])

        # Test 1 second after expiry - should be expired
        auth_time_after = expires_ts3 + dt.timedelta(seconds=1)
        result_after = confirm_subscriber_token(temp_db, token=token3, now_utc=auth_time_after)
        assert result_after.get("error") == "token_expired", "Token should be expired 1 second after expiry"

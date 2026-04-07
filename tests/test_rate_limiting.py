"""Unit tests for rate limiting functionality.

Tests Requirements 17.1, 17.2, 17.4, 17.7
"""

import time
from datetime import timedelta

import pytest

from trader_koo.ratelimit.service import RateLimiter, RateLimitConfig


class TestRateLimiter:
    """Test rate limiter service."""

    def test_basic_rate_limit_enforcement(self):
        """Test that rate limits are enforced correctly (Requirement 17.1)."""
        config = RateLimitConfig(
            public_limit=5,
            public_window=timedelta(seconds=1),
        )
        limiter = RateLimiter(config)

        # First 5 requests should be allowed
        for i in range(5):
            result = limiter.check_rate_limit("test_key", 5, timedelta(seconds=1))
            assert result.allowed, f"Request {i+1} should be allowed"
            assert result.remaining == 4 - i

        # 6th request should be denied
        result = limiter.check_rate_limit("test_key", 5, timedelta(seconds=1))
        assert not result.allowed, "6th request should be denied"
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_sliding_window_algorithm(self):
        """Test sliding window algorithm for accurate rate calculation (Requirement 17.4)."""
        config = RateLimitConfig(
            public_limit=3,
            public_window=timedelta(seconds=2),
        )
        limiter = RateLimiter(config)

        # Make 3 requests at t=0
        for _ in range(3):
            result = limiter.check_rate_limit("test_key", 3, timedelta(seconds=2))
            assert result.allowed

        # 4th request should be denied
        result = limiter.check_rate_limit("test_key", 3, timedelta(seconds=2))
        assert not result.allowed

        # Wait for window to slide (2.1 seconds to be safe)
        time.sleep(2.1)

        # Now requests should be allowed again
        result = limiter.check_rate_limit("test_key", 3, timedelta(seconds=2))
        assert result.allowed, "Request should be allowed after window expires"

    def test_per_ip_limits(self):
        """Test per-IP rate limits for public endpoints (Requirement 17.1)."""
        config = RateLimitConfig(
            public_limit=100,
            public_window=timedelta(minutes=1),
        )
        limiter = RateLimiter(config)

        # Different IPs should have independent limits
        ip1_key = "ip:192.168.1.1"
        ip2_key = "ip:192.168.1.2"

        # Use up limit for IP1
        for _ in range(100):
            result = limiter.check_rate_limit(ip1_key, 100, timedelta(minutes=1))
            assert result.allowed

        # IP1 should be rate limited
        result = limiter.check_rate_limit(ip1_key, 100, timedelta(minutes=1))
        assert not result.allowed

        # IP2 should still be allowed
        result = limiter.check_rate_limit(ip2_key, 100, timedelta(minutes=1))
        assert result.allowed

    def test_per_user_limits(self):
        """Test per-user rate limits for authenticated endpoints (Requirement 17.1)."""
        config = RateLimitConfig(
            authenticated_limit=1000,
            authenticated_window=timedelta(hours=1),
        )
        limiter = RateLimiter(config)

        # Different users should have independent limits
        user1_key = "user:user1"
        user2_key = "user:user2"

        # Make some requests for user1
        for _ in range(10):
            result = limiter.check_rate_limit(user1_key, 1000, timedelta(hours=1))
            assert result.allowed

        # User1 should have 990 remaining
        result = limiter.check_rate_limit(user1_key, 1000, timedelta(hours=1))
        assert result.allowed
        assert result.remaining == 989

        # User2 should have full limit
        result = limiter.check_rate_limit(user2_key, 1000, timedelta(hours=1))
        assert result.allowed
        assert result.remaining == 999

    def test_admin_override(self):
        """Test admin override to temporarily increase limits (Requirement 17.7)."""
        config = RateLimitConfig(
            public_limit=5,
            public_window=timedelta(seconds=1),
        )
        limiter = RateLimiter(config)

        key = "test_key"

        # Use up normal limit
        for _ in range(5):
            result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
            assert result.allowed

        # Should be rate limited
        result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
        assert not result.allowed

        # Set admin override to increase limit
        limiter.set_override(
            key=key,
            limit=100,
            window=timedelta(seconds=1),
            duration=timedelta(seconds=10),
        )

        # Now should be allowed with higher limit
        result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
        assert result.allowed, "Request should be allowed with admin override"

    def test_admin_override_expiration(self):
        """Test that admin overrides expire correctly."""
        config = RateLimitConfig(
            public_limit=5,
            public_window=timedelta(seconds=1),
        )
        limiter = RateLimiter(config)

        key = "test_key"

        # Set override with short duration
        limiter.set_override(
            key=key,
            limit=100,
            window=timedelta(seconds=1),
            duration=timedelta(seconds=1),
        )

        # Should use override
        result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
        assert result.allowed

        # Wait for override to expire
        time.sleep(1.1)

        # Use up normal limit
        for _ in range(5):
            result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
            assert result.allowed

        # Should be rate limited with normal limit
        result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
        assert not result.allowed, "Override should have expired"

    def test_remove_override(self):
        """Test removing admin override."""
        config = RateLimitConfig(
            public_limit=5,
            public_window=timedelta(seconds=1),
        )
        limiter = RateLimiter(config)

        key = "test_key"

        # Set override
        limiter.set_override(
            key=key,
            limit=100,
            window=timedelta(seconds=1),
            duration=timedelta(seconds=60),
        )

        # Remove override
        removed = limiter.remove_override(key)
        assert removed, "Override should be removed"

        # Use up normal limit
        for _ in range(5):
            result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
            assert result.allowed

        # Should be rate limited with normal limit
        result = limiter.check_rate_limit(key, 5, timedelta(seconds=1))
        assert not result.allowed

    def test_get_status(self):
        """Test getting rate limit status for a key."""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        key = "test_key"

        # No status initially
        status = limiter.get_status(key)
        assert status is None

        # Make some requests
        for _ in range(3):
            limiter.check_rate_limit(key, 10, timedelta(seconds=1))

        # Should have status now
        status = limiter.get_status(key)
        assert status is not None
        assert status["key"] == key
        assert status["request_count"] == 3
        assert status["oldest_request"] is not None
        assert status["newest_request"] is not None
        assert not status["has_override"]

    def test_get_status_with_override(self):
        """Test getting status when override is active."""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        key = "test_key"

        # Make a request to initialize
        limiter.check_rate_limit(key, 10, timedelta(seconds=1))

        # Set override
        limiter.set_override(
            key=key,
            limit=100,
            window=timedelta(seconds=60),
            duration=timedelta(seconds=300),
        )

        # Check status
        status = limiter.get_status(key)
        assert status is not None
        assert status["has_override"]
        assert status["override"] is not None
        assert status["override"]["limit"] == 100
        assert status["override"]["window_seconds"] == 60

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        config = RateLimitConfig()
        limiter = RateLimiter(config)

        # Add some entries
        limiter.check_rate_limit("key1", 10, timedelta(seconds=1))
        limiter.check_rate_limit("key2", 10, timedelta(seconds=1))

        # Cleanup with very short max_age should remove entries
        time.sleep(0.1)
        removed = limiter.cleanup_expired(max_age=timedelta(milliseconds=50))
        assert removed == 2, "Both entries should be removed"

        # Status should be None after cleanup
        assert limiter.get_status("key1") is None
        assert limiter.get_status("key2") is None

    def test_retry_after_header(self):
        """Test that retry_after is calculated correctly (Requirement 17.2)."""
        config = RateLimitConfig(
            public_limit=2,
            public_window=timedelta(seconds=2),
        )
        limiter = RateLimiter(config)

        # Use up limit
        for _ in range(2):
            result = limiter.check_rate_limit("test_key", 2, timedelta(seconds=2))
            assert result.allowed

        # Next request should be denied with retry_after
        result = limiter.check_rate_limit("test_key", 2, timedelta(seconds=2))
        assert not result.allowed
        assert result.retry_after is not None
        assert 0 < result.retry_after <= 2, "retry_after should be within window"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

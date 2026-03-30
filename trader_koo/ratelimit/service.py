"""Rate limiting service with sliding window algorithm.

Implements Requirements 17.1, 17.2, 17.4, 17.5, 17.7, 17.8
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

LOG = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    
    # Per-IP limits for public endpoints (Requirement 17.1)
    public_limit: int = 300  # requests per minute
    public_window: timedelta = timedelta(minutes=1)
    
    # Per-user limits for authenticated endpoints (Requirement 17.1)
    authenticated_limit: int = 1000  # requests per hour
    authenticated_window: timedelta = timedelta(hours=1)
    
    # Export endpoint limits
    export_limit: int = 10  # requests per hour
    export_window: timedelta = timedelta(hours=1)


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    
    allowed: bool
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None  # seconds until reset


class RateLimiter:
    """Rate limiter using sliding window algorithm.
    
    Uses in-memory storage for rate limit state. For production with multiple
    instances, this should be backed by Redis (Requirement 17.5).
    
    The sliding window algorithm provides accurate rate calculation by tracking
    individual request timestamps (Requirement 17.4).
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.
        
        Args:
            config: Rate limit configuration. Uses defaults if not provided.
        """
        self.config = config or RateLimitConfig()
        # Storage: key -> list of request timestamps
        self._storage: Dict[str, List[float]] = {}
        # Admin overrides: key -> (limit, window_seconds, expires_at)
        self._overrides: Dict[str, tuple[int, float, float]] = {}
        LOG.info(
            "RateLimiter initialized: public=%d/%s, auth=%d/%s",
            self.config.public_limit,
            self.config.public_window,
            self.config.authenticated_limit,
            self.config.authenticated_window,
        )
    
    def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: timedelta,
    ) -> RateLimitResult:
        """Check if request is within rate limit using sliding window algorithm.
        
        Args:
            key: Unique identifier (IP address or user_id)
            limit: Maximum number of requests allowed
            window: Time window for the limit
            
        Returns:
            RateLimitResult with allowed status and metadata
        """
        now = time.time()
        window_seconds = window.total_seconds()
        
        # Check for admin override (Requirement 17.7)
        override = self._overrides.get(key)
        if override:
            override_limit, override_window, expires_at = override
            if now < expires_at:
                limit = override_limit
                window_seconds = override_window
                LOG.debug("Using admin override for key=%s: limit=%d, window=%ds", key, limit, window_seconds)
            else:
                # Override expired, remove it
                del self._overrides[key]
                LOG.info("Admin override expired for key=%s", key)
        
        # Get or initialize request history for this key
        if key not in self._storage:
            self._storage[key] = []
        
        timestamps = self._storage[key]
        
        # Sliding window: remove timestamps outside the window
        cutoff = now - window_seconds
        timestamps[:] = [ts for ts in timestamps if ts > cutoff]
        
        # Check if limit exceeded
        current_count = len(timestamps)
        allowed = current_count < limit
        
        if allowed:
            # Add current request timestamp
            timestamps.append(now)
            remaining = limit - current_count - 1
        else:
            remaining = 0
        
        # Calculate reset time (when oldest request will expire)
        if timestamps:
            oldest_timestamp = timestamps[0]
            reset_at = datetime.fromtimestamp(oldest_timestamp + window_seconds)
            retry_after = max(1, int(oldest_timestamp + window_seconds - now)) if not allowed else None
        else:
            reset_at = datetime.fromtimestamp(now + window_seconds)
            retry_after = None
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )
    
    def set_override(
        self,
        key: str,
        limit: int,
        window: timedelta,
        duration: timedelta,
    ) -> None:
        """Set temporary rate limit override for a specific key.
        
        Allows admin to temporarily increase rate limits for specific users
        (Requirement 17.7).
        
        Args:
            key: Unique identifier (IP address or user_id)
            limit: New rate limit
            window: Time window for the limit
            duration: How long the override should last
        """
        expires_at = time.time() + duration.total_seconds()
        self._overrides[key] = (limit, window.total_seconds(), expires_at)
        LOG.info(
            "Admin override set for key=%s: limit=%d, window=%s, duration=%s",
            key,
            limit,
            window,
            duration,
        )
    
    def remove_override(self, key: str) -> bool:
        """Remove rate limit override for a specific key.
        
        Args:
            key: Unique identifier (IP address or user_id)
            
        Returns:
            True if override was removed, False if no override existed
        """
        if key in self._overrides:
            del self._overrides[key]
            LOG.info("Admin override removed for key=%s", key)
            return True
        return False
    
    def get_status(self, key: str) -> Optional[Dict]:
        """Get current rate limit status for a key.
        
        Provides admin endpoint to view current rate limit status per IP/user
        (Requirement 17.6).
        
        Args:
            key: Unique identifier (IP address or user_id)
            
        Returns:
            Dictionary with status information, or None if key not found
        """
        if key not in self._storage:
            return None
        
        now = time.time()
        timestamps = self._storage[key]
        
        # Check for override
        override = self._overrides.get(key)
        has_override = False
        override_info = None
        if override:
            override_limit, override_window, expires_at = override
            if now < expires_at:
                has_override = True
                override_info = {
                    "limit": override_limit,
                    "window_seconds": override_window,
                    "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
                }
        
        return {
            "key": key,
            "request_count": len(timestamps),
            "oldest_request": datetime.fromtimestamp(timestamps[0]).isoformat() if timestamps else None,
            "newest_request": datetime.fromtimestamp(timestamps[-1]).isoformat() if timestamps else None,
            "has_override": has_override,
            "override": override_info,
        }
    
    def get_all_status(self) -> List[Dict]:
        """Get rate limit status for all tracked keys.
        
        Returns:
            List of status dictionaries for all keys
        """
        return [
            status
            for key in self._storage.keys()
            if (status := self.get_status(key)) is not None
        ]
    
    def cleanup_expired(self, max_age: timedelta = timedelta(hours=24)) -> int:
        """Clean up old entries from storage.
        
        Args:
            max_age: Remove entries older than this
            
        Returns:
            Number of entries removed
        """
        now = time.time()
        cutoff = now - max_age.total_seconds()
        
        keys_to_remove = []
        for key, timestamps in self._storage.items():
            # Remove old timestamps
            timestamps[:] = [ts for ts in timestamps if ts > cutoff]
            # If no recent requests, remove the key entirely
            if not timestamps:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._storage[key]
        
        # Clean up expired overrides
        expired_overrides = [
            key for key, (_, _, expires_at) in self._overrides.items()
            if now >= expires_at
        ]
        for key in expired_overrides:
            del self._overrides[key]
        
        removed = len(keys_to_remove) + len(expired_overrides)
        if removed > 0:
            LOG.info("Cleaned up %d expired rate limit entries", removed)
        
        return removed

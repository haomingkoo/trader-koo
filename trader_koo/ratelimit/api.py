"""Admin API endpoints for rate limiting.

Implements Requirements 17.6, 17.7
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from trader_koo.ratelimit.service import RateLimiter

LOG = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/ratelimit", tags=["admin", "ratelimit"])


class RateLimitOverrideRequest(BaseModel):
    """Request to set rate limit override."""
    
    key: str = Field(..., description="IP address or user_id to override")
    limit: int = Field(..., gt=0, description="New rate limit")
    window_seconds: int = Field(..., gt=0, description="Time window in seconds")
    duration_seconds: int = Field(..., gt=0, description="How long override should last")


class RateLimitOverrideResponse(BaseModel):
    """Response after setting rate limit override."""
    
    success: bool
    message: str
    key: str
    limit: int
    window_seconds: int
    duration_seconds: int


class RateLimitStatusResponse(BaseModel):
    """Rate limit status for a key."""
    
    key: str
    request_count: int
    oldest_request: Optional[str]
    newest_request: Optional[str]
    has_override: bool
    override: Optional[Dict[str, Any]]


# Global rate limiter instance (will be set by main.py)
_rate_limiter: Optional[RateLimiter] = None


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set the global rate limiter instance.
    
    Args:
        limiter: RateLimiter instance to use for API endpoints
    """
    global _rate_limiter
    _rate_limiter = limiter
    LOG.info("Rate limiter instance set for API endpoints")


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance.
    
    Returns:
        RateLimiter instance
        
    Raises:
        HTTPException: If rate limiter not initialized
    """
    if _rate_limiter is None:
        raise HTTPException(
            status_code=500,
            detail="Rate limiter not initialized",
        )
    return _rate_limiter


@router.get(
    "/status",
    response_model=List[RateLimitStatusResponse],
    summary="Get rate limit status for all keys",
    description="View current rate limit status per IP/user (Requirement 17.6)",
)
def get_all_status() -> List[RateLimitStatusResponse]:
    """Get rate limit status for all tracked keys.
    
    Returns:
        List of rate limit status for all keys
    """
    limiter = get_rate_limiter()
    statuses = limiter.get_all_status()
    return [RateLimitStatusResponse(**status) for status in statuses]


@router.get(
    "/status/{key}",
    response_model=RateLimitStatusResponse,
    summary="Get rate limit status for specific key",
    description="View current rate limit status for a specific IP/user (Requirement 17.6)",
)
def get_status(key: str) -> RateLimitStatusResponse:
    """Get rate limit status for a specific key.
    
    Args:
        key: IP address or user_id (e.g., "ip:192.168.1.1" or "user:123")
        
    Returns:
        Rate limit status for the key
        
    Raises:
        HTTPException: If key not found
    """
    limiter = get_rate_limiter()
    status = limiter.get_status(key)
    
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"No rate limit data found for key: {key}",
        )
    
    return RateLimitStatusResponse(**status)


@router.post(
    "/override",
    response_model=RateLimitOverrideResponse,
    summary="Set rate limit override",
    description="Temporarily increase rate limits for specific users (Requirement 17.7)",
)
def set_override(request: RateLimitOverrideRequest) -> RateLimitOverrideResponse:
    """Set temporary rate limit override for a specific key.
    
    Allows admin to temporarily increase rate limits for specific users
    (Requirement 17.7).
    
    Args:
        request: Override configuration
        
    Returns:
        Confirmation of override being set
    """
    limiter = get_rate_limiter()
    
    limiter.set_override(
        key=request.key,
        limit=request.limit,
        window=timedelta(seconds=request.window_seconds),
        duration=timedelta(seconds=request.duration_seconds),
    )
    
    return RateLimitOverrideResponse(
        success=True,
        message=f"Rate limit override set for {request.key}",
        key=request.key,
        limit=request.limit,
        window_seconds=request.window_seconds,
        duration_seconds=request.duration_seconds,
    )


@router.delete(
    "/override/{key}",
    summary="Remove rate limit override",
    description="Remove temporary rate limit override for a specific key",
)
def remove_override(key: str) -> Dict[str, Any]:
    """Remove rate limit override for a specific key.
    
    Args:
        key: IP address or user_id (e.g., "ip:192.168.1.1" or "user:123")
        
    Returns:
        Confirmation of override removal
    """
    limiter = get_rate_limiter()
    removed = limiter.remove_override(key)
    
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No override found for key: {key}",
        )
    
    return {
        "success": True,
        "message": f"Rate limit override removed for {key}",
        "key": key,
    }


@router.post(
    "/cleanup",
    summary="Clean up expired rate limit entries",
    description="Remove old entries from rate limit storage",
)
def cleanup_expired(
    max_age_hours: int = Query(24, gt=0, description="Remove entries older than this many hours"),
) -> Dict[str, Any]:
    """Clean up expired rate limit entries.
    
    Args:
        max_age_hours: Remove entries older than this many hours
        
    Returns:
        Number of entries removed
    """
    limiter = get_rate_limiter()
    removed = limiter.cleanup_expired(max_age=timedelta(hours=max_age_hours))
    
    return {
        "success": True,
        "message": f"Cleaned up {removed} expired entries",
        "removed_count": removed,
        "max_age_hours": max_age_hours,
    }

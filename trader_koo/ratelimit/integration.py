"""Integration module for rate limiting with existing backend."""

import logging
import os
from fastapi import FastAPI

from trader_koo.ratelimit.service import RateLimiter, RateLimitConfig
from trader_koo.ratelimit.api import router as ratelimit_router, set_rate_limiter

LOG = logging.getLogger(__name__)


def initialize_rate_limiting(app: FastAPI) -> RateLimiter:
    """Initialize rate limiting system and integrate with FastAPI app.
    
    This function:
    1. Creates rate limiter with configuration from environment
    2. Registers rate limiting API endpoints
    3. Stores rate limiter in app state
    
    Note: Middleware should be added separately before lifespan.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        RateLimiter instance
    """
    # Get configuration from environment
    public_limit = int(os.getenv("RATE_LIMIT_PUBLIC_PER_MINUTE", "300"))
    authenticated_limit = int(os.getenv("RATE_LIMIT_AUTH_PER_HOUR", "1000"))
    export_limit = int(os.getenv("RATE_LIMIT_EXPORT_PER_HOUR", "10"))
    
    # Create configuration
    from datetime import timedelta
    config = RateLimitConfig(
        public_limit=public_limit,
        public_window=timedelta(minutes=1),
        authenticated_limit=authenticated_limit,
        authenticated_window=timedelta(hours=1),
        export_limit=export_limit,
        export_window=timedelta(hours=1),
    )
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(config)
    
    # Register API endpoints
    set_rate_limiter(rate_limiter)
    app.include_router(ratelimit_router)
    LOG.info("Rate limiting API endpoints registered")
    
    # Store rate limiter in app state for access
    app.state.rate_limiter = rate_limiter
    
    return rate_limiter

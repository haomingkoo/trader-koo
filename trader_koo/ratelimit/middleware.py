"""Rate limiting middleware for FastAPI.

Implements Requirements 17.2, 17.3, 17.8
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from trader_koo.backend.utils import client_ip as _client_ip
from trader_koo.ratelimit.service import RateLimiter, RateLimitConfig

LOG = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits on API endpoints.

    Intercepts all requests before route handlers and checks rate limits.
    Returns HTTP 429 with Retry-After header when exceeded (Requirement 17.2).
    Logs rate limit violations (Requirement 17.8).
    """

    def __init__(
        self,
        app,
        rate_limiter: Optional[RateLimiter] = None,
        config: Optional[RateLimitConfig] = None,
    ):
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application
            rate_limiter: RateLimiter instance. Creates new one if not provided.
            config: Rate limit configuration. Uses defaults if not provided.
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter(config)
        self.config = config or RateLimitConfig()
        LOG.info("RateLimitMiddleware initialized")

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        return _client_ip(request)

    def _get_rate_limit_key(self, request: Request) -> tuple[str, int, timedelta]:
        """Determine rate limit key and limits based on request.

        Returns different limits for:
        - Public endpoints: per-IP, 100/min (Requirement 17.1)
        - Authenticated endpoints: per-user, 1000/hour (Requirement 17.1)
        - Export endpoints: per-user, 10/hour

        Args:
            request: FastAPI request object

        Returns:
            Tuple of (key, limit, window)
        """
        path = request.url.path

        # Check if user is authenticated
        user_id = None
        if hasattr(request.state, "admin_identity"):
            user_id = request.state.admin_identity.get("user_id")
        elif hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "id", None)

        # Export endpoints have stricter limits
        if "/export" in path:
            key = f"user:{user_id}" if user_id else f"ip:{self._get_client_ip(request)}"
            return (key, self.config.export_limit, self.config.export_window)

        # Authenticated endpoints use per-user limits
        if user_id:
            return (
                f"user:{user_id}",
                self.config.authenticated_limit,
                self.config.authenticated_window,
            )

        # Public endpoints use per-IP limits
        return (
            f"ip:{self._get_client_ip(request)}",
            self.config.public_limit,
            self.config.public_window,
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and enforce rate limits.

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response from next handler, or 429 if rate limit exceeded
        """
        # Only rate-limit /api/* paths — skip SPA routes, static assets, health
        path = request.url.path
        if not path.startswith("/api/") or path in ("/api/health", "/api/status"):
            return await call_next(request)

        # Get rate limit parameters
        key, limit, window = self._get_rate_limit_key(request)

        # Check rate limit
        result = self.rate_limiter.check_rate_limit(key, limit, window)

        # Add rate limit headers to response
        if result.allowed:
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
            response.headers["X-RateLimit-Reset"] = result.reset_at.isoformat()
            return response

        # Rate limit exceeded - log violation (Requirement 17.8)
        client_ip = self._get_client_ip(request)
        user_id = key.split(":", 1)[1] if ":" in key else None

        LOG.warning(
            "Rate limit exceeded: key=%s, ip=%s, user=%s, endpoint=%s, method=%s",
            key,
            client_ip,
            user_id,
            request.url.path,
            request.method,
        )

        # Return HTTP 429 with Retry-After header (Requirement 17.2, 17.3)
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please try again later.",
                "limit": limit,
                "window_seconds": int(window.total_seconds()),
                "reset_at": result.reset_at.isoformat(),
            },
            headers={
                "Retry-After": str(result.retry_after) if result.retry_after else "60",
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": result.reset_at.isoformat(),
            },
        )

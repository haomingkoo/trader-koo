"""Rate limiting middleware for FastAPI.

Implements Requirements 17.2, 17.3, 17.8
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable, Optional

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from trader_koo.backend.utils import client_ip as _client_ip
from trader_koo.middleware.auth import AdminAuthenticator
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
        self.rate_limiter = rate_limiter
        self.config = config or (rate_limiter.config if rate_limiter else RateLimitConfig())
        LOG.info("RateLimitMiddleware initialized")

    def _resolve_rate_limiter(self, request: Request) -> RateLimiter:
        """Use the application service so admin controls affect enforcement."""
        if self.rate_limiter is not None:
            return self.rate_limiter
        app_limiter = getattr(request.app.state, "rate_limiter", None)
        if isinstance(app_limiter, RateLimiter):
            return app_limiter
        # Standalone/test apps without the production initializer still use one
        # stable limiter instead of creating a new instance per request.
        self.rate_limiter = RateLimiter(self.config)
        return self.rate_limiter

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        return _client_ip(request)

    def _get_rate_limit_key(
        self,
        request: Request,
        config: RateLimitConfig,
    ) -> tuple[str, int, timedelta]:
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
            return (key, config.export_limit, config.export_window)

        # Authenticated endpoints use per-user limits
        if user_id:
            return (
                f"user:{user_id}",
                config.authenticated_limit,
                config.authenticated_window,
            )

        # Public endpoints use per-IP limits
        return (
            f"ip:{self._get_client_ip(request)}",
            config.public_limit,
            config.public_window,
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
        if request.method == "OPTIONS":
            return await call_next(request)

        # Router dependencies run after HTTP middleware.  Authenticate the
        # admin surface here so valid admins receive the per-user quota instead
        # of being pooled into an anonymous proxy-IP bucket.  ``require_admin``
        # reuses this identity and therefore does not authenticate or audit the
        # request twice.
        if path == "/api/admin" or path.startswith("/api/admin/"):
            authenticator = getattr(request.app.state, "admin_authenticator", None)
            if isinstance(authenticator, AdminAuthenticator):
                try:
                    authenticator.authenticate(
                        request, request.headers.get("X-API-Key")
                    )
                except HTTPException as exc:
                    # Exceptions raised outside the router dependency layer do
                    # not pass through FastAPI's exception handler. Preserve
                    # the same public response contract at this earlier seam.
                    return JSONResponse(
                        status_code=exc.status_code,
                        content={"detail": exc.detail},
                        headers=exc.headers,
                    )

        # Get rate limit parameters
        rate_limiter = self._resolve_rate_limiter(request)
        key, limit, window = self._get_rate_limit_key(request, rate_limiter.config)

        # Check rate limit
        result = rate_limiter.check_rate_limit(key, limit, window)

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

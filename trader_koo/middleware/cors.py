"""CORS middleware with restrictive defaults and security logging.

This module implements CORS (Cross-Origin Resource Sharing) middleware with
secure-by-default configuration that rejects unauthorized cross-origin requests
and logs all rejections for security monitoring.
"""

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from trader_koo.backend.utils import client_ip as _client_ip

LOG = logging.getLogger("trader_koo.middleware.cors")


class RestrictiveCORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware with restrictive defaults and rejection logging.
    
    This middleware enforces CORS policies by:
    1. Rejecting requests from origins not in the allowed list
    2. Logging all rejected CORS requests for security monitoring
    3. Supporting development mode with localhost exceptions
    4. Setting secure default headers (Access-Control-Allow-Credentials: false)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: list[str],
        development_mode: bool = False,
    ):
        """Initialize CORS middleware.
        
        Args:
            app: The ASGI application.
            allowed_origins: List of allowed origin strings.
            development_mode: If True, allow http://localhost:* origins.
        """
        super().__init__(app)
        self.allowed_origins = set(allowed_origins)
        self.development_mode = development_mode
        
        LOG.info(
            "CORS middleware initialized: allowed_origins=%s development_mode=%s",
            list(self.allowed_origins) if self.allowed_origins else "[]",
            development_mode,
        )
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request and enforce CORS policy.
        
        Args:
            request: The incoming request.
            call_next: The next middleware/handler in the chain.
            
        Returns:
            Response with appropriate CORS headers or rejection.
        """
        origin = request.headers.get("origin")
        
        # If no origin header, this is not a CORS request
        if not origin:
            response = await call_next(request)
            # Still set the credentials header for consistency
            response.headers["Access-Control-Allow-Credentials"] = "false"
            return response
        
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            # Check if origin is allowed for preflight
            if not self._is_origin_allowed(origin):
                LOG.warning(
                    "CORS preflight rejected: origin=%s path=%s client_ip=%s",
                    origin,
                    request.url.path,
                    self._get_client_ip(request),
                )
                return Response(
                    content="CORS policy: Origin not allowed",
                    status_code=403,
                    headers={
                        "Access-Control-Allow-Credentials": "false",
                    },
                )
            
            # Return preflight response
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "false",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*, X-API-Key",
                    "Access-Control-Max-Age": "600",
                },
            )
        
        # Check if origin is allowed for actual requests
        if not self._is_origin_allowed(origin):
            # Log rejected CORS request
            LOG.warning(
                "CORS request rejected: origin=%s path=%s method=%s client_ip=%s",
                origin,
                request.url.path,
                request.method,
                self._get_client_ip(request),
            )
            
            # Return 403 Forbidden for rejected CORS requests
            return Response(
                content="CORS policy: Origin not allowed",
                status_code=403,
                headers={
                    "Access-Control-Allow-Credentials": "false",
                },
            )
        
        # Origin is allowed, proceed with request
        response = await call_next(request)
        
        # Add CORS headers to response
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "false"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*, X-API-Key"
        
        return response
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed by CORS policy.
        
        Args:
            origin: The origin string from the request.
            
        Returns:
            True if origin is allowed, False otherwise.
        """
        # Check exact match in allowed origins
        if origin in self.allowed_origins:
            return True
        
        # In development mode, allow any localhost origin
        if self.development_mode and origin.startswith("http://localhost"):
            return True
        
        # Not allowed
        return False
    
    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP from request.
        
        Args:
            request: The incoming request.
            
        Returns:
            Client IP address string.
        """
        return _client_ip(request)

"""
Audit logging middleware for FastAPI.

Automatically logs all admin API requests, authentication attempts,
and provides request correlation IDs.
"""

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from trader_koo.audit.logger import AuditLogger
from trader_koo.backend.utils import client_ip as _client_ip


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically log admin API requests and authentication events.
    """
    
    def __init__(self, app, db_path: Path):
        """
        Initialize audit middleware.
        
        Args:
            app: FastAPI application
            db_path: Path to database file
        """
        super().__init__(app)
        self.db_path = db_path
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log to audit trail.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        # Generate correlation ID for request tracing
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Extract request metadata
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        
        # Start timing
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log admin API requests (Requirement 15.1)
        if request.url.path.startswith("/api/admin/"):
            try:
                # Create connection for this request
                conn = sqlite3.connect(str(self.db_path))
                audit_logger = AuditLogger(conn)
                
                user_id = None
                username = None
                
                # Extract user identity if available
                if hasattr(request.state, "admin_identity"):
                    identity = request.state.admin_identity
                    username = identity.get("username")
                    user_id = identity.get("user_id", username)
                
                # Log the API request
                audit_logger.log_api_request(
                    request_method=request.method,
                    request_path=request.url.path,
                    status_code=response.status_code,
                    response_time_ms=response_time_ms,
                    user_id=user_id,
                    username=username,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    correlation_id=correlation_id,
                    request_params=dict(request.query_params) if request.query_params else None,
                )
                
                conn.close()
            except Exception:
                # Don't fail the request if audit logging fails
                pass
        
        # Add correlation ID to response headers for tracing
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        return _client_ip(request)


def log_auth_attempt(
    audit_logger: AuditLogger,
    success: bool,
    username: str | None,
    ip_address: str,
    user_agent: str | None = None,
    auth_method: str = "api_key",
    reason: str | None = None,
) -> str:
    """
    Log authentication attempt (success or failure).
    
    Args:
        audit_logger: AuditLogger instance
        success: Whether authentication succeeded
        username: Username attempting to authenticate
        ip_address: Client IP address
        user_agent: Client user agent
        auth_method: Authentication method used
        reason: Failure reason (if applicable)
        
    Returns:
        Correlation ID for the event
    """
    if success:
        return audit_logger.log_auth_success(
            user_id=username or "unknown",
            username=username or "unknown",
            ip_address=ip_address,
            user_agent=user_agent,
            auth_method=auth_method,
        )
    else:
        return audit_logger.log_auth_failure(
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            reason=reason or "invalid_credentials",
        )

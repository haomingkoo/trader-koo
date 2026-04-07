"""Error response sanitization middleware.

This module provides FastAPI middleware that sanitizes error responses
to prevent secret exposure in stack traces and error details.

Requirement: 6.4
"""

import logging
import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from trader_koo.security.redaction import sanitize_error_response, sanitize_stack_trace


LOG = logging.getLogger(__name__)


class ErrorSanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware that sanitizes error responses to prevent secret exposure.

    This middleware intercepts exceptions and ensures that error responses
    do not contain environment variables, configuration values, or secrets
    in stack traces.
    """

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process the request and sanitize any error responses.

        Args:
            request: The incoming request.
            call_next: The next middleware or route handler.

        Returns:
            The response, with sanitized error details if an exception occurred.
        """
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Log the original exception for debugging (will be redacted by logging filter)
            LOG.error(
                "Request failed: %s %s",
                request.method,
                request.url.path,
                exc_info=True
            )

            # Create sanitized error response
            error_detail = {
                "detail": str(exc),
                "type": type(exc).__name__,
            }

            # In development mode, include sanitized stack trace
            # In production, omit stack trace entirely
            include_trace = request.app.state.config.development_mode if hasattr(request.app.state, 'config') else False

            if include_trace:
                # Get stack trace and sanitize it
                tb = traceback.format_exc()
                sanitized_tb = sanitize_stack_trace(tb)
                error_detail["trace"] = sanitized_tb

            # Sanitize the error response
            sanitized_error = sanitize_error_response(error_detail)

            # Return appropriate status code
            status_code = 500
            if hasattr(exc, 'status_code'):
                status_code = exc.status_code

            return JSONResponse(
                content=sanitized_error,
                status_code=status_code
            )

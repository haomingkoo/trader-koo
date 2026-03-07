"""Authentication middleware and decorators for admin endpoints.

This module provides authentication decorators and utilities to ensure
all admin endpoints are properly protected.

Requirements:
- 5.1: Maintain registry of all /api/admin/* endpoint paths
- 5.2: Verify authentication middleware is applied to all admin endpoints
- 5.6: Require explicit authentication decorator for new endpoints
"""

import logging
from functools import wraps
from typing import Any, Callable

from fastapi import Request, HTTPException

LOG = logging.getLogger(__name__)

# Registry of all admin endpoints with their authentication status
_ADMIN_ENDPOINT_REGISTRY: dict[str, dict[str, Any]] = {}


def register_admin_endpoint(path: str, method: str, has_auth: bool = True) -> None:
    """Register an admin endpoint in the global registry.
    
    Args:
        path: The endpoint path (e.g., "/api/admin/trigger-update")
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        has_auth: Whether the endpoint has authentication applied
    """
    key = f"{method}:{path}"
    _ADMIN_ENDPOINT_REGISTRY[key] = {
        "path": path,
        "method": method,
        "has_auth": has_auth,
    }
    LOG.debug(f"Registered admin endpoint: {key} (has_auth={has_auth})")


def get_admin_endpoint_registry() -> dict[str, dict[str, Any]]:
    """Get the complete admin endpoint registry.
    
    Returns:
        Dictionary mapping endpoint keys to their metadata
    """
    return _ADMIN_ENDPOINT_REGISTRY.copy()


def verify_all_admin_endpoints_protected() -> tuple[bool, list[str]]:
    """Verify all admin endpoints have authentication applied.
    
    Returns:
        Tuple of (all_protected, unprotected_endpoints)
        - all_protected: True if all endpoints are protected
        - unprotected_endpoints: List of endpoint keys without authentication
    """
    unprotected = [
        key for key, info in _ADMIN_ENDPOINT_REGISTRY.items()
        if not info["has_auth"]
    ]
    return len(unprotected) == 0, unprotected


def require_admin_auth(func: Callable) -> Callable:
    """Decorator to mark an endpoint as requiring admin authentication.
    
    This decorator:
    1. Registers the endpoint in the admin registry
    2. Marks it as having authentication
    3. Validates that the request has admin identity (set by middleware)
    
    Usage:
        @app.get("/api/admin/some-endpoint")
        @require_admin_auth
        def some_endpoint():
            return {"status": "ok"}
    
    Requirements:
    - 5.6: Require explicit authentication decorator for new endpoints
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Extract request from kwargs (FastAPI injects it)
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if request is None:
            for value in kwargs.values():
                if isinstance(value, Request):
                    request = value
                    break
        
        # Verify admin identity is set by middleware
        if request and not hasattr(request.state, "admin_identity"):
            LOG.error(
                f"Admin endpoint {func.__name__} called without admin_identity. "
                "This indicates the authentication middleware is not working."
            )
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Admin authentication required"
            )
        
        # Call the original function
        if hasattr(func, "__wrapped__"):
            # Handle already wrapped functions
            return await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
            # Handle both sync and async functions
            if hasattr(result, "__await__"):
                return await result
            return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Extract request from kwargs
        request = None
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        if request is None:
            for value in kwargs.values():
                if isinstance(value, Request):
                    request = value
                    break
        
        # Verify admin identity is set by middleware
        if request and not hasattr(request.state, "admin_identity"):
            LOG.error(
                f"Admin endpoint {func.__name__} called without admin_identity. "
                "This indicates the authentication middleware is not working."
            )
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Admin authentication required"
            )
        
        return func(*args, **kwargs)
    
    # Determine if function is async
    import inspect
    if inspect.iscoroutinefunction(func):
        wrapper = async_wrapper
    else:
        wrapper = sync_wrapper
    
    # Mark the wrapper so we can identify decorated endpoints
    wrapper._requires_admin_auth = True  # type: ignore
    wrapper._original_func = func  # type: ignore
    
    return wrapper


def auto_register_admin_endpoints(app: Any) -> None:
    """Automatically register all admin endpoints from FastAPI app routes.
    
    This function scans all routes in the FastAPI app and registers
    any that start with /api/admin/.
    
    Args:
        app: FastAPI application instance
        
    Requirements:
    - 5.1: Maintain registry of all /api/admin/* endpoint paths
    """
    for route in app.routes:
        if hasattr(route, "path") and hasattr(route, "methods"):
            path = route.path
            if path.startswith("/api/admin/"):
                for method in route.methods:
                    # Check if endpoint has authentication decorator
                    has_auth = False
                    if hasattr(route, "endpoint"):
                        endpoint = route.endpoint
                        has_auth = getattr(endpoint, "_requires_admin_auth", False)
                    
                    register_admin_endpoint(path, method, has_auth)
                    
                    if not has_auth:
                        LOG.warning(
                            f"Admin endpoint {method}:{path} does not have "
                            "@require_admin_auth decorator"
                        )

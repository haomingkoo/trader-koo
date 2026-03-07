"""RBAC middleware for enforcing role-based permissions."""

from typing import Callable, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from trader_koo.auth.models import User, UserRole
from trader_koo.auth.service import AuthService


class RBACMiddleware:
    """Middleware for enforcing role-based access control (Requirements 16.4-16.7)."""
    
    def __init__(self, auth_service: AuthService):
        """Initialize RBAC middleware.
        
        Args:
            auth_service: Authentication service instance
        """
        self.auth_service = auth_service
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process request and enforce RBAC.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Authenticate user
        user = await self._authenticate_request(request)
        if not user:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication required"}
            )
        
        # Store user in request state
        request.state.user = user
        
        # Check permissions
        if not self._check_permission(user, request.url.path, request.method):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "Insufficient permissions"}
            )
        
        return await call_next(request)
    
    async def _authenticate_request(self, request: Request) -> Optional[User]:
        """Authenticate request using various methods.
        
        Tries in order:
        1. JWT token from Authorization header
        2. API key from X-API-Key header
        3. Email token from query parameter
        
        Args:
            request: FastAPI request
            
        Returns:
            User object if authenticated
        """
        # Try JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = self.auth_service.authenticate_jwt(token)
            if user:
                return user
        
        # Try API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user = self.auth_service.authenticate_api_key(api_key)
            if user:
                return user
        
        # Try email token
        email_token = request.query_params.get("token")
        if email_token:
            user = self.auth_service.authenticate_email_token(email_token)
            if user:
                return user
        
        return None
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (no auth required).
        
        Args:
            path: Request path
            
        Returns:
            True if endpoint is public
        """
        public_prefixes = [
            "/",
            "/api/health",
            "/api/status",
            "/api/login",
            "/api/register",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        
        return any(path.startswith(prefix) for prefix in public_prefixes)
    
    def _check_permission(self, user: User, path: str, method: str) -> bool:
        """Check if user has permission for path and method.
        
        Args:
            user: Authenticated user
            path: Request path
            method: HTTP method
            
        Returns:
            True if user has permission
        """
        # Map HTTP methods to actions
        action_map = {
            "GET": "read",
            "HEAD": "read",
            "OPTIONS": "read",
            "POST": "write",
            "PUT": "write",
            "PATCH": "write",
            "DELETE": "delete",
        }
        
        action = action_map.get(method, "read")
        return user.has_permission(path, action)


def require_role(*allowed_roles: UserRole):
    """Decorator to require specific roles for an endpoint.
    
    Usage:
        @app.get("/api/admin/users")
        @require_role(UserRole.ADMIN)
        async def list_users(request: Request):
            ...
    
    Args:
        allowed_roles: Roles that are allowed to access the endpoint
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        async def wrapper(request: Request, *args, **kwargs):
            user = getattr(request.state, "user", None)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

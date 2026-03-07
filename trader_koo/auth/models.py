"""Data models for authentication."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User role types for RBAC."""
    ADMIN = "admin"      # Full access to all endpoints
    ANALYST = "analyst"  # Read/write analysis, no admin access
    VIEWER = "viewer"    # Read-only access


@dataclass
class User:
    """User account model."""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    is_active: bool
    failed_login_attempts: int
    locked_until: datetime | None
    created_at: datetime
    last_login: datetime | None
    
    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    def has_permission(self, resource: str, action: str) -> bool:
        """Check if user has permission for resource/action.
        
        Args:
            resource: Resource path (e.g., "/api/admin/users")
            action: Action type (e.g., "read", "write", "delete")
            
        Returns:
            True if user has permission
        """
        if not self.is_active:
            return False
        
        if self.role == UserRole.ADMIN:
            return True
        
        # Admin endpoints require admin role
        if resource.startswith("/api/admin"):
            return False
        
        # Analyst can read/write analysis and reports
        if self.role == UserRole.ANALYST:
            if resource.startswith(("/api/analysis", "/api/reports", "/api/dashboard")):
                return action in ("read", "write")
            return action == "read"
        
        # Viewer is read-only
        if self.role == UserRole.VIEWER:
            return action == "read"
        
        return False


@dataclass
class Session:
    """User session model."""
    id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime
    last_activity: datetime
    ip_address: str | None
    user_agent: str | None


@dataclass
class EmailToken:
    """Email authentication token model."""
    id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime
    used_at: datetime | None
    revoked_at: datetime | None
    
    def is_valid(self) -> bool:
        """Check if token is valid for use."""
        now = datetime.utcnow()
        return (
            self.used_at is None
            and self.revoked_at is None
            and now < self.expires_at
        )

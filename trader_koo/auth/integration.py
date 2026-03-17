"""Integration module for RBAC with existing backend."""

import logging
import os
import sqlite3
from pathlib import Path
from fastapi import FastAPI

from trader_koo.auth.schema import ensure_auth_schema
from trader_koo.auth.service import AuthService
from trader_koo.auth.user_management import UserManagementService
from trader_koo.auth.api import create_auth_router
from trader_koo.audit.logger import AuditLogger


def initialize_rbac(app: FastAPI, db_path: Path) -> tuple[AuthService, UserManagementService]:
    """Initialize RBAC system and integrate with FastAPI app.
    
    This function:
    1. Creates auth database tables
    2. Initializes auth and user management services
    3. Registers auth API endpoints
    
    Args:
        app: FastAPI application instance
        db_path: Path to SQLite database
        
    Returns:
        Tuple of (AuthService, UserManagementService)
    """
    # Ensure auth schema exists
    ensure_auth_schema(db_path)
    
    # Get configuration from environment
    jwt_secret = os.getenv("JWT_SECRET_KEY", "")
    if not jwt_secret or len(jwt_secret) < 32:
        raise ValueError(
            "JWT_SECRET_KEY must be set and at least 32 characters long. "
            "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    
    jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    email_token_expiration_days = int(os.getenv("EMAIL_TOKEN_EXPIRATION_DAYS", "7"))
    
    # Legacy API key for backward compatibility
    legacy_api_key = os.getenv("TRADER_KOO_API_KEY", "")
    
    # Initialize services
    auth_service = AuthService(
        db_path=db_path,
        jwt_secret=jwt_secret,
        jwt_algorithm=jwt_algorithm,
        jwt_expiration_hours=jwt_expiration_hours,
        email_token_expiration_days=email_token_expiration_days,
        api_key=legacy_api_key if legacy_api_key else None,
    )
    
    user_mgmt_service = UserManagementService(db_path=db_path)
    
    # Create audit logger
    conn = sqlite3.connect(str(db_path))
    audit_logger = AuditLogger(conn)
    
    # Register auth API endpoints
    auth_router = create_auth_router(
        auth_service=auth_service,
        user_mgmt_service=user_mgmt_service,
        audit_logger=audit_logger,
    )
    app.include_router(auth_router)
    
    # Store services in app state for middleware access
    app.state.auth_service = auth_service
    app.state.user_mgmt_service = user_mgmt_service
    
    return auth_service, user_mgmt_service


def create_default_admin_user(user_mgmt_service: UserManagementService) -> None:
    """Create default admin user if no users exist.
    
    This is a convenience function for initial setup. In production,
    users should be created through the API.
    
    Args:
        user_mgmt_service: User management service
    """
    log = logging.getLogger("trader_koo.auth.integration")
    users = user_mgmt_service.list_users()
    if len(users) == 0:
        # No users exist, create default admin
        from trader_koo.auth.models import UserRole

        default_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
        default_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@trader-koo.local")
        default_password = os.getenv("TRADER_KOO_ADMIN_PASSWORD", "")

        if not default_password:
            import secrets
            default_password = secrets.token_urlsafe(16)
            log.warning(
                "Auto-generated admin password (set TRADER_KOO_ADMIN_PASSWORD to override): %s",
                default_password,
            )

        user, error = user_mgmt_service.create_user(
            username=default_username,
            email=default_email,
            password=default_password,
            role=UserRole.ADMIN,
        )

        if error:
            log.warning("Failed to create default admin user: %s", error)
        else:
            log.info("Default admin user created: %s", default_username)

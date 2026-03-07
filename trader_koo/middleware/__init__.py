"""Middleware modules for trader_koo platform."""

from trader_koo.middleware.auth import (
    require_admin_auth,
    register_admin_endpoint,
    get_admin_endpoint_registry,
    verify_all_admin_endpoints_protected,
    auto_register_admin_endpoints,
)

__all__ = [
    "require_admin_auth",
    "register_admin_endpoint",
    "get_admin_endpoint_registry",
    "verify_all_admin_endpoints_protected",
    "auto_register_admin_endpoints",
]

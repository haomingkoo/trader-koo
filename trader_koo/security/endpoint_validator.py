"""Endpoint validator for secret exposure prevention.

This module provides utilities to validate that public endpoints like
/api/status, /api/health, and /api/config do not expose secret values.

Requirements: 6.5, 6.6
"""

import logging
from typing import Any

from trader_koo.security.redaction import _is_secret_key, REDACTED_VALUE


LOG = logging.getLogger(__name__)


def validate_response_no_secrets(
    response_data: dict[str, Any],
    endpoint: str,
    max_depth: int = 10
) -> tuple[bool, list[str]]:
    """Validate that a response does not contain secret values.
    
    This function recursively checks a response dictionary to ensure
    no secret keys or values are present.
    
    Args:
        response_data: The response data to validate.
        endpoint: The endpoint name (for logging).
        max_depth: Maximum recursion depth.
        
    Returns:
        Tuple of (is_valid, violations) where violations is a list of
        paths to secret keys found in the response.
    """
    violations = []
    _check_for_secrets(response_data, "", violations, max_depth)
    
    if violations:
        LOG.error(
            "Endpoint %s exposes secrets at paths: %s",
            endpoint,
            ", ".join(violations)
        )
        return False, violations
    
    return True, []


def _check_for_secrets(
    data: Any,
    path: str,
    violations: list[str],
    max_depth: int
) -> None:
    """Recursively check data structure for secret keys.
    
    Args:
        data: The data to check.
        path: Current path in the data structure (for reporting).
        violations: List to append violations to.
        max_depth: Maximum recursion depth.
    """
    if max_depth <= 0:
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Check if the key itself is a secret
            # But exclude metadata fields that describe authentication
            # (e.g., "admin_api_key_required" is metadata, not a secret)
            if _is_secret_key(key):
                # Additional check: if the value is a boolean or the key ends with
                # "_required", "_header", "_enabled", it's likely metadata, not a secret
                if isinstance(value, bool):
                    # Boolean values are metadata, not secrets
                    pass
                elif key.endswith(('_required', '_header', '_enabled', '_configured')):
                    # These are metadata fields about authentication, not secrets
                    pass
                else:
                    violations.append(current_path)
            
            # Recursively check the value
            _check_for_secrets(value, current_path, violations, max_depth - 1)
    
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            _check_for_secrets(item, current_path, violations, max_depth - 1)


def sanitize_public_response(response_data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a public endpoint response to ensure no secrets are exposed.
    
    This function removes any keys that match secret patterns and replaces
    their values with [REDACTED].
    
    Args:
        response_data: The response data to sanitize.
        
    Returns:
        Sanitized response data.
    """
    sanitized = {}
    
    for key, value in response_data.items():
        if _is_secret_key(key):
            # Replace secret values with redacted marker
            sanitized[key] = REDACTED_VALUE
        elif isinstance(value, dict):
            sanitized[key] = sanitize_public_response(value)
        elif isinstance(value, (list, tuple)):
            sanitized[key] = [
                sanitize_public_response(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized

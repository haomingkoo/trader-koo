"""Secret redaction utilities for logs and error responses.

This module provides comprehensive protection against secret leakage by:
1. Redacting secret values in log output
2. Sanitizing error responses to strip environment variables and config
3. Maintaining a list of secret patterns to detect sensitive data

Requirements: 6.1, 6.2, 6.3, 6.4
"""

import re
from typing import Any


# List of secret environment variable names and patterns (Requirement 6.1)
SECRET_PATTERNS = [
    # API Keys
    "API_KEY",
    "TRADER_KOO_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    
    # Passwords and tokens
    "PASSWORD",
    "PASS",
    "SECRET",
    "TOKEN",
    "JWT_SECRET_KEY",
    "HMAC_SECRET",
    
    # SMTP credentials
    "SMTP_PASS",
    "SMTP_PASSWORD",
    "SMTP_USER",
    
    # Database credentials
    "DATABASE_URL",
    "DB_PASSWORD",
    "POSTGRES_PASSWORD",
    
    # Other sensitive data
    "PRIVATE_KEY",
    "CERTIFICATE",
    "CREDENTIALS",
]

# Compiled regex patterns for efficient matching
_SECRET_KEY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in SECRET_PATTERNS
]

REDACTED_VALUE = "[REDACTED]"


def _is_secret_key(key: str) -> bool:
    """Check if a key name matches any secret pattern.
    
    Args:
        key: The key name to check.
        
    Returns:
        True if the key matches a secret pattern, False otherwise.
    """
    if not key:
        return False
    
    key_upper = str(key).upper()
    
    # Check exact matches and pattern matches
    for pattern in _SECRET_KEY_PATTERNS:
        if pattern.search(key_upper):
            return True
    
    return False


def redact_secrets(data: Any, max_depth: int = 10) -> Any:
    """Redact secret values from a data structure.
    
    This function recursively traverses dictionaries, lists, and other data
    structures to find and redact values associated with secret keys.
    
    Requirements: 6.2, 6.3
    
    Args:
        data: The data structure to redact (dict, list, str, etc.).
        max_depth: Maximum recursion depth to prevent infinite loops.
        
    Returns:
        A copy of the data structure with secret values replaced by [REDACTED].
    """
    if max_depth <= 0:
        return data
    
    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            if _is_secret_key(key):
                redacted[key] = REDACTED_VALUE
            else:
                redacted[key] = redact_secrets(value, max_depth - 1)
        return redacted
    
    elif isinstance(data, (list, tuple)):
        redacted_items = [redact_secrets(item, max_depth - 1) for item in data]
        return type(data)(redacted_items)
    
    elif isinstance(data, str):
        # Check if the string itself looks like a secret (long alphanumeric strings)
        # This is a heuristic to catch secrets that might not be in a dict
        if len(data) >= 32 and re.match(r'^[A-Za-z0-9+/=_-]+$', data):
            # Could be a base64-encoded secret or API key
            # Only redact if it's very long to avoid false positives
            if len(data) >= 40:
                return REDACTED_VALUE
        return data
    
    else:
        # For other types (int, float, bool, None, etc.), return as-is
        return data


def sanitize_error_response(error_data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize error response to strip environment variables and config.
    
    This function removes sensitive information from error responses including:
    - Environment variables
    - Configuration values
    - File paths that might reveal system structure
    - Stack traces containing secrets
    
    Requirement: 6.4
    
    Args:
        error_data: The error response dictionary.
        
    Returns:
        Sanitized error response with sensitive data removed.
    """
    sanitized = {}
    
    for key, value in error_data.items():
        # Skip environment and config keys entirely
        if key.lower() in {"env", "environment", "config", "configuration", "settings"}:
            continue
        
        # Redact secrets in the value
        if isinstance(value, (dict, list)):
            sanitized[key] = redact_secrets(value)
        elif isinstance(value, str):
            # Remove file paths from error messages
            sanitized_value = re.sub(
                r'/[a-zA-Z0-9_/.-]+',
                '[PATH]',
                value
            )
            # Redact potential secrets in error messages
            sanitized[key] = redact_secrets(sanitized_value)
        else:
            sanitized[key] = value
    
    return sanitized


_URL_TOKEN_RE = re.compile(
    r'([?&])(api_key|token|key|secret|apikey)=([^&\s"\']+)',
    re.IGNORECASE,
)
# Telegram bot tokens: /bot<id>:<secret>/
_TELEGRAM_BOT_RE = re.compile(r'/bot(\d+):([A-Za-z0-9_-]+)/')


def redact_url_tokens(text: str) -> str:
    """Redact API tokens and keys from URLs embedded in a string."""
    result = _URL_TOKEN_RE.sub(r'\1\2=' + REDACTED_VALUE, text)
    result = _TELEGRAM_BOT_RE.sub(r'/bot\1:' + REDACTED_VALUE + '/', result)
    return result


def sanitize_stack_trace(stack_trace: str) -> str:
    """Sanitize a stack trace to remove sensitive information.
    
    Args:
        stack_trace: The stack trace string.
        
    Returns:
        Sanitized stack trace with secrets removed.
    """
    if not stack_trace:
        return stack_trace
    
    # Remove environment variable values from stack traces
    # Pattern: ENV_VAR='value' or ENV_VAR="value"
    sanitized = re.sub(
        r'([A-Z_]+)=["\']([^"\']+)["\']',
        lambda m: f'{m.group(1)}="{REDACTED_VALUE}"' if _is_secret_key(m.group(1)) else m.group(0),
        stack_trace
    )
    
    # Remove potential API keys and tokens from URLs
    # Pattern: ?api_key=xxx or &token=xxx or token=xxx
    sanitized = re.sub(
        r'([?&\s])(api_key|token|key|secret)=([^&\s]+)',
        r'\1\2=' + REDACTED_VALUE,
        sanitized,
        flags=re.IGNORECASE
    )
    
    return sanitized

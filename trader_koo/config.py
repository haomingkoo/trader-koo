"""Configuration validation module for trader_koo platform.

This module provides secure-by-default configuration validation to prevent
accidental deployments without proper authentication.
"""

import os
from typing import Any


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """Application configuration with validation.
    
    This class validates critical security configuration at startup to ensure
    the platform is deployed with proper authentication and security settings.
    """
    
    def __init__(self):
        """Initialize and validate configuration."""
        # Load environment variables
        self.admin_strict_api_key = self._as_bool(
            os.getenv("ADMIN_STRICT_API_KEY", "1")
        )
        self.trader_koo_api_key = os.getenv("TRADER_KOO_API_KEY", "")
        
        # Email token expiration (7 days = 168 hours)
        self.email_token_expiry_hours = self._parse_int(
            os.getenv("EMAIL_TOKEN_EXPIRY_HOURS", "168"),
            default=168,
            min_value=1,
            max_value=8760  # 1 year max
        )
        
        # CORS configuration - restrictive defaults
        self.cors_allowed_origins = self._parse_cors_origins(
            os.getenv("TRADER_KOO_CORS_ORIGINS", "")
        )
        
        # Development mode detection
        self.development_mode = self._as_bool(
            os.getenv("TRADER_KOO_DEVELOPMENT_MODE", "0")
        )
        
        # Validate configuration
        self.validate()
    
    def validate(self) -> None:
        """Validate all configuration requirements.
        
        Raises:
            ConfigError: If any validation check fails.
        """
        self._validate_api_key_presence()
        self._validate_api_key_length()
        self._validate_cors_origins()
    
    def _validate_api_key_presence(self) -> None:
        """Validate that TRADER_KOO_API_KEY is set when strict mode is enabled.
        
        When ADMIN_STRICT_API_KEY=1 (default), the platform requires
        TRADER_KOO_API_KEY to be set to prevent accidental deployments
        without authentication.
        
        Raises:
            ConfigError: If strict mode is enabled and API key is not set.
        """
        if self.admin_strict_api_key and not self.trader_koo_api_key:
            raise ConfigError(
                "TRADER_KOO_API_KEY is required when ADMIN_STRICT_API_KEY=1.\n"
                "\n"
                "Setup instructions:\n"
                "1. Generate a secure API key (minimum 32 characters):\n"
                "   python -c 'import secrets; print(secrets.token_urlsafe(32))'\n"
                "\n"
                "2. Set the environment variable:\n"
                "   export TRADER_KOO_API_KEY='your-generated-key'\n"
                "\n"
                "3. For local development without authentication, set:\n"
                "   export ADMIN_STRICT_API_KEY=0\n"
                "\n"
                "Note: ADMIN_STRICT_API_KEY defaults to 1 for security."
            )
    
    def _validate_api_key_length(self) -> None:
        """Validate that TRADER_KOO_API_KEY meets minimum length requirement.
        
        API keys must be at least 32 characters to ensure sufficient entropy
        and prevent brute-force attacks.
        
        Raises:
            ConfigError: If API key is provided but too short.
        """
        if self.trader_koo_api_key and len(self.trader_koo_api_key) < 32:
            raise ConfigError(
                f"TRADER_KOO_API_KEY must be at least 32 characters long "
                f"(current length: {len(self.trader_koo_api_key)}).\n"
                "\n"
                "Generate a secure API key:\n"
                "  python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
    
    @staticmethod
    def _as_bool(value: Any) -> bool:
        """Convert environment variable value to boolean.
        
        Args:
            value: The value to convert.
            
        Returns:
            True if value is "1", "true", "yes", or "on" (case-insensitive).
            False otherwise.
        """
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
    
    @staticmethod
    def _parse_int(value: str, default: int, min_value: int, max_value: int) -> int:
        """Parse integer from environment variable with bounds checking.
        
        Args:
            value: The string value to parse.
            default: Default value if parsing fails.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            
        Returns:
            Parsed integer within bounds, or default if parsing fails.
        """
        try:
            parsed = int(str(value or "").strip())
            return max(min_value, min(max_value, parsed))
        except ValueError:
            return default
    
    @staticmethod
    def _parse_cors_origins(value: str) -> list[str]:
        """Parse comma-separated CORS origins from environment variable.
        
        Args:
            value: Comma-separated list of origins.
            
        Returns:
            List of origin strings (empty list if value is empty).
        """
        if not value or not value.strip():
            return []
        
        origins = []
        for origin in value.split(","):
            origin = origin.strip()
            if origin:
                origins.append(origin)
        
        return origins
    
    def _validate_cors_origins(self) -> None:
        """Validate that all CORS origins follow allowed patterns.
        
        Valid patterns:
        - https://* (any HTTPS origin)
        - http://localhost* (localhost with any port)
        
        Raises:
            ConfigError: If any origin doesn't match allowed patterns.
        """
        for origin in self.cors_allowed_origins:
            if not self._is_valid_cors_origin(origin):
                raise ConfigError(
                    f"Invalid CORS origin: {origin}\n"
                    "\n"
                    "CORS origins must follow one of these patterns:\n"
                    "  - https://example.com (HTTPS origins)\n"
                    "  - http://localhost (localhost for development)\n"
                    "  - http://localhost:3000 (localhost with port)\n"
                    "\n"
                    "Current TRADER_KOO_CORS_ORIGINS: {}\n".format(
                        ",".join(self.cors_allowed_origins)
                    )
                )
    
    @staticmethod
    def _is_valid_cors_origin(origin: str) -> bool:
        """Check if an origin follows allowed CORS patterns.
        
        Args:
            origin: The origin string to validate.
            
        Returns:
            True if origin is valid, False otherwise.
        """
        if not origin:
            return False
        
        # Allow HTTPS origins
        if origin.startswith("https://"):
            return True
        
        # Allow http://localhost with optional port
        if origin.startswith("http://localhost"):
            return True
        
        return False


def validate_config() -> Config:
    """Validate configuration and return Config instance.
    
    This function should be called at application startup to ensure
    all configuration requirements are met before the server starts.
    
    Returns:
        Config: Validated configuration instance.
        
    Raises:
        ConfigError: If any validation check fails.
    """
    return Config()

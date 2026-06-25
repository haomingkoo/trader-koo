"""Configuration validation module for trader_koo platform.

This module provides secure-by-default configuration validation to prevent
accidental deployments without proper authentication.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = (PACKAGE_DIR / "data" / "trader_koo.db").resolve()
DEFAULT_REPORT_DIR = (PACKAGE_DIR / "data" / "reports").resolve()
DEFAULT_LOG_DIR = (PACKAGE_DIR / "data" / "logs").resolve()

OPTIONS_SOURCE = "yfinance_options_iv"
OPTIONS_SOURCE_NOTE = (
    "Yahoo/yfinance option-chain snapshot; delayed/unofficial. "
    "Not real-time sweeps, block trades, or net premium."
)
OPTIONS_PREMIUM_PROXY_NOTE = (
    "Call premium minus put premium is estimated from the latest option-chain "
    "snapshot using volume/open-interest times mid price. It is not signed "
    "buyer/seller flow and should not be read as live net premium."
)
OPTIONS_PREMIUM_SORT_CHOICES = ("volume_premium", "oi_premium", "ticker")
OPTIONS_PREMIUM_SORT_PATTERN = "^(volume_premium|oi_premium|ticker)$"
OPTIONS_PREMIUM_DEFAULT_SORT_BY = "volume_premium"
OPTIONS_PREMIUM_DEFAULT_LIMIT = 100
OPTIONS_PREMIUM_PAGE_LIMIT = 150
OPTIONS_PREMIUM_MIN_LIMIT = 1
OPTIONS_PREMIUM_MAX_LIMIT = 500
OPTIONS_PREMIUM_BALANCED_SKEW_RATIO = 0.05
OPTIONS_PREMIUM_BIAS_UNKNOWN = "unknown"
OPTIONS_PREMIUM_BIAS_BALANCED = "balanced"
OPTIONS_PREMIUM_BIAS_CALL = "call_premium_skew"
OPTIONS_PREMIUM_BIAS_PUT = "put_premium_skew"
OPTIONS_CONTRACT_MULTIPLIER = 100.0
OPTIONS_PERCENT_MULTIPLIER = 100.0
OPTIONS_SCORE_MIN = 0.0
OPTIONS_SCORE_MAX = 100.0
OPTIONS_SMART_TAG_TOP_SCORE = "top_score"
OPTIONS_SMART_TAG_STRONG_FLOW = "strong_flow"
OPTIONS_SMART_TAG_LIQUID = "liquid"
OPTIONS_SMART_TAG_RELATIVE_VALUE_IV = "relative_value_iv"
OPTIONS_SMART_TAG_HOT_IV = "hot_iv"
OPTIONS_SMART_TAG_LIMITED_HISTORY = "limited_history"
OPTIONS_SMART_TAG_CALL_OI_LEAD = "call_oi_lead"
OPTIONS_SMART_TAG_PUT_OI_HEAVY = "put_oi_heavy"
OPTIONS_SMART_TAG_OI_CONFIRMED = "oi_confirmed"
OPTIONS_SMART_SIGNAL_BULLISH = "bullish_candidate"
OPTIONS_SMART_SIGNAL_BEARISH_OR_HEDGE = "bearish_or_hedge"
OPTIONS_SMART_SIGNAL_RELATIVE_VALUE = "relative_value"
OPTIONS_SMART_SIGNAL_MOMENTUM_CHASE = "momentum_chase"
OPTIONS_SMART_SIGNAL_WATCH = "watch"
OPTIONS_SMART_STRONG_SCORE = 65.0
OPTIONS_SMART_HIGH_SCORE = 75.0
OPTIONS_SMART_STRONG_SKEW_PCT = 20.0
OPTIONS_SMART_LOW_PUT_CALL_OI = 0.7
OPTIONS_SMART_HIGH_PUT_CALL_OI = 1.3
OPTIONS_SMART_HOT_IV_PCT = 100.0
OPTIONS_SMART_LOW_HISTORY_SNAPSHOTS = 3
OPTIONS_SMART_SINGLE_NAME_SCORE = 50.0
OPTIONS_SMART_OI_MISMATCH_SCORE = 25.0
OPTIONS_SMART_SCORE_WEIGHTS = {
    "volume_rank": 0.35,
    "liquidity": 0.25,
    "iv_value": 0.20,
    "flow_quality": 0.20,
}
OPTIONS_DIGEST_DEFAULT_LIMIT = 5
OPTIONS_SNAPSHOT_DEFAULT_HOUR_UTC = 21
OPTIONS_SNAPSHOT_DEFAULT_MINUTE_UTC = 40
OPTIONS_SNAPSHOT_DEFAULT_MAX_TICKERS = 40
OPTIONS_SNAPSHOT_DEFAULT_MAX_EXPIRIES = 2
OPTIONS_SNAPSHOT_DEFAULT_MIN_MONEYNESS = 0.7
OPTIONS_SNAPSHOT_DEFAULT_MAX_MONEYNESS = 1.3
OPTIONS_SNAPSHOT_DEFAULT_MIN_INTERVAL_HOURS = 20.0
OPTIONS_SNAPSHOT_DEFAULT_SLEEP_SEC = 0.25
OPTIONS_SNAPSHOT_DEFAULT_TICKERS = (
    "SPY",
    "QQQ",
    "IWM",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMD",
    "META",
    "AMZN",
    "GOOGL",
)


def env_str(
    name: str,
    default: str = "",
    *,
    allow_blank: bool = True,
    strip: bool = True,
) -> str:
    """Read a string env var without silently replacing invalid configured values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip() if strip else str(raw)
    if not value and not allow_blank:
        raise ConfigError(f"{name} cannot be blank")
    return value


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var and reject invalid configured values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ConfigError(
        f"{name} must be one of {sorted(TRUE_VALUES | FALSE_VALUES)}; got {raw!r}"
    )


def env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Read an integer env var and fail fast when configured out of range."""
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        text = str(raw).strip()
        if not text:
            raise ConfigError(f"{name} cannot be blank")
        try:
            value = int(text)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer; got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}; got {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"{name} must be <= {max_value}; got {value}")
    return value


def env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read a float env var and fail fast when configured out of range."""
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        text = str(raw).strip()
        if not text:
            raise ConfigError(f"{name} cannot be blank")
        try:
            value = float(text)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number; got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}; got {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"{name} must be <= {max_value}; got {value}")
    return value


def env_optional_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read an optional float env var; blank means intentionally unset."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return env_float(name, default, min_value=min_value, max_value=max_value)


def env_path(name: str, default: Path) -> Path:
    """Read a path env var from one place."""
    return Path(env_str(name, str(default), allow_blank=False))


@dataclass(frozen=True)
class OptionsSnapshotConfig:
    """Options snapshot job configuration."""

    enabled: bool
    hour_utc: int
    minute_utc: int
    tickers: str
    latest_report_path: Path
    max_tickers: int
    max_expiries: int
    min_moneyness: float
    max_moneyness: float
    min_interval_hours: float
    sleep_sec: float
    log_path: Path


@dataclass(frozen=True)
class OptionsDigestConfig:
    """Options Telegram digest configuration."""

    enabled: bool
    limit: int


@dataclass(frozen=True)
class OptionsPremiumConfig:
    """Options premium API/query configuration."""

    default_limit: int
    page_limit: int
    min_limit: int
    max_limit: int
    default_sort_by: str
    sort_choices: tuple[str, ...]
    sort_pattern: str


@dataclass(frozen=True)
class OptionsConfig:
    """Single source of truth for options research, API, scheduler, and digest."""

    source: str
    source_note: str
    premium_proxy_note: str
    default_tickers: tuple[str, ...]
    premium: OptionsPremiumConfig
    snapshot: OptionsSnapshotConfig
    digest: OptionsDigestConfig


def normalize_options_limit(limit: int, *, name: str = "limit") -> int:
    """Validate a premium-proxy limit without clamping hidden bad inputs."""
    try:
        value = int(limit)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer; got {limit!r}") from exc
    if value < OPTIONS_PREMIUM_MIN_LIMIT or value > OPTIONS_PREMIUM_MAX_LIMIT:
        raise ConfigError(
            f"{name} must be between {OPTIONS_PREMIUM_MIN_LIMIT} "
            f"and {OPTIONS_PREMIUM_MAX_LIMIT}; got {value}"
        )
    return value


def normalize_options_sort(sort_by: str) -> str:
    """Validate a premium-proxy sort key."""
    value = str(sort_by or "").strip()
    if value not in OPTIONS_PREMIUM_SORT_CHOICES:
        raise ConfigError(
            "sort_by must be one of "
            f"{', '.join(OPTIONS_PREMIUM_SORT_CHOICES)}; got {sort_by!r}"
        )
    return value


def get_options_config() -> OptionsConfig:
    """Build options config from env with explicit defaults and strict parsing."""
    report_dir = env_path("TRADER_KOO_REPORT_DIR", DEFAULT_REPORT_DIR)
    log_dir = env_path("TRADER_KOO_LOG_DIR", DEFAULT_LOG_DIR)
    latest_report_default = report_dir / "daily_report_latest.json"
    snapshot_log_default = log_dir / "options_iv_snapshot.log"
    default_sort_by = normalize_options_sort(
        env_str(
            "TRADER_KOO_OPTIONS_PREMIUM_SORT_BY",
            OPTIONS_PREMIUM_DEFAULT_SORT_BY,
            allow_blank=False,
        )
    )
    default_limit = env_int(
        "TRADER_KOO_OPTIONS_PREMIUM_LIMIT",
        OPTIONS_PREMIUM_DEFAULT_LIMIT,
        min_value=OPTIONS_PREMIUM_MIN_LIMIT,
        max_value=OPTIONS_PREMIUM_MAX_LIMIT,
    )
    page_limit = env_int(
        "TRADER_KOO_OPTIONS_PREMIUM_PAGE_LIMIT",
        OPTIONS_PREMIUM_PAGE_LIMIT,
        min_value=OPTIONS_PREMIUM_MIN_LIMIT,
        max_value=OPTIONS_PREMIUM_MAX_LIMIT,
    )
    config = OptionsConfig(
        source=OPTIONS_SOURCE,
        source_note=OPTIONS_SOURCE_NOTE,
        premium_proxy_note=OPTIONS_PREMIUM_PROXY_NOTE,
        default_tickers=OPTIONS_SNAPSHOT_DEFAULT_TICKERS,
        premium=OptionsPremiumConfig(
            default_limit=default_limit,
            page_limit=page_limit,
            min_limit=OPTIONS_PREMIUM_MIN_LIMIT,
            max_limit=OPTIONS_PREMIUM_MAX_LIMIT,
            default_sort_by=default_sort_by,
            sort_choices=OPTIONS_PREMIUM_SORT_CHOICES,
            sort_pattern=OPTIONS_PREMIUM_SORT_PATTERN,
        ),
        snapshot=OptionsSnapshotConfig(
            enabled=env_bool("TRADER_KOO_OPTIONS_SNAPSHOT_ENABLED", True),
            hour_utc=env_int(
                "TRADER_KOO_OPTIONS_SNAPSHOT_HOUR_UTC",
                OPTIONS_SNAPSHOT_DEFAULT_HOUR_UTC,
                min_value=0,
                max_value=23,
            ),
            minute_utc=env_int(
                "TRADER_KOO_OPTIONS_SNAPSHOT_MINUTE_UTC",
                OPTIONS_SNAPSHOT_DEFAULT_MINUTE_UTC,
                min_value=0,
                max_value=59,
            ),
            tickers=env_str("TRADER_KOO_OPTIONS_SNAPSHOT_TICKERS", ""),
            latest_report_path=env_path(
                "TRADER_KOO_OPTIONS_SNAPSHOT_REPORT",
                latest_report_default,
            ),
            max_tickers=env_int(
                "TRADER_KOO_OPTIONS_SNAPSHOT_MAX_TICKERS",
                OPTIONS_SNAPSHOT_DEFAULT_MAX_TICKERS,
                min_value=1,
                max_value=500,
            ),
            max_expiries=env_int(
                "TRADER_KOO_OPTIONS_MAX_EXPIRIES",
                OPTIONS_SNAPSHOT_DEFAULT_MAX_EXPIRIES,
                min_value=1,
                max_value=12,
            ),
            min_moneyness=env_float(
                "TRADER_KOO_OPTIONS_MIN_MONEYNESS",
                OPTIONS_SNAPSHOT_DEFAULT_MIN_MONEYNESS,
                min_value=0.01,
                max_value=5.0,
            ),
            max_moneyness=env_float(
                "TRADER_KOO_OPTIONS_MAX_MONEYNESS",
                OPTIONS_SNAPSHOT_DEFAULT_MAX_MONEYNESS,
                min_value=0.01,
                max_value=5.0,
            ),
            min_interval_hours=env_float(
                "TRADER_KOO_OPTIONS_MIN_INTERVAL_HOURS",
                OPTIONS_SNAPSHOT_DEFAULT_MIN_INTERVAL_HOURS,
                min_value=0.0,
                max_value=168.0,
            ),
            sleep_sec=env_float(
                "TRADER_KOO_OPTIONS_SNAPSHOT_SLEEP_SEC",
                OPTIONS_SNAPSHOT_DEFAULT_SLEEP_SEC,
                min_value=0.0,
                max_value=10.0,
            ),
            log_path=env_path(
                "TRADER_KOO_OPTIONS_SNAPSHOT_LOG",
                snapshot_log_default,
            ),
        ),
        digest=OptionsDigestConfig(
            enabled=env_bool("TRADER_KOO_OPTIONS_DIGEST_ENABLED", True),
            limit=env_int(
                "TRADER_KOO_OPTIONS_DIGEST_LIMIT",
                OPTIONS_DIGEST_DEFAULT_LIMIT,
                min_value=1,
                max_value=50,
            ),
        ),
    )
    if config.snapshot.min_moneyness > config.snapshot.max_moneyness:
        raise ConfigError(
            "TRADER_KOO_OPTIONS_MIN_MONEYNESS must be <= "
            "TRADER_KOO_OPTIONS_MAX_MONEYNESS"
        )
    return config


class Config:
    """Application configuration with validation.

    This class validates critical security configuration at startup to ensure
    the platform is deployed with proper authentication and security settings.
    """

    def __init__(self):
        """Initialize and validate configuration."""
        # Load environment variables
        self.admin_strict_api_key = env_bool("ADMIN_STRICT_API_KEY", True)
        self.trader_koo_api_key = env_str("TRADER_KOO_API_KEY", "", strip=False)

        # Email token expiration (7 days = 168 hours)
        self.email_token_expiry_hours = env_int(
            "EMAIL_TOKEN_EXPIRY_HOURS",
            168,
            min_value=1,
            max_value=8760  # 1 year max
        )

        # CORS configuration - restrictive defaults
        self.cors_allowed_origins = self._parse_cors_origins(
            env_str("TRADER_KOO_CORS_ORIGINS", "")
        )

        # Development mode detection
        self.development_mode = env_bool("TRADER_KOO_DEVELOPMENT_MODE", False)

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

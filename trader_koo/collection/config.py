"""Tunables for the data-accumulation jobs (option chains, earnings history)."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from trader_koo.config import DEFAULT_LOG_DIR, env_bool, env_float, env_int, env_path

# Full-universe option-chain job. 16:10 New York follows the US close through
# DST changes. The job refuses to run at or after 21:25 UTC because the nightly
# report window starts at 21:30 UTC.
UNIVERSE_CRON_HOUR_ET = 16
UNIVERSE_CRON_MINUTE_ET = 10
UNIVERSE_WINDOW_OPEN_ET = dt.time(16, 5)
UNIVERSE_WINDOW_CLOSE_UTC = dt.time(21, 25)
WEEKEND_DAYS = (5, 6)
UNIVERSE_DEFAULT_MAX_TICKERS = 600
UNIVERSE_MAX_TICKERS_LIMIT = 1000
UNIVERSE_DEFAULT_WORKERS = 2
UNIVERSE_MAX_WORKERS = 3
UNIVERSE_DEFAULT_SLEEP_SEC = 0.25
UNIVERSE_NEAREST_EXPIRIES = 2
UNIVERSE_TARGET_DAYS = (30, 60, 90)

# options_daily_metrics math.
DAYS_PER_YEAR = 365.0
MIN_VALID_IV = 0.01  # yfinance reports ~1e-5 for dead quotes
MAX_VALID_IV = 5.0
SKEW_TARGET_DELTA = -0.25
SKEW_DELTA_TOLERANCE = 0.10  # NULL when no put lies within this of the target
SKEW_MATURITY_DAYS = 30
CP_SPREAD_MONEYNESS_BAND = 0.05
RATE_SYMBOL = "^IRX"  # 13-week T-bill discount rate, quoted in percent

# Earnings history.
FINNHUB_MIN_INTERVAL_SEC = 1.1  # free tier allows 60 calls/min
BMO_BEFORE = dt.time(9, 30)
AMC_FROM = dt.time(16, 0)
DATE_TEXT_LEN = 10  # YYYY-MM-DD
YFINANCE_EARNINGS_LIMIT = 40
BACKFILL_DEFAULT_YEARS = 5
BACKFILL_LOOKAHEAD_DAYS = 120


@dataclass(frozen=True)
class OptionsUniverseConfig:
    enabled: bool
    max_tickers: int
    workers: int
    sleep_sec: float
    nearest_expiries: int
    target_days: tuple[int, ...]
    log_path: Path


def get_options_universe_config() -> OptionsUniverseConfig:
    return OptionsUniverseConfig(
        enabled=env_bool("TRADER_KOO_OPTIONS_UNIVERSE_ENABLED", True),
        max_tickers=env_int(
            "TRADER_KOO_OPTIONS_UNIVERSE_MAX_TICKERS",
            UNIVERSE_DEFAULT_MAX_TICKERS,
            min_value=1,
            max_value=UNIVERSE_MAX_TICKERS_LIMIT,
        ),
        workers=env_int(
            "TRADER_KOO_OPTIONS_UNIVERSE_WORKERS",
            UNIVERSE_DEFAULT_WORKERS,
            min_value=1,
            max_value=UNIVERSE_MAX_WORKERS,
        ),
        sleep_sec=env_float(
            "TRADER_KOO_OPTIONS_UNIVERSE_SLEEP_SEC",
            UNIVERSE_DEFAULT_SLEEP_SEC,
            min_value=0.0,
            max_value=10.0,
        ),
        nearest_expiries=UNIVERSE_NEAREST_EXPIRIES,
        target_days=UNIVERSE_TARGET_DAYS,
        log_path=env_path("TRADER_KOO_LOG_DIR", DEFAULT_LOG_DIR) / "options_iv_universe.log",
    )

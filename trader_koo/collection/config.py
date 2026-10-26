"""Tunables for the data-accumulation jobs (option chains, earnings history).

Values come from config/research.toml (trader_koo/research/settings.py). The
names below are kept so the scheduler and the scripts import them unchanged.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from trader_koo.config import DEFAULT_LOG_DIR, env_bool, env_float, env_int, env_path
from trader_koo.research.settings import get as _settings

_C = _settings().collection

# Full-universe option-metrics job. End-of-day option quotes do not change
# between the close and the next pre-market, so it runs after the nightly report
# window (21:30 to 23:30 UTC) on XNYS-session evenings, stamps rows with that
# evening's US session date, and hard-stops at the window end.
UNIVERSE_CRON_HOUR_UTC = _C.universe_cron_hour_utc
UNIVERSE_CRON_MINUTE_UTC = _C.universe_cron_minute_utc
UNIVERSE_WINDOW_START_UTC = _C.window_start()
UNIVERSE_WINDOW_END_UTC = _C.window_end()
UNIVERSE_WAIT_POLL_SEC = _C.universe_wait_poll_sec
UNIVERSE_DEFAULT_MAX_TICKERS = _C.universe_default_max_tickers
UNIVERSE_MAX_TICKERS_LIMIT = _C.universe_max_tickers_limit
UNIVERSE_DEFAULT_WORKERS = _C.universe_default_workers
UNIVERSE_MAX_WORKERS = _C.universe_max_workers
UNIVERSE_DEFAULT_SLEEP_SEC = _C.universe_default_sleep_sec
UNIVERSE_MAX_SLEEP_SEC = _C.universe_max_sleep_sec
UNIVERSE_NEAREST_EXPIRIES = _C.universe_nearest_expiries
UNIVERSE_TARGET_DAYS = _C.universe_target_days

# options_daily_metrics math.
DAYS_PER_YEAR = _C.days_per_year
MIN_VALID_IV = _C.min_valid_iv
MAX_VALID_IV = _C.max_valid_iv
SKEW_TARGET_DELTA = _C.skew_target_delta
SKEW_DELTA_TOLERANCE = _C.skew_delta_tolerance
SKEW_MATURITY_DAYS = _C.skew_maturity_days
CP_SPREAD_MONEYNESS_BAND = _C.cp_spread_moneyness_band
RATE_SYMBOL = _C.rate_symbol
HISTORY_PERIOD = _C.history_period

# Earnings history.
FINNHUB_MIN_INTERVAL_SEC = _C.finnhub_min_interval_sec
BMO_BEFORE = dt.time.fromisoformat(_C.bmo_before_et)
AMC_FROM = dt.time.fromisoformat(_C.amc_from_et)
DATE_TEXT_LEN = 10  # YYYY-MM-DD, an ISO format width
YFINANCE_EARNINGS_LIMIT = _C.yfinance_earnings_limit
BACKFILL_DEFAULT_YEARS = _C.backfill_default_years
BACKFILL_LOOKAHEAD_DAYS = _C.backfill_lookahead_days


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
            max_value=UNIVERSE_MAX_SLEEP_SEC,
        ),
        nearest_expiries=UNIVERSE_NEAREST_EXPIRIES,
        target_days=UNIVERSE_TARGET_DAYS,
        log_path=env_path("TRADER_KOO_LOG_DIR", DEFAULT_LOG_DIR) / "options_iv_universe.log",
    )

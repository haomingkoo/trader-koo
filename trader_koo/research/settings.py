"""Typed, validated settings for the research and data-collection code.

``config/research.toml`` is the only place these values live. Each entry is
``{ value, unit, reason }``. The file is read once into frozen dataclasses; a
missing section or key, an unknown key, a wrong type or a missing unit/reason
raises ``SettingsError``. There is no default for any value in code.

The numbers inside pre-registered hypothesis signals are not settings; see
``trader_koo/research/bench/hypotheses.py``.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import typing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 dev environments; deployment runs 3.11+
    import tomli as tomllib

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "research.toml"
ENTRY_KEYS = frozenset({"value", "unit", "reason"})


class SettingsError(ValueError):
    """The settings file is missing, malformed or does not match the typed schema."""


@dataclass(frozen=True)
class BenchSettings:
    cost_bps: float
    stress_bps: float
    n_random: int
    top_n: int
    buffer_n: int
    review_every: int
    min_hold: int
    seal_fraction: float
    trading_days_per_year: int
    random_pct_low: int
    random_pct_mid: int
    random_pct_high: int
    summary_digits: int
    move_digits: int
    cost_solve_iterations: int
    benchmark_ticker: str
    universe: str
    split_ratio_round_digits: int
    keep_real_move_flags: tuple[str, ...]
    share_check_filings: int
    share_check_min_prior: int
    share_scale_limit: float
    share_history_rows: int
    lookahead_seed: int
    lookahead_noise_low: float
    lookahead_noise_high: float
    lookahead_price_factor: float
    lookahead_fact_scale: float
    lookahead_fact_shift: float


@dataclass(frozen=True)
class QualitySettings:
    zero_volume_run_sessions: int
    seam_move_fraction: float
    erratic_move_fraction: float
    erratic_moves_per_year: int


@dataclass(frozen=True)
class TotalReturnSettings:
    version: str
    max_abs_daily_move: float
    max_dividend_fraction: float


@dataclass(frozen=True)
class FundamentalsSettings:
    sec_user_agent: str
    sec_tickers_url: str
    sec_facts_url: str
    requests_per_second: float
    http_timeout_sec: int
    max_attempts: int
    backoff_base_sec: float
    backoff_cap_sec: float
    retry_statuses: tuple[int, ...]
    quarter_days: tuple[int, ...]
    ytd_days: tuple[int, ...]
    annual_days: tuple[int, ...]
    match_tolerance_days: int
    days_per_year: int


@dataclass(frozen=True)
class PriceBackfillSettings:
    floor: str
    warmup_days: int
    departure_buffer_days: int
    floor_tolerance_days: int
    retries: int
    backoff_sec: float
    pace_sec: float
    seed_batch: int
    detail_max_chars: int
    probe_days: int
    http_timeout_sec: int
    busy_timeout_ms: int
    sha_prefix_chars: int
    gap_days: int
    late_start_days: int


@dataclass(frozen=True)
class MembershipSettings:
    count_min: int
    count_max: int
    date_tolerance_days: int
    first_year: int
    fetch_timeout_sec: int
    user_agent: str


@dataclass(frozen=True)
class CollectionSettings:
    universe_cron_hour_utc: int
    universe_cron_minute_utc: int
    universe_window_start_utc: str
    universe_window_end_utc: str
    universe_wait_poll_sec: int
    universe_default_max_tickers: int
    universe_max_tickers_limit: int
    universe_default_workers: int
    universe_max_workers: int
    universe_default_sleep_sec: float
    universe_max_sleep_sec: float
    universe_nearest_expiries: int
    universe_target_days: tuple[int, ...]
    days_per_year: float
    min_valid_iv: float
    max_valid_iv: float
    skew_target_delta: float
    skew_delta_tolerance: float
    skew_maturity_days: int
    cp_spread_moneyness_band: float
    rate_symbol: str
    history_period: str
    finnhub_min_interval_sec: float
    bmo_before_et: str
    amc_from_et: str
    yfinance_earnings_limit: int
    backfill_default_years: int
    backfill_lookahead_days: int

    def window_start(self) -> dt.time:
        return dt.time.fromisoformat(self.universe_window_start_utc)

    def window_end(self) -> dt.time:
        return dt.time.fromisoformat(self.universe_window_end_utc)


@dataclass(frozen=True)
class ResearchSettings:
    bench: BenchSettings
    quality: QualitySettings
    total_return: TotalReturnSettings
    fundamentals: FundamentalsSettings
    price_backfill: PriceBackfillSettings
    membership: MembershipSettings
    collection: CollectionSettings
    sha256: str
    path: str


def _check(value: object, annotation: object, where: str) -> object:
    """Return ``value`` coerced to ``annotation`` or raise; bool is never an int and a str is never a number."""
    if annotation is bool or annotation is str:
        if not isinstance(value, annotation):
            raise SettingsError(f"{where}: expected {annotation.__name__}, got {type(value).__name__}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(f"{where}: expected int, got {type(value).__name__}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{where}: expected a number, got {type(value).__name__}")
        return float(value)
    if typing.get_origin(annotation) is tuple:
        (item,) = (a for a in typing.get_args(annotation) if a is not Ellipsis)
        if not isinstance(value, list) or not value:
            raise SettingsError(f"{where}: expected a non-empty list")
        return tuple(_check(v, item, f"{where}[{i}]") for i, v in enumerate(value))
    raise SettingsError(f"{where}: unsupported setting type {annotation!r}")


def _section(cls: type, table: object, name: str) -> object:
    if not isinstance(table, dict):
        raise SettingsError(f"section [{name}] is missing")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    missing = [n for n in names if n not in table]
    unknown = sorted(set(table) - set(names))
    if missing or unknown:
        raise SettingsError(f"[{name}]: missing keys {missing}, unknown keys {unknown}")
    values = {}
    for key in names:
        where = f"[{name}].{key}"
        entry = table[key]
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise SettingsError(f"{where}: must be a table with exactly value, unit and reason")
        for text in ("unit", "reason"):
            if not isinstance(entry[text], str) or not entry[text].strip():
                raise SettingsError(f"{where}: {text} must be a non-empty string")
        values[key] = _check(entry["value"], hints[key], where)
    return cls(**values)


def _validate(s: ResearchSettings) -> None:
    b, q, c, f = s.bench, s.quality, s.collection, s.fundamentals
    problems = []
    if not 0.0 < b.seal_fraction < 1.0:
        problems.append("bench.seal_fraction must be between 0 and 1")
    if not b.random_pct_low < b.random_pct_mid < b.random_pct_high:
        problems.append("bench.random_pct_* must increase")
    if not 0 < b.top_n <= b.buffer_n:
        problems.append("bench needs 0 < top_n <= buffer_n")
    if min(b.review_every, b.n_random, b.share_check_filings) < 1 or b.min_hold < 0:
        problems.append("bench counts must be positive")
    if q.zero_volume_run_sessions < 1 or q.erratic_moves_per_year < 1:
        problems.append("quality counts must be positive")
    for name in ("quarter_days", "ytd_days", "annual_days"):
        pair = getattr(f, name)
        if len(pair) != 2 or pair[0] > pair[1]:
            problems.append(f"fundamentals.{name} must be [low, high]")
    if (c.universe_cron_hour_utc, c.universe_cron_minute_utc) != (c.window_start().hour, c.window_start().minute):
        problems.append("collection cron time must equal universe_window_start_utc")
    if problems:
        raise SettingsError("; ".join(problems))


def load_settings(path: str | Path = DEFAULT_PATH) -> ResearchSettings:
    """Read and validate a settings file. Raises ``SettingsError`` on any problem."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise SettingsError(f"cannot read settings file {p}: {exc}") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise SettingsError(f"{p}: not valid TOML: {exc}") from exc
    sections = {f.name: f.type for f in dataclasses.fields(ResearchSettings) if f.name not in ("sha256", "path")}
    unknown = sorted(set(data) - set(sections))
    if unknown:
        raise SettingsError(f"unknown sections {unknown}")
    hints = typing.get_type_hints(ResearchSettings)
    built = {name: _section(hints[name], data.get(name), name) for name in sections}
    settings = ResearchSettings(**built, sha256=hashlib.sha256(raw).hexdigest(), path=str(p))
    _validate(settings)
    return settings


@lru_cache(maxsize=1)
def get() -> ResearchSettings:
    """The default settings file, loaded once per process."""
    return load_settings(DEFAULT_PATH)

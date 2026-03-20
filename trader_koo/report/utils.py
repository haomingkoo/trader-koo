"""Shared utility functions for report generation."""
from __future__ import annotations

import calendar
import datetime as dt
import logging
import math
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LOG = logging.getLogger(__name__)

MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")
try:
    MARKET_TZ = ZoneInfo(MARKET_TZ_NAME)
except Exception as exc:
    LOG.warning("Failed to load market timezone %r, falling back to UTC: %s", MARKET_TZ_NAME, exc)
    MARKET_TZ = dt.timezone.utc
MARKET_CLOSE_HOUR = min(23, max(0, int(os.getenv("TRADER_KOO_MARKET_CLOSE_HOUR", "16"))))
TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY_VALUES


def _normalize_report_kind(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "weekly":
        return "weekly"
    return "daily"


def table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def parse_iso_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        ts = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def hours_since(value: str | None, now: dt.datetime) -> float | None:
    ts = parse_iso_utc(value)
    if ts is None:
        return None
    return (now - ts).total_seconds() / 3600.0


def days_since_date(value: str | None, now: dt.datetime) -> float | None:
    if not value:
        return None
    try:
        market_date = dt.date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None
    market_close = dt.datetime.combine(market_date, dt.time(hour=MARKET_CLOSE_HOUR), tzinfo=MARKET_TZ)
    now_market = now.astimezone(MARKET_TZ)
    age_days = (now_market - market_close).total_seconds() / 86400.0
    return max(0.0, age_days)


def _observed_holiday(day: dt.date) -> dt.date:
    # NYSE-style weekend observation for fixed-date holidays.
    if day.weekday() == 5:  # Saturday -> Friday
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> Monday
        return day + dt.timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=delta + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    last_day = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last_day)
    delta = (d.weekday() - weekday) % 7
    return d - dt.timedelta(days=delta)


def _easter_sunday(year: int) -> dt.date:
    # Anonymous Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def nyse_holidays_for_year(year: int) -> dict[dt.date, str]:
    out: dict[dt.date, str] = {}
    out[_observed_holiday(dt.date(year, 1, 1))] = "New Year's Day"
    # Include observed New Year of next year if it lands in this year (e.g. Dec 31).
    next_new_year_obs = _observed_holiday(dt.date(year + 1, 1, 1))
    if next_new_year_obs.year == year:
        out[next_new_year_obs] = "New Year's Day (Observed)"
    out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"  # 3rd Monday Jan
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"  # 3rd Monday Feb
    out[_easter_sunday(year) - dt.timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"  # last Monday May
    out[_observed_holiday(dt.date(year, 6, 19))] = "Juneteenth National Independence Day"
    out[_observed_holiday(dt.date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"  # 4th Thursday Nov
    out[_observed_holiday(dt.date(year, 12, 25))] = "Christmas Day"
    return out


def nyse_early_closes_for_year(year: int) -> dict[dt.date, str]:
    out: dict[dt.date, str] = {}
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    friday_after_thanksgiving = thanksgiving + dt.timedelta(days=1)
    if friday_after_thanksgiving.weekday() < 5:
        out[friday_after_thanksgiving] = "Day after Thanksgiving (1:00 PM ET close)"

    christmas_eve = dt.date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in nyse_holidays_for_year(year):
        out[christmas_eve] = "Christmas Eve (1:00 PM ET close)"

    july3 = dt.date(year, 7, 3)
    if july3.weekday() < 5 and july3 not in nyse_holidays_for_year(year):
        out[july3] = "Pre-Independence Day (1:00 PM ET close)"
    return out


def market_calendar_context(now_utc: dt.datetime) -> dict[str, Any]:
    now_et = now_utc.astimezone(MARKET_TZ)
    today = now_et.date()
    years = [today.year - 1, today.year, today.year + 1]
    holidays: dict[dt.date, str] = {}
    early_closes: dict[dt.date, str] = {}
    for year in years:
        holidays.update(nyse_holidays_for_year(year))
        early_closes.update(nyse_early_closes_for_year(year))

    today_holiday = holidays.get(today)
    today_early_close = early_closes.get(today)
    next_holiday = None
    next_early_close = None
    for day in sorted(holidays):
        if day >= today:
            next_holiday = {"date": day.isoformat(), "name": holidays[day]}
            break
    for day in sorted(early_closes):
        if day >= today:
            next_early_close = {"date": day.isoformat(), "name": early_closes[day]}
            break

    return {
        "market_tz": MARKET_TZ_NAME,
        "as_of_market_ts": now_et.replace(microsecond=0).isoformat(),
        "market_date": today.isoformat(),
        "is_holiday": bool(today_holiday),
        "holiday_name": today_holiday,
        "is_early_close": bool(today_early_close),
        "early_close_name": today_early_close,
        "next_holiday": next_holiday,
        "next_early_close": next_early_close,
    }


def tail_text(path: Path, lines: int = 80, max_bytes: int = 96_000) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]
    except Exception as exc:
        LOG.warning("tail_text failed for %s: %s", path, exc)
        return []


def row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(row)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _setup_tier(score: float) -> str:
    if score >= 80.0:
        return "A"
    if score >= 65.0:
        return "B"
    if score >= 50.0:
        return "C"
    return "D"


def _stdev(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _fmt_pct_short(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    num = float(value)
    sign = "+" if num > 0 else ""
    return f"{sign}{num:.1f}%"


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n % 2 == 1:
        return sorted_v[n // 2]
    return (sorted_v[(n // 2) - 1] + sorted_v[n // 2]) / 2.0


def _percentile_rank(values: list[float], current: float | None) -> float | None:
    if current is None or not values:
        return None
    rank_le = sum(1 for v in values if v <= current)
    return (rank_le / len(values)) * 100.0


def _parse_iso_date(value: Any) -> dt.date | None:
    if not value:
        return None
    raw = str(value).strip()
    if len(raw) < 10:
        return None
    try:
        return dt.date.fromisoformat(raw[:10])
    except ValueError:
        return None

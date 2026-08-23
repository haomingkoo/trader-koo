"""Shared point-in-time rules for paper execution and replay."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

_MARKET_TZ = ZoneInfo("America/New_York")


@lru_cache(maxsize=1)
def _nyse_calendar():
    return xcals.get_calendar("XNYS")


def next_scheduled_session_after(report_date: str) -> str | None:
    """Return the immediate scheduled NYSE session after the report date."""
    try:
        candidate = dt.date.fromisoformat(report_date) + dt.timedelta(days=1)
        session = _nyse_calendar().date_to_session(
            candidate.isoformat(), direction="next"
        )
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None
    return session.date().isoformat()


def publication_precedes_session_open(
    published_ts: str,
    session_date: str,
) -> bool:
    """Return whether verified publication existed strictly before 09:30 ET."""
    try:
        published = dt.datetime.fromisoformat(published_ts.replace("Z", "+00:00"))
        if published.tzinfo is None:
            return False
        session = dt.date.fromisoformat(session_date)
        market_open = dt.datetime.combine(
            session,
            dt.time(hour=9, minute=30),
            tzinfo=_MARKET_TZ,
        ).astimezone(dt.timezone.utc)
    except (TypeError, ValueError):
        return False
    return published.astimezone(dt.timezone.utc) < market_open

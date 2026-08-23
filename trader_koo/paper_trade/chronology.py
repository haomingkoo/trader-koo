"""Shared point-in-time rules for paper execution and replay."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from zoneinfo import ZoneInfo

_MARKET_TZ = ZoneInfo("America/New_York")


def next_session_after(report_date: str, sessions: Iterable[str]) -> str | None:
    """Return the immediate observed session after the report date."""
    try:
        cutoff = dt.date.fromisoformat(report_date)
        parsed = sorted(
            (dt.date.fromisoformat(str(value)), str(value)) for value in set(sessions)
        )
    except (TypeError, ValueError):
        return None
    return next((raw for date, raw in parsed if date > cutoff), None)


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

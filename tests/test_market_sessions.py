from __future__ import annotations

import datetime as dt

from trader_koo.backend.services.market_data import market_session_completion
from trader_koo.report.utils import completed_nyse_period_through


def _utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_friday_week_is_not_complete_before_new_york_close() -> None:
    before = market_session_completion(_utc("2026-08-21T12:00:00Z"))
    after = market_session_completion(_utc("2026-08-21T21:00:00Z"))
    assert before["completed_week_through"] == "2026-08-14"
    assert after["completed_week_through"] == "2026-08-21"


def test_month_ending_on_weekend_completes_on_friday_close() -> None:
    assert completed_nyse_period_through(
        "monthly", _utc("2026-01-30T21:30:00Z")
    ) == dt.date(2026, 1, 30)


def test_early_close_session_is_complete_after_one_pm_new_york() -> None:
    before = market_session_completion(_utc("2026-11-27T17:30:00Z"))
    after = market_session_completion(_utc("2026-11-27T18:30:00Z"))
    assert before["last_completed_session"] == "2026-11-25"
    assert after["last_completed_session"] == "2026-11-27"

"""SEC 8-K events: point-in-time filtering, labels, failure reporting.

The submissions fixture is a trimmed copy of EDGAR's real response for Apple
(CIK 320193), so the parser runs against SEC's actual field layout.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from trader_koo import sec_events

FIXTURE = json.loads((Path(__file__).parent / "fixtures/sec/submissions_aapl_trimmed.json").read_text())
CIK_MAP = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
           "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"}}
UTC = dt.timezone.utc


@pytest.fixture(autouse=True)
def _fresh_cache():
    sec_events._cik_cache.update(loaded_at=0.0, map={})
    yield
    sec_events._cik_cache.update(loaded_at=0.0, map={})


def _fake_get(calls: list[str], *, fail: set[str] = frozenset()):
    def get_json(url: str):
        calls.append(url)
        if any(token in url for token in fail):
            raise RuntimeError("boom")
        if url == sec_events.sec_cfg.SEC_TICKERS_URL:
            return CIK_MAP
        if url.endswith("CIK0000320193.json"):
            return FIXTURE
        return {"filings": {"recent": {"form": []}}}
    return get_json


def test_only_8k_filings_accepted_before_the_cutoff_are_returned():
    cutoff = dt.datetime(2026, 9, 1, 20, 30, 35, tzinfo=UTC)
    out = sec_events.filings_from_submissions(FIXTURE, cik=320193, cutoff=cutoff, lookback_days=60)
    assert [f["accepted_at"] for f in out] == ["2026-09-01T20:30:35+00:00", "2026-07-30T20:30:28+00:00"]
    assert {f["form"] for f in out} <= {"8-K", "8-K/A"}
    one_second_earlier = cutoff - dt.timedelta(seconds=1)
    out = sec_events.filings_from_submissions(FIXTURE, cik=320193, cutoff=one_second_earlier, lookback_days=60)
    assert [f["accepted_at"] for f in out] == ["2026-07-30T20:30:28+00:00"]


def test_lookback_window_excludes_older_filings():
    cutoff = dt.datetime(2026, 9, 2, tzinfo=UTC)
    out = sec_events.filings_from_submissions(FIXTURE, cik=320193, cutoff=cutoff, lookback_days=30)
    assert [f["form"] for f in out] == ["8-K/A"]


def test_items_become_plain_labels_and_link_points_at_the_filing():
    cutoff = dt.datetime(2026, 8, 1, tzinfo=UTC)
    [earnings] = sec_events.filings_from_submissions(FIXTURE, cik=320193, cutoff=cutoff, lookback_days=10)
    assert earnings["items"] == ["Earnings results"]  # 9.01 exhibits dropped beside a real item
    assert earnings["url"] == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm"
    )
    assert sec_events._labels("9.01") == ["Exhibits filed"]
    assert sec_events._labels("6.10") == ["Item 6.10"]
    assert sec_events._labels("") == ["Filing (no item listed)"]


def test_cutoff_is_end_of_market_day_in_new_york_and_never_after_now():
    now = dt.datetime(2026, 10, 31, 2, 0, tzinfo=UTC)
    assert sec_events.cutoff_for("2026-10-30", now) == now
    later = dt.datetime(2026, 11, 5, tzinfo=UTC)
    assert sec_events.cutoff_for("2026-10-30", later) == dt.datetime(2026, 10, 31, 4, 0, tzinfo=UTC)


def test_surface_maps_tickers_skips_indexes_and_reports_unmapped():
    calls: list[str] = []
    surface = sec_events.build_setup_sec_events(
        ["AAPL", "^GSPC", "BRK-B", "ZZZZ", "AAPL"], as_of_date="2026-09-01",
        now_utc=dt.datetime(2026, 9, 10, tzinfo=UTC), get_json=_fake_get(calls), enabled=True, max_tickers=12,
    )
    assert surface["available"] is True
    assert surface["tickers"]["AAPL"][0]["form"] == "8-K/A"
    assert surface["tickers"]["BRK-B"] == []
    assert surface["unmapped"] == ["ZZZZ"]
    assert not any("GSPC" in url for url in calls)
    assert sum(url.endswith("CIK0000320193.json") for url in calls) == 1


def test_ticker_map_failure_is_reported_not_hidden():
    calls: list[str] = []
    surface = sec_events.build_setup_sec_events(
        ["AAPL"], as_of_date="2026-09-01", now_utc=dt.datetime(2026, 9, 10, tzinfo=UTC),
        get_json=_fake_get(calls, fail={"company_tickers"}), enabled=True, max_tickers=12,
    )
    assert surface["available"] is False
    assert surface["error"] == "cik_map_failed: RuntimeError"
    assert surface["tickers"] == {}


def test_one_ticker_failing_keeps_the_others():
    calls: list[str] = []
    surface = sec_events.build_setup_sec_events(
        ["AAPL", "BRK-B"], as_of_date="2026-09-01", now_utc=dt.datetime(2026, 9, 10, tzinfo=UTC),
        get_json=_fake_get(calls, fail={"CIK0001067983"}), enabled=True, max_tickers=12,
    )
    assert surface["errors"] == {"BRK-B": "RuntimeError"}
    assert "AAPL" in surface["tickers"] and surface["available"] is True


def test_disabled_makes_no_requests():
    calls: list[str] = []
    surface = sec_events.build_setup_sec_events(
        ["AAPL"], as_of_date="2026-09-01", get_json=_fake_get(calls), enabled=False,
    )
    assert calls == [] and surface["enabled"] is False and surface["available"] is False


def test_max_tickers_bounds_the_requests():
    calls: list[str] = []
    sec_events.build_setup_sec_events(
        ["AAPL", "BRK-B"], as_of_date="2026-09-01", now_utc=dt.datetime(2026, 9, 10, tzinfo=UTC),
        get_json=_fake_get(calls), enabled=True, max_tickers=1,
    )
    assert len(calls) == 2  # ticker map + one submissions file


def test_repeated_failures_stop_further_requests():
    calls: list[str] = []
    big_map = {str(i): {"cik_str": 1000 + i, "ticker": f"T{i}", "title": "x"} for i in range(6)}

    def get_json(url: str):
        calls.append(url)
        if url == sec_events.sec_cfg.SEC_TICKERS_URL:
            return big_map
        raise TimeoutError("edgar down")

    surface = sec_events.build_setup_sec_events(
        [f"T{i}" for i in range(6)], as_of_date="2026-09-01", now_utc=dt.datetime(2026, 9, 10, tzinfo=UTC),
        get_json=get_json, enabled=True, max_tickers=12,
    )
    assert len(calls) == 1 + sec_events.MAX_FAILURES
    assert list(surface["errors"].values()).count("skipped_after_failures") == 6 - sec_events.MAX_FAILURES
    assert surface["available"] is False

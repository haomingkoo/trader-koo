"""Market sentiment is refreshed in the background and served from the last result."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.structure import fear_greed

_NO_PROVIDER = {"available": False, "items": [], "provider": None}


@pytest.fixture()
def file_conn(seeded_conn, tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "sentiment.db"))
    seeded_conn.backup(conn)
    fear_greed._FG_CACHE.clear()
    yield conn
    fear_greed._FG_CACHE.clear()
    conn.close()


def _no_external_sentiment():
    return (
        patch.object(fear_greed, "get_external_news_sentiment", return_value=dict(_NO_PROVIDER)),
        patch.object(fear_greed, "get_social_sentiment", return_value=dict(_NO_PROVIDER)),
    )


def test_first_call_computes_and_reports_when(file_conn) -> None:
    news, social = _no_external_sentiment()
    before = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    with news, social:
        result = fear_greed.latest_fear_greed_index(file_conn)

    assert result["score"] is not None
    assert dt.datetime.fromisoformat(result["computed_at"]) >= before


def test_latest_serves_stale_result_without_blocking_on_providers(file_conn) -> None:
    news, social = _no_external_sentiment()
    with news, social:
        first = fear_greed.refresh_fear_greed_index(file_conn)
    key = fear_greed._db_cache_key(file_conn)
    fear_greed._FG_CACHE[key] = (0.0, fear_greed._FG_CACHE[key][1])  # older than any TTL

    with (
        patch.object(fear_greed, "get_external_news_sentiment", side_effect=AssertionError("blocked")),
        patch.object(fear_greed, "get_social_sentiment", side_effect=AssertionError("blocked")),
    ):
        served = fear_greed.latest_fear_greed_index(file_conn)

    assert served["computed_at"] == first["computed_at"]
    assert served["score"] == first["score"]


def test_market_sentiment_endpoint_serves_latest_result(file_conn) -> None:
    from trader_koo.backend.routers import report

    news, social = _no_external_sentiment()
    with news, social:
        stored = fear_greed.refresh_fear_greed_index(file_conn)

    with (
        patch.object(report, "get_conn", return_value=MagicMock(wraps=file_conn)),
        patch.object(fear_greed, "_compute_fear_greed_index_uncached", side_effect=AssertionError("recomputed")),
    ):
        body = report.fear_greed_index()

    assert body == {"ok": True, **stored}


def test_scheduler_refreshes_market_sentiment_every_ten_minutes() -> None:
    from trader_koo.backend.services.scheduler import create_scheduler

    job = create_scheduler().get_job("market_sentiment_refresh")

    assert job is not None
    assert job.trigger.interval.total_seconds() == 10 * 60

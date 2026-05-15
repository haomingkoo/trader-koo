from __future__ import annotations

import datetime as dt
import sqlite3

from trader_koo.rss_news import (
    _score_headline,
    _score_to_100,
    fetch_rss_headlines,
    load_rss_headline_snapshot,
    persist_rss_headline_snapshot,
)


class TestHeadlineScoring:
    def test_bullish_headline(self):
        raw, bullish, bearish = _score_headline("Markets rally on strong earnings beat")
        assert raw is not None
        assert raw > 0
        assert bullish > bearish

    def test_bearish_headline(self):
        raw, bullish, bearish = _score_headline("Stocks crash amid recession fears and selloff")
        assert raw is not None
        assert raw < 0
        assert bearish > bullish

    def test_neutral_headline(self):
        raw, bullish, bearish = _score_headline("Company announces quarterly results")
        assert raw == 0.0
        assert bullish == 0
        assert bearish == 0

    def test_empty_headline(self):
        raw, bullish, bearish = _score_headline("")
        assert raw is None

    def test_score_to_100_mapping(self):
        assert _score_to_100(-1.0) == 0
        assert _score_to_100(0.0) == 50
        assert _score_to_100(1.0) == 100
        assert _score_to_100(None) is None


class TestFetchRssHeadlines:
    def test_returns_unavailable_when_disabled(self, monkeypatch):
        monkeypatch.setenv("TRADER_KOO_RSS_ENABLED", "0")

        result = fetch_rss_headlines(
            tickers=["SPY"],
            now_utc=dt.datetime(2026, 3, 18, 12, 0, tzinfo=dt.timezone.utc),
        )

        assert result["available"] is False
        assert "disabled" in result["note"]

    def test_returns_scored_headlines(self, monkeypatch):
        _RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>SPY surges as markets rally on optimism</title>
              <description>Strong gains across the board.</description>
              <link>https://example.com/article1</link>
              <pubDate>Wed, 18 Mar 2026 12:00:00 +0000</pubDate>
            </item>
            <item>
              <title>Recession fears grow amid weak data</title>
              <description>Markets fall on economic weakness.</description>
              <link>https://example.com/article2</link>
              <pubDate>Wed, 18 Mar 2026 11:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        class _FakeResp:
            def read(self):
                return _RSS_XML
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        monkeypatch.setenv("TRADER_KOO_RSS_ENABLED", "1")
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=15: _FakeResp())

        result = fetch_rss_headlines(
            tickers=["SPY"],
            include_market_feeds=False,
            now_utc=dt.datetime(2026, 3, 18, 12, 0, tzinfo=dt.timezone.utc),
        )

        assert result["available"] is True
        assert result["article_count"] == 2
        assert len(result["headlines"]) == 2
        # First headline is bullish (surges, rally, optimism)
        bullish_hl = [h for h in result["headlines"] if (h.get("score") or 0) > 50]
        assert len(bullish_hl) >= 1
        assert result["score"] is not None

    def test_market_feeds_include_official_macro_sources(self, monkeypatch):
        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <item>
              <title>CPI rises as Federal Reserve policy rate stays unchanged</title>
              <description>Inflation and rate data moved the market.</description>
              <link>https://example.com/macro</link>
              <pubDate>Fri, 10 Apr 2026 12:30:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""

        class _FakeResp:
            def read(self):
                return rss_xml
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        monkeypatch.setenv("TRADER_KOO_RSS_ENABLED", "1")
        monkeypatch.setenv("TRADER_KOO_MACRO_RSS_ENABLED", "1")
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=15: _FakeResp())

        result = fetch_rss_headlines(
            tickers=[],
            include_market_feeds=True,
            now_utc=dt.datetime(2026, 4, 10, 12, 30, tzinfo=dt.timezone.utc),
        )

        assert result["available"] is True
        assert result["article_count"] == 1
        assert result["headlines"][0]["macro_relevant"] is True
        assert "8 macro feeds" in result["note"]


class TestRssHeadlineSnapshots:
    def test_persist_and_load_snapshot_for_point_in_time_context(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        payload = {
            "provider": "rss_aggregator",
            "available": True,
            "headlines": [
                {
                    "title": "AMD rallies on strong demand",
                    "source": "Yahoo Finance",
                    "url": "https://example.com/amd",
                    "time_published": "2026-03-18T12:00:00Z",
                    "score": 82,
                    "label": "Extreme Greed",
                    "feed_ticker": "AMD",
                    "macro_relevant": False,
                },
                {
                    "title": "CPI risk pressures rate expectations",
                    "source": "BLS CPI",
                    "url": "https://example.com/cpi",
                    "time_published": "2026-03-18T13:00:00Z",
                    "score": 35,
                    "label": "Fear",
                    "feed_ticker": None,
                    "macro_relevant": True,
                },
            ],
        }

        inserted = persist_rss_headline_snapshot(
            conn,
            payload,
            snapshot_date="2026-03-18",
            snapshot_ts="2026-03-18T14:00:00Z",
        )
        conn.commit()
        loaded = load_rss_headline_snapshot(
            conn,
            tickers=["AMD"],
            as_of_date="2026-03-18",
            max_headlines=10,
        )

        assert inserted == 2
        assert loaded["available"] is True
        assert loaded["provider"] == "rss_snapshot"
        assert loaded["article_count"] == 2
        assert {h["title"] for h in loaded["headlines"]} == {
            "AMD rallies on strong demand",
            "CPI risk pressures rate expectations",
        }

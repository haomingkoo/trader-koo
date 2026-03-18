from __future__ import annotations

import datetime as dt

from trader_koo.rss_news import _score_headline, _score_to_100, fetch_rss_headlines


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

from __future__ import annotations

import datetime as dt
import json

from trader_koo.news_sentiment import get_external_news_sentiment


class _FakeResponse:
    def __init__(self, payload: dict | list):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TestExternalNewsSentiment:
    def test_falls_back_to_rss_when_no_keys_configured(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.delenv("TRADER_KOO_ALPHA_VANTAGE_KEY", raising=False)
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        monkeypatch.setenv("TRADER_KOO_RSS_ENABLED", "0")  # disable RSS to test empty path

        payload = get_external_news_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 0, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["provider"] == "rss_aggregator"
        assert "FINNHUB_API_KEY" in payload["note"]

    def test_finnhub_news_sentiment(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        monkeypatch.setenv("TRADER_KOO_SENTIMENT_TICKERS", "SPY")
        monkeypatch.setenv("TRADER_KOO_RSS_ENABLED", "0")

        def _fake_urlopen(req, timeout=20):  # noqa: ARG001
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "company-news" in url:
                return _FakeResponse([
                    {
                        "headline": "Markets rally on strong earnings beat",
                        "summary": "Stocks surged as companies reported gains above expectations.",
                        "source": "Reuters",
                        "url": "https://example.com/article-1",
                        "datetime": 1710676200,
                    },
                    {
                        "headline": "Tech sector leads broad market rally",
                        "summary": "Optimism about growth drove buying momentum.",
                        "source": "Bloomberg",
                        "url": "https://example.com/article-2",
                        "datetime": 1710672600,
                    },
                ])
            return _FakeResponse({})

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

        payload = get_external_news_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 30, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is True
        assert payload["provider"] == "finnhub"
        assert payload["source_type"] == "news"
        assert payload["score"] is not None
        assert payload["score"] > 50  # bullish headlines should score above neutral
        assert payload["label"] in {
            "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed",
        }
        assert payload["article_count"] >= 1
        assert len(payload["headlines"]) >= 1

    def test_falls_back_to_alpha_vantage(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        monkeypatch.setenv("TRADER_KOO_ALPHA_VANTAGE_KEY", "demo-key")

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=20: _FakeResponse(
                {
                    "feed": [
                        {
                            "title": "Risk appetite improves after cooler inflation data",
                            "source": "Reuters",
                            "url": "https://example.com/article-1",
                            "time_published": "20260317T120000",
                            "overall_sentiment_score": "0.18",
                            "ticker_sentiment": [
                                {
                                    "ticker": "SPY",
                                    "relevance_score": "0.8",
                                    "ticker_sentiment_score": "0.25",
                                }
                            ],
                        },
                    ]
                }
            ),
        )

        payload = get_external_news_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 30, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is True
        assert payload["provider"] == "alpha_vantage"
        assert payload["article_count"] == 1
        assert payload["score"] is not None

from __future__ import annotations

import datetime as dt
import json

from trader_koo.news_sentiment import get_external_news_sentiment


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TestExternalNewsSentiment:
    def test_returns_unavailable_when_alpha_vantage_key_missing(self, monkeypatch):
        monkeypatch.delenv("TRADER_KOO_ALPHA_VANTAGE_KEY", raising=False)
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

        payload = get_external_news_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 0, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is False
        assert payload["provider"] == "alpha_vantage"
        assert payload["score"] is None
        assert "TRADER_KOO_ALPHA_VANTAGE_KEY" in payload["note"]

    def test_aggregates_alpha_vantage_news_sentiment(self, monkeypatch):
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
                        {
                            "title": "Equity traders stay cautious into the Fed decision",
                            "source": "Bloomberg",
                            "url": "https://example.com/article-2",
                            "time_published": "20260317T103000",
                            "overall_sentiment_score": "-0.04",
                            "ticker_sentiment": [
                                {
                                    "ticker": "QQQ",
                                    "relevance_score": "0.7",
                                    "ticker_sentiment_score": "0.05",
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
        assert payload["source_type"] == "news"
        assert payload["article_count"] == 2
        assert payload["score"] is not None
        assert payload["label"] in {
            "Extreme Fear",
            "Fear",
            "Neutral",
            "Greed",
            "Extreme Greed",
        }
        assert len(payload["headlines"]) == 2
        assert payload["headlines"][0]["title"].startswith("Risk appetite improves")

from __future__ import annotations

import datetime as dt

from trader_koo.social_sentiment import get_social_sentiment


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class TestSocialSentiment:
    def test_aggregates_reddit_posts(self, monkeypatch):
        def _fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
            if url.endswith("/top.json"):
                return _FakeResponse(
                    {
                        "data": {
                            "children": [
                                {
                                    "data": {
                                        "title": "Bullish breakout, buying calls into the rally",
                                        "selftext": "Strong upside and support holding.",
                                        "score": 420,
                                        "num_comments": 88,
                                        "permalink": "/r/stocks/comments/demo1/bullish_breakout/",
                                        "created_utc": 1773748800,
                                    }
                                },
                                {
                                    "data": {
                                        "title": "Market looks weak, loading puts for downside",
                                        "selftext": "Bearish breakdown risk is growing.",
                                        "score": 280,
                                        "num_comments": 53,
                                        "permalink": "/r/stocks/comments/demo2/market_looks_weak/",
                                        "created_utc": 1773745200,
                                    }
                                },
                            ]
                        }
                    }
                )
            if "/comments/demo1/" in url:
                return _FakeResponse(
                    [
                        {
                            "data": {
                                "children": [
                                    {
                                        "data": {
                                            "selftext": "Strong upside and support holding.",
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "data": {
                                "children": [
                                    {
                                        "data": {
                                            "body": "Still bullish, breakout looks real.",
                                            "score": 18,
                                        }
                                    }
                                ]
                            }
                        },
                    ]
                )
            if "/comments/demo2/" in url:
                return _FakeResponse(
                    [
                        {
                            "data": {
                                "children": [
                                    {
                                        "data": {
                                            "selftext": "Bearish breakdown risk is growing.",
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "data": {
                                "children": [
                                    {
                                        "data": {
                                            "body": "puts are tempting if support fails",
                                            "score": 12,
                                        }
                                    }
                                ]
                            }
                        },
                    ]
                )
            return _FakeResponse(
                {"data": {"children": []}}
            )

        monkeypatch.setattr("requests.get", _fake_get)
        monkeypatch.setenv("TRADER_KOO_REDDIT_SUBREDDITS", "stocks")
        monkeypatch.setenv("TRADER_KOO_REDDIT_POST_LIMIT", "5")
        monkeypatch.setenv("TRADER_KOO_REDDIT_MIN_SCORE", "10")

        payload = get_social_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 0, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is True
        assert payload["provider"] == "reddit_public_json"
        assert payload["source_type"] == "social"
        assert payload["post_count"] == 2
        assert payload["score"] is not None
        assert len(payload["posts"]) == 2
        assert payload["posts"][0]["title"]
        assert payload["posts"][0]["excerpt"] is not None

    def test_returns_unavailable_when_posts_do_not_pass_filter(self, monkeypatch):
        def _fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
            if url.endswith("/top.json"):
                return _FakeResponse(
                    {
                        "data": {
                            "children": [
                                {
                                    "data": {
                                        "title": "Low engagement thread",
                                        "selftext": "",
                                        "score": 1,
                                        "num_comments": 0,
                                        "permalink": "/r/stocks/comments/demo3/low_engagement/",
                                        "created_utc": 1773745200,
                                    }
                                }
                            ]
                        }
                    }
                )
            if "/comments/demo3/" in url:
                return _FakeResponse(
                    [
                        {"data": {"children": [{"data": {"selftext": ""}}]}},
                        {"data": {"children": []}},
                    ]
                )
            return _FakeResponse(
                {"data": {"children": []}}
            )

        monkeypatch.setattr("requests.get", _fake_get)
        monkeypatch.setenv("TRADER_KOO_REDDIT_SUBREDDITS", "stocks")
        monkeypatch.setenv("TRADER_KOO_REDDIT_MIN_SCORE", "50")

        payload = get_social_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 0, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is False
        assert payload["post_count"] == 0
        assert "No Reddit posts passed" in payload["note"]

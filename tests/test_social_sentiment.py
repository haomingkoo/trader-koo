from __future__ import annotations

import datetime as dt
import json

from trader_koo.social_sentiment import get_social_sentiment


class _FakeResponse:
    def __init__(self, payload: dict | list):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TestSocialSentiment:
    def test_aggregates_stocktwits_messages(self, monkeypatch):
        monkeypatch.setenv("TRADER_KOO_SOCIAL_TICKERS", "SPY")

        def _fake_urlopen(req, timeout=15):  # noqa: ARG001
            return _FakeResponse({
                "symbol": {"symbol": "SPY"},
                "messages": [
                    {
                        "body": "SPY looking strong, buying calls",
                        "created_at": "2026-03-17T12:00:00Z",
                        "entities": {"sentiment": {"basic": "Bullish"}},
                        "likes": {"total": 5},
                        "user": {"username": "trader1"},
                    },
                    {
                        "body": "Markets are topping, puts loaded",
                        "created_at": "2026-03-17T11:30:00Z",
                        "entities": {"sentiment": {"basic": "Bearish"}},
                        "likes": {"total": 3},
                        "user": {"username": "trader2"},
                    },
                    {
                        "body": "Just watching for now",
                        "created_at": "2026-03-17T11:00:00Z",
                        "entities": {"sentiment": None},
                        "likes": {"total": 1},
                        "user": {"username": "trader3"},
                    },
                ],
            })

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

        payload = get_social_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 0, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is True
        assert payload["provider"] == "stocktwits"
        assert payload["source_type"] == "social"
        assert payload["post_count"] == 2  # only 2 have sentiment tags
        assert payload["bullish_terms_total"] == 1
        assert payload["bearish_terms_total"] == 1
        assert payload["score"] == 50  # 1 bullish, 1 bearish = neutral
        assert len(payload["posts"]) == 2

    def test_returns_unavailable_when_no_sentiment_tagged(self, monkeypatch):
        monkeypatch.setenv("TRADER_KOO_SOCIAL_TICKERS", "SPY")

        def _fake_urlopen(req, timeout=15):  # noqa: ARG001
            return _FakeResponse({
                "symbol": {"symbol": "SPY"},
                "messages": [
                    {
                        "body": "Random message without sentiment tag",
                        "entities": {},
                    },
                ],
            })

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

        payload = get_social_sentiment(
            now_utc=dt.datetime(2026, 3, 17, 12, 0, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is False
        assert payload["post_count"] == 0

    def test_handles_stocktwits_api_failure(self, monkeypatch):
        monkeypatch.setenv("TRADER_KOO_SOCIAL_TICKERS", "SPY,QQQ")

        def _fake_urlopen(req, timeout=15):  # noqa: ARG001
            raise Exception("Connection refused")

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

        payload = get_social_sentiment(
            now_utc=dt.datetime(2026, 3, 18, 0, 0, tzinfo=dt.timezone.utc),
            force_refresh=True,
        )

        assert payload["available"] is False
        assert payload["post_count"] == 0
        assert len(payload["source_breakdown"]) == 2
        assert all("Connection refused" in (b.get("note") or "") for b in payload["source_breakdown"])

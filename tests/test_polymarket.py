from __future__ import annotations

import io
import json

from trader_koo.polymarket import (
    _is_trading_relevant_polymarket_event,
    fetch_polymarket_events,
)


def test_polymarket_relevance_uses_source_tags_not_description_words():
    gpt_event = {
        "title": "GPT-6 released by...?",
        "description": "A release could affect the economy and stock market.",
        "tags": [{"slug": "openai"}, {"slug": "ai"}, {"slug": "tech"}],
    }
    fed_event = {
        "title": "How many Fed rate cuts in 2026?",
        "description": "",
        "tags": [{"slug": "fed-rates"}, {"slug": "finance"}],
    }

    assert _is_trading_relevant_polymarket_event(gpt_event) is False
    assert _is_trading_relevant_polymarket_event(fed_event) is True


def test_untagged_polymarket_event_fails_closed():
    assert _is_trading_relevant_polymarket_event({"title": "Bitcoin doubles"}) is False


def test_polymarket_payload_returns_only_active_contracts(monkeypatch):
    event = {
        "id": "event-1",
        "title": "Bitcoin price in 2026?",
        "slug": "bitcoin-price-2026",
        "tags": [{"slug": "crypto-prices"}],
        "markets": [
            {
                "id": "active",
                "question": "Will Bitcoin reach $100,000?",
                "active": True,
                "closed": False,
                "volume24hr": 100,
                "liquidity": 500,
            },
            {
                "id": "resolved",
                "question": "Did Bitcoin reach $80,000?",
                "active": False,
                "closed": True,
                "volume24hr": 0,
            },
        ],
    }
    payload = json.dumps([event]).encode()
    monkeypatch.setattr(
        "trader_koo.polymarket.urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    result = fetch_polymarket_events(limit=15, use_cache=False)

    assert result[0]["market_count"] == 1
    assert result[0]["active_count"] == 1
    assert result[0]["resolved_count"] == 1
    assert [market["market_id"] for market in result[0]["markets"]] == ["active"]


def _ranked_events(count: int) -> list[dict]:
    return [
        {
            "id": f"event-{i}",
            "title": f"Bitcoin above {i}?",
            "slug": f"bitcoin-above-{i}",
            "tags": [{"slug": "crypto-prices"}],
            "markets": [
                {"id": f"m-{i}", "question": f"Bitcoin above {i}?", "active": True,
                 "closed": False, "volume24hr": i, "liquidity": 10},
            ],
        }
        for i in range(count)
    ]


def test_scheduler_fetch_populates_cache_served_to_smaller_limits(monkeypatch):
    from trader_koo import polymarket

    monkeypatch.setattr(polymarket, "_polymarket_cache", {})
    payload = json.dumps(_ranked_events(60)).encode()
    calls: list[int] = []

    def urlopen(*_args, **_kwargs):
        calls.append(1)
        return io.BytesIO(payload)

    monkeypatch.setattr("trader_koo.polymarket.urllib.request.urlopen", urlopen)
    scheduler_events = fetch_polymarket_events(limit=50, use_cache=False)
    fetches = len(calls)

    page_events = fetch_polymarket_events(limit=15)

    assert len(calls) == fetches
    assert polymarket.polymarket_fetch_status()["cache_hit"] is True
    assert len(scheduler_events) == 50
    assert page_events == scheduler_events[:15]
    assert [e["slug"] for e in page_events] == [f"bitcoin-above-{i}" for i in range(59, 44, -1)]


def test_polymarket_route_bounds_limit():
    from unittest.mock import patch

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from trader_koo.backend.routers import system

    app = FastAPI()
    app.include_router(system.router)
    client = TestClient(app)
    with patch("trader_koo.polymarket.fetch_polymarket_events", return_value=[]) as fetch:
        assert client.get("/api/polymarket?limit=0").status_code == 422
        assert client.get("/api/polymarket?limit=51").status_code == 422
        assert client.get("/api/polymarket?limit=50").status_code == 200

    fetch.assert_called_once_with(limit=50)

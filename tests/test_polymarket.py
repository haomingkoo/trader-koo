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

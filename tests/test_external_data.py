from __future__ import annotations

from trader_koo.ml.external_data import (
    _is_trading_relevant_polymarket_event,
    _redact_url_secrets,
)


def test_redact_url_secrets_hides_fred_api_key():
    text = (
        "500 Server Error for url: "
        "https://api.stlouisfed.org/fred/series/observations?"
        "series_id=T10Y2Y&api_key=secret-123&file_type=json"
    )

    redacted = _redact_url_secrets(text)

    assert "secret-123" not in redacted
    assert "api_key=<redacted>" in redacted
    assert "series_id=T10Y2Y" in redacted


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

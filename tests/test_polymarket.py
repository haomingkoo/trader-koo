from __future__ import annotations

import gzip
import http.server
import io
import json
import threading
import urllib.parse
from collections.abc import Iterator

import pytest

from trader_koo.polymarket import (
    _is_trading_relevant_polymarket_event,
    fetch_polymarket_events,
)


class _IdentityResponse(io.BytesIO):
    """An uncompressed urlopen response: a body plus the headers urllib exposes."""

    headers: dict[str, str] = {}


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
        lambda *_args, **_kwargs: _IdentityResponse(payload),
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
        return _IdentityResponse(payload)

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


def _event(i: int) -> dict:
    return {
        "id": f"event-{i}",
        "title": f"Fed decision {i}?",
        "slug": f"fed-decision-{i}",
        "tags": [{"slug": "fed-rates"}],
        "markets": [
            {"id": f"m-{i}", "question": f"Fed cut {i}?", "active": True,
             "closed": False, "volume24hr": 1000 + i, "liquidity": 10},
        ],
    }


class _GammaLikeHandler(http.server.BaseHTTPRequestHandler):
    """Reproduces Gamma's 2026-11-12 contract for /events.

    limit above 50 without Accept-Encoding: gzip is refused with 406
    gzip_required; otherwise the page is served gzip-compressed. Offset 0
    returns a full page, offset 100 a short final page.
    """

    seen: list[dict[str, str]] = []

    def do_GET(self) -> None:  # noqa: N802
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.path).query))
        type(self).seen.append({**query, "accept_encoding": self.headers.get("Accept-Encoding", "")})
        limit, offset = int(query["limit"]), int(query["offset"])
        if limit > 50 and "gzip" not in self.headers.get("Accept-Encoding", ""):
            body = json.dumps({"error": "gzip_required"}).encode()
            self.send_response(406)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        count = limit if offset == 0 else 30
        body = gzip.compress(json.dumps([_event(offset + i) for i in range(count)]).encode())
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


@pytest.fixture()
def gamma_server(monkeypatch) -> Iterator[type[_GammaLikeHandler]]:
    from trader_koo import polymarket

    _GammaLikeHandler.seen = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _GammaLikeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(polymarket, "_POLYMARKET_GAMMA", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(polymarket, "_polymarket_cache", {})
    try:
        yield _GammaLikeHandler
    finally:
        server.shutdown()
        server.server_close()


def test_gzip_only_events_pages_are_fetched_and_paged(gamma_server):
    from trader_koo import polymarket

    events = fetch_polymarket_events(limit=50, use_cache=False)

    status = polymarket.polymarket_fetch_status()
    assert status["ok"] is True, status
    assert status["error"] is None
    assert [(r["limit"], r["offset"]) for r in gamma_server.seen] == [("100", "0"), ("100", "100")]
    assert all("gzip" in r["accept_encoding"] for r in gamma_server.seen)
    # 130 distinct events across both pages, ranked by 24h volume: the top
    # entry comes from the second page, so paging past offset 0 happened.
    assert len(events) == 50
    assert events[0]["slug"] == "fed-decision-129"


def test_unknown_content_encoding_fails_visibly(monkeypatch):
    from trader_koo import polymarket

    class _BrotliResponse(io.BytesIO):
        headers = {"Content-Encoding": "br"}

    monkeypatch.setattr(polymarket, "_polymarket_cache", {})
    monkeypatch.setattr(
        "trader_koo.polymarket.urllib.request.urlopen",
        lambda *_args, **_kwargs: _BrotliResponse(b"\x00\x01"),
    )

    assert fetch_polymarket_events(limit=15, use_cache=False) == []
    status = polymarket.polymarket_fetch_status()
    assert status["ok"] is False
    assert status["error"] == "ValueError"

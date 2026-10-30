"""Write routes outside /api/admin still require the admin key."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers import paper_trades as paper_trades_module
from trader_koo.backend.routers import streaming as streaming_module
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator
from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.streaming.finnhub_ws import MAX_SUBSCRIPTIONS, FinnhubWSClient
from tests.test_routers.test_paper_trades_router import _publish_report_fixture

KEY = "k" * 32
ADMIN = {"X-API-Key": KEY}


def _app(*routers) -> TestClient:
    app = FastAPI()
    app.state.admin_authenticator = AdminAuthenticator(AdminAuthConfig(api_key=KEY))
    for router in routers:
        app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def notes_client(tmp_path, monkeypatch):
    db_path = tmp_path / "notes.db"
    conn = sqlite3.connect(db_path)
    ensure_paper_trade_schema(conn)
    # The shared report fixture pins its run to 2026-08-21; the trade must follow it.
    _publish_report_fixture(
        conn, tmp_path, run_id="notes-run", generated_ts="2026-08-21T11:00:00Z",
    )
    conn.execute(
        """INSERT INTO paper_trades
           (id, report_date, ticker, direction, entry_price, entry_date, status,
            current_price, generated_ts, report_run_id, campaign_id)
           VALUES (7, '2026-08-21', 'REJECT', 'long', 100, '2026-08-21', 'open', 101,
                   '2026-08-21T12:00:00Z', 'notes-run', 'paper-v2')"""
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(paper_trades_module, "get_conn", lambda: sqlite3.connect(db_path))

    def stored_notes():
        with sqlite3.connect(db_path) as check:
            row = check.execute(
                "SELECT notes FROM paper_trade_annotations WHERE trade_id = 7"
            ).fetchone()
        return row[0] if row else None

    return _app(paper_trades_module.router), stored_notes


def test_trade_notes_reject_anonymous_writes(notes_client):
    client, stored_notes = notes_client
    response = client.patch("/api/paper-trades/7/notes", json={"notes": "defaced"})
    assert response.status_code == 401
    assert stored_notes() is None

    wrong = client.patch(
        "/api/paper-trades/7/notes", json={"notes": "defaced"},
        headers={"X-API-Key": "wrong"},
    )
    assert wrong.status_code == 401
    assert stored_notes() is None


def test_trade_notes_accept_admin_key(notes_client):
    client, stored_notes = notes_client
    response = client.patch(
        "/api/paper-trades/7/notes", json={"notes": "breakout held"}, headers=ADMIN,
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "trade_id": 7, "notes": "breakout held"}
    assert stored_notes() == "breakout held"


@pytest.fixture()
def streaming_calls(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_subscribe(sym):
        calls.append(("subscribe", sym))
        return True

    def fake_unsubscribe(sym):
        calls.append(("unsubscribe", sym))
        return True

    monkeypatch.setattr(streaming_module, "subscribe_symbol", fake_subscribe)
    monkeypatch.setattr(streaming_module, "unsubscribe_symbol", fake_unsubscribe)
    monkeypatch.setattr(streaming_module, "get_equity_price", lambda sym: None)
    monkeypatch.setattr(
        streaming_module, "get_subscription_info",
        lambda: {"connected": True, "subscribed_count": 3, "max_symbols": 50, "symbols": []},
    )
    return calls


@pytest.mark.parametrize("action", ["subscribe", "unsubscribe"])
def test_streaming_subscription_routes_reject_anonymous_calls(streaming_calls, action):
    client = _app(streaming_module.router)
    response = client.post(f"/api/streaming/{action}/nvda")
    assert response.status_code == 401
    assert streaming_calls == []


@pytest.mark.parametrize("action", ["subscribe", "unsubscribe"])
def test_streaming_subscription_routes_accept_admin_key(streaming_calls, action):
    client = _app(streaming_module.router)
    response = client.post(f"/api/streaming/{action}/nvda", headers=ADMIN)
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert streaming_calls == [(action, "NVDA")]


def test_public_price_route_subscribes_server_side(streaming_calls):
    client = _app(streaming_module.router)
    response = client.get("/api/streaming/price/nvda")
    assert response.status_code == 200
    assert response.json()["subscribed"] is True
    assert streaming_calls == [("subscribe", "NVDA")]


def test_full_feed_evicts_least_recently_requested_on_demand_symbol():
    client = FinnhubWSClient(api_key="unused", always_on=["SPY", "QQQ"])
    client._subscribed.update({"SPY", "QQQ"})
    on_demand = [f"T{i:02d}" for i in range(MAX_SUBSCRIPTIONS - 2)]
    for sym in on_demand:
        assert client.subscribe(sym)
    assert client.get_subscription_count() == MAX_SUBSCRIPTIONS

    assert client.subscribe("T00")  # viewed again, now the most recent
    assert client.subscribe("NVDA")

    symbols = set(client.get_subscribed_symbols())
    assert len(symbols) == MAX_SUBSCRIPTIONS
    assert "NVDA" in symbols
    assert "T00" in symbols
    assert "T01" not in symbols
    assert {"SPY", "QQQ"} <= symbols


def test_always_on_symbols_are_never_evicted():
    always_on = [f"A{i:02d}" for i in range(MAX_SUBSCRIPTIONS)]
    client = FinnhubWSClient(api_key="unused", always_on=always_on)
    client._subscribed.update(always_on)
    assert client.subscribe("NVDA") is False
    assert set(client.get_subscribed_symbols()) == set(always_on)

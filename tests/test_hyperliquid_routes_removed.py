"""The unauthenticated Hyperliquid write routes stay removed."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.hyperliquid.routes import router


def test_unauthenticated_poll_and_collect_routes_are_gone():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    assert client.post("/api/hyperliquid/poll").status_code in {404, 405}
    assert client.post("/api/hyperliquid/collect/machibro").status_code in {404, 405}

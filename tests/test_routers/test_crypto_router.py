from __future__ import annotations

import datetime as dt

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers.crypto import router
from trader_koo.crypto.models import CryptoBar


def _bars(count: int = 120) -> list[CryptoBar]:
    base = dt.datetime(2026, 3, 17, 0, 0, tzinfo=dt.timezone.utc)
    bars: list[CryptoBar] = []
    for idx in range(count):
        close = 3000.0 + idx * 0.8 + ((idx % 9) - 4) * 3.0
        bars.append(
            CryptoBar(
                symbol="ETH-USD",
                timestamp=base + dt.timedelta(minutes=idx),
                interval="1m",
                open=close - 2.0,
                high=close + 3.0,
                low=close - 4.0,
                close=close,
                volume=500.0 + idx,
            )
        )
    return bars


def test_crypto_structure_endpoint_returns_payload(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    bars = _bars()

    monkeypatch.setattr(
        "trader_koo.backend.routers.crypto.get_crypto_history",
        lambda symbol, interval="1m", limit=100: bars[:limit],
    )

    with TestClient(app) as client:
        response = client.get("/api/crypto/structure/ETH-USD?interval=1m&limit=120")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["symbol"] == "ETH-USD"
    assert "levels" in data
    assert "context" in data

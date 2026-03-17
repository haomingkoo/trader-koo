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


def test_crypto_correlation_endpoint_returns_payload(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    base = dt.datetime(2026, 2, 10, 0, 0, tzinfo=dt.timezone.utc)
    bars = [
        CryptoBar(
            symbol="BTC-USD",
            timestamp=base + dt.timedelta(days=idx),
            interval="1d",
            open=90000.0 + idx * 800.0,
            high=90400.0 + idx * 800.0,
            low=89600.0 + idx * 800.0,
            close=90200.0 + idx * 800.0,
            volume=1000.0 + idx,
        )
        for idx in range(40)
    ]

    monkeypatch.setattr(
        "trader_koo.backend.routers.crypto.get_crypto_history",
        lambda symbol, interval="1d", limit=40: bars[:limit],
    )

    class _FakeConn:
        def execute(self, query, params=()):
            ticker = params[0] if params else "SPY"
            if ticker != "SPY":
                return []
            rows = []
            base = dt.date(2026, 3, 17)
            for idx in range(30):
                date_str = (base - dt.timedelta(days=idx)).isoformat()
                rows.append((date_str, 560.0 + idx * 1.5))

            class _Cursor:
                def fetchall(self_nonlocal):
                    return list(reversed(rows))

            return _Cursor()

        def close(self):
            return None

    monkeypatch.setattr(
        "trader_koo.backend.routers.crypto.get_conn",
        lambda: _FakeConn(),
    )

    with TestClient(app) as client:
        response = client.get("/api/crypto/correlation/BTC-USD?benchmark=SPY&limit=30")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["symbol"] == "BTC-USD"
    assert data["benchmark"] == "SPY"
    assert "windows" in data
    assert "20d" in data["windows"]


def test_crypto_market_structure_endpoint_returns_payload(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    monkeypatch.setattr(
        "trader_koo.backend.routers.crypto.get_crypto_history",
        lambda symbol, interval="1h", limit=240: _bars(count=min(limit, 80)),
    )
    monkeypatch.setattr(
        "trader_koo.backend.routers.crypto.get_crypto_summary",
        lambda: {
            "prices": {
                "BTC-USD": {"symbol": "BTC-USD", "price": 93000.0, "volume_24h": 1.0, "change_pct_24h": 2.0},
                "ETH-USD": {"symbol": "ETH-USD", "price": 3400.0, "volume_24h": 1.0, "change_pct_24h": -1.0},
                "SOL-USD": {"symbol": "SOL-USD", "price": 190.0, "volume_24h": 1.0, "change_pct_24h": 0.5},
                "XRP-USD": {"symbol": "XRP-USD", "price": 2.5, "volume_24h": 1.0, "change_pct_24h": 0.2},
                "DOGE-USD": {"symbol": "DOGE-USD", "price": 0.3, "volume_24h": 1.0, "change_pct_24h": -0.4},
            }
        },
    )

    with TestClient(app) as client:
        response = client.get("/api/crypto/market-structure?interval=1h&limit=80")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "overview" in data
    assert "leaders" in data
    assert isinstance(data["symbols"], list)

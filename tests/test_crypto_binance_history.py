from __future__ import annotations

import datetime as dt

from trader_koo.crypto.binance_history import fetch_recent_klines


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_fetch_recent_klines_parses_payload(monkeypatch):
    base = dt.datetime(2026, 3, 17, 0, 0, tzinfo=dt.timezone.utc)
    rows = []
    for idx in range(3):
        ts_ms = int((base + dt.timedelta(days=idx)).timestamp() * 1000)
        rows.append([ts_ms, "100.0", "110.0", "95.0", "105.0", "1234.5"])

    monkeypatch.setattr(
        "trader_koo.crypto.binance_history.requests.get",
        lambda *args, **kwargs: _FakeResponse(rows),
    )

    bars = fetch_recent_klines("BTC-USD", "1w", 3)

    assert len(bars) == 3
    assert bars[0].symbol == "BTC-USD"
    assert bars[0].interval == "1w"
    assert bars[0].timestamp == base
    assert bars[-1].close == 105.0


def test_fetch_recent_klines_ignores_unknown_symbol():
    assert fetch_recent_klines("ABC-USD", "1d", 10) == []

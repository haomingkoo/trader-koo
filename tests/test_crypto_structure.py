from __future__ import annotations

import datetime as dt
import math

from trader_koo.crypto.models import CryptoBar
from trader_koo.crypto.structure import build_crypto_structure


def _sample_bars(count: int = 180) -> list[CryptoBar]:
    bars: list[CryptoBar] = []
    base = dt.datetime(2026, 3, 10, 0, 0, tzinfo=dt.timezone.utc)
    for idx in range(count):
        wave = math.sin(idx / 9.0) * 18.0
        close = 50000.0 + wave + idx * 1.8
        open_ = close - math.cos(idx / 7.0) * 4.0
        high = max(open_, close) + 6.0
        low = min(open_, close) - 6.0
        bars.append(
            CryptoBar(
                symbol="BTC-USD",
                timestamp=base + dt.timedelta(minutes=idx),
                interval="1m",
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + idx * 3.0,
            )
        )
    return bars


def test_build_crypto_structure_returns_levels_and_context():
    payload = build_crypto_structure("BTC-USD", _sample_bars(), interval="1m")

    assert payload["symbol"] == "BTC-USD"
    assert payload["bar_count"] >= 100
    assert isinstance(payload["levels"], list)
    assert len(payload["levels"]) > 0
    assert "context" in payload
    assert payload["context"]["latest_close"] is not None
    assert payload["context"]["level_context"] in {
        "below_support",
        "above_resistance",
        "at_support",
        "at_resistance",
        "closer_support",
        "closer_resistance",
        "mid_range",
    }

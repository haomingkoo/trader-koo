from __future__ import annotations

import datetime as dt
import math
import warnings

import pandas as pd

from trader_koo.structure.hmm_regime import predict_regimes


def _price_frame(count: int = 180) -> pd.DataFrame:
    rows = []
    start = dt.date(2025, 1, 1)
    for idx in range(count):
        close = 100.0 + idx * 0.15 + math.sin(idx / 8.0) * 1.8
        open_ = close - math.cos(idx / 6.0) * 0.6
        rows.append(
            {
                "date": (start + dt.timedelta(days=idx)).isoformat(),
                "open": open_,
                "high": max(open_, close) + 0.9,
                "low": min(open_, close) - 0.9,
                "close": close,
                "volume": 1_000_000 + idx * 5_000,
            }
        )
    return pd.DataFrame(rows)


def test_predict_regimes_avoids_runtime_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = predict_regimes(_price_frame(), ticker="SPY")

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert result is not None
    assert result["current_state"] in {"low_vol", "normal", "high_vol"}

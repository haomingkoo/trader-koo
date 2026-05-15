from __future__ import annotations

import sqlite3

from trader_koo.ml import features as feature_mod
from trader_koo.ml.features import _get_news_sentiment_scores, extract_features_for_universe


def _seed_prices(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE price_daily (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        """
    )
    rows = []
    for i in range(1, 31):
        date = f"2025-01-{i:02d}"
        close = 100.0 + i
        rows.append(("AAPL", date, close - 0.5, close + 1.0, close - 1.0, close, 1_000_000 + i))
        rows.append(("SPY", date, close - 0.5, close + 1.0, close - 1.0, close, 2_000_000 + i))
        rows.append(("^VIX", date, 20.0, 21.0, 19.0, 20.0 + i / 10.0, 0))
    conn.executemany(
        "INSERT INTO price_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def test_historical_news_sentiment_does_not_call_live_api_without_cache():
    conn = sqlite3.connect(":memory:")

    scores = _get_news_sentiment_scores(["AAPL"], "2025-01-15", conn=conn)

    assert scores == {}


def test_extract_features_skips_news_when_not_requested(monkeypatch):
    conn = sqlite3.connect(":memory:")
    _seed_prices(conn)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("news sentiment should not be computed")

    monkeypatch.setattr(feature_mod, "_get_news_sentiment_scores", fail_if_called)

    out = extract_features_for_universe(
        conn,
        as_of_date="2025-01-30",
        feature_columns=["ret_1d"],
        strict=True,
    )

    assert list(out.columns) == ["ret_1d"]
    assert "AAPL" in out.index
    assert "SPY" not in out.index
    assert "^VIX" not in out.index


def test_extract_features_allows_context_tickers_when_explicit():
    conn = sqlite3.connect(":memory:")
    _seed_prices(conn)

    out = extract_features_for_universe(
        conn,
        as_of_date="2025-01-30",
        tickers=["SPY", "^VIX"],
        feature_columns=["ret_1d"],
        strict=True,
    )

    assert "SPY" in out.index
    assert "^VIX" in out.index

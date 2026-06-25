from __future__ import annotations

import datetime as dt
import math
import sqlite3

import pandas as pd

from trader_koo.ml.rule_baseline import (
    RULE_BASELINE_FEATURE_COLUMNS,
    run_rule_baseline,
    score_rule_candidates,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
        """
    )
    return conn


def _insert_prices(conn: sqlite3.Connection, ticker: str, closes: list[float]) -> list[str]:
    start = dt.date(2025, 1, 1)
    dates: list[str] = []
    for idx, close in enumerate(closes):
        day = start + dt.timedelta(days=idx)
        date_s = day.isoformat()
        dates.append(date_s)
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                date_s,
                close * 0.998,
                close * 1.012,
                close * 0.988,
                close,
                1_000_000,
            ),
        )
    conn.commit()
    return dates


def _features_frame() -> pd.DataFrame:
    rows = {
        "AAA": {
            "ret_5d": -0.02,
            "ret_21d": -0.04,
            "ret_63d": 0.06,
            "atr_pct_14": 3.5,
            "dist_ma50_pct": -4.0,
            "xrank_dist_ma50_pct": 0.20,
            "xrank_mean_reversion_5d": 0.15,
            "xrank_atr_pct_14": 0.50,
            "xrank_ret_21d": 0.45,
            "xrank_vol_21d": 0.50,
            "xrank_volume_ratio_20d": 0.55,
        },
        "BBB": {
            "ret_5d": -0.02,
            "ret_21d": -0.07,
            "ret_63d": -0.08,
            "atr_pct_14": 4.0,
            "dist_ma50_pct": -6.0,
            "xrank_dist_ma50_pct": 0.20,
            "xrank_mean_reversion_5d": 0.30,
            "xrank_atr_pct_14": 0.55,
            "xrank_ret_21d": 0.20,
            "xrank_vol_21d": 0.52,
            "xrank_volume_ratio_20d": 0.60,
        },
    }
    return pd.DataFrame.from_dict(rows, orient="index").reindex(
        columns=RULE_BASELINE_FEATURE_COLUMNS
    )


def test_score_rule_candidates_ranks_current_edge_families():
    scored = score_rule_candidates(_features_frame(), min_score=68.0)

    assert list(scored["ticker"]) == ["AAA", "BBB"]
    assert scored.loc[0, "setup_family"] == "bullish_reversal"
    assert scored.loc[0, "direction"] == "long"
    assert scored.loc[1, "setup_family"] == "bearish_continuation"
    assert scored.loc[1, "direction"] == "short"


def test_run_rule_baseline_produces_registry_ready_summary(monkeypatch):
    conn = _conn()
    dates = _insert_prices(conn, "SPY", [100 + i * 0.08 for i in range(70)])
    _insert_prices(conn, "AAA", [50 + i * 0.35 + math.sin(i) * 0.8 for i in range(70)])
    _insert_prices(conn, "BBB", [90 - i * 0.28 + math.sin(i) * 0.9 for i in range(70)])

    def fake_extract_features(*args, **kwargs):
        return _features_frame()

    monkeypatch.setattr(
        "trader_koo.ml.rule_baseline.extract_features_for_universe",
        fake_extract_features,
    )

    result = run_rule_baseline(
        conn,
        start_date=dates[25],
        end_date=dates[62],
        rebalance_frequency=5,
        max_positions=2,
        position_pct=20.0,
        min_score=68.0,
    )

    assert result["ok"] is True
    assert result["return_pct"] == result["summary"]["return_pct"]
    assert result["summary"]["method"] == "current_rule_technical_proxy"
    assert result["summary"]["total_trades"] > 0
    assert "spy_return_pct" in result["summary"]

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from trader_koo.analysis.green_barrier import compute_williams_percent_r, resample_ohlcv
from trader_koo.backend.services.market_data import get_data_sources
from trader_koo.db.sources import DataSourceManager
from trader_koo.scripts.update_market_db import ensure_schema, write_price_daily
from trader_koo.db.price_contract import research_price_contract
from trader_koo.ml.benchmark import run_benchmark
from trader_koo.ml.trainer import build_dataset
from trader_koo.report import generator as report_generator


def _raw_frame(
    continuous: pd.DataFrame,
    *,
    action_date: str,
    factor: float,
    dividend: float = 0.0,
) -> pd.DataFrame:
    raw = continuous.copy()
    before = raw.index < pd.Timestamp(action_date)
    for column in ("Open", "High", "Low", "Close"):
        raw.loc[before, column] *= factor
    raw.loc[before, "Volume"] /= factor
    raw["Stock Splits"] = 0.0
    raw.loc[pd.Timestamp(action_date), "Stock Splits"] = factor
    raw["Dividends"] = 0.0
    raw.loc[pd.Timestamp(action_date), "Dividends"] = dividend
    return raw


def _continuous_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", "2026-06-30", name="Date")
    close = 100 + np.linspace(0, 35, len(dates)) + np.sin(np.arange(len(dates)) / 9) * 4
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(len(dates), 1_000_000.0),
        },
        index=dates,
    )


@pytest.mark.parametrize(
    ("ticker", "factor", "action_type"),
    [("BKNG", 20.0, "split"), ("SNDK", 0.2, "reverse_split")],
)
def test_declared_action_keeps_all_williams_timeframes_on_one_basis(
    ticker: str,
    factor: float,
    action_type: str,
) -> None:
    continuous = _continuous_prices()
    raw = _raw_frame(
        continuous,
        action_date="2025-07-01",
        factor=factor,
        dividend=0.42,
    )
    normalized = DataSourceManager._normalize_ohlcv(raw)
    expected = DataSourceManager._normalize_ohlcv(continuous)

    assert normalized.attrs["basis_status"] == "verified"
    assert {action["action_type"] for action in normalized.attrs["corporate_actions"]} == {
        action_type,
        "dividend",
    }
    for timeframe in ("daily", "weekly", "monthly"):
        actual_bars = (
            normalized
            if timeframe == "daily"
            else resample_ohlcv(
                normalized,
                timeframe,
                completed_only=True,
                as_of=pd.Timestamp("2026-06-30").date(),
            )
        )
        expected_bars = (
            expected
            if timeframe == "daily"
            else resample_ohlcv(
                expected,
                timeframe,
                completed_only=True,
                as_of=pd.Timestamp("2026-06-30").date(),
            )
        )
        assert compute_williams_percent_r(actual_bars).iloc[-1] == pytest.approx(
            compute_williams_percent_r(expected_bars).iloc[-1]
        )


def test_genuine_crash_is_preserved_but_fails_research_closed() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-08-20", "2026-08-21"]),
            "Open": [100.0, 44.0],
            "High": [102.0, 46.0],
            "Low": [98.0, 40.0],
            "Close": [100.0, 44.0],
            "Volume": [1000.0, 4000.0],
        }
    ).set_index("Date")
    normalized = DataSourceManager._normalize_ohlcv(frame)

    assert normalized["close"].tolist() == [100.0, 44.0]
    assert normalized.attrs["corporate_actions"] == []
    assert normalized.attrs["basis_status"] == "unresolved"
    assert normalized.attrs["unresolved_discontinuities"][0]["reason"] == (
        "unexplained_adjacent_price_discontinuity"
    )
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    write_price_daily(conn, "CRASH", normalized)
    assert get_data_sources(conn, "CRASH")["research_eligible"] is False


def test_action_persistence_is_idempotent_and_visible_in_chart_contract() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    normalized = DataSourceManager._normalize_ohlcv(
        _raw_frame(_continuous_prices(), action_date="2025-07-01", factor=20.0)
    )

    write_price_daily(conn, "BKNG", normalized, fetch_timestamp="2026-08-22T00:00:00Z")
    write_price_daily(conn, "BKNG", normalized, fetch_timestamp="2026-08-22T00:00:00Z")
    conn.commit()

    assert conn.execute(
        "SELECT COUNT(*) FROM price_corporate_actions WHERE ticker = 'BKNG'"
    ).fetchone()[0] == 1
    contract = get_data_sources(conn, "BKNG")
    assert contract["adjustment_basis"] == "split_adjusted_price_only"
    assert contract["adjustment_version"] == "yfinance-actions-v1"
    assert contract["basis_status"] == "verified"
    assert contract["research_eligible"] is True
    assert contract["corporate_actions"][0]["action_type"] == "split"


def test_mixing_price_bases_marks_whole_ticker_unresolved() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    prices = DataSourceManager._normalize_ohlcv(_continuous_prices().iloc[:30])
    write_price_daily(conn, "SNDK", prices)
    incompatible = prices.tail(2).copy()
    incompatible.attrs.update(prices.attrs)
    incompatible.attrs["adjustment_basis"] = "total_return"
    write_price_daily(conn, "SNDK", incompatible)

    statuses = conn.execute(
        "SELECT DISTINCT basis_status FROM price_daily WHERE ticker = 'SNDK'"
    ).fetchall()
    assert statuses == [("unresolved",)]
    assert get_data_sources(conn, "SNDK")["research_eligible"] is False


def test_incremental_batch_boundary_discontinuity_fails_closed() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)

    for date, close in (("2026-08-20", 100.0), ("2026-08-21", 40.0)):
        frame = pd.DataFrame(
            [{"date": date, "open": close, "high": close, "low": close, "close": close, "volume": 1000.0}]
        )
        frame.attrs.update(
            adjustment_basis="split_adjusted_price_only",
            adjustment_version="yfinance-actions-v1",
            basis_status="verified",
            unresolved_discontinuities=[],
            corporate_actions=[],
        )
        write_price_daily(conn, "CRASH", frame)

    rows = conn.execute(
        "SELECT close, basis_status, unresolved_reason FROM price_daily WHERE ticker='CRASH' ORDER BY date"
    ).fetchall()
    assert [row[0] for row in rows] == [100.0, 40.0]
    assert {row[1] for row in rows} == {"unresolved"}
    assert "unexplained_adjacent_price_discontinuity" in rows[0][2]
    assert research_price_contract(conn, ["CRASH"])["eligible"] is False


def test_narrow_refetch_does_not_erase_applied_split_provenance() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    full = DataSourceManager._normalize_ohlcv(
        _raw_frame(_continuous_prices(), action_date="2025-07-01", factor=20.0)
    )
    write_price_daily(conn, "BKNG", full, fetch_timestamp="2026-08-20T00:00:00Z")

    raw_post_only = _raw_frame(
        _continuous_prices(), action_date="2025-07-01", factor=20.0
    ).loc["2025-07-01":]
    narrow = DataSourceManager._normalize_ohlcv(raw_post_only)
    assert narrow.attrs["corporate_actions"][0]["applied_to_prices"] is False
    write_price_daily(conn, "BKNG", narrow, fetch_timestamp="2026-08-21T00:00:00Z")

    action = conn.execute(
        """SELECT applied_to_prices, fetch_timestamp, evidence_json
        FROM price_corporate_actions WHERE ticker='BKNG' AND action_type='split'"""
    ).fetchone()
    assert action[0] == 1
    assert action[1] == "2026-08-20T00:00:00Z"
    assert '"applied_to_prices": true' in action[2]


def test_requested_missing_ticker_fails_contract_closed() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    prices = DataSourceManager._normalize_ohlcv(_continuous_prices().iloc[:5])
    write_price_daily(conn, "SPY", prices)

    contract = research_price_contract(conn, ["SPY", "MISSING"])

    assert contract["eligible"] is False
    assert contract["reason"] == "missing_requested_tickers"
    assert contract["missing_tickers"] == ["MISSING"]


def test_report_and_ml_consumers_fail_closed_for_unresolved_basis() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    prices = DataSourceManager._normalize_ohlcv(_continuous_prices().iloc[:30])
    prices.attrs["basis_status"] = "unresolved"
    prices.attrs["unresolved_discontinuities"] = [
        {"reason": "unexplained_adjacent_price_discontinuity"}
    ]
    write_price_daily(conn, "SPY", prices)

    report_generator._report_warnings = []
    signals = report_generator.fetch_signals(conn)
    assert signals["setup_quality_top"] == []
    assert "hmm_regime_by_ticker" not in signals
    assert signals["price_contract"]["eligible"] is False
    assert "price_basis_unresolved" in report_generator._report_warnings

    with pytest.raises(ValueError, match="not research eligible"):
        build_dataset(
            conn,
            start_date="2024-01-01",
            end_date="2026-01-01",
        )
    benchmark = run_benchmark(conn, start_date="2024-01-01")
    assert benchmark["ok"] is False
    assert benchmark["price_contract"]["eligible"] is False

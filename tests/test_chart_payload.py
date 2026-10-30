"""Chart payload compaction and the separate HMM regime endpoint."""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from trader_koo.backend.services import chart_builder, chart_regime
from trader_koo.backend.services.chart_compact import (
    CHART_ROW_KEYS,
    HMM_CHART_DAYS,
    compact_chart_rows,
    compact_hmm_regime,
)
from trader_koo.db.price_contract import record_price_series_revision

from tests.conftest import _create_test_db

BARS = 320


def _make_price_db(path: Path, ticker: str = "ACME", bars: int = BARS) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    schema = _create_test_db()  # the shared test schema, copied to a file DB
    schema.backup(conn)
    schema.close()
    end = dt.date.today() - dt.timedelta(days=1)
    dates = pd.bdate_range(end=end, periods=bars)
    rows = []
    for i, d in enumerate(dates):
        # float32-style noise like yfinance rows (e.g. 86.12999725341797)
        close = 100.0 + 10.0 * math.sin(i / 9.0) + i * 0.05 + 0.0000123
        rows.append((ticker, d.date().isoformat(), close - 0.4, close + 1.1, close - 1.3, close, 1_000_000.0 + i))
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    _record_revision(conn, ticker)
    return conn


def _record_revision(conn: sqlite3.Connection, ticker: str) -> None:
    now_ts = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    record_price_series_revision(
        conn,
        ticker,
        evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
        fetch_timestamp=now_ts,
    )
    conn.commit()


def _fake_regime(dates: list[str]) -> dict:
    return {
        "regimes": [
            {"date": d, "state": 0, "label": "low_vol", "color": "#38d39f",
             "prob_low": 0.9, "prob_normal": 0.07, "prob_high": 0.03}
            for d in dates
        ],
        "current_state": "low_vol",
        "current_probs": {"low_vol": 0.9, "normal": 0.07, "high_vol": 0.03},
        "transition_risk_pct": 1.2,
        "transition_matrix": [[0.99, 0.01, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]],
        "days_in_current": 12,
    }


@pytest.fixture(autouse=True)
def _clear_caches():
    chart_builder._PREPARED_CACHE.clear()
    chart_regime._REGIME_CACHE.clear()
    yield
    chart_builder._PREPARED_CACHE.clear()
    chart_regime._REGIME_CACHE.clear()


def test_compact_chart_rows_rounds_prices_and_drops_unused_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05"],
            "open": [86.12999725341797, 87.0],
            "high": [87.5, 88.00000762939453],
            "low": [85.0, 86.0],
            "close": [86.12999725341797, 87.91000366210938],
            "volume": [102223600.0, float("nan")],
            "ma20": [float("nan"), 86.5],
            "ema200": [float("nan"), 80.0],
        }
    )
    rows = compact_chart_rows(frame)
    assert rows == [
        {"date": "2026-01-02", "open": 86.13, "high": 87.5, "low": 85.0, "close": 86.13, "volume": 102223600},
        {"date": "2026-01-05", "open": 87.0, "high": 88.0, "low": 86.0, "close": 87.91, "volume": None},
    ]
    assert isinstance(rows[0]["volume"], int)
    assert compact_chart_rows(pd.DataFrame()) == []


def test_quick_payload_chart_is_compact_and_keeps_full_history(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    with patch.object(chart_builder, "get_ticker_earnings_markers", return_value=[]):
        payload = chart_builder.build_dashboard_quick_payload(conn, "ACME", 0)
    chart = payload["chart"]
    assert len(chart) == BARS
    assert all(set(row) == set(CHART_ROW_KEYS) for row in chart)
    assert all(row["close"] == round(row["close"], 4) for row in chart)
    assert '""' not in json.dumps(chart)
    # Levels and patterns still come from the full-history model.
    assert payload["levels"]
    assert payload["asof"] == chart[-1]["date"]


def test_compact_hmm_regime_keeps_plotted_fields_for_recent_window() -> None:
    dates = [f"d{i:04d}" for i in range(HMM_CHART_DAYS + 50)]
    original = _fake_regime(dates)
    snapshot = json.dumps(original, sort_keys=True)
    trimmed = compact_hmm_regime(original)
    assert json.dumps(original, sort_keys=True) == snapshot  # input untouched
    assert len(trimmed["regimes"]) == HMM_CHART_DAYS
    assert trimmed["regimes"][-1] == {"date": dates[-1], "prob_low": 0.9, "prob_normal": 0.07, "prob_high": 0.03}
    assert trimmed["regimes"][0]["date"] == dates[50]
    for key in ("current_state", "current_probs", "transition_risk_pct", "transition_matrix", "days_in_current"):
        assert trimmed[key] == original[key]
    assert compact_hmm_regime(None) is None


def test_commentary_payload_does_not_fit_hmm(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    with patch.object(
        chart_builder, "hmm_predict_regimes", side_effect=AssertionError("commentary must not fit HMM")
    ):
        payload = chart_builder.build_commentary_payload(conn, "ACME", 0, report_dir=tmp_path)
    assert payload["chart_commentary"]
    assert "hmm_regime" not in payload


def test_regime_payload_is_trimmed_and_cached_per_last_bar(tmp_path: Path) -> None:
    db_path = tmp_path / "prices.db"
    conn = _make_price_db(db_path)
    calls: list[int] = []

    def _predict(prices: pd.DataFrame, ticker: str = "") -> dict:
        calls.append(len(prices))
        dates = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d").tolist()
        return _fake_regime(dates)

    with patch.object(chart_regime, "hmm_predict_regimes", side_effect=_predict):
        first = chart_regime.build_regime_payload(conn, "ACME", report_dir=tmp_path)
        second = chart_regime.build_regime_payload(conn, "ACME", report_dir=tmp_path)
        assert calls == [BARS]
        assert first == second
        assert len(first["hmm_regime"]["regimes"]) == min(BARS, HMM_CHART_DAYS)
        assert set(first["hmm_regime"]["regimes"][0]) == {"date", "prob_low", "prob_normal", "prob_high"}

        next_day = dt.date.today().isoformat()
        conn.execute(
            "INSERT INTO price_daily (ticker, date, open, high, low, close, volume) VALUES ('ACME', ?, 1, 2, 0.5, 1.5, 10)",
            (next_day,),
        )
        _record_revision(conn, "ACME")
        chart_regime.build_regime_payload(conn, "ACME", report_dir=tmp_path)
        assert calls == [BARS, BARS + 1]


def test_regime_payload_caches_failed_fit(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    with patch.object(
        chart_regime, "hmm_predict_regimes", side_effect=ValueError("HMM regime: no stable fit found")
    ) as predict:
        first = chart_regime.build_regime_payload(conn, "ACME", report_dir=tmp_path)
        second = chart_regime.build_regime_payload(conn, "ACME", report_dir=tmp_path)
    assert first["hmm_regime"] is None and second["hmm_regime"] is None
    assert predict.call_count == 1


def test_regime_payload_prefers_fresh_report_artifact(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    stored = _fake_regime(["2026-01-02", "2026-01-05"])
    with (
        patch.object(chart_regime, "latest_report_hmm_for_ticker", return_value=stored),
        patch.object(chart_regime, "hmm_predict_regimes", side_effect=AssertionError("must not fit")),
    ):
        payload = chart_regime.build_regime_payload(conn, "ACME", report_dir=tmp_path)
    assert payload["hmm_regime"]["regimes"][0] == {
        "date": "2026-01-02", "prob_low": 0.9, "prob_normal": 0.07, "prob_high": 0.03,
    }
    assert len(stored["regimes"][0]) == 7  # stored artifact is not modified


def test_regime_payload_excluded_ticker_skips_fit(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    conn.execute("UPDATE price_daily SET basis_status = 'unresolved'")
    conn.commit()
    with patch.object(chart_regime, "hmm_predict_regimes", side_effect=AssertionError("must not fit")):
        payload = chart_regime.build_regime_payload(conn, "ACME", report_dir=tmp_path)
    assert payload["hmm_regime"] is None
    assert payload["excluded_reason"] == "price_basis_not_research_eligible"


def test_regime_route_returns_payload(test_app) -> None:
    with patch(
        "trader_koo.backend.routers.dashboard.build_regime_payload",
        return_value={"ticker": "SPY", "hmm_regime": None},
    ) as build:
        response = test_app.get("/api/dashboard/SPY/regime")
    assert response.status_code == 200
    assert response.json() == {"ticker": "SPY", "hmm_regime": None}
    assert build.call_args.kwargs["ticker"] == "SPY"

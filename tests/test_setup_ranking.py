"""Setup ranking and per-run report warnings, tested through the report's real schema."""
from __future__ import annotations

import datetime as dt
import json
import math
import socket
import sqlite3
from pathlib import Path

import pytest
import time_machine

from trader_koo.db.price_contract import record_price_series_revision
from trader_koo.report import generator, market_context, setup_ranking
from trader_koo.scripts.run_yolo_patterns import ensure_yolo_table
from trader_koo.scripts.update_market_db import ensure_schema

SESSION = dt.date(2026, 10, 23)
NOW = dt.datetime(2026, 10, 23, 22, 0, tzinfo=dt.timezone.utc)
# ticker: (start close, drift per bar, wave amplitude, wave period, discount %, PEG, sector)
SERIES = {
    "AAA": (50.0, 0.004, 1.5, 9.0, 18.0, 0.7, "Technology"),
    "BBB": (120.0, -0.003, 3.0, 7.0, -6.0, 3.2, "Energy"),
    "CCC": (80.0, 0.001, 0.4, 13.0, 4.0, 1.2, "Technology"),
    "DDD": (30.0, 0.006, 0.9, 5.0, 25.0, None, "Health Care"),
    "EEE": (200.0, -0.001, 6.0, 11.0, -15.0, 5.0, "Financials"),
    "FFF": (65.0, 0.0, 2.5, 4.0, 0.0, 1.9, "Energy"),
    "GGG": (15.0, 0.008, 0.2, 17.0, 9.0, 0.9, "Utilities"),
    "HHH": (95.0, -0.005, 1.0, 6.0, None, 2.4, "Financials"),
}
YOLO = {"AAA": ("double_bottom", 0.82), "BBB": ("head_and_shoulders_top", 0.74), "GGG": ("triangle", 0.55)}
SETUP_TICKERS = list(SERIES)


def _sessions(count: int) -> list[str]:
    days, cursor = [], SESSION
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor -= dt.timedelta(days=1)
    return days[::-1]


def _bars(start: float, drift: float, amp: float, period: float, dates: list[str]) -> list[tuple]:
    rows = []
    for k, date in enumerate(dates):
        close = round(start * (1 + drift * k) + amp * math.sin(k * 2 * math.pi / period), 2)
        rows.append((date, round(close * 0.995, 2), round(close * 1.012, 2), round(close * 0.988, 2), close,
                     1_000_000 + 7_919 * ((k * 31) % 97)))
    return rows


def _build_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    ensure_yolo_table(conn)
    dates = _sessions(90)
    series = {t: _bars(*SERIES[t][:4], dates) for t in SETUP_TICKERS}
    series["^VIX"] = _bars(17.0, 0.0, 3.0, 8.0, dates)
    for ticker, rows in series.items():
        conn.executemany(
            "INSERT INTO price_daily (ticker, date, open, high, low, close, volume, data_source, fetch_timestamp, "
            "adjustment_basis, adjustment_version, basis_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'yfinance', '2026-10-23T21:30:00Z', "
            "'split_adjusted_price_only', 'test-v1', 'verified')",
            [(ticker, *row) for row in rows],
        )
        record_price_series_revision(
            conn, ticker,
            evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
            fetch_timestamp="2026-10-23T21:30:00Z",
        )
    conn.execute(
        "INSERT INTO ingest_runs (run_id, started_ts, finished_ts, status, tickers_total, tickers_ok, "
        "tickers_failed, args_json) VALUES ('r1', '2026-10-23T21:00:00Z', '2026-10-23T21:30:00Z', 'ok', ?, ?, 0, "
        "'{\"use_sp500\": 1}')",
        (len(SETUP_TICKERS), len(SETUP_TICKERS)),
    )
    for ticker, (_s, _d, _a, _p, discount, peg, sector) in SERIES.items():
        conn.execute(
            "INSERT INTO ingest_ticker_status (run_id, ticker, started_ts, status) "
            "VALUES ('r1', ?, '2026-10-23T21:00:00Z', 'ok')",
            (ticker,),
        )
        conn.execute(
            "INSERT INTO finviz_fundamentals (snapshot_ts, ticker, price, peg, discount_pct, raw_json) "
            "VALUES ('2026-10-23T21:00:00Z', ?, ?, ?, ?, ?)",
            (ticker, series[ticker][-1][4], peg, discount, json.dumps({"Sector": sector})),
        )
    for ticker, (pattern, confidence) in YOLO.items():
        conn.execute(
            "INSERT INTO yolo_patterns (ticker, timeframe, pattern, confidence, x0_date, x1_date, y0, y1, "
            "lookback_days, as_of_date, detected_ts) VALUES (?, 'daily', ?, ?, ?, ?, 1, 2, 120, ?, "
            "'2026-10-23T21:45:00Z')",
            (ticker, pattern, confidence, dates[-30], dates[-4], SESSION.isoformat()),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_a, **_k):
        raise OSError("network blocked in test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture
def ranked_db(tmp_path: Path, no_network: None) -> sqlite3.Connection:
    path = tmp_path / "ranking.db"
    _build_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def _signals(conn: sqlite3.Connection, warnings: list[str] | None = None) -> dict:
    with time_machine.travel(NOW, tick=False):
        return generator.fetch_signals(conn, [] if warnings is None else warnings)


def _final_key(row: dict) -> tuple:
    return (
        float(row.get("score") or 0.0),
        float(row.get("confirmation_count") or 0.0),
        -float(row.get("contradiction_count") or 0.0),
        float(row.get("pct_change") or 0.0),
    )


# Recorded from origin/main (2ffd0e91) before the setup-ranking module existed.
EXPECTED_ORDER = [
    ("BBB", 100.0, "B"), ("AAA", 84.7, "B"), ("DDD", 80.1, "B"), ("GGG", 62.2, "D"),
    ("EEE", 58.2, "D"), ("FFF", 57.2, "D"), ("HHH", 52.2, "D"), ("CCC", 32.0, "D"),
]


def test_setup_order_matches_the_ranking_before_the_refactor(ranked_db: sqlite3.Connection) -> None:
    rows = _signals(ranked_db)["setup_quality_all"]

    assert [(r["ticker"], r["score"], r["setup_tier"]) for r in rows] == EXPECTED_ORDER


def test_final_order_drives_every_setup_surface(ranked_db: sqlite3.Connection) -> None:
    signals = _signals(ranked_db)
    rows = signals["setup_quality_all"]

    assert rows == sorted(rows, key=_final_key, reverse=True)
    assert signals["setup_quality_top"] == rows[:40]
    assert [w["ticker"] for w in signals["watchlist_candidates"]] == [r["ticker"] for r in rows[:20]]
    assert set(signals["setup_quality_lookup"]) == {r["ticker"] for r in rows}
    assert signals["options_research"]["enabled"] is setup_ranking.OPTIONS_RESEARCH_ENABLED


def test_calibrate_setups_reorders_top_and_mirrors_scores_into_watchlist(ranked_db: sqlite3.Connection) -> None:
    signals = _signals(ranked_db)
    signals["setup_quality_top"].reverse()

    calibration = setup_ranking.calibrate_setups(
        signals, reliability_lookup={}, min_sample=20, baseline_hit_rate_pct=None, calibration_state={},
    )

    top = signals["setup_quality_top"]
    assert isinstance(calibration, dict)
    assert top == sorted(top, key=_final_key, reverse=True)
    by_ticker = {r["ticker"]: r for r in top}
    for item in signals["watchlist_candidates"]:
        assert item["score"] == by_ticker[item["ticker"]]["score"]
    assert signals["suggestions"] == setup_ranking.build_suggestions(top)


def test_warnings_collector_receives_this_runs_warnings(
    ranked_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_conn):
        raise RuntimeError("volatility source down")

    first: list[str] = []
    with monkeypatch.context() as patch:
        patch.setattr(generator, "_fetch_volatility_inputs", broken)
        _signals(ranked_db, first)
    second: list[str] = []
    _signals(ranked_db, second)

    assert "volatility_inputs_failed" in first
    assert "volatility_inputs_failed" not in second
    assert not hasattr(generator, "_report_warnings")


def _break_vix_levels(patch: pytest.MonkeyPatch) -> None:
    def broken(*_a, **_k):
        raise RuntimeError("level engine down")

    patch.setattr(market_context, "add_basic_features", broken)


def test_regime_context_reports_level_failure_to_the_callers_collector(
    ranked_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _break_vix_levels(monkeypatch)
    collected: list[str] = []

    market_context._build_regime_context(ranked_db, collected)

    assert collected == ["levels_computation_failed"]


def test_api_regime_calls_leave_no_residue_for_later_report_runs(
    ranked_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        _break_vix_levels(patch)
        # The API calls the regime builder without a collector, as the report routers do.
        market_context._build_regime_context(ranked_db)
        market_context._build_regime_context(ranked_db)
    warnings: list[str] = []

    _signals(ranked_db, warnings)

    assert not hasattr(market_context, "_report_warnings")
    assert "levels_computation_failed" not in warnings

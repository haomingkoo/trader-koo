from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import DEFAULT, patch

import pandas as pd
import pytest

from trader_koo.analysis.green_barrier import (
    build_green_barrier_chart_png,
    compute_williams_percent_r,
    resample_ohlcv,
    scan_green_barrier_snapshot,
    scan_green_barriers,
)
from trader_koo.backend.services import chart_builder
from trader_koo.db.price_contract import record_price_series_revision
from trader_koo.notifications.morning_summary import (
    _select_green_barrier_attachments,
    send_morning_summary,
)


def _make_price_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            data_source TEXT DEFAULT 'yfinance',
            fetch_timestamp TEXT DEFAULT '2026-08-22T00:00:00Z',
            adjustment_basis TEXT NOT NULL DEFAULT 'split_adjusted_price_only',
            adjustment_version TEXT NOT NULL DEFAULT 'test-v1',
            basis_status TEXT NOT NULL DEFAULT 'verified',
            unresolved_reason TEXT
        )
        """
    )
    rows = []
    dates = pd.date_range("2024-01-31", periods=18, freq="ME")
    for idx, date in enumerate(dates):
        rows.append(("HIT", date.date().isoformat(), 110, 120, 80, 80 if idx == 17 else 110, 1000))
        rows.append(("CLEAR", date.date().isoformat(), 110, 120, 80, 120, 1000))
    conn.executemany(
        """INSERT INTO price_daily
        (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    for ticker in ("HIT", "CLEAR"):
        record_price_series_revision(
            conn,
            ticker,
            evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
            fetch_timestamp="2026-08-22T00:00:00Z",
        )
    conn.commit()
    return conn


def test_williams_percent_r_reaches_minus_100_at_range_low() -> None:
    frame = pd.DataFrame(
        {
            "high": [120.0] * 14,
            "low": [80.0] * 14,
            "close": [110.0] * 13 + [80.0],
        }
    )
    result = compute_williams_percent_r(frame)
    assert result.iloc[:13].isna().all()
    assert result.iloc[-1] == -100.0


def test_resample_uses_last_observed_date_not_future_period_end() -> None:
    daily = pd.DataFrame(
        {
            "date": ["2026-03-16", "2026-03-17", "2026-03-18"],
            "open": [10, 11, 12],
            "high": [12, 13, 14],
            "low": [9, 10, 11],
            "close": [11, 12, 13],
            "volume": [100, 200, 300],
        }
    )
    weekly = resample_ohlcv(daily, "weekly")
    assert weekly.iloc[-1]["date"].date().isoformat() == "2026-03-18"


def test_scan_and_render_green_barrier(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    try:
        hits = scan_green_barriers(
            conn,
            timeframes=("monthly",),
            as_of=pd.Timestamp("2025-06-30").date(),
        )
        assert [row["ticker"] for row in hits] == ["HIT"]
        assert hits[0]["value"] == -100.0
        assert hits[0]["distance_to_barrier"] == 0.0

        png = build_green_barrier_chart_png(conn, ticker="HIT", timeframe="monthly")
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(png) > 10_000
    finally:
        conn.close()


def test_attachment_selection_prefers_monthly_and_deduplicates_ticker() -> None:
    hits = [
        {"ticker": "AAA", "timeframe": "weekly", "value": -100.0},
        {"ticker": "AAA", "timeframe": "monthly", "value": -98.5},
        {"ticker": "BBB", "timeframe": "weekly", "value": -99.0},
    ]
    selected = _select_green_barrier_attachments(hits, limit=2)
    assert [(row["ticker"], row["timeframe"]) for row in selected] == [
        ("AAA", "monthly"),
        ("BBB", "weekly"),
    ]


def test_morning_summary_sends_native_photo_attachment(tmp_path: Path) -> None:
    db_path = tmp_path / "prices.db"
    conn = _make_price_db(db_path)
    conn.close()
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    report = {
        "signals": {
            "green_barrier_hits": [
                {
                    "ticker": "HIT",
                    "timeframe": "monthly",
                    "value": -100.0,
                    "threshold": -98.0,
                    "asof": "2025-06-30",
                }
            ]
        }
    }

    with (
        patch("trader_koo.notifications.morning_summary.is_configured", return_value=True),
        patch("trader_koo.notifications.morning_summary.generate_morning_summary", return_value="brief"),
        patch("trader_koo.notifications.morning_summary.send_message", return_value=True) as send_text,
        patch("trader_koo.notifications.morning_summary.send_photo", return_value=True) as send_image,
        patch(
            "trader_koo.backend.services.report_loader.latest_daily_report_json",
            return_value=(report_dir / "daily_report_latest.json", report),
        ),
    ):
        assert send_morning_summary(db_path, report_dir) is True

    send_text.assert_called_once_with("brief")
    assert send_image.call_count == 1
    assert send_image.call_args.kwargs["filename"] == "green-barrier-HIT-monthly.png"
    assert send_image.call_args.kwargs["caption"].startswith("🟢")
    chart_url = send_image.call_args.kwargs["reply_markup"]["inline_keyboard"][0][0]["url"]
    assert chart_url.endswith(
        "/chart?ticker=HIT&timeframe=monthly&threshold=-98&asof=2025-06-30&value=-100"
    )
    assert "Current Condition" in send_image.call_args.kwargs["caption"]


def test_scan_skips_stale_price_data(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    try:
        assert scan_green_barriers(
            conn,
            timeframes=("monthly",),
            as_of=pd.Timestamp("2025-07-08").date(),
            max_age_days=7,
        ) == []
    finally:
        conn.close()


def test_scan_snapshot_reports_incomplete_coverage(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    try:
        snapshot = scan_green_barrier_snapshot(
            conn,
            timeframes=("monthly",),
            as_of=pd.Timestamp("2025-07-08").date(),
            max_age_days=7,
        )
    finally:
        conn.close()

    assert snapshot["hits"] == []
    assert snapshot["coverage"] == {
        "scan_asof": "2025-07-08",
        "threshold": -98.0,
        "max_age_days": 7,
        "source_ticker_count": 2,
        "scanned_ticker_count": 0,
        "stale_skipped_count": 2,
        "stale_skipped_tickers": ["CLEAR", "HIT"],
        "invalid_date_skipped_count": 0,
        "insufficient_history_skipped_count": 0,
        "basis_unresolved_skipped_count": 0,
        "basis_unresolved_skipped_tickers": [],
    }


def test_scan_snapshot_counts_insufficient_timeframe_history(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "short.db")
    conn.execute(
        """CREATE TABLE price_daily (
        ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
        adjustment_basis TEXT DEFAULT 'split_adjusted_price_only',
        adjustment_version TEXT DEFAULT 'test-v1', basis_status TEXT DEFAULT 'verified',
        unresolved_reason TEXT)"""
    )
    rows = [
        ("SHORT", date.date().isoformat(), 10, 11, 9, 10, 100)
        for date in pd.date_range("2026-08-01", periods=10, freq="D")
    ]
    conn.executemany(
        """INSERT INTO price_daily
        (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    record_price_series_revision(
        conn,
        "SHORT",
        evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
        fetch_timestamp="2026-08-22T00:00:00Z",
    )
    conn.commit()
    try:
        snapshot = scan_green_barrier_snapshot(
            conn,
            timeframes=("weekly", "monthly"),
            as_of=pd.Timestamp("2026-08-10").date(),
        )
    finally:
        conn.close()

    assert snapshot["coverage"]["scanned_ticker_count"] == 1
    assert snapshot["coverage"]["insufficient_history_skipped_count"] == 2


def test_scan_fails_closed_for_unresolved_price_basis(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "unresolved.db")
    conn.execute("UPDATE price_daily SET basis_status = 'unresolved' WHERE ticker = 'HIT'")
    conn.commit()
    try:
        snapshot = scan_green_barrier_snapshot(
            conn,
            timeframes=("monthly",),
            as_of=pd.Timestamp("2025-06-30").date(),
        )
    finally:
        conn.close()

    assert snapshot["hits"] == []
    assert snapshot["coverage"]["basis_unresolved_skipped_tickers"] == ["HIT"]


def test_chart_renderer_fails_closed_for_unresolved_price_basis(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "unresolved-chart.db")
    conn.execute("UPDATE price_daily SET basis_status = 'unresolved' WHERE ticker = 'HIT'")
    conn.commit()
    try:
        with pytest.raises(ValueError, match="not research eligible"):
            build_green_barrier_chart_png(conn, ticker="HIT", timeframe="monthly")
    finally:
        conn.close()


def test_dashboard_and_commentary_do_not_compute_unresolved_series(
    tmp_path: Path,
) -> None:
    conn = _make_price_db(tmp_path / "unresolved-dashboard.db")
    conn.execute("UPDATE price_daily SET basis_status = 'unresolved' WHERE ticker = 'HIT'")
    conn.commit()
    with patch.object(
        chart_builder,
        "_prepare_model_and_features",
        side_effect=AssertionError("indicator computation must not run"),
    ):
        quick = chart_builder.build_dashboard_quick_payload(conn, "HIT", 12)
        full = chart_builder.build_dashboard_payload(
            conn,
            "HIT",
            12,
            report_dir=tmp_path,
        )
        commentary = chart_builder.build_commentary_payload(
            conn,
            "HIT",
            12,
            report_dir=tmp_path,
        )
    conn.close()

    assert quick["chart"] == []
    assert full["chart_commentary"] is None
    assert commentary["hmm_regime"] is None
    assert quick["data_sources"]["research_eligible"] is False


def test_quick_dashboard_uses_one_sqlite_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "snapshot.db"
    conn = _make_price_db(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    writer = sqlite3.connect(db_path)
    original_prepare = chart_builder._prepare_model_and_features

    def repair_after_admission(*args, **kwargs):
        writer.execute(
            "UPDATE price_daily SET close=999, basis_status='unresolved' WHERE ticker='HIT'"
        )
        writer.commit()
        return original_prepare(*args, **kwargs)

    try:
        with patch.multiple(
            chart_builder,
            _prepare_model_and_features=DEFAULT,
            get_latest_fundamentals=DEFAULT,
            get_ticker_earnings_markers=DEFAULT,
            get_yolo_patterns=DEFAULT,
            get_yolo_audit=DEFAULT,
            get_latest_options_summary=DEFAULT,
        ) as mocked:
            mocked["_prepare_model_and_features"].side_effect = repair_after_admission
            mocked["get_latest_fundamentals"].return_value = None
            mocked["get_ticker_earnings_markers"].return_value = []
            mocked["get_yolo_patterns"].return_value = []
            mocked["get_yolo_audit"].return_value = []
            mocked["get_latest_options_summary"].return_value = {}
            payload = chart_builder.build_dashboard_quick_payload(conn, "HIT", 24)
        assert payload["chart"][-1]["close"] == 80
        assert payload["data_sources"]["research_eligible"] is True
        assert writer.execute(
            "SELECT close FROM price_daily WHERE ticker='HIT' ORDER BY date DESC LIMIT 1"
        ).fetchone()[0] == 999
    finally:
        writer.close()
        conn.close()


def test_chart_is_bound_to_report_asof_and_value(tmp_path: Path) -> None:
    conn = _make_price_db(tmp_path / "prices.db")
    try:
        conn.execute(
            """INSERT INTO price_daily
            (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("HIT", "2025-07-31", 110, 120, 80, 120, 1000),
        )
        record_price_series_revision(
            conn,
            "HIT",
            evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
            fetch_timestamp="2026-08-22T00:00:00Z",
        )
        conn.commit()

        png = build_green_barrier_chart_png(
            conn,
            ticker="HIT",
            timeframe="monthly",
            as_of="2025-06-30",
            threshold=-95.0,
            expected_value=-100.0,
        )
        assert png.startswith(b"\x89PNG\r\n\x1a\n")

        with pytest.raises(ValueError, match="snapshot mismatch"):
            build_green_barrier_chart_png(
                conn,
                ticker="HIT",
                timeframe="monthly",
                as_of="2025-07-31",
                threshold=-95.0,
                expected_value=-100.0,
            )
    finally:
        conn.close()


def test_photo_failure_is_explicit_partial_without_text_retry(
    tmp_path: Path,
    caplog,
) -> None:
    db_path = tmp_path / "prices.db"
    conn = _make_price_db(db_path)
    conn.close()
    report = {
        "signals": {
            "green_barrier_hits": [
                {
                    "ticker": "HIT",
                    "timeframe": "monthly",
                    "value": -100.0,
                    "threshold": -95.0,
                    "asof": "2025-06-30",
                }
            ]
        }
    }

    with (
        patch("trader_koo.notifications.morning_summary.is_configured", return_value=True),
        patch("trader_koo.notifications.morning_summary.generate_morning_summary", return_value="brief"),
        patch("trader_koo.notifications.morning_summary.send_message", return_value=True) as send_text,
        patch("trader_koo.notifications.morning_summary.send_photo", return_value=False),
        patch(
            "trader_koo.backend.services.report_loader.latest_daily_report_json",
            return_value=(tmp_path / "daily_report_latest.json", report),
        ),
    ):
        assert send_morning_summary(db_path, tmp_path) is True

    send_text.assert_called_once_with("brief")
    assert "MORNING_SUMMARY_PARTIAL" in caplog.text

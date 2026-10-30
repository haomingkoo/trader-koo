"""Daily-close data must say which close it is, never claim to be live."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trader_koo.backend.routers import report as report_router
from trader_koo.notifications import macro_monitor
from trader_koo.scripts.update_market_db import (
    ensure_schema,
    reconcile_vendor_action_ledger,
    write_price_daily,
)
from trader_koo.structure.vix_metrics import compute_vix_metrics


def _recent_dates(count: int) -> list[str]:
    today = dt.datetime.now(dt.timezone.utc).date()
    return [(today - dt.timedelta(days=count - i)).isoformat() for i in range(count)]


def _seed(conn: sqlite3.Connection, ticker: str, closes: list[float]) -> list[str]:
    dates = _recent_dates(len(closes))
    frame = pd.DataFrame(
        [
            {"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1000}
            for d, c in zip(dates, closes)
        ]
    )
    frame.attrs.update(
        adjustment_basis="split_adjusted_price_only",
        adjustment_version="fixture-v1",
        basis_status="verified",
    )
    reconcile_vendor_action_ledger(frame, [])
    write_price_daily(conn, ticker, frame, fetch_timestamp=f"{dates[-1]}T22:00:00Z")
    conn.commit()
    return dates


def test_vix_metrics_report_the_close_date_they_used() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    dates = _seed(conn, "^VIX", [16.0, 17.0, 18.5])

    metrics = compute_vix_metrics(conn)

    assert metrics["as_of"] == dates[-1]
    assert metrics["vix_close"] == 18.5


def test_vix_metrics_endpoint_does_not_claim_live_source(tmp_path: Path) -> None:
    db_path = tmp_path / "vix.db"
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    dates = _seed(conn, "^VIX", [16.0, 17.0])
    conn.close()

    with patch.object(report_router, "get_conn", lambda: sqlite3.connect(db_path)):
        payload = report_router.vix_metrics()

    assert payload["ok"] is True
    assert payload["source"] == "daily_close"
    assert payload["as_of"] == dates[-1]


def test_macro_live_labels_close_fallback_with_its_date(tmp_path: Path) -> None:
    db_path = tmp_path / "macro.db"
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    dates = _seed(conn, "^VIX", [17.0, 18.0])
    conn.close()

    # The live provider has no index quote on this plan.
    with (
        patch.object(macro_monitor, "_get_finnhub_key", return_value="KEY"),
        patch.object(macro_monitor, "_fetch_quote", return_value=None),
    ):
        payload = macro_monitor._load_macro_live(db_path)

    vix = next(i for i in payload["instruments"] if i["ticker"] == "^VIX")
    assert vix["current"] is None
    assert vix["prev_close"] == 18.0
    assert vix["prev_close_date"] == dates[-1]
    assert vix["quote_source"] is None


def test_macro_live_marks_streamed_quotes_live(tmp_path: Path) -> None:
    db_path = tmp_path / "macro.db"
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)
    _seed(conn, "GLD", [300.0, 310.0])
    conn.close()

    with (
        patch.object(macro_monitor, "_get_finnhub_key", return_value="KEY"),
        patch.object(macro_monitor, "_fetch_quote", return_value=303.8),
    ):
        payload = macro_monitor._load_macro_live(db_path)

    gld = next(i for i in payload["instruments"] if i["ticker"] == "GLD")
    assert gld["quote_source"] == "live"
    assert gld["change_pct"] == -2.0

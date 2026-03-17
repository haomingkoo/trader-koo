"""Shared fixtures for the trader_koo v2 backend test suite.

Provides an in-memory SQLite database pre-loaded with the real schema,
a FastAPI TestClient wired to all routers, and helper factories.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers.system import router as system_router
from trader_koo.backend.routers.dashboard import router as dashboard_router
from trader_koo.backend.routers.report import router as report_router
from trader_koo.backend.routers.opportunities import router as opportunities_router
from trader_koo.backend.routers.paper_trades import router as paper_trades_router
from trader_koo.backend.routers.email import router as email_router
from trader_koo.backend.routers.usage import router as usage_router
from trader_koo.backend.routers.admin import router as admin_router


# ---------------------------------------------------------------------------
# In-memory DB helpers
# ---------------------------------------------------------------------------

def _create_test_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the minimal schema needed by
    the backend services.  This avoids hitting the real disk database.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            data_source TEXT,
            fetch_timestamp TEXT,
            UNIQUE(ticker, date)
        );

        CREATE TABLE finviz_fundamentals (
            ticker TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
            price REAL,
            pe REAL,
            peg REAL,
            eps_ttm REAL,
            eps_growth_5y REAL,
            target_price REAL,
            discount_pct REAL,
            target_reason TEXT,
            raw_json TEXT,
            UNIQUE(ticker, snapshot_ts)
        );

        CREATE TABLE options_iv (
            ticker TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
            option_type TEXT,
            implied_vol REAL,
            open_interest INTEGER,
            UNIQUE(ticker, snapshot_ts, option_type)
        );

        CREATE TABLE ingest_runs (
            run_id TEXT PRIMARY KEY,
            started_ts TEXT,
            finished_ts TEXT,
            status TEXT DEFAULT 'running',
            tickers_total INTEGER DEFAULT 0,
            tickers_ok INTEGER DEFAULT 0,
            tickers_failed INTEGER DEFAULT 0,
            error_message TEXT
        );

        CREATE TABLE ingest_ticker_status (
            run_id TEXT,
            ticker TEXT,
            status TEXT,
            UNIQUE(run_id, ticker)
        );

        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT,
            ticker TEXT NOT NULL,
            direction TEXT,
            entry_price REAL,
            entry_date TEXT,
            target_price REAL,
            stop_loss REAL,
            atr_at_entry REAL,
            exit_price REAL,
            exit_date TEXT,
            exit_reason TEXT,
            status TEXT DEFAULT 'open',
            current_price REAL,
            unrealized_pnl_pct REAL,
            pnl_pct REAL,
            r_multiple REAL,
            high_water_mark REAL,
            low_water_mark REAL,
            setup_family TEXT,
            setup_tier TEXT,
            score REAL,
            signal_bias TEXT,
            actionability TEXT,
            observation TEXT,
            action_text TEXT,
            risk_note TEXT,
            yolo_pattern TEXT,
            yolo_recency TEXT,
            debate_agreement_score REAL,
            last_mtm_date TEXT,
            created_ts TEXT,
            updated_ts TEXT,
            generated_ts TEXT
        );

        CREATE TABLE paper_portfolio_snapshots (
            snapshot_date TEXT PRIMARY KEY,
            open_trades INTEGER DEFAULT 0,
            total_unrealized_pnl_pct REAL DEFAULT 0.0,
            snapshot_ts TEXT
        );

        CREATE TABLE ui_usage_sessions (
            session_id TEXT PRIMARY KEY,
            visitor_id TEXT NOT NULL,
            started_ts TEXT,
            last_seen_ts TEXT,
            active_ms INTEGER NOT NULL DEFAULT 0,
            page_views_total INTEGER NOT NULL DEFAULT 0,
            guide_views INTEGER NOT NULL DEFAULT 0,
            report_views INTEGER NOT NULL DEFAULT 0,
            earnings_views INTEGER NOT NULL DEFAULT 0,
            chart_views INTEGER NOT NULL DEFAULT 0,
            opportunities_views INTEGER NOT NULL DEFAULT 0,
            chart_loads INTEGER NOT NULL DEFAULT 0,
            last_tab TEXT,
            last_ticker TEXT,
            market TEXT,
            path TEXT,
            tz TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE setup_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT NOT NULL,
            asof TEXT,
            verdict TEXT NOT NULL CHECK (verdict IN ('good', 'bad', 'neutral')),
            source_surface TEXT,
            note TEXT,
            setup_tier TEXT,
            setup_score REAL,
            setup_family TEXT,
            signal_bias TEXT,
            actionability TEXT,
            yolo_role TEXT,
            yolo_recency TEXT,
            visitor_id TEXT,
            session_id TEXT,
            client_ip TEXT,
            user_agent TEXT,
            context_json TEXT
        );
    """)
    return conn


def _seed_price_data(conn: sqlite3.Connection) -> None:
    """Insert realistic SPY price data for the last 250 trading days."""
    base_date = dt.date(2026, 3, 10)
    base_price = 580.0
    rows = []
    for i in range(250):
        d = base_date - dt.timedelta(days=i)
        day_offset = (i % 7) - 3
        close = round(base_price + day_offset * 1.5 - i * 0.02, 2)
        open_ = round(close - 0.5, 2)
        high = round(close + 1.2, 2)
        low = round(close - 1.3, 2)
        volume = 80_000_000 + i * 10_000
        rows.append(("SPY", d.isoformat(), open_, high, low, close, volume, "yfinance", None))
    conn.executemany(
        "INSERT OR IGNORE INTO price_daily (ticker, date, open, high, low, close, volume, data_source, fetch_timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _seed_fundamentals(conn: sqlite3.Connection) -> None:
    """Insert a Finviz fundamentals row for SPY."""
    snap = "2026-03-10T00:00:00Z"
    conn.execute(
        "INSERT OR IGNORE INTO finviz_fundamentals "
        "(ticker, snapshot_ts, price, pe, peg, eps_ttm, eps_growth_5y, target_price, discount_pct, target_reason, raw_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SPY", snap, 580.0, 22.0, 1.5, 26.0, 12.0, 610.0, 5.2, "FINVIZ_TARGET", '{"Sector":"ETF","Industry":"Index Fund"}'),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem_conn():
    """In-memory SQLite connection with full schema, no data."""
    conn = _create_test_db()
    yield conn
    conn.close()


@pytest.fixture()
def seeded_conn():
    """In-memory SQLite connection with schema + SPY price/fundamentals data."""
    conn = _create_test_db()
    _seed_price_data(conn)
    _seed_fundamentals(conn)
    yield conn
    conn.close()


@pytest.fixture()
def test_app(tmp_path: Path, seeded_conn: sqlite3.Connection):
    """FastAPI TestClient with all routers included.

    Patches ``get_conn`` to return the in-memory seeded DB,
    and ``DB_PATH`` to point to a real (but temporary) file so
    ``DB_PATH.exists()`` returns True.
    """
    # Write a dummy DB file so DB_PATH.exists() is True
    tmp_db_file = tmp_path / "test_trader_koo.db"
    tmp_db_file.write_bytes(b"")

    app = FastAPI(title="test_trader_koo")
    app.include_router(system_router)
    app.include_router(dashboard_router)
    app.include_router(report_router)
    app.include_router(opportunities_router)
    app.include_router(paper_trades_router)
    app.include_router(email_router)
    app.include_router(usage_router)
    app.include_router(admin_router)

    def _fake_get_conn(db_path=None):
        """Return a new in-memory connection sharing the same data as seeded_conn."""
        return seeded_conn

    patches = [
        patch("trader_koo.backend.services.database.DB_PATH", tmp_db_file),
        patch("trader_koo.backend.services.database.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.system.DB_PATH", tmp_db_file),
        patch("trader_koo.backend.routers.dashboard.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.report.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.opportunities.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.paper_trades.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.usage.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.usage.DB_PATH", tmp_db_file),
        patch("trader_koo.backend.routers.email.DB_PATH", tmp_db_file),
        patch("trader_koo.backend.routers.email.get_conn", _fake_get_conn),
    ]
    for p in patches:
        p.start()

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    for p in patches:
        p.stop()

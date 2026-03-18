"""Schema helpers for paper trades."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    ddl: str,
) -> None:
    columns = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def decode_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        payload = json.loads(str(raw))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def ensure_paper_trade_schema(conn: sqlite3.Connection) -> None:
    """Create paper_trades and paper_portfolio_snapshots tables."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL,
            generated_ts TEXT,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            target_price REAL,
            stop_loss REAL,
            atr_at_entry REAL,
            exit_price REAL,
            exit_date TEXT,
            exit_reason TEXT,
            status TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'closed', 'stopped_out', 'target_hit', 'expired')),
            current_price REAL,
            unrealized_pnl_pct REAL,
            last_mtm_date TEXT,
            high_water_mark REAL,
            low_water_mark REAL,
            pnl_pct REAL,
            r_multiple REAL,
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
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_date, ticker, direction)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_status "
        "ON paper_trades(status, entry_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker "
        "ON paper_trades(ticker, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_family "
        "ON paper_trades(setup_family, direction, status)"
    )
    _ensure_column(conn, "paper_trades", "decision_version", "decision_version TEXT")
    _ensure_column(conn, "paper_trades", "decision_state", "decision_state TEXT")
    _ensure_column(conn, "paper_trades", "analyst_stage", "analyst_stage TEXT")
    _ensure_column(conn, "paper_trades", "debate_stage", "debate_stage TEXT")
    _ensure_column(conn, "paper_trades", "risk_stage", "risk_stage TEXT")
    _ensure_column(conn, "paper_trades", "portfolio_decision", "portfolio_decision TEXT")
    _ensure_column(conn, "paper_trades", "decision_summary", "decision_summary TEXT")
    _ensure_column(conn, "paper_trades", "decision_reasons", "decision_reasons TEXT")
    _ensure_column(conn, "paper_trades", "risk_flags", "risk_flags TEXT")
    _ensure_column(conn, "paper_trades", "position_size_pct", "position_size_pct REAL")
    _ensure_column(conn, "paper_trades", "risk_budget_pct", "risk_budget_pct REAL")
    _ensure_column(conn, "paper_trades", "stop_distance_pct", "stop_distance_pct REAL")
    _ensure_column(conn, "paper_trades", "expected_reward_pct", "expected_reward_pct REAL")
    _ensure_column(conn, "paper_trades", "expected_r_multiple", "expected_r_multiple REAL")
    _ensure_column(conn, "paper_trades", "entry_plan", "entry_plan TEXT")
    _ensure_column(conn, "paper_trades", "exit_plan", "exit_plan TEXT")
    _ensure_column(conn, "paper_trades", "sizing_summary", "sizing_summary TEXT")
    _ensure_column(conn, "paper_trades", "review_status", "review_status TEXT")
    _ensure_column(conn, "paper_trades", "review_summary", "review_summary TEXT")
    _ensure_column(conn, "paper_trades", "bot_version", "bot_version TEXT")
    _ensure_column(conn, "paper_trades", "vix_at_entry", "vix_at_entry REAL")
    _ensure_column(conn, "paper_trades", "vix_percentile_at_entry", "vix_percentile_at_entry REAL")
    _ensure_column(conn, "paper_trades", "regime_state_at_entry", "regime_state_at_entry TEXT")
    _ensure_column(conn, "paper_trades", "hmm_regime_at_entry", "hmm_regime_at_entry TEXT")
    _ensure_column(conn, "paper_trades", "hmm_confidence_at_entry", "hmm_confidence_at_entry REAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_version TEXT NOT NULL UNIQUE,
            decision_version TEXT,
            strategy_kind TEXT NOT NULL DEFAULT 'paper_rules',
            status TEXT NOT NULL DEFAULT 'active',
            config_json TEXT,
            notes TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bot_versions_status "
        "ON bot_versions(status, created_ts)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL UNIQUE,
            open_trades INTEGER NOT NULL DEFAULT 0,
            closed_trades_total INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            win_rate_pct REAL,
            avg_pnl_pct REAL,
            avg_r_multiple REAL,
            total_pnl_pct REAL,
            max_drawdown_pct REAL,
            sharpe_ratio REAL,
            profit_factor REAL,
            equity_index REAL NOT NULL DEFAULT 100.0,
            best_trade_pct REAL,
            worst_trade_pct REAL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_portfolio_date "
        "ON paper_portfolio_snapshots(snapshot_date)"
    )
    conn.commit()


def register_bot_version(
    conn: sqlite3.Connection,
    *,
    bot_version: str,
    decision_version: str | None,
    config_json: str | None = None,
    notes: str | None = None,
) -> None:
    ensure_paper_trade_schema(conn)
    if not bot_version:
        return
    conn.execute(
        """
        INSERT INTO bot_versions (
            bot_version, decision_version, strategy_kind, status, config_json, notes
        ) VALUES (?, ?, 'paper_rules', 'active', ?, ?)
        ON CONFLICT(bot_version) DO UPDATE SET
            decision_version = excluded.decision_version,
            config_json = COALESCE(excluded.config_json, bot_versions.config_json),
            notes = COALESCE(excluded.notes, bot_versions.notes)
        """,
        (bot_version, decision_version, config_json, notes),
    )

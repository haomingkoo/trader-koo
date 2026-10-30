"""Campaign v3: draft-only registration of the declared campaign and the time-windowed expectancy brake."""
from __future__ import annotations

import dataclasses
import sqlite3

import pytest

from trader_koo.paper_trade import critic
from trader_koo.paper_trade.registry import DECLARED_CAMPAIGNS, register_declared_campaign
from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trades import (
    PAPER_TRADE_CAMPAIGN_ID,
    _PAPER_DECISION_VERSION,
    _build_config,
    register_runtime_campaign,
)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(c)
    return c


def test_runtime_policy_is_campaign_v3_with_a_windowed_brake():
    config = _build_config()
    assert (config.campaign_id, config.decision_version) == ("paper-v3", "paper-campaign-v3.0")
    assert (PAPER_TRADE_CAMPAIGN_ID, _PAPER_DECISION_VERSION) == ("paper-v3", "paper-campaign-v3.0")
    assert config.rolling_expectancy_window_days == 90
    assert DECLARED_CAMPAIGNS["paper-v3"].policy_version == config.decision_version


def test_registration_inserts_a_draft_once_and_never_touches_other_campaigns(conn):
    conn.execute("UPDATE paper_campaigns SET status='active' WHERE campaign_id='paper-v2'")
    before = conn.execute("SELECT campaign_id,status,policy_version FROM paper_campaigns ORDER BY campaign_id").fetchall()
    assert register_declared_campaign(conn, _build_config()) is True
    assert register_declared_campaign(conn, _build_config()) is False  # idempotent
    rows = conn.execute("SELECT campaign_id,status,policy_version FROM paper_campaigns ORDER BY campaign_id").fetchall()
    assert rows == before + [("paper-v3", "draft", "paper-campaign-v3.0")]
    assert conn.execute("SELECT COUNT(*) FROM paper_campaign_audit").fetchone() == (0,)  # no transition recorded


def test_an_undeclared_campaign_is_not_registered(conn):
    typo = dataclasses.replace(_build_config(), campaign_id="paper-v33")
    assert register_declared_campaign(conn, typo) is False
    assert conn.execute("SELECT 1 FROM paper_campaigns WHERE campaign_id='paper-v33'").fetchone() is None


def test_a_declared_campaign_with_another_policy_version_fails_closed(conn):
    drifted = dataclasses.replace(_build_config(), decision_version="paper-campaign-v3.1")
    with pytest.raises(ValueError, match="declared for paper-campaign-v3.0"):
        register_declared_campaign(conn, drifted)


def test_startup_registration_writes_nothing_while_paper_writes_are_paused(conn, monkeypatch):
    import trader_koo.paper_trades as facade

    monkeypatch.setattr(facade, "PAPER_TRADE_ENABLED", False)
    assert register_runtime_campaign(conn) is False
    assert conn.execute("SELECT 1 FROM paper_campaigns WHERE campaign_id='paper-v3'").fetchone() is None
    monkeypatch.setattr(facade, "PAPER_TRADE_ENABLED", True)
    assert register_runtime_campaign(conn) is True


@pytest.fixture()
def trades():
    """Only the columns the expectancy brake reads (as tests/test_critic.py does)."""
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE paper_trades (campaign_id TEXT, status TEXT, pnl_pct REAL, exit_date TEXT)")
    return c


def _closed(conn, exit_date: str, pnl: float, campaign: str = "paper-v3") -> None:
    conn.execute("INSERT INTO paper_trades VALUES (?, 'closed', ?, ?)", (campaign, pnl, exit_date))


def test_losses_older_than_the_window_no_longer_block(trades):
    conn = trades
    for day in ("2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"):
        _closed(conn, day, -3.0)  # five losing trades, the v2 deadlock shape
    blocked, reason = critic._check_rolling_expectancy(conn, "paper-v3", as_of="2026-02-01", window_days=90)
    assert blocked is False and "-3.00%" in reason
    released, reason = critic._check_rolling_expectancy(conn, "paper-v3", as_of="2026-04-10", window_days=90)
    assert released is True and "Insufficient history (0 trades" in reason


def test_window_counts_from_the_report_date_and_ignores_later_exits(trades):
    conn = trades
    for day in ("2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"):
        _closed(conn, day, 1.0)
    _closed(conn, "2026-03-20", -50.0)  # after the report date: not knowable at the decision
    ok, reason = critic._check_rolling_expectancy(conn, "paper-v3", as_of="2026-03-10", window_days=90)
    assert ok is True and "1.00% avg over last 5 trades" in reason


def test_without_a_window_the_brake_is_unchanged(trades):
    conn = trades
    for day in ("2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"):
        _closed(conn, day, -3.0)
    assert critic._check_rolling_expectancy(conn, "paper-v3")[0] is False

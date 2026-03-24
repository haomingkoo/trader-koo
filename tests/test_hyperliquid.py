"""Tests for Hyperliquid whale tracker."""
from __future__ import annotations

import sqlite3

import pytest

from trader_koo.hyperliquid.tracker import (
    WalletPosition,
    WalletSnapshot,
    ensure_hyperliquid_schema,
    generate_counter_signals,
    save_counter_signals,
    save_snapshot,
    seed_default_wallets,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    ensure_hyperliquid_schema(c)
    yield c
    c.close()


class TestSchema:
    def test_creates_tables(self, conn):
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "hyperliquid_wallets" in tables
        assert "hyperliquid_snapshots" in tables
        assert "hyperliquid_counter_signals" in tables

    def test_seed_default_wallets(self, conn):
        seed_default_wallets(conn)
        row = conn.execute(
            "SELECT label, address FROM hyperliquid_wallets WHERE label = 'machibro'"
        ).fetchone()
        assert row is not None
        assert "0x020c" in row[1]


class TestCounterSignals:
    def _make_snapshot(self, side="long", leverage=25, size=5000.0, upnl=-50000.0):
        pos = WalletPosition(
            wallet_label="machibro",
            wallet_address="0x020c",
            coin="ETH",
            side=side,
            size=size,
            entry_price=2100.0,
            mark_price=2000.0,
            unrealized_pnl=upnl,
            leverage_type="cross",
            leverage_value=leverage,
            notional_usd=size * 2000.0,
            liquidation_price=1850.0,
        )
        return WalletSnapshot(
            wallet_label="machibro",
            wallet_address="0x020c",
            account_value=500000.0,
            total_margin_used=400000.0,
            margin_ratio=0.8,
            positions=[pos],
            timestamp="2026-03-24T00:00:00Z",
        )

    def test_generates_counter_for_long(self):
        snapshot = self._make_snapshot(side="long")
        signals = generate_counter_signals(snapshot)
        assert len(signals) == 1
        assert signals[0]["counter_side"] == "short"
        assert signals[0]["their_side"] == "long"
        assert signals[0]["coin"] == "ETH"

    def test_generates_counter_for_short(self):
        snapshot = self._make_snapshot(side="short")
        signals = generate_counter_signals(snapshot)
        assert signals[0]["counter_side"] == "long"

    def test_higher_leverage_higher_confidence(self):
        low_lev = generate_counter_signals(self._make_snapshot(leverage=5))
        high_lev = generate_counter_signals(self._make_snapshot(leverage=25))
        assert high_lev[0]["confidence"] > low_lev[0]["confidence"]

    def test_larger_notional_higher_confidence(self):
        small = generate_counter_signals(self._make_snapshot(size=100.0))
        large = generate_counter_signals(self._make_snapshot(size=10000.0))
        assert large[0]["confidence"] > small[0]["confidence"]


class TestPersistence:
    def test_save_snapshot(self, conn):
        snapshot = WalletSnapshot(
            wallet_label="test",
            wallet_address="0xabc",
            account_value=100000.0,
            total_margin_used=50000.0,
            margin_ratio=0.5,
            positions=[],
            timestamp="2026-03-24T00:00:00Z",
        )
        save_snapshot(conn, snapshot)
        row = conn.execute("SELECT COUNT(*) FROM hyperliquid_snapshots").fetchone()
        assert row[0] == 1

    def test_save_counter_signals(self, conn):
        signals = [{
            "wallet_label": "test",
            "coin": "ETH",
            "counter_side": "short",
            "their_side": "long",
            "their_size": 100.0,
            "their_leverage": 25,
            "their_notional_usd": 200000.0,
            "confidence": 80.0,
            "reasoning": "test signal",
            "timestamp": "2026-03-24T00:00:00Z",
        }]
        count = save_counter_signals(conn, signals)
        assert count == 1

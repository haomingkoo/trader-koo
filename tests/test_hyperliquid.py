"""Tests for Hyperliquid whale tracker."""
from __future__ import annotations

import sqlite3

import pytest

import json

from trader_koo.hyperliquid.tracker import (
    PositionChange,
    WalletPosition,
    WalletSnapshot,
    _diff_positions,
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

    def test_larger_notional_higher_score(self):
        small = generate_counter_signals(self._make_snapshot(size=100.0))
        large = generate_counter_signals(self._make_snapshot(size=10000.0))
        # Larger positions have higher concentration → higher score
        assert large[0]["score"] >= small[0]["score"]


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


# ── Counter Signal Overhaul (v2) ──────────────────────────────


def _make_position(
    coin: str = "ETH",
    side: str = "long",
    size: float = 5000.0,
    leverage: int = 25,
    notional: float = 10_000_000.0,
    upnl: float = -50000.0,
    liq_price: float | None = 1850.0,
    mark_price: float = 2000.0,
) -> WalletPosition:
    return WalletPosition(
        wallet_label="machibro",
        wallet_address="0x020c",
        coin=coin,
        side=side,
        size=size,
        entry_price=2100.0,
        mark_price=mark_price,
        unrealized_pnl=upnl,
        leverage_type="cross",
        leverage_value=leverage,
        notional_usd=notional,
        liquidation_price=liq_price,
    )


def _make_multi_snapshot(position_count: int, **kwargs: Any) -> WalletSnapshot:
    """Create a snapshot with N identical positions (different coins)."""
    coins = [f"COIN{i}" for i in range(position_count)]
    positions = [
        _make_position(coin=c, notional=kwargs.get("notional", 10_000_000.0) / position_count, **{
            k: v for k, v in kwargs.items() if k != "notional"
        })
        for c in coins
    ]
    total_notional = sum(p.notional_usd for p in positions)
    return WalletSnapshot(
        wallet_label="machibro",
        wallet_address="0x020c",
        account_value=500000.0,
        total_margin_used=400000.0,
        margin_ratio=0.8,
        positions=positions,
        timestamp="2026-04-07T00:00:00Z",
    )


from typing import Any


class TestCounterSignalOverhaul:
    """Tests for v2 counter-trade signal enhancements."""

    def test_position_count_discount_many(self):
        """With >8 positions, scores should be halved."""
        # Single position baseline
        single_snap = WalletSnapshot(
            wallet_label="machibro", wallet_address="0x020c",
            account_value=500000.0, total_margin_used=400000.0,
            margin_ratio=0.8,
            positions=[_make_position(leverage=25, notional=10_000_000.0)],
            timestamp="2026-04-07T00:00:00Z",
        )
        single_signals = generate_counter_signals(single_snap)
        single_score = single_signals[0]["score"]

        # Same position but with 10 total
        multi_snap = _make_multi_snapshot(10, leverage=25)
        multi_signals = generate_counter_signals(multi_snap)
        # Each individual score should be lower due to 0.5x multiplier
        any_score = multi_signals[0]["score"]
        assert any_score < single_score

    def test_position_count_boost_few(self):
        """With <=3 positions, scores get 1.5x boost."""
        # 4 positions (no boost)
        snap_4 = _make_multi_snapshot(4, leverage=25)
        sig_4 = generate_counter_signals(snap_4)

        # 2 positions (1.5x boost)
        snap_2 = _make_multi_snapshot(2, leverage=25)
        sig_2 = generate_counter_signals(snap_2)

        # Boosted scores should be higher
        assert sig_2[0]["score"] >= sig_4[0]["score"]

    def test_notional_gate_small_downgraded(self):
        """Small notional position with high score gets LEAN_COUNTER, not COUNTER."""
        # $2M position with high leverage (should score high)
        snap = WalletSnapshot(
            wallet_label="machibro", wallet_address="0x020c",
            account_value=100000.0, total_margin_used=90000.0,
            margin_ratio=0.9,
            positions=[_make_position(
                leverage=25, notional=2_000_000.0, upnl=-100000.0,
                liq_price=1960.0, mark_price=2000.0,
            )],
            timestamp="2026-04-07T00:00:00Z",
        )
        signals = generate_counter_signals(snap)
        assert signals[0]["score"] >= 6  # high score
        assert signals[0]["action"] != "COUNTER"  # but downgraded due to notional
        assert signals[0]["action"] == "LEAN_COUNTER"

    def test_notional_gate_large_stays_counter(self):
        """Large notional position with high score stays COUNTER."""
        snap = WalletSnapshot(
            wallet_label="machibro", wallet_address="0x020c",
            account_value=100000.0, total_margin_used=90000.0,
            margin_ratio=0.9,
            positions=[_make_position(
                leverage=25, notional=10_000_000.0, upnl=-500000.0,
                liq_price=1960.0, mark_price=2000.0,
            )],
            timestamp="2026-04-07T00:00:00Z",
        )
        signals = generate_counter_signals(snap)
        assert signals[0]["score"] >= 6
        assert signals[0]["action"] == "COUNTER"

    def test_concentration_boost_70pct(self):
        """Position with >70% concentration gets +2 instead of +1."""
        # One big position + one tiny
        big = _make_position(coin="ETH", notional=8_000_000.0, leverage=5, upnl=0)
        small = _make_position(coin="BTC", notional=1_000_000.0, leverage=5, upnl=0)
        snap = WalletSnapshot(
            wallet_label="machibro", wallet_address="0x020c",
            account_value=5_000_000.0, total_margin_used=1_000_000.0,
            margin_ratio=0.2,
            positions=[big, small],
            timestamp="2026-04-07T00:00:00Z",
        )
        signals = generate_counter_signals(snap)
        eth_signal = next(s for s in signals if s["coin"] == "ETH")
        assert any("extreme concentration" in r for r in eth_signal["reasons"])

    def test_position_age_no_conn_skips(self):
        """Without conn, position age scoring is skipped (no crash)."""
        snap = WalletSnapshot(
            wallet_label="machibro", wallet_address="0x020c",
            account_value=500000.0, total_margin_used=400000.0,
            margin_ratio=0.8,
            positions=[_make_position()],
            timestamp="2026-04-07T00:00:00Z",
        )
        signals = generate_counter_signals(snap)  # no conn
        assert signals[0]["position_age_hours"] is None
        assert not any("held" in r for r in signals[0]["reasons"])

    def test_position_age_with_conn(self, conn):
        """With snapshot history, position age should add to score."""
        import datetime as dt

        # Seed 5 snapshots over 4 days showing ETH long
        for i in range(5):
            ts = (dt.datetime(2026, 4, 3, tzinfo=dt.timezone.utc) + dt.timedelta(hours=i * 24)).isoformat()
            pos_json = json.dumps([{
                "coin": "ETH", "side": "long", "size": 5000.0,
                "entry_price": 2100.0, "mark_price": 2000.0,
            }])
            conn.execute(
                "INSERT INTO hyperliquid_snapshots (wallet_label, wallet_address, account_value, "
                "total_margin_used, margin_ratio, positions_json, snapshot_ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("machibro", "0x020c", 500000.0, 400000.0, 0.8, pos_json, ts),
            )
        conn.commit()

        snap = WalletSnapshot(
            wallet_label="machibro", wallet_address="0x020c",
            account_value=500000.0, total_margin_used=400000.0,
            margin_ratio=0.8,
            positions=[_make_position()],
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        signals = generate_counter_signals(snap, conn=conn)
        # Position held > 72h → should have age scoring
        assert signals[0]["position_age_hours"] is not None
        assert signals[0]["position_age_hours"] > 72
        assert any("held" in r for r in signals[0]["reasons"])


class TestDiffPartialLiquidation:
    """Tests for partial liquidation detection in _diff_positions."""

    def test_partial_liq_underwater_near_liq(self):
        """Position reduced while underwater and near liq → partial_liq."""
        prev = {
            "ETH": {"side": "long", "size": 5000.0},
        }
        current = [_make_position(
            coin="ETH", side="long", size=2000.0,
            upnl=-100000.0, liq_price=1960.0, mark_price=2000.0,
        )]
        changes = _diff_positions(prev, current)
        assert len(changes) == 1
        assert changes[0].change_type == "partial_liq"
        assert changes[0].size_delta_pct < -5

    def test_partial_close_healthy_pnl(self):
        """Position reduced but healthy PnL → partial_close (not liq)."""
        prev = {
            "ETH": {"side": "long", "size": 5000.0},
        }
        current = [_make_position(
            coin="ETH", side="long", size=2000.0,
            upnl=50000.0, liq_price=1500.0, mark_price=2000.0,
        )]
        changes = _diff_positions(prev, current)
        assert len(changes) == 1
        assert changes[0].change_type == "partial_close"

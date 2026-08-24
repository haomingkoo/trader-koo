"""Tests for Hyperliquid whale tracker."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any

import pytest

from trader_koo.hyperliquid.tracker import (
    PositionChange,
    WalletPosition,
    WalletSnapshot,
    _check_reload,
    _diff_positions,
    _recent_reload_context,
    _send_telegram_signal_alert,
    ensure_hyperliquid_schema,
    fetch_wallet_state,
    generate_counter_signals,
    save_counter_signals,
    save_snapshot,
    seed_default_wallets,
)
from trader_koo.hyperliquid.wallets import get_tracked_wallets


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
        assert "hyperliquid_reload_events" in tables
        assert "hyperliquid_alert_state" in tables

    def test_seed_default_wallets(self, conn):
        seed_default_wallets(conn)
        machi = conn.execute(
            "SELECT label, address FROM hyperliquid_wallets WHERE label = 'machibro'"
        ).fetchone()
        james = conn.execute(
            "SELECT label, address FROM hyperliquid_wallets WHERE label = 'james_wynn'"
        ).fetchone()
        assert machi is not None
        assert "0x020c" in machi[1]
        assert james == ("james_wynn", "0x5078c2fbea2b2ad61bc840bc023e35fce56bedb6")

    def test_seed_default_wallets_honours_env_wallets(self, conn, monkeypatch):
        monkeypatch.setenv(
            "TRADER_KOO_HL_TRACKED_WALLETS",
            "james_wynn=0x1111111111111111111111111111111111111111",
        )
        seed_default_wallets(conn)
        row = conn.execute(
            "SELECT label, address FROM hyperliquid_wallets WHERE label = 'james_wynn'"
        ).fetchone()
        assert row == ("james_wynn", "0x1111111111111111111111111111111111111111")


class TestWalletConfig:
    def test_env_wallet_json_merges_defaults(self, monkeypatch):
        monkeypatch.setenv(
            "TRADER_KOO_HL_TRACKED_WALLETS",
            json.dumps({"james_wynn": "0x1111111111111111111111111111111111111111"}),
        )
        wallets = get_tracked_wallets()
        assert wallets["machibro"].startswith("0x020c")
        assert wallets["james_wynn"] == "0x1111111111111111111111111111111111111111"

    def test_env_wallet_invalid_entries_are_ignored(self, monkeypatch):
        monkeypatch.setenv(
            "TRADER_KOO_HL_TRACKED_WALLETS",
            "bad=not-an-address,good=0x2222222222222222222222222222222222222222",
        )
        wallets = get_tracked_wallets()
        assert "bad" not in wallets
        assert wallets["good"] == "0x2222222222222222222222222222222222222222"

    def test_invalid_json_logs_warning_and_keeps_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "TRADER_KOO_HL_TRACKED_WALLETS",
            '{"broken": "0x1111111111111111111111111111111111111111"',
        )
        wallets = get_tracked_wallets()
        assert wallets["machibro"].startswith("0x020c")
        assert "Invalid TRADER_KOO_HL_TRACKED_WALLETS JSON" in caplog.text


class TestWalletStateParsing:
    def test_derives_mark_price_from_position_value(self, monkeypatch):
        class FakeInfo:
            def user_state(self, address):
                return {
                    "marginSummary": {
                        "accountValue": "1000",
                        "totalMarginUsed": "200",
                    },
                    "assetPositions": [
                        {
                            "position": {
                                "coin": "BTC",
                                "szi": "2",
                                "entryPx": "100",
                                "positionValue": "220",
                                "unrealizedPnl": "20",
                                "leverage": {"type": "cross", "value": "10"},
                                "liquidationPx": "80",
                            }
                        }
                    ],
                }

        monkeypatch.setattr(
            "trader_koo.hyperliquid.tracker._get_info_client",
            lambda: FakeInfo(),
        )

        snapshot = fetch_wallet_state("0xabc", wallet_label="test")

        assert snapshot is not None
        assert snapshot.account_value == 1000.0
        assert snapshot.margin_ratio == 0.2
        assert len(snapshot.positions) == 1
        pos = snapshot.positions[0]
        assert pos.mark_price == 110.0
        assert pos.notional_usd == 220.0
        assert pos.mark_price_source == "positionValue"
        assert pos.notional_source == "positionValue"
        assert pos.data_warnings == ()

    def test_missing_mark_and_position_value_are_reported(self, monkeypatch):
        class FakeInfo:
            def user_state(self, address):
                return {
                    "marginSummary": {
                        "accountValue": "1000",
                        "totalMarginUsed": "200",
                    },
                    "assetPositions": [
                        {
                            "position": {
                                "coin": "ETH",
                                "szi": "3",
                                "entryPx": "100",
                                "unrealizedPnl": "0",
                                "leverage": {"type": "cross", "value": "5"},
                            }
                        }
                    ],
                }

        monkeypatch.setattr(
            "trader_koo.hyperliquid.tracker._get_info_client",
            lambda: FakeInfo(),
        )

        snapshot = fetch_wallet_state("0xabc", wallet_label="test")

        assert snapshot is not None
        pos = snapshot.positions[0]
        assert pos.mark_price == 0.0
        assert pos.notional_usd == 0.0
        assert pos.mark_price_source == "missing"
        assert pos.notional_source == "missing"
        assert pos.data_warnings == ("missing_mark_price", "missing_position_value")


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

    def test_telegram_alert_escapes_external_html(self, conn, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
        posted: dict[str, Any] = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

        def fake_post(url, json, timeout):
            posted["url"] = url
            posted["json"] = json
            posted["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr("httpx.post", fake_post)

        snapshot = WalletSnapshot(
            wallet_label="machi<b>bad</b>",
            wallet_address="0x020c",
            account_value=500000.0,
            total_margin_used=400000.0,
            margin_ratio=0.8,
            positions=[
                WalletPosition(
                    wallet_label="machi<b>bad</b>",
                    wallet_address="0x020c",
                    coin="ETH<script>",
                    side="long",
                    size=5000.0,
                    entry_price=2100.0,
                    mark_price=2000.0,
                    unrealized_pnl=-50000.0,
                    leverage_type="cross",
                    leverage_value=25,
                    notional_usd=10_000_000.0,
                    liquidation_price=1850.0,
                )
            ],
            timestamp="2026-03-24T00:00:00Z",
        )
        _send_telegram_signal_alert(
            conn,
            snapshot,
            [{
                "coin": "ETH<script>",
                "action": "COUNTER",
                "score": "9</b><b>",
                "counter_side": "SHORT",
                "reasons": ["<b>fake</b> & reason"],
            }],
        )

        text = posted["json"]["text"]
        assert "machi&lt;b&gt;bad&lt;/b&gt;" in text
        assert "ETH&lt;script&gt;" in text
        assert "&lt;b&gt;fake&lt;/b&gt; &amp; reason" in text
        assert "9&lt;/b&gt;&lt;b&gt;" in text
        assert "machi<b>bad</b>" not in text
        assert "<b>fake</b>" not in text


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


def _make_alert_snapshot(side: str = "long") -> WalletSnapshot:
    return WalletSnapshot(
        wallet_label="machibro",
        wallet_address="0x020c",
        account_value=500000.0,
        total_margin_used=400000.0,
        margin_ratio=0.8,
        positions=[_make_position(coin="ETH", side=side)],
        timestamp="2026-04-07T00:00:00Z",
    )


def _counter_signal(counter_side: str = "short") -> dict[str, Any]:
    return {
        "coin": "ETH",
        "action": "COUNTER",
        "score": 6,
        "counter_side": counter_side,
        "reasons": ["high leverage", "concentrated position"],
    }


class TestTelegramAlertSemantics:
    @pytest.fixture(autouse=True)
    def _telegram(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
        self.posts: list[dict[str, Any]] = []

        class FakeResponse:
            def raise_for_status(self):
                return None

        def fake_post(url, json, timeout):
            self.posts.append({"url": url, "json": json, "timeout": timeout})
            return FakeResponse()

        monkeypatch.setattr("httpx.post", fake_post)

    def test_position_resize_does_not_page(self, conn):
        resize = PositionChange(
            coin="ETH",
            change_type="increased",
            prev_side="long",
            prev_size=100.0,
            curr_side="long",
            curr_size=160.0,
            size_delta_pct=60.0,
        )

        sent = _send_telegram_signal_alert(conn, _make_alert_snapshot(), [], [resize])

        assert sent is False
        assert self.posts == []

    def test_unchanged_counter_signal_pages_once(self, conn):
        snapshot = _make_alert_snapshot()
        signal = _counter_signal()

        assert _send_telegram_signal_alert(conn, snapshot, [signal]) is True
        changed_metrics = {
            **signal,
            "score": 9,
            "reasons": ["different measurements"],
        }
        assert _send_telegram_signal_alert(conn, snapshot, [changed_metrics]) is False

        assert len(self.posts) == 1
        text = self.posts[0]["json"]["text"]
        assert "Affected positions" in text
        assert "Open positions" not in text

    def test_counter_direction_change_pages_again(self, conn):
        snapshot = _make_alert_snapshot()

        assert _send_telegram_signal_alert(
            conn, snapshot, [_counter_signal("short")]
        ) is True
        assert _send_telegram_signal_alert(
            conn, snapshot, [_counter_signal("long")]
        ) is True

        assert len(self.posts) == 2

    def test_counter_reentry_after_non_counter_state_pages_again(self, conn):
        snapshot = _make_alert_snapshot()
        counter = _counter_signal()
        monitor = {**counter, "action": "MONITOR"}

        assert _send_telegram_signal_alert(conn, snapshot, [counter]) is True
        assert _send_telegram_signal_alert(conn, snapshot, [monitor]) is False
        assert _send_telegram_signal_alert(conn, snapshot, [counter]) is True

        assert len(self.posts) == 2

    def test_lifecycle_event_pages(self, conn):
        opened = PositionChange(
            coin="ETH",
            change_type="new",
            prev_side=None,
            prev_size=None,
            curr_side="long",
            curr_size=5000.0,
            size_delta_pct=None,
        )

        assert _send_telegram_signal_alert(
            conn, _make_alert_snapshot(), [], [opened]
        ) is True
        assert "NEW LONG" in self.posts[0]["json"]["text"]

    def test_signal_state_survives_connection_restart(self, tmp_path):
        db_path = tmp_path / "alerts.db"
        first = sqlite3.connect(db_path)
        ensure_hyperliquid_schema(first)
        assert _send_telegram_signal_alert(
            first, _make_alert_snapshot(), [_counter_signal()]
        ) is True
        first.close()

        second = sqlite3.connect(db_path)
        ensure_hyperliquid_schema(second)
        assert _send_telegram_signal_alert(
            second, _make_alert_snapshot(), [_counter_signal()]
        ) is False
        second.close()

        assert len(self.posts) == 1

    def test_failed_send_does_not_create_in_memory_fallback(self, conn, monkeypatch):
        class FailedResponse:
            def raise_for_status(self):
                raise RuntimeError("telegram unavailable")

        monkeypatch.setattr("httpx.post", lambda *args, **kwargs: FailedResponse())

        assert _send_telegram_signal_alert(
            conn, _make_alert_snapshot(), [_counter_signal()]
        ) is False
        state_count = conn.execute(
            "SELECT COUNT(*) FROM hyperliquid_alert_state"
        ).fetchone()[0]
        assert state_count == 0

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
        """Position under $25M notional with high score gets LEAN_COUNTER, not COUNTER."""
        # $10M position with high leverage (should score high but under $25M gate)
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
        assert signals[0]["score"] >= 6  # high score
        assert signals[0]["action"] != "COUNTER"  # but downgraded due to notional
        assert signals[0]["action"] == "LEAN_COUNTER"

    def test_notional_gate_large_stays_counter(self):
        """Position >= $25M notional with high score stays COUNTER."""
        snap = WalletSnapshot(
            wallet_label="machibro", wallet_address="0x020c",
            account_value=100000.0, total_margin_used=90000.0,
            margin_ratio=0.9,
            positions=[_make_position(
                leverage=25, notional=30_000_000.0, upnl=-1500000.0,
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
        small = _make_position(coin="SOL", notional=1_000_000.0, leverage=5, upnl=0)
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
        """Position reduced while underwater and near critical liq (<2%) → partial_liq."""
        prev = {
            "ETH": {"side": "long", "size": 5000.0},
        }
        current = [_make_position(
            coin="ETH", side="long", size=2000.0,
            upnl=-100000.0, liq_price=1985.0, mark_price=2000.0,  # liq_dist = 0.75%
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


class TestReloadAndCrowdContext:
    def test_reload_event_persists_and_boosts_signals(self, conn):
        conn.execute(
            """
            INSERT INTO hyperliquid_snapshots
                (wallet_label, wallet_address, account_value, total_margin_used,
                 margin_ratio, positions_json, snapshot_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("machibro", "0x020c", 50.0, 0.0, 0.0, "[]", "2026-04-07T00:00:00+00:00"),
        )
        conn.commit()

        snapshot = WalletSnapshot(
            wallet_label="machibro",
            wallet_address="0x020c",
            account_value=250000.0,
            total_margin_used=120000.0,
            margin_ratio=0.48,
            positions=[_make_position(coin="ETH", notional=30_000_000.0)],
            timestamp="2026-04-07T04:00:00+00:00",
        )

        save_snapshot(conn, snapshot)
        _check_reload(conn, snapshot, "machibro")

        row = conn.execute(
            "SELECT wallet_label, position_count FROM hyperliquid_reload_events ORDER BY detected_ts DESC LIMIT 1"
        ).fetchone()
        assert row == ("machibro", 1)

        recent = _recent_reload_context(conn, "machibro", snapshot.timestamp)
        assert recent is not None
        assert recent["score_boost"] == 3

        signals = generate_counter_signals(snapshot, conn=conn)
        assert any("post-reload" in reason for reason in signals[0]["reasons"])
        assert signals[0]["wallet_context"]["recent_reload"] is not None

    def test_crowded_longs_boost_counter_short(self, conn):
        from trader_koo.crypto.derivatives import ensure_derivatives_schema

        ensure_derivatives_schema(conn)
        now_ts = "2026-04-07T08:00:00+00:00"
        conn.execute(
            """
            INSERT INTO crypto_funding_rates
                (symbol, funding_rate, funding_time, mark_price, snapshot_ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("ETH-USD", 0.0006, now_ts, 2100.0, now_ts),
        )
        conn.execute(
            """
            INSERT INTO crypto_long_short_ratio
                (symbol, long_account, short_account, long_short_ratio, timestamp, snapshot_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ETH-USD", 0.71, 0.29, 2.45, now_ts, now_ts),
        )
        conn.commit()

        snapshot = WalletSnapshot(
            wallet_label="machibro",
            wallet_address="0x020c",
            account_value=500000.0,
            total_margin_used=400000.0,
            margin_ratio=0.8,
            positions=[_make_position(coin="ETH", side="long", notional=30_000_000.0)],
            timestamp=now_ts,
        )

        signals = generate_counter_signals(snapshot, conn=conn)
        signal = signals[0]
        assert signal["market_context"] is not None
        assert signal["market_context"]["aligns_with_counter"] is True
        assert signal["market_context"]["score_boost"] == 2
        assert any("funding" in reason.lower() or "binance top traders" in reason.lower() for reason in signal["reasons"])


class TestWalletSeedUpdates:
    def test_seed_default_wallets_updates_existing_address(self, conn, monkeypatch):
        conn.execute(
            """
            INSERT INTO hyperliquid_wallets (label, address, track_mode, notes, active)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("james_wynn", "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "counter", "existing", 1),
        )
        conn.commit()

        monkeypatch.setenv(
            "TRADER_KOO_HL_TRACKED_WALLETS",
            json.dumps({"james_wynn": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}),
        )
        seed_default_wallets(conn)

        row = conn.execute(
            "SELECT address, track_mode, notes, active FROM hyperliquid_wallets WHERE label = 'james_wynn'"
        ).fetchone()
        assert row == ("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "counter", "existing", 1)

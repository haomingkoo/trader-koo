"""Hyperliquid whale tracker - monitor and counter-trade tracked wallets.

Polls tracked wallets for position changes. Stores snapshots in SQLite
for historical analysis. Generates counter-trade signals when tracked
traders open large positions (inverse direction, scaled by their size).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

LOG = logging.getLogger(__name__)

# Default tracked wallets
TRACKED_WALLETS: dict[str, str] = {
    "machibro": "0x020ca66c30bec2c4fe3861a94e4db4a498a35872",
}


@dataclass(frozen=True)
class WalletPosition:
    """A single position for a tracked wallet."""

    wallet_label: str
    wallet_address: str
    coin: str
    side: str  # "long" | "short"
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage_type: str  # "cross" | "isolated"
    leverage_value: int
    notional_usd: float
    liquidation_price: float | None


@dataclass(frozen=True)
class WalletSnapshot:
    """Full account snapshot for a tracked wallet."""

    wallet_label: str
    wallet_address: str
    account_value: float
    total_margin_used: float
    margin_ratio: float  # margin_used / account_value (>1.0 = danger)
    positions: list[WalletPosition]
    timestamp: str


def _get_info_client():
    """Lazy import to avoid import errors if SDK not installed."""
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    return Info(constants.MAINNET_API_URL, skip_ws=True)


def fetch_wallet_state(
    wallet_address: str,
    wallet_label: str = "",
) -> WalletSnapshot | None:
    """Fetch current positions and account state for a wallet."""
    try:
        info = _get_info_client()
        state = info.user_state(wallet_address)

        margin = state.get("marginSummary", {})
        account_value = float(margin.get("accountValue", 0))
        total_margin = float(margin.get("totalMarginUsed", 0))
        margin_ratio = total_margin / account_value if account_value > 0 else 0

        positions: list[WalletPosition] = []
        for asset_pos in state.get("assetPositions", []):
            p = asset_pos.get("position", {})
            sz = float(p.get("szi", 0))
            if sz == 0:
                continue

            lev = p.get("leverage", {})
            entry_px = float(p.get("entryPx", 0))
            mark_px = float(p.get("markPx") or p.get("entryPx", 0))
            liq_px = float(p.get("liquidationPx")) if p.get("liquidationPx") else None

            positions.append(WalletPosition(
                wallet_label=wallet_label,
                wallet_address=wallet_address,
                coin=p.get("coin", ""),
                side="long" if sz > 0 else "short",
                size=abs(sz),
                entry_price=entry_px,
                mark_price=mark_px,
                unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                leverage_type=str(lev.get("type", "cross")),
                leverage_value=int(lev.get("value", 1)),
                notional_usd=abs(sz) * mark_px,
                liquidation_price=liq_px,
            ))

        return WalletSnapshot(
            wallet_label=wallet_label,
            wallet_address=wallet_address,
            account_value=account_value,
            total_margin_used=total_margin,
            margin_ratio=round(margin_ratio, 4),
            positions=positions,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
    except Exception as exc:
        LOG.warning("Failed to fetch wallet state for %s (%s): %s", wallet_label, wallet_address, exc)
        return None


def fetch_wallet_fills(
    wallet_address: str,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Fetch recent trade fills for a wallet (up to 2000 from API)."""
    try:
        info = _get_info_client()
        fills = info.user_fills(wallet_address)
        return fills[:limit]
    except Exception as exc:
        LOG.warning("Failed to fetch fills for %s: %s", wallet_address, exc)
        return []


def fetch_wallet_open_orders(
    wallet_address: str,
) -> list[dict[str, Any]]:
    """Fetch open/pending orders for a wallet."""
    try:
        info = _get_info_client()
        return info.open_orders(wallet_address)
    except Exception as exc:
        LOG.warning("Failed to fetch open orders for %s: %s", wallet_address, exc)
        return []


def fetch_wallet_history(
    wallet_address: str,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Fetch full trade history and compute performance stats.

    Returns historical fills with aggregated PnL, win rate, and per-coin breakdown.
    Uses user_fills_by_time for deeper history (up to 10K fills).
    """
    import time as _time

    try:
        info = _get_info_client()
        start_ms = int((_time.time() - lookback_days * 86400) * 1000)
        fills = info.user_fills_by_time(wallet_address, start_ms)

        if not fills:
            return {"fills": [], "stats": {}, "by_coin": {}}

        total_pnl = sum(float(f.get("closedPnl", 0)) for f in fills)
        fees = sum(float(f.get("fee", 0)) for f in fills)
        wins = sum(1 for f in fills if float(f.get("closedPnl", 0)) > 0)
        losses = sum(1 for f in fills if float(f.get("closedPnl", 0)) < 0)
        liqs = sum(1 for f in fills if f.get("liquidation"))

        # Per-coin breakdown
        by_coin: dict[str, dict[str, float | int]] = {}
        for f in fills:
            coin = f.get("coin", "?")
            pnl = float(f.get("closedPnl", 0))
            if coin not in by_coin:
                by_coin[coin] = {"pnl": 0.0, "fills": 0, "wins": 0, "losses": 0}
            by_coin[coin]["pnl"] += pnl
            by_coin[coin]["fills"] += 1
            if pnl > 0:
                by_coin[coin]["wins"] += 1
            elif pnl < 0:
                by_coin[coin]["losses"] += 1

        return {
            "fill_count": len(fills),
            "lookback_days": lookback_days,
            "stats": {
                "total_pnl": round(total_pnl, 2),
                "total_fees": round(fees, 2),
                "net_pnl": round(total_pnl - fees, 2),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
                "liquidations": liqs,
            },
            "by_coin": {
                coin: {
                    "pnl": round(data["pnl"], 2),
                    "fills": data["fills"],
                    "win_rate_pct": round(
                        data["wins"] / (data["wins"] + data["losses"]) * 100, 1
                    ) if (data["wins"] + data["losses"]) > 0 else 0,
                }
                for coin, data in sorted(by_coin.items(), key=lambda x: x[1]["pnl"])
            },
        }
    except Exception as exc:
        LOG.warning("Failed to fetch wallet history for %s: %s", wallet_address, exc)
        return {"fills": [], "stats": {}, "by_coin": {}}


def generate_counter_signals(snapshot: WalletSnapshot) -> list[dict[str, Any]]:
    """Generate counter-trade signals using backtested ETH-normalized logic.

    Strategy: counter-trade when position notional exceeds ETH-normalized
    threshold (ref $10M at ETH $4400). Confidence scales with how far
    above threshold + leverage + liquidation proximity.

    Based on 8-month backtest: 22 trades, 63.6% WR, +16.6% return.
    """
    # ETH-normalized threshold: $10M ref at ETH $4400
    # Current ETH price approximated from snapshot ETH position or default
    REF_NOTIONAL = 10_000_000
    REF_ETH = 4400
    eth_price = 2100  # default fallback
    for pos in snapshot.positions:
        if pos.coin == "ETH" and pos.mark_price > 0:
            eth_price = pos.mark_price
            break
    counter_threshold = REF_NOTIONAL * (eth_price / REF_ETH)

    # Account leverage = total notional / account value
    total_notional = sum(p.notional_usd for p in snapshot.positions)
    account_leverage = total_notional / snapshot.account_value if snapshot.account_value > 0 else 0

    signals: list[dict[str, Any]] = []
    for pos in snapshot.positions:
        counter_side = "short" if pos.side == "long" else "long"

        # Is this position above the ETH-normalized counter threshold?
        above_threshold = pos.notional_usd >= counter_threshold
        threshold_ratio = pos.notional_usd / counter_threshold if counter_threshold > 0 else 0

        # Confidence: base 30, scale up based on signals
        confidence = 30.0

        # +25 if above ETH-normalized threshold (the primary signal)
        if above_threshold:
            confidence += 25 + min(threshold_ratio - 1, 2) * 10  # up to +45

        # +15 if account leverage > 10x (overextended)
        if account_leverage > 10:
            confidence += min((account_leverage - 10) / 10, 1.0) * 15

        # +10 if high individual leverage
        lev_factor = min(pos.leverage_value / 25.0, 1.0)
        confidence += lev_factor * 10

        # Proximity to liquidation boosts confidence
        liq_distance_pct = None
        if pos.liquidation_price and pos.mark_price > 0:
            if pos.side == "long":
                liq_distance_pct = (pos.mark_price - pos.liquidation_price) / pos.mark_price * 100
            else:
                liq_distance_pct = (pos.liquidation_price - pos.mark_price) / pos.mark_price * 100
            if liq_distance_pct < 5:
                confidence += 15

        confidence = round(min(95, max(30, confidence)), 1)

        signals.append({
            "source": "hyperliquid_counter",
            "wallet_label": pos.wallet_label,
            "coin": pos.coin,
            "counter_side": counter_side,
            "their_side": pos.side,
            "their_size": pos.size,
            "their_leverage": pos.leverage_value,
            "their_notional_usd": round(pos.notional_usd, 2),
            "their_entry_price": pos.entry_price,
            "their_unrealized_pnl": round(pos.unrealized_pnl, 2),
            "their_liq_distance_pct": round(liq_distance_pct, 2) if liq_distance_pct else None,
            "confidence": confidence,
            "reasoning": (
                f"Counter-trade {pos.wallet_label}: {pos.side} {pos.coin} "
                f"{pos.size} @ ${pos.entry_price:.2f} ({pos.leverage_value}x lev, "
                f"uPnL ${pos.unrealized_pnl:+,.0f})"
            ),
            "timestamp": snapshot.timestamp,
        })

    return signals


# ---------------------------------------------------------------------------
# Database persistence
# ---------------------------------------------------------------------------

def ensure_hyperliquid_schema(conn: sqlite3.Connection) -> None:
    """Create tables for Hyperliquid tracking."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hyperliquid_wallets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            address TEXT NOT NULL,
            track_mode TEXT NOT NULL DEFAULT 'counter',
            notes TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hyperliquid_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_label TEXT NOT NULL,
            wallet_address TEXT NOT NULL,
            account_value REAL,
            total_margin_used REAL,
            margin_ratio REAL,
            positions_json TEXT,
            snapshot_ts TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hl_snapshots_wallet_ts "
        "ON hyperliquid_snapshots(wallet_label, snapshot_ts DESC)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hyperliquid_counter_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_label TEXT NOT NULL,
            coin TEXT NOT NULL,
            counter_side TEXT NOT NULL,
            their_side TEXT NOT NULL,
            their_size REAL,
            their_leverage INTEGER,
            their_notional_usd REAL,
            confidence REAL,
            reasoning TEXT,
            signal_ts TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hl_signals_coin_ts "
        "ON hyperliquid_counter_signals(coin, signal_ts DESC)"
    )
    conn.commit()


def save_snapshot(conn: sqlite3.Connection, snapshot: WalletSnapshot) -> None:
    """Persist a wallet snapshot to the database."""
    ensure_hyperliquid_schema(conn)
    positions_json = json.dumps([asdict(p) for p in snapshot.positions])
    conn.execute(
        """
        INSERT INTO hyperliquid_snapshots
            (wallet_label, wallet_address, account_value, total_margin_used,
             margin_ratio, positions_json, snapshot_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot.wallet_label, snapshot.wallet_address,
            snapshot.account_value, snapshot.total_margin_used,
            snapshot.margin_ratio, positions_json, snapshot.timestamp,
        ),
    )
    conn.commit()


def save_counter_signals(
    conn: sqlite3.Connection,
    signals: list[dict[str, Any]],
) -> int:
    """Persist counter-trade signals to the database."""
    ensure_hyperliquid_schema(conn)
    inserted = 0
    for sig in signals:
        conn.execute(
            """
            INSERT INTO hyperliquid_counter_signals
                (wallet_label, coin, counter_side, their_side, their_size,
                 their_leverage, their_notional_usd, confidence, reasoning, signal_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig["wallet_label"], sig["coin"], sig["counter_side"],
                sig["their_side"], sig["their_size"], sig["their_leverage"],
                sig["their_notional_usd"], sig["confidence"],
                sig["reasoning"], sig["timestamp"],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


def seed_default_wallets(conn: sqlite3.Connection) -> None:
    """Insert default tracked wallets if not already present."""
    ensure_hyperliquid_schema(conn)
    for label, address in TRACKED_WALLETS.items():
        conn.execute(
            "INSERT OR IGNORE INTO hyperliquid_wallets (label, address) VALUES (?, ?)",
            (label, address),
        )
    conn.commit()


def poll_all_wallets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Poll all active tracked wallets, save snapshots, generate signals."""
    ensure_hyperliquid_schema(conn)
    seed_default_wallets(conn)

    wallets = conn.execute(
        "SELECT label, address, track_mode FROM hyperliquid_wallets WHERE active = 1"
    ).fetchall()

    all_signals: list[dict[str, Any]] = []
    for label, address, track_mode in wallets:
        snapshot = fetch_wallet_state(address, wallet_label=label)
        if not snapshot:
            continue

        save_snapshot(conn, snapshot)
        LOG.info(
            "HL snapshot: %s | $%,.0f acct | %d positions | margin ratio %.2f",
            label, snapshot.account_value, len(snapshot.positions), snapshot.margin_ratio,
        )

        if track_mode == "counter" and snapshot.positions:
            signals = generate_counter_signals(snapshot)
            saved = save_counter_signals(conn, signals)
            all_signals.extend(signals)
            LOG.info("HL counter signals: %d generated for %s", saved, label)

        # Send Telegram alert for notable positions
        _send_telegram_whale_alert(snapshot)

    return all_signals


def _send_telegram_whale_alert(snapshot: WalletSnapshot) -> None:
    """Send Telegram alert if whale has notable positions.

    Includes counter-trade signal when ETH-normalized threshold is exceeded.
    """
    import os

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    if not snapshot.positions:
        return

    # ETH-normalized threshold
    REF_NOTIONAL = 10_000_000
    REF_ETH = 4400
    eth_price = 2100
    for pos in snapshot.positions:
        if pos.coin == "ETH" and pos.mark_price > 0:
            eth_price = pos.mark_price
            break
    counter_threshold = REF_NOTIONAL * (eth_price / REF_ETH)
    total_notional = sum(p.notional_usd for p in snapshot.positions)
    acct_leverage = total_notional / snapshot.account_value if snapshot.account_value > 0 else 0

    # Check if any position is in counter zone
    has_counter_signal = any(p.notional_usd >= counter_threshold for p in snapshot.positions)

    lines = [f"{'🔴 COUNTER SIGNAL' if has_counter_signal else '🐋'} <b>{snapshot.wallet_label}</b> — ${snapshot.account_value:,.0f}"]
    lines.append(f"Leverage: {acct_leverage:.0f}x | Margin: {snapshot.margin_ratio:.0%}")
    if has_counter_signal:
        lines.append(f"Threshold: ${counter_threshold:,.0f} (ETH ${eth_price:,.0f})")

    for pos in snapshot.positions:
        in_zone = pos.notional_usd >= counter_threshold
        emoji = "🎯" if in_zone else ("🟢" if pos.side == "long" else "🔴")
        counter_side = "SHORT" if pos.side == "long" else "LONG"
        pnl_emoji = "✅" if pos.unrealized_pnl > 0 else "❌"
        lines.append(
            f"{emoji} {pos.coin} {pos.side.upper()} {pos.size:,.1f} "
            f"@ ${pos.entry_price:,.2f} ({pos.leverage_value}x)"
        )
        lines.append(
            f"   {pnl_emoji} uPnL: ${pos.unrealized_pnl:+,.0f} "
            f"| ${pos.notional_usd:,.0f} notional"
        )
        if in_zone:
            lines.append(f"   ➡️ Counter: {counter_side} {pos.coin} ({pos.notional_usd/counter_threshold:.1f}x threshold)")
        if pos.liquidation_price:
            liq_dist = abs(pos.mark_price - pos.liquidation_price) / pos.mark_price * 100
            if liq_dist < 10:
                lines.append(f"   ⚠️ Liq: ${pos.liquidation_price:,.2f} ({liq_dist:.1f}% away)")

    if has_counter_signal:
        lines.append("")
        lines.append("Research signal only. Not financial advice.")

    text = "\n".join(lines)

    try:
        import httpx

        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        LOG.debug("Telegram whale alert failed: %s", exc)

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


def _estimate_position_age_hours(
    conn: sqlite3.Connection | None,
    wallet_label: str,
    coin: str,
    current_side: str,
) -> float | None:
    """Estimate how long a position has been open by scanning snapshot history.

    Walks backward through snapshots to find the earliest consecutive one
    where this coin+side appears. Returns hours since that snapshot, or None.
    """
    if conn is None:
        return None
    rows = conn.execute(
        """
        SELECT positions_json, snapshot_ts
        FROM hyperliquid_snapshots
        WHERE wallet_label = ?
        ORDER BY snapshot_ts DESC
        LIMIT 100
        """,
        (wallet_label,),
    ).fetchall()
    if not rows:
        return None

    earliest_ts: str | None = None
    for positions_json, snapshot_ts in rows:
        if not positions_json:
            break
        try:
            positions = json.loads(positions_json)
        except (json.JSONDecodeError, TypeError):
            break
        found = any(
            p.get("coin") == coin and p.get("side") == current_side
            for p in positions
        )
        if found:
            earliest_ts = snapshot_ts
        else:
            break  # position wasn't present in this older snapshot

    if earliest_ts is None:
        return None
    try:
        earliest_dt = dt.datetime.fromisoformat(earliest_ts.replace("Z", "+00:00"))
        now = dt.datetime.now(dt.timezone.utc)
        return max(0, (now - earliest_dt).total_seconds() / 3600)
    except (ValueError, TypeError):
        return None


# Minimum notional to promote a signal to COUNTER.
# Study (595K fills, Jul 2025–Apr 2026): he wins 71-87% on <$15M positions.
# Only >$25M positions show counter-trade edge (33% WR, -$34M total PnL).
_MIN_COUNTER_NOTIONAL_USD = 25_000_000

# Coins where he consistently wins — do NOT counter-trade these.
# BTC: 94.7% WR over 19 cycles (+$489K). He's skilled at BTC.
_SKIP_COUNTER_COINS: frozenset[str] = frozenset({"BTC"})


def generate_counter_signals(
    snapshot: WalletSnapshot,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Generate counter-trade signals using expert panel validated logic.

    Scoring system from ML expert + quant + critic panel analysis
    of 575K fills over 216 trading days.

    Enhancements (v2):
    - Position count discount: >8 positions halves scores, <=3 boosts 1.5x
    - Position age: extended holds (+1 >24h, +2 >72h) — his worst win rate
    - Notional gate: only COUNTER for positions >$5M
    - Concentration boost: >70% in one position doubles concentration score

    Win rate is ~51% daily (coin flip). Edge comes from payoff
    asymmetry — his losses are much bigger than his wins.
    Top 10 loss days = 105% of all returns.
    """
    total_notional = sum(p.notional_usd for p in snapshot.positions)
    account_leverage = total_notional / snapshot.account_value if snapshot.account_value > 0 else 0
    position_count = len(snapshot.positions)

    # Position count multiplier: concentrated bets = stronger signal
    if position_count > 8:
        count_multiplier = 0.5
    elif position_count <= 3:
        count_multiplier = 1.5
    else:
        count_multiplier = 1.0

    signals: list[dict[str, Any]] = []
    for pos in snapshot.positions:
        counter_side = "short" if pos.side == "long" else "long"

        # Skip coins where he consistently wins (study-validated)
        if pos.coin in _SKIP_COUNTER_COINS:
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
                "their_liq_distance_pct": None,
                "confidence": 30.0,
                "score": 0,
                "action": "SKIP",
                "reasons": [f"{pos.coin} on skip list (high WR)"],
                "position_age_hours": None,
                "position_count": position_count,
                "reasoning": f"[SKIP] {pos.coin} excluded — he wins this coin",
                "timestamp": snapshot.timestamp,
            })
            continue

        # Scoring system (expert panel validated + v2 enhancements)
        score = 0
        reasons: list[str] = []

        # Account leverage > 10x (overextended)
        if account_leverage > 20:
            score += 3
            reasons.append(f"extreme leverage {account_leverage:.0f}x")
        elif account_leverage > 10:
            score += 2
            reasons.append(f"high leverage {account_leverage:.0f}x")

        # Position concentration: enhanced with 70% extreme tier
        if total_notional > 0:
            concentration_pct = pos.notional_usd / total_notional * 100
            if concentration_pct > 70:
                score += 2
                reasons.append(f"extreme concentration {concentration_pct:.0f}% in {pos.coin}")
            elif concentration_pct > 50:
                score += 1
                reasons.append(f"concentrated {concentration_pct:.0f}% in {pos.coin}")

        # Liquidation proximity
        liq_distance_pct = None
        if pos.liquidation_price and pos.mark_price > 0:
            if pos.side == "long":
                liq_distance_pct = (pos.mark_price - pos.liquidation_price) / pos.mark_price * 100
            else:
                liq_distance_pct = (pos.liquidation_price - pos.mark_price) / pos.mark_price * 100
            if liq_distance_pct < 2:
                score += 3
                reasons.append(f"liq {liq_distance_pct:.1f}% away (critical)")
            elif liq_distance_pct < 5:
                score += 2
                reasons.append(f"liq {liq_distance_pct:.1f}% away")
            elif liq_distance_pct < 10:
                score += 1
                reasons.append(f"liq {liq_distance_pct:.1f}% away")

        # High individual leverage
        if pos.leverage_value >= 25:
            score += 2
            reasons.append(f"{pos.leverage_value}x leverage")
        elif pos.leverage_value >= 10:
            score += 1
            reasons.append(f"{pos.leverage_value}x leverage")

        # Position is underwater (unrealized loss)
        if pos.unrealized_pnl < 0:
            loss_pct = abs(pos.unrealized_pnl) / pos.notional_usd * 100 if pos.notional_usd > 0 else 0
            if loss_pct > 5:
                score += 2
                reasons.append(f"underwater {loss_pct:.1f}%")
            elif loss_pct > 1:
                score += 1
                reasons.append(f"underwater {loss_pct:.1f}%")

        # Position age: extended holds have his worst win rate
        age_hours: float | None = None
        if conn is not None:
            age_hours = _estimate_position_age_hours(
                conn, snapshot.wallet_label, pos.coin, pos.side,
            )
            if age_hours is not None:
                if age_hours > 72:
                    score += 2
                    reasons.append(f"held {age_hours / 24:.0f}d (stubborn)")
                elif age_hours > 24:
                    score += 1
                    reasons.append(f"held {age_hours / 24:.0f}d")

        # Apply position count multiplier
        score = round(score * count_multiplier)

        # Convert score to confidence (30-95)
        confidence = round(min(95, max(30, 30 + score * 8)), 1)

        # Action based on score + notional gate
        if score >= 6 and pos.notional_usd >= _MIN_COUNTER_NOTIONAL_USD:
            action = "COUNTER"
        elif score >= 6:
            # High score but small position — downgrade
            action = "LEAN_COUNTER"
            reasons.append(f"notional ${pos.notional_usd / 1e6:.1f}M < $25M gate")
        elif score >= 3:
            action = "LEAN_COUNTER"
        else:
            action = "MONITOR"

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
            "score": score,
            "action": action,
            "reasons": reasons,
            "position_age_hours": round(age_hours, 1) if age_hours is not None else None,
            "position_count": position_count,
            "reasoning": (
                f"[{action}] score={score} ({position_count} pos) | {', '.join(reasons[:4])}"
                if reasons else f"[{action}] score={score}"
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


def _load_previous_positions(
    conn: sqlite3.Connection,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Load positions from the previous snapshot keyed by coin."""
    row = conn.execute(
        """
        SELECT positions_json FROM hyperliquid_snapshots
        WHERE wallet_label = ? ORDER BY snapshot_ts DESC LIMIT 1
        """,
        (label,),
    ).fetchone()
    if not row or not row[0]:
        return {}
    try:
        return {p["coin"]: p for p in json.loads(row[0])}
    except (json.JSONDecodeError, KeyError):
        return {}


@dataclass(frozen=True)
class PositionChange:
    """Describes how a position changed between two snapshots."""

    coin: str
    change_type: str  # "new" | "closed" | "partial_close" | "partial_liq" | "increased" | "flipped" | "unchanged"
    prev_side: str | None
    prev_size: float | None
    curr_side: str | None
    curr_size: float | None
    size_delta_pct: float | None  # % change in absolute size


def _diff_positions(
    prev: dict[str, dict[str, Any]],
    current: list[WalletPosition],
) -> list[PositionChange]:
    """Compare previous and current positions to detect changes."""
    changes: list[PositionChange] = []
    seen_coins: set[str] = set()

    for pos in current:
        seen_coins.add(pos.coin)
        old = prev.get(pos.coin)
        if old is None:
            changes.append(PositionChange(
                coin=pos.coin, change_type="new",
                prev_side=None, prev_size=None,
                curr_side=pos.side, curr_size=pos.size,
                size_delta_pct=None,
            ))
            continue

        old_side = old.get("side", "")
        old_size = float(old.get("size", 0))

        if old_side != pos.side:
            changes.append(PositionChange(
                coin=pos.coin, change_type="flipped",
                prev_side=old_side, prev_size=old_size,
                curr_side=pos.side, curr_size=pos.size,
                size_delta_pct=None,
            ))
        elif old_size > 0:
            delta_pct = (pos.size - old_size) / old_size * 100
            if delta_pct < -5:
                # Distinguish partial liquidation from voluntary close:
                # partial liq = underwater + close to liquidation price
                change = "partial_close"
                if pos.unrealized_pnl < 0 and pos.liquidation_price:
                    if pos.side == "long":
                        liq_dist = (pos.mark_price - pos.liquidation_price) / pos.mark_price * 100 if pos.mark_price > 0 else 100
                    else:
                        liq_dist = (pos.liquidation_price - pos.mark_price) / pos.mark_price * 100 if pos.mark_price > 0 else 100
                    if liq_dist < 5:
                        change = "partial_liq"
                changes.append(PositionChange(
                    coin=pos.coin, change_type=change,
                    prev_side=old_side, prev_size=old_size,
                    curr_side=pos.side, curr_size=pos.size,
                    size_delta_pct=round(delta_pct, 1),
                ))
            elif delta_pct > 5:
                changes.append(PositionChange(
                    coin=pos.coin, change_type="increased",
                    prev_side=old_side, prev_size=old_size,
                    curr_side=pos.side, curr_size=pos.size,
                    size_delta_pct=round(delta_pct, 1),
                ))
            # else: unchanged (within 5% noise band)

    # Coins that were in previous but not in current = fully closed
    for coin, old in prev.items():
        if coin not in seen_coins:
            changes.append(PositionChange(
                coin=coin, change_type="closed",
                prev_side=old.get("side"), prev_size=float(old.get("size", 0)),
                curr_side=None, curr_size=None,
                size_delta_pct=None,
            ))

    return changes


def poll_all_wallets(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Poll all active tracked wallets, save snapshots, generate signals."""
    ensure_hyperliquid_schema(conn)
    seed_default_wallets(conn)

    wallets = conn.execute(
        "SELECT label, address, track_mode FROM hyperliquid_wallets WHERE active = 1"
    ).fetchall()

    all_signals: list[dict[str, Any]] = []
    for label, address, track_mode in wallets:
        # Load previous positions BEFORE saving new snapshot
        prev_positions = _load_previous_positions(conn, label)

        snapshot = fetch_wallet_state(address, wallet_label=label)
        if not snapshot:
            continue

        save_snapshot(conn, snapshot)
        LOG.info(
            "HL snapshot: %s | $%,.0f acct | %d positions | margin ratio %.2f",
            label, snapshot.account_value, len(snapshot.positions), snapshot.margin_ratio,
        )

        # Detect liquidation: account was active, now empty
        _check_liquidation(conn, snapshot, label)

        # Detect reload: account was empty, now has positions again
        _check_reload(conn, snapshot, label)

        signals: list[dict[str, Any]] = []
        if track_mode == "counter" and snapshot.positions:
            signals = generate_counter_signals(snapshot, conn=conn)
            saved = save_counter_signals(conn, signals)
            all_signals.extend(signals)
            LOG.info("HL counter signals: %d generated for %s", saved, label)

        # Diff positions and only alert on meaningful changes
        changes = _diff_positions(prev_positions, snapshot.positions)
        _send_telegram_signal_alert(snapshot, signals, changes)

    return all_signals


def _check_liquidation(
    conn: sqlite3.Connection,
    snapshot: WalletSnapshot,
    label: str,
) -> None:
    """Detect liquidation by comparing current snapshot to previous.

    If the previous snapshot had positions and account value > $1000,
    but now the account is empty ($0 or near-zero with no positions),
    that's a liquidation event.
    """
    if snapshot.account_value > 100 or snapshot.positions:
        return  # Account still active, not liquidated

    # Check the previous snapshot
    row = conn.execute(
        """
        SELECT account_value, positions_json, snapshot_ts
        FROM hyperliquid_snapshots
        WHERE wallet_label = ?
        ORDER BY snapshot_ts DESC
        LIMIT 1 OFFSET 1
        """,
        (label,),
    ).fetchone()

    if not row:
        return  # No previous snapshot to compare

    prev_value, prev_positions_json, prev_ts = row
    prev_positions = json.loads(prev_positions_json) if prev_positions_json else []

    if float(prev_value) < 1000 or not prev_positions:
        return  # Previous snapshot was already empty

    # This looks like a liquidation
    LOG.warning(
        "LIQUIDATION DETECTED: %s went from $%,.0f (%d positions) to $%,.0f (0 positions)",
        label, float(prev_value), len(prev_positions), snapshot.account_value,
    )
    _send_telegram_liquidation_alert(label, float(prev_value), prev_positions, prev_ts)


def _check_reload(
    conn: sqlite3.Connection,
    snapshot: WalletSnapshot,
    label: str,
) -> None:
    """Detect when a wallet reloads after being empty."""
    if snapshot.account_value < 100 or not snapshot.positions:
        return  # Still empty

    row = conn.execute(
        """
        SELECT account_value, positions_json
        FROM hyperliquid_snapshots
        WHERE wallet_label = ?
        ORDER BY snapshot_ts DESC
        LIMIT 1 OFFSET 1
        """,
        (label,),
    ).fetchone()

    if not row:
        return

    prev_value, prev_positions_json = row
    prev_positions = json.loads(prev_positions_json) if prev_positions_json else []

    if float(prev_value) > 100 or prev_positions:
        return  # Previous snapshot was active, not a reload

    LOG.info("RELOAD DETECTED: %s back with $%,.0f and %d positions",
             label, snapshot.account_value, len(snapshot.positions))

    _send_telegram_reload_alert(snapshot)


def _send_telegram_reload_alert(snapshot: WalletSnapshot) -> None:
    """Send Telegram alert when a tracked wallet reloads after being empty."""
    import os

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    total_notional = sum(p.notional_usd for p in snapshot.positions)
    leverage = total_notional / snapshot.account_value if snapshot.account_value > 0 else 0

    lines = [
        f"<b>RELOAD: {snapshot.wallet_label}</b>",
        f"Back with ${snapshot.account_value:,.0f} | {leverage:.0f}x leverage",
        "",
    ]

    for p in snapshot.positions:
        lines.append(f"  {p.coin} {p.side.upper()} ${p.notional_usd:,.0f} at {p.leverage_value}x")

    lines.append("")
    lines.append("Watch for new counter signals.")

    text = "\n".join(lines)

    try:
        import httpx

        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        LOG.debug("Telegram reload alert failed: %s", exc)


def _send_telegram_liquidation_alert(
    label: str,
    prev_value: float,
    prev_positions: list[dict],
    prev_ts: str,
) -> None:
    """Send Telegram alert when a tracked wallet gets liquidated."""
    import os

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    lines = [
        f"<b>LIQUIDATED: {label}</b>",
        f"Account went from ${prev_value:,.0f} to $0",
        f"Last seen: {prev_ts}",
        "",
    ]

    for p in prev_positions[:5]:
        coin = p.get("coin", "?")
        side = p.get("side", "?").upper()
        notional = float(p.get("notional_usd", 0))
        leverage = p.get("leverage_value", "?")
        liq_px = p.get("liquidation_price")
        entry_px = float(p.get("entry_price", 0))
        lines.append(f"  {coin} {side} ${notional:,.0f} at {leverage}x (entry ${entry_px:,.2f})")
        if liq_px:
            lines.append(f"  Liq price was ${float(liq_px):,.2f}")

    lines.append("")
    lines.append("Counter-trade strategy validated.")

    text = "\n".join(lines)

    try:
        import httpx

        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        LOG.info("Liquidation alert sent for %s", label)
    except Exception as exc:
        LOG.debug("Telegram liquidation alert failed: %s", exc)


def _send_telegram_signal_alert(
    snapshot: WalletSnapshot,
    signals: list[dict[str, Any]],
    changes: list[PositionChange] | None = None,
) -> None:
    """Send Telegram alert only when positions meaningfully change.

    Alert triggers:
    - Position opened, closed, partially closed, increased, or flipped
    - High-score signal (COUNTER >= 6) even without size change (liq proximity)
    Stays silent when nothing changed and no critical scores.
    """
    import os

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    changes = changes or []
    changed_coins = {c.coin for c in changes}
    has_position_changes = len(changes) > 0

    # Only alert on critical scores (COUNTER) if no position changes
    critical_signals = [s for s in signals if s.get("score", 0) >= 6]
    if not has_position_changes and not critical_signals:
        return

    total_notional = sum(p.notional_usd for p in snapshot.positions)
    acct_leverage = total_notional / snapshot.account_value if snapshot.account_value > 0 else 0

    sig_by_coin = {s["coin"]: s for s in signals}

    lines = [f"<b>{snapshot.wallet_label}</b>"]
    lines.append(
        f"Account ${snapshot.account_value:,.0f} | {acct_leverage:.0f}x leverage"
        f" | {len(snapshot.positions)} positions"
    )
    lines.append("")

    # Position changes section
    if changes:
        for ch in changes:
            _CHANGE_EMOJI = {
                "new": "\U0001f7e2",       # green circle
                "closed": "\u274c",         # red X
                "partial_close": "\U0001f4c9",  # chart decreasing
                "partial_liq": "\U0001f4a5",    # explosion — suspected partial liquidation
                "increased": "\U0001f4c8",  # chart increasing
                "flipped": "\U0001f504",    # arrows
            }
            emoji = _CHANGE_EMOJI.get(ch.change_type, "\u2022")

            if ch.change_type == "closed":
                lines.append(f"{emoji} <b>{ch.coin}</b> CLOSED (was {ch.prev_side} {ch.prev_size:,.2f})")
            elif ch.change_type in ("partial_close", "partial_liq"):
                label = "PARTIAL LIQ" if ch.change_type == "partial_liq" else "PARTIAL CLOSE"
                lines.append(
                    f"{emoji} <b>{ch.coin}</b> {label} {ch.size_delta_pct:+.0f}%"
                    f" ({ch.prev_size:,.2f} \u2192 {ch.curr_size:,.2f} {ch.curr_side})"
                )
            elif ch.change_type == "new":
                lines.append(f"{emoji} <b>{ch.coin}</b> NEW {ch.curr_side.upper()} {ch.curr_size:,.2f}")
            elif ch.change_type == "increased":
                lines.append(
                    f"{emoji} <b>{ch.coin}</b> INCREASED {ch.size_delta_pct:+.0f}%"
                    f" ({ch.prev_size:,.2f} \u2192 {ch.curr_size:,.2f} {ch.curr_side})"
                )
            elif ch.change_type == "flipped":
                lines.append(
                    f"{emoji} <b>{ch.coin}</b> FLIPPED"
                    f" {ch.prev_side} \u2192 {ch.curr_side} ({ch.curr_size:,.2f})"
                )

            # Add liquidation info for current positions
            sig = sig_by_coin.get(ch.coin)
            if sig and sig.get("their_liq_distance_pct") is not None:
                liq_dist = sig["their_liq_distance_pct"]
                pos = next((p for p in snapshot.positions if p.coin == ch.coin), None)
                liq_str = f"  Liq: {liq_dist:.1f}% away"
                if pos and pos.liquidation_price:
                    liq_str += f" (${pos.liquidation_price:,.2f})"
                lines.append(liq_str)

        lines.append("")

    # Current positions summary with liquidation prices
    if snapshot.positions:
        lines.append("<b>Open positions:</b>")
        for pos in snapshot.positions:
            liq_info = ""
            if pos.liquidation_price:
                if pos.mark_price > 0:
                    if pos.side == "long":
                        dist = (pos.mark_price - pos.liquidation_price) / pos.mark_price * 100
                    else:
                        dist = (pos.liquidation_price - pos.mark_price) / pos.mark_price * 100
                    liq_info = f" | liq ${pos.liquidation_price:,.2f} ({dist:.1f}%)"
            coin_marker = " \u26a0\ufe0f" if pos.coin in changed_coins else ""
            lines.append(
                f"  {pos.coin}{coin_marker} {pos.side.upper()} ${pos.notional_usd:,.0f}"
                f" ({pos.leverage_value}x) uPnL ${pos.unrealized_pnl:+,.0f}{liq_info}"
            )
        lines.append("")

    # Counter signals for critical scores only
    if critical_signals:
        best = max(critical_signals, key=lambda s: s.get("score", 0))
        lines.append(f"\U0001f6a8 <b>COUNTER signal</b> (score {best['score']})")
        for sig in critical_signals:
            reasons = sig.get("reasons", [])
            age_str = ""
            age_h = sig.get("position_age_hours")
            if age_h is not None and age_h > 1:
                if age_h >= 24:
                    age_str = f" [{age_h / 24:.0f}d held]"
                else:
                    age_str = f" [{age_h:.0f}h held]"
            lines.append(f"  {sig['counter_side'].upper()} {sig['coin']}: {', '.join(reasons[:4])}{age_str}")
        lines.append("")

    lines.append("<i>NFA</i>")
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

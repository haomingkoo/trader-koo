"""Position cycle reconstruction from Hyperliquid fills.

Reconstructs complete position lifecycles from raw fills,
handling pre-existing positions, liquidations, shorts, and
mid-stream data.

Algorithm:
1. Process fills chronologically per coin
2. Track running net position (allow negative for pre-existing positions)
3. Detect zero crossings - every time net returns to ~0, a cycle completes
4. Accumulate PnL, fees, notional, fill counts between zero crossings
5. Track peak notional exposure per cycle for threshold-based strategies

Excludes spot trades (Buy/Sell dirs, @-prefixed coins).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

LOG = logging.getLogger(__name__)

# Spot/non-perp coins to exclude
_SPOT_PREFIXES = ("@",)


def reconstruct_cycles(
    conn: sqlite3.Connection,
    wallet: str,
) -> list[dict[str, Any]]:
    """Reconstruct position cycles from fill history."""
    rows = conn.execute(
        """
        SELECT coin, dir, size, price, closed_pnl, fee, is_liquidation,
               fill_time_ms, fill_date
        FROM hyperliquid_fills
        WHERE wallet_label = ?
        ORDER BY fill_time_ms ASC
        """,
        (wallet,),
    ).fetchall()

    if not rows:
        return []

    # Per-coin state
    state: dict[str, dict[str, Any]] = {}
    cycles: list[dict[str, Any]] = []

    for row in rows:
        coin = row[0]
        direction = str(row[1] or "")
        size = float(row[2] or 0)
        price = float(row[3] or 0)
        pnl = float(row[4] or 0)
        fee = float(row[5] or 0)
        is_liq = bool(row[6])
        fill_ts = int(row[7] or 0)
        fill_date = row[8]

        # Skip spot trades
        if any(coin.startswith(p) for p in _SPOT_PREFIXES):
            continue
        if direction in ("Buy", "Sell", "Spot Dust Conversion"):
            continue

        # Classify fill
        is_open = "Open" in direction
        is_close = "Close" in direction
        is_long_open = direction == "Open Long"
        is_short_open = direction == "Open Short"

        # Handle "Long > Short" as close long + open short
        if direction == "Long > Short":
            is_close = True
            is_open = False  # treat as close for now

        if coin not in state:
            state[coin] = _new_state()

        s = state[coin]

        # Update net position
        if is_open:
            s["net_position"] += size
            s["total_open_notional"] += size * price
            s["open_fills"] += 1
        elif is_close:
            s["net_position"] -= size
            s["close_fills"] += 1

        # Always accumulate PnL and fees
        s["total_pnl"] += pnl
        s["total_fees"] += fee

        # Track liquidation
        if is_liq:
            s["was_liquidated"] = True
            s["liq_fills"] += 1

        # Track peak notional (for threshold strategies)
        current_notional = abs(s["net_position"] * price)
        if current_notional > s["peak_notional"]:
            s["peak_notional"] = current_notional

        # Determine position direction
        if not s["direction"]:
            if is_long_open:
                s["direction"] = "Long"
            elif is_short_open:
                s["direction"] = "Short"
            elif "Long" in direction:
                s["direction"] = "Long"
            elif "Short" in direction:
                s["direction"] = "Short"

        # Track timing
        if s["first_fill_ts"] == 0:
            s["first_fill_ts"] = fill_ts
            s["first_fill_date"] = fill_date
        s["last_fill_ts"] = fill_ts
        s["last_fill_date"] = fill_date

        # Detect cycle completion: net position crosses through zero
        if _crossed_zero(s):
            _record_cycle(cycles, coin, s)
            state[coin] = _new_state()

    return cycles


def _new_state() -> dict[str, Any]:
    return {
        "net_position": 0.0,
        "total_open_notional": 0.0,
        "peak_notional": 0.0,
        "total_pnl": 0.0,
        "total_fees": 0.0,
        "open_fills": 0,
        "close_fills": 0,
        "liq_fills": 0,
        "was_liquidated": False,
        "direction": None,
        "first_fill_ts": 0,
        "first_fill_date": None,
        "last_fill_ts": 0,
        "last_fill_date": None,
    }


def _crossed_zero(s: dict[str, Any]) -> bool:
    """Check if position has returned to approximately zero."""
    # Only complete if we've actually seen fills
    if s["open_fills"] + s["close_fills"] < 2:
        return False
    return abs(s["net_position"]) < 0.01


def _record_cycle(
    cycles: list[dict[str, Any]],
    coin: str,
    s: dict[str, Any],
) -> None:
    """Record a completed cycle."""
    duration_hours = 0.0
    if s["first_fill_ts"] > 0 and s["last_fill_ts"] > 0:
        duration_hours = (s["last_fill_ts"] - s["first_fill_ts"]) / 3_600_000

    cycles.append({
        "coin": coin,
        "direction": s["direction"] or "Long",
        "entry_notional_usd": round(s["total_open_notional"], 2),
        "peak_notional_usd": round(s["peak_notional"], 2),
        "closed_pnl": round(s["total_pnl"], 2),
        "total_fees": round(s["total_fees"], 2),
        "duration_hours": round(max(0, duration_hours), 2),
        "was_liquidated": s["was_liquidated"],
        "liq_fills": s["liq_fills"],
        "open_fills": s["open_fills"],
        "close_fills": s["close_fills"],
        "cycle_start": s["first_fill_date"],
        "cycle_end": s["last_fill_date"],
    })

"""API routes for Hyperliquid whale tracking."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from trader_koo.backend.services.database import get_conn
from trader_koo.hyperliquid.tracker import (
    ensure_hyperliquid_schema,
    fetch_wallet_fills,
    fetch_wallet_history,
    fetch_wallet_open_orders,
    fetch_wallet_state,
    generate_counter_signals,
    poll_all_wallets,
    seed_default_wallets,
)

LOG = logging.getLogger(__name__)
router = APIRouter(tags=["hyperliquid"])


@router.get("/api/hyperliquid/wallets")
def list_tracked_wallets() -> dict[str, Any]:
    """List all tracked Hyperliquid wallets."""
    conn = get_conn()
    try:
        ensure_hyperliquid_schema(conn)
        seed_default_wallets(conn)
        rows = conn.execute(
            "SELECT label, address, track_mode, active, notes FROM hyperliquid_wallets"
        ).fetchall()
        return {
            "ok": True,
            "wallets": [
                {
                    "label": r[0], "address": r[1], "track_mode": r[2],
                    "active": bool(r[3]), "notes": r[4],
                }
                for r in rows
            ],
        }
    finally:
        conn.close()


@router.get("/api/hyperliquid/live/{label}")
def get_live_wallet(label: str) -> dict[str, Any]:
    """Fetch live positions + counter signals for a tracked wallet."""
    conn = get_conn()
    try:
        ensure_hyperliquid_schema(conn)
        row = conn.execute(
            "SELECT address, track_mode FROM hyperliquid_wallets WHERE label = ? AND active = 1",
            (label,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": f"Wallet '{label}' not found or inactive"}

        address, track_mode = row[0], row[1]
        snapshot = fetch_wallet_state(address, wallet_label=label)
        if not snapshot:
            return {"ok": False, "error": "Failed to fetch wallet state"}

        counter_signals = []
        if track_mode == "counter":
            counter_signals = generate_counter_signals(snapshot)

        return {
            "ok": True,
            "wallet": {
                "label": label,
                "address": address,
                "account_value": snapshot.account_value,
                "total_margin_used": snapshot.total_margin_used,
                "margin_ratio": snapshot.margin_ratio,
                "timestamp": snapshot.timestamp,
            },
            "positions": [
                {
                    "coin": p.coin,
                    "side": p.side,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "mark_price": p.mark_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": f"{p.leverage_value}x {p.leverage_type}",
                    "notional_usd": p.notional_usd,
                    "liquidation_price": p.liquidation_price,
                }
                for p in snapshot.positions
            ],
            "counter_signals": counter_signals,
        }
    finally:
        conn.close()


@router.get("/api/hyperliquid/fills/{label}")
def get_wallet_fills(label: str, limit: int = 50) -> dict[str, Any]:
    """Fetch recent trade fills for a tracked wallet."""
    conn = get_conn()
    try:
        ensure_hyperliquid_schema(conn)
        row = conn.execute(
            "SELECT address FROM hyperliquid_wallets WHERE label = ? AND active = 1",
            (label,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": f"Wallet '{label}' not found"}

        fills = fetch_wallet_fills(row[0], limit=limit)
        return {"ok": True, "fills": fills, "count": len(fills)}
    finally:
        conn.close()


@router.get("/api/hyperliquid/history/{label}")
def get_trade_history(label: str, days: int = 7) -> dict[str, Any]:
    """Fetch full trade history with PnL stats and per-coin breakdown."""
    conn = get_conn()
    try:
        ensure_hyperliquid_schema(conn)
        row = conn.execute(
            "SELECT address FROM hyperliquid_wallets WHERE label = ? AND active = 1",
            (label,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": f"Wallet '{label}' not found"}

        history = fetch_wallet_history(row[0], lookback_days=days)
        return {"ok": True, "label": label, **history}
    finally:
        conn.close()


@router.get("/api/hyperliquid/orders/{label}")
def get_open_orders(label: str) -> dict[str, Any]:
    """Fetch pending/open orders for a tracked wallet."""
    conn = get_conn()
    try:
        ensure_hyperliquid_schema(conn)
        row = conn.execute(
            "SELECT address FROM hyperliquid_wallets WHERE label = ? AND active = 1",
            (label,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": f"Wallet '{label}' not found"}

        orders = fetch_wallet_open_orders(row[0])
        return {"ok": True, "label": label, "orders": orders, "count": len(orders)}
    finally:
        conn.close()


@router.get("/api/hyperliquid/snapshots/{label}")
def get_wallet_history(label: str, limit: int = 100) -> dict[str, Any]:
    """Get historical snapshots for a tracked wallet."""
    conn = get_conn()
    try:
        ensure_hyperliquid_schema(conn)
        rows = conn.execute(
            "SELECT account_value, total_margin_used, margin_ratio, "
            "positions_json, snapshot_ts FROM hyperliquid_snapshots "
            "WHERE wallet_label = ? ORDER BY snapshot_ts DESC LIMIT ?",
            (label, limit),
        ).fetchall()
        return {
            "ok": True,
            "snapshots": [
                {
                    "account_value": r[0],
                    "margin_used": r[1],
                    "margin_ratio": r[2],
                    "positions": r[3],
                    "timestamp": r[4],
                }
                for r in rows
            ],
        }
    finally:
        conn.close()


@router.get("/api/hyperliquid/signals")
def get_counter_signals(limit: int = 50) -> dict[str, Any]:
    """Get recent counter-trade signals from all tracked wallets."""
    conn = get_conn()
    try:
        ensure_hyperliquid_schema(conn)
        rows = conn.execute(
            "SELECT wallet_label, coin, counter_side, their_side, their_size, "
            "their_leverage, their_notional_usd, confidence, reasoning, signal_ts "
            "FROM hyperliquid_counter_signals ORDER BY signal_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {
            "ok": True,
            "signals": [
                {
                    "wallet": r[0], "coin": r[1], "counter_side": r[2],
                    "their_side": r[3], "their_size": r[4], "their_leverage": r[5],
                    "their_notional_usd": r[6], "confidence": r[7],
                    "reasoning": r[8], "timestamp": r[9],
                }
                for r in rows
            ],
        }
    finally:
        conn.close()


@router.post("/api/hyperliquid/poll")
def trigger_poll() -> dict[str, Any]:
    """Manually trigger a poll of all tracked wallets."""
    conn = get_conn()
    try:
        signals = poll_all_wallets(conn)
        return {
            "ok": True,
            "signals_generated": len(signals),
            "signals": signals,
        }
    finally:
        conn.close()

"""Backtest counter-trading machibro (or any tracked wallet).

Strategy: When the tracked trader opens a position, we take the opposite
side. When they close (or get liquidated), we close too.

Their "Open Long" -> we open SHORT at same price
Their "Close Long" (or liquidation) -> we close SHORT at same price
Vice versa for shorts.

We size at a fixed notional (default $10K per entry) regardless of
their leverage. No leverage on our side.

Usage:
    python -m trader_koo.hyperliquid.backtest_counter
    python -m trader_koo.hyperliquid.backtest_counter --wallet machibro --size 10000
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.getenv(
    "TRADER_KOO_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "trader_koo.db"),
))


@dataclass
class CounterTrade:
    coin: str
    our_side: str  # "short" or "long" (opposite of theirs)
    entry_price: float
    entry_time: str
    their_dir: str  # what they did: "Open Long", etc.
    size_usd: float  # our notional
    size_coins: float  # our position in coins


@dataclass
class ClosedCounterTrade:
    coin: str
    our_side: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    their_exit_reason: str  # "Close Long", liquidation, etc.
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    holding_minutes: float


@dataclass
class BacktestState:
    open_positions: dict[str, CounterTrade] = field(default_factory=dict)  # coin -> position
    closed_trades: list[ClosedCounterTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    equity_curve: list[tuple[str, float]] = field(default_factory=list)


def run_backtest(
    conn: sqlite3.Connection,
    wallet_label: str = "machibro",
    position_size_usd: float = 10_000.0,
    coins: list[str] | None = None,
) -> dict:
    """Run counter-trade backtest on stored fill history.

    For each "Open Long" by the tracked wallet, we open a SHORT.
    For each "Close Long" (or liquidation), we close our SHORT.
    """
    state = BacktestState()

    query = """
        SELECT coin, side, size, price, closed_pnl, is_liquidation,
               fill_time_ms, fill_date, dir, start_position
        FROM hyperliquid_fills
        WHERE wallet_label = ?
        ORDER BY fill_time_ms ASC
    """
    params: list = [wallet_label]

    fills = conn.execute(query, params).fetchall()
    LOG.info("Processing %d fills for %s", len(fills), wallet_label)

    for coin, side, size, price, closed_pnl, is_liq, time_ms, date, dir_str, start_pos in fills:
        if coins and coin not in coins:
            continue
        if price <= 0:
            continue

        dir_lower = (dir_str or "").lower()
        time_str = dt.datetime.fromtimestamp(time_ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M")

        # They OPEN a position -> we open the OPPOSITE
        if "open long" in dir_lower:
            if coin not in state.open_positions:
                coins_amount = position_size_usd / price
                state.open_positions[coin] = CounterTrade(
                    coin=coin, our_side="short", entry_price=price,
                    entry_time=time_str, their_dir=dir_str,
                    size_usd=position_size_usd, size_coins=coins_amount,
                )

        elif "open short" in dir_lower:
            if coin not in state.open_positions:
                coins_amount = position_size_usd / price
                state.open_positions[coin] = CounterTrade(
                    coin=coin, our_side="long", entry_price=price,
                    entry_time=time_str, their_dir=dir_str,
                    size_usd=position_size_usd, size_coins=coins_amount,
                )

        # They CLOSE (or get liquidated) -> we close our counter-position
        elif "close" in dir_lower or is_liq:
            pos = state.open_positions.get(coin)
            if not pos:
                continue

            # Calculate our P&L
            if pos.our_side == "short":
                pnl_per_coin = pos.entry_price - price
            else:
                pnl_per_coin = price - pos.entry_price

            pnl_usd = pnl_per_coin * pos.size_coins
            pnl_pct = (pnl_usd / pos.size_usd) * 100

            entry_ms = dt.datetime.strptime(pos.entry_time, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc).timestamp() * 1000
            holding_min = (time_ms - entry_ms) / 60_000

            exit_reason = "liquidation" if is_liq else dir_str

            closed = ClosedCounterTrade(
                coin=coin, our_side=pos.our_side,
                entry_price=pos.entry_price, exit_price=price,
                entry_time=pos.entry_time, exit_time=time_str,
                their_exit_reason=exit_reason,
                size_usd=pos.size_usd,
                pnl_usd=round(pnl_usd, 2), pnl_pct=round(pnl_pct, 2),
                holding_minutes=round(holding_min, 1),
            )
            state.closed_trades.append(closed)
            state.total_pnl += pnl_usd
            del state.open_positions[coin]

            # Track equity curve + drawdown
            equity = 100_000 + state.total_pnl  # start with $100K
            state.peak_equity = max(state.peak_equity, equity)
            dd = (state.peak_equity - equity) / state.peak_equity * 100 if state.peak_equity > 0 else 0
            state.max_drawdown = max(state.max_drawdown, dd)
            state.equity_curve.append((time_str, round(equity, 2)))

    # Compile results
    trades = state.closed_trades
    if not trades:
        return {"error": "No closed trades", "fills_processed": len(fills)}

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd < 0]
    win_pnls = [t.pnl_usd for t in wins]
    loss_pnls = [t.pnl_usd for t in losses]

    # By coin
    by_coin: dict[str, dict] = {}
    for t in trades:
        if t.coin not in by_coin:
            by_coin[t.coin] = {"trades": 0, "pnl": 0, "wins": 0, "losses": 0}
        by_coin[t.coin]["trades"] += 1
        by_coin[t.coin]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            by_coin[t.coin]["wins"] += 1
        elif t.pnl_usd < 0:
            by_coin[t.coin]["losses"] += 1

    # Monthly
    by_month: dict[str, dict] = {}
    for t in trades:
        month = t.exit_time[:7]
        if month not in by_month:
            by_month[month] = {"trades": 0, "pnl": 0, "wins": 0}
        by_month[month]["trades"] += 1
        by_month[month]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            by_month[month]["wins"] += 1

    avg_hold_min = sum(t.holding_minutes for t in trades) / len(trades)

    return {
        "wallet": wallet_label,
        "position_size_usd": position_size_usd,
        "fills_processed": len(fills),
        "closed_trades": len(trades),
        "open_positions": len(state.open_positions),
        "total_pnl_usd": round(state.total_pnl, 2),
        "total_return_pct": round(state.total_pnl / 100_000 * 100, 2),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "avg_win_usd": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        "avg_loss_usd": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        "biggest_win_usd": round(max(win_pnls), 2) if win_pnls else 0,
        "biggest_loss_usd": round(min(loss_pnls), 2) if loss_pnls else 0,
        "profit_factor": round(sum(win_pnls) / abs(sum(loss_pnls)), 2) if loss_pnls else float("inf"),
        "max_drawdown_pct": round(state.max_drawdown, 2),
        "avg_holding_minutes": round(avg_hold_min, 1),
        "by_coin": {
            coin: {
                "trades": d["trades"],
                "pnl_usd": round(d["pnl"], 2),
                "win_rate_pct": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0,
            }
            for coin, d in sorted(by_coin.items(), key=lambda x: x[1]["pnl"], reverse=True)
        },
        "by_month": {
            month: {
                "trades": d["trades"],
                "pnl_usd": round(d["pnl"], 2),
                "win_rate_pct": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0,
            }
            for month, d in sorted(by_month.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest counter-trading a Hyperliquid whale")
    parser.add_argument("--wallet", default="machibro")
    parser.add_argument("--size", type=float, default=10_000, help="Position size in USD per entry")
    parser.add_argument("--coin", default=None, help="Filter to specific coin (e.g. ETH)")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        coins = [args.coin] if args.coin else None
        result = run_backtest(conn, args.wallet, args.size, coins=coins)
        print(json.dumps(result, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

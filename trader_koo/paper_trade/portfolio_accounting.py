"""One reconciled paper-account projection from persisted trade facts."""

from __future__ import annotations

import sqlite3
from typing import Any


def reconcile_portfolio(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    starting_capital: float,
) -> dict[str, Any]:
    rows = conn.execute(
        """SELECT id,ticker,direction,status,entry_price,current_price,exit_price,
                  quantity,entry_notional,entry_commission,exit_commission,
                  borrow_cost,realized_pnl_usd,entry_date,exit_date,last_mtm_date
           FROM paper_trades
           WHERE campaign_id=? AND accounting_status='reconciled'
           ORDER BY entry_date,id""",
        (campaign_id,),
    ).fetchall()
    legacy_count = int(conn.execute(
        """SELECT COUNT(*) FROM paper_trades
           WHERE campaign_id=? AND accounting_status!='reconciled'""",
        (campaign_id,),
    ).fetchone()[0])
    cash = float(starting_capital)
    realized = 0.0
    unrealized = 0.0
    gross_exposure = 0.0
    open_positions = 0
    closed_trades = 0
    latest_date: str | None = None
    breaks: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []

    for row in rows:
        (
            trade_id, ticker, direction, status, entry_price, current_price,
            exit_price, quantity, entry_notional, entry_commission,
            exit_commission, borrow_cost, realized_pnl, entry_date, exit_date,
            last_mtm_date,
        ) = row
        values = (quantity, entry_notional, entry_commission)
        if any(value is None for value in values):
            breaks.append({"trade_id": int(trade_id), "reason": "missing_entry_accounting"})
            continue
        qty = float(quantity)
        reserve = float(entry_notional)
        entry_fee = float(entry_commission)
        cash -= reserve + entry_fee
        latest_date = max(filter(None, (latest_date, str(entry_date or "")))) or latest_date
        latest_date = max(
            filter(None, (latest_date, str(last_mtm_date or "")))
        ) or latest_date

        if str(status) != "open":
            if exit_price is None or realized_pnl is None:
                breaks.append({"trade_id": int(trade_id), "reason": "missing_close_accounting"})
                continue
            exit_fee = float(exit_commission or 0.0)
            borrow = float(borrow_cost or 0.0)
            gross = (
                (float(exit_price) - float(entry_price)) * qty
                if str(direction) == "long"
                else (float(entry_price) - float(exit_price)) * qty
            )
            settlement = reserve + gross - exit_fee - borrow
            cash += settlement
            realized += float(realized_pnl)
            closed_trades += 1
            latest_date = max(filter(None, (latest_date, str(exit_date or "")))) or latest_date
            continue

        mark = float(current_price) if current_price is not None else None
        if mark is None:
            breaks.append({"trade_id": int(trade_id), "reason": "missing_open_mark"})
            continue
        gross_pnl = (
            (mark - float(entry_price)) * qty
            if str(direction) == "long"
            else (float(entry_price) - mark) * qty
        )
        position_value = mark * qty if str(direction) == "long" else reserve + gross_pnl
        unrealized += gross_pnl - entry_fee
        gross_exposure += mark * qty
        open_positions += 1
        positions.append({
            "trade_id": int(trade_id),
            "ticker": str(ticker),
            "direction": str(direction),
            "quantity": qty,
            "mark": mark,
            "market_value": position_value,
            "unrealized_pnl_usd": gross_pnl - entry_fee,
        })

    equity = cash + sum(float(item["market_value"]) for item in positions)
    invariant_delta = equity - (starting_capital + realized + unrealized)
    if abs(invariant_delta) > 0.01:
        breaks.append({"reason": "equity_pnl_invariant", "delta_usd": invariant_delta})
    return {
        "campaign_id": campaign_id,
        "as_of_date": latest_date,
        "starting_capital": float(starting_capital),
        "cash": cash,
        "equity": equity,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": unrealized,
        "gross_exposure_usd": gross_exposure,
        "gross_exposure_pct": gross_exposure / equity * 100 if equity > 0 else None,
        "open_positions": open_positions,
        "closed_trades": closed_trades,
        "legacy_unreconciled_count": legacy_count,
        "accounting_breaks": breaks,
        "reconciled": not breaks,
        "positions": positions,
    }

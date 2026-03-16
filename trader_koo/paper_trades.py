"""Paper trade tracking for trader-koo.

Automatically creates simulated positions from daily report setups,
marks them to market, and tracks portfolio-level performance metrics.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import sqlite3
from typing import Any

LOG = logging.getLogger(__name__)

# ── Configuration (env vars) ────────────────────────────────────────
PAPER_TRADE_ENABLED = os.getenv("TRADER_KOO_PAPER_TRADE_ENABLED", "1") == "1"
PAPER_TRADE_MIN_TIER = os.getenv("TRADER_KOO_PAPER_TRADE_MIN_TIER", "B")
PAPER_TRADE_MIN_SCORE = float(os.getenv("TRADER_KOO_PAPER_TRADE_MIN_SCORE", "60.0"))
PAPER_TRADE_MAX_OPEN = int(os.getenv("TRADER_KOO_PAPER_TRADE_MAX_OPEN", "20"))
PAPER_TRADE_EXPIRY_DAYS = int(os.getenv("TRADER_KOO_PAPER_TRADE_EXPIRY_DAYS", "10"))
PAPER_TRADE_STOP_ATR_MULT = float(os.getenv("TRADER_KOO_PAPER_TRADE_STOP_ATR_MULT", "1.5"))
PAPER_TRADE_DEFAULT_STOP_PCT = float(os.getenv("TRADER_KOO_PAPER_TRADE_DEFAULT_STOP_PCT", "3.0"))

_QUALIFYING_TIERS = {"A", "B"}
_QUALIFYING_ACTIONABILITY = {"higher-probability", "conditional"}
_QUALIFYING_DIRECTIONS = {"long", "short"}

_TIER_RANK = {"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}


# ── Schema ──────────────────────────────────────────────────────────

def ensure_paper_trade_schema(conn: sqlite3.Connection) -> None:
    """Create paper_trades and paper_portfolio_snapshots tables."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL,
            generated_ts TEXT,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            target_price REAL,
            stop_loss REAL,
            atr_at_entry REAL,
            exit_price REAL,
            exit_date TEXT,
            exit_reason TEXT,
            status TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'closed', 'stopped_out', 'target_hit', 'expired')),
            current_price REAL,
            unrealized_pnl_pct REAL,
            last_mtm_date TEXT,
            high_water_mark REAL,
            low_water_mark REAL,
            pnl_pct REAL,
            r_multiple REAL,
            setup_family TEXT,
            setup_tier TEXT,
            score REAL,
            signal_bias TEXT,
            actionability TEXT,
            observation TEXT,
            action_text TEXT,
            risk_note TEXT,
            yolo_pattern TEXT,
            yolo_recency TEXT,
            debate_agreement_score REAL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_date, ticker, direction)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_status "
        "ON paper_trades(status, entry_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker "
        "ON paper_trades(ticker, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_family "
        "ON paper_trades(setup_family, direction, status)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL UNIQUE,
            open_trades INTEGER NOT NULL DEFAULT 0,
            closed_trades_total INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            win_rate_pct REAL,
            avg_pnl_pct REAL,
            avg_r_multiple REAL,
            total_pnl_pct REAL,
            max_drawdown_pct REAL,
            sharpe_ratio REAL,
            profit_factor REAL,
            equity_index REAL NOT NULL DEFAULT 100.0,
            best_trade_pct REAL,
            worst_trade_pct REAL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_portfolio_date "
        "ON paper_portfolio_snapshots(snapshot_date)"
    )
    conn.commit()


# ── Qualification ───────────────────────────────────────────────────

def _direction_from_row(row: dict[str, Any]) -> str:
    family = str(row.get("setup_family") or "").strip().lower()
    bias = str(row.get("signal_bias") or "").strip().lower()
    if family.startswith("bullish") or bias == "bullish":
        return "long"
    if family.startswith("bearish") or bias == "bearish":
        return "short"
    return "neutral"


def qualify_setup_for_paper_trade(row: dict[str, Any]) -> bool:
    """Return True if a setup row qualifies as a paper trade entry."""
    tier = str(row.get("setup_tier") or "").strip().upper()
    if tier not in _QUALIFYING_TIERS:
        min_rank = _TIER_RANK.get(PAPER_TRADE_MIN_TIER, 2)
        if _TIER_RANK.get(tier, 99) > min_rank:
            return False

    score = row.get("score")
    if not isinstance(score, (int, float)) or float(score) < PAPER_TRADE_MIN_SCORE:
        return False

    actionability = str(row.get("actionability") or "").strip().lower()
    if actionability not in _QUALIFYING_ACTIONABILITY:
        return False

    direction = _direction_from_row(row)
    if direction not in _QUALIFYING_DIRECTIONS:
        return False

    close = row.get("close")
    if not isinstance(close, (int, float)) or float(close) <= 0:
        return False

    return True


# ── Stop / Target ───────────────────────────────────────────────────

def compute_stop_and_target(
    row: dict[str, Any],
    direction: str,
) -> dict[str, float | None]:
    """Compute stop_loss, target_price, and atr_at_entry from a setup row."""
    entry = float(row["close"])
    atr_pct = row.get("atr_pct_14")
    support = row.get("support_level")
    resistance = row.get("resistance_level")

    # ATR-based stop distance
    if isinstance(atr_pct, (int, float)) and float(atr_pct) > 0:
        atr_distance = (float(atr_pct) / 100.0) * entry * PAPER_TRADE_STOP_ATR_MULT
    else:
        atr_distance = entry * (PAPER_TRADE_DEFAULT_STOP_PCT / 100.0)

    if direction == "long":
        stop_loss = entry - atr_distance
        # Use support level if tighter and within 5%
        if isinstance(support, (int, float)) and float(support) > 0:
            support_stop = float(support) * 0.99  # 1% below support
            if entry * 0.95 < support_stop < entry:
                stop_loss = max(stop_loss, support_stop)

        risk = entry - stop_loss
        # Target: resistance or 2R
        if isinstance(resistance, (int, float)) and float(resistance) > entry:
            target_price = float(resistance)
        else:
            target_price = entry + (risk * 2.0)
    else:
        stop_loss = entry + atr_distance
        # Use resistance level if tighter and within 5%
        if isinstance(resistance, (int, float)) and float(resistance) > 0:
            resist_stop = float(resistance) * 1.01  # 1% above resistance
            if entry < resist_stop < entry * 1.05:
                stop_loss = min(stop_loss, resist_stop)

        risk = stop_loss - entry
        # Target: support or 2R
        if isinstance(support, (int, float)) and 0 < float(support) < entry:
            target_price = float(support)
        else:
            target_price = entry - (risk * 2.0)

    return {
        "stop_loss": round(stop_loss, 2),
        "target_price": round(target_price, 2),
        "atr_at_entry": round(float(atr_pct), 2) if isinstance(atr_pct, (int, float)) else None,
    }


# ── Trade Creation ──────────────────────────────────────────────────

def create_paper_trades_from_report(
    conn: sqlite3.Connection,
    *,
    setup_rows: list[dict[str, Any]],
    report_date: str,
    generated_ts: str,
) -> int:
    """Create paper trades from qualifying daily report setups."""
    if not report_date or not setup_rows:
        return 0

    ensure_paper_trade_schema(conn)

    open_count = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
    ).fetchone()[0]

    if open_count >= PAPER_TRADE_MAX_OPEN:
        LOG.info(
            "Paper trades: %d open trades already at max (%d), skipping creation",
            open_count, PAPER_TRADE_MAX_OPEN,
        )
        return 0

    remaining_slots = PAPER_TRADE_MAX_OPEN - open_count
    inserted = 0

    for row in setup_rows:
        if inserted >= remaining_slots:
            break
        if not isinstance(row, dict):
            continue
        if not qualify_setup_for_paper_trade(row):
            continue

        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue

        direction = _direction_from_row(row)
        entry_price = float(row["close"])
        levels = compute_stop_and_target(row, direction)

        before_changes = conn.total_changes
        conn.execute(
            """
            INSERT INTO paper_trades (
                report_date, generated_ts, ticker, direction,
                entry_price, entry_date, target_price, stop_loss, atr_at_entry,
                status, current_price, unrealized_pnl_pct,
                high_water_mark, low_water_mark,
                setup_family, setup_tier, score, signal_bias, actionability,
                observation, action_text, risk_note,
                yolo_pattern, yolo_recency, debate_agreement_score
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                'open', ?, 0.0,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(report_date, ticker, direction) DO NOTHING
            """,
            (
                report_date,
                generated_ts,
                ticker,
                direction,
                entry_price,
                report_date,
                levels["target_price"],
                levels["stop_loss"],
                levels["atr_at_entry"],
                entry_price,
                entry_price,
                entry_price,
                row.get("setup_family"),
                row.get("setup_tier"),
                row.get("score"),
                row.get("signal_bias"),
                row.get("actionability"),
                row.get("observation"),
                row.get("action"),
                row.get("risk_note"),
                row.get("yolo_pattern"),
                row.get("yolo_recency"),
                row.get("debate_agreement_score"),
            ),
        )
        if conn.total_changes > before_changes:
            inserted += 1
            LOG.info(
                "Paper trade created: %s %s @ %.2f (stop=%.2f target=%.2f)",
                direction.upper(), ticker, entry_price,
                levels["stop_loss"], levels["target_price"],
            )

    return inserted


# ── Mark to Market ──────────────────────────────────────────────────

def _compute_pnl(
    direction: str,
    entry_price: float,
    current_price: float,
) -> float:
    """Return P&L percentage."""
    if direction == "long":
        return ((current_price / entry_price) - 1.0) * 100.0
    return (1.0 - (current_price / entry_price)) * 100.0


def _compute_r_multiple(
    direction: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float | None,
) -> float | None:
    """Return R-multiple (profit / initial risk)."""
    if stop_loss is None:
        risk = entry_price * (PAPER_TRADE_DEFAULT_STOP_PCT / 100.0)
    else:
        risk = abs(entry_price - stop_loss)
    if risk <= 0:
        return None

    if direction == "long":
        pnl_per_share = exit_price - entry_price
    else:
        pnl_per_share = entry_price - exit_price
    return round(pnl_per_share / risk, 2)


def _close_trade(
    conn: sqlite3.Connection,
    trade_id: int,
    exit_price: float,
    exit_date: str,
    exit_reason: str,
    direction: str,
    entry_price: float,
    stop_loss: float | None,
) -> None:
    pnl = round(_compute_pnl(direction, entry_price, exit_price), 2)
    r_mult = _compute_r_multiple(direction, entry_price, exit_price, stop_loss)
    status = exit_reason if exit_reason in ("stopped_out", "target_hit", "expired") else "closed"
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    conn.execute(
        """
        UPDATE paper_trades SET
            status = ?,
            exit_price = ?,
            exit_date = ?,
            exit_reason = ?,
            pnl_pct = ?,
            r_multiple = ?,
            current_price = ?,
            unrealized_pnl_pct = NULL,
            updated_ts = ?
        WHERE id = ?
        """,
        (status, exit_price, exit_date, exit_reason, pnl, r_mult, exit_price, now, trade_id),
    )


def mark_to_market(conn: sqlite3.Connection) -> dict[str, Any]:
    """Update all open paper trades with latest prices."""
    ensure_paper_trade_schema(conn)

    open_rows = conn.execute(
        """
        SELECT id, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, high_water_mark, low_water_mark
        FROM paper_trades WHERE status = 'open'
        """
    ).fetchall()

    if not open_rows:
        _update_portfolio_snapshot(conn)
        return {"open_trades": 0, "updated": 0, "closed": 0}

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    updated = 0
    closed = 0

    for row in open_rows:
        trade_id, ticker, direction, entry_price, entry_date = row[:5]
        target_price, stop_loss, hwm, lwm = row[5:]

        price_row = conn.execute(
            "SELECT CAST(close AS REAL), date FROM price_daily "
            "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()

        if not price_row or price_row[0] is None:
            continue

        current_price = float(price_row[0])
        price_date = price_row[1]
        unrealized = round(_compute_pnl(direction, entry_price, current_price), 2)
        new_hwm = max(hwm or current_price, current_price)
        new_lwm = min(lwm or current_price, current_price)

        # Check stop loss
        hit_stop = False
        if stop_loss is not None:
            if direction == "long" and current_price <= stop_loss:
                hit_stop = True
            elif direction == "short" and current_price >= stop_loss:
                hit_stop = True

        # Check target
        hit_target = False
        if target_price is not None:
            if direction == "long" and current_price >= target_price:
                hit_target = True
            elif direction == "short" and current_price <= target_price:
                hit_target = True

        # Check expiry
        expired = False
        try:
            entry_dt = dt.datetime.strptime(entry_date, "%Y-%m-%d")
            today_dt = dt.datetime.strptime(today, "%Y-%m-%d")
            if (today_dt - entry_dt).days >= PAPER_TRADE_EXPIRY_DAYS:
                expired = True
        except (ValueError, TypeError):
            pass

        if hit_stop:
            _close_trade(conn, trade_id, current_price, today, "stopped_out",
                         direction, entry_price, stop_loss)
            closed += 1
        elif hit_target:
            _close_trade(conn, trade_id, current_price, today, "target_hit",
                         direction, entry_price, stop_loss)
            closed += 1
        elif expired:
            _close_trade(conn, trade_id, current_price, today, "expired",
                         direction, entry_price, stop_loss)
            closed += 1
        else:
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE paper_trades SET
                    current_price = ?, unrealized_pnl_pct = ?,
                    last_mtm_date = ?, high_water_mark = ?, low_water_mark = ?,
                    updated_ts = ?
                WHERE id = ?
                """,
                (current_price, unrealized, price_date, new_hwm, new_lwm, now, trade_id),
            )
        updated += 1

    _update_portfolio_snapshot(conn)
    return {"open_trades": len(open_rows) - closed, "updated": updated, "closed": closed}


# ── Portfolio Snapshot ──────────────────────────────────────────────

def _update_portfolio_snapshot(conn: sqlite3.Connection) -> None:
    """Compute and persist daily portfolio metrics."""
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    open_count = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
    ).fetchone()[0]

    closed_rows = conn.execute(
        "SELECT pnl_pct, r_multiple FROM paper_trades WHERE status != 'open' AND pnl_pct IS NOT NULL"
    ).fetchall()

    total_closed = len(closed_rows)
    if total_closed == 0:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_portfolio_snapshots
                (snapshot_date, open_trades, closed_trades_total, wins, losses,
                 equity_index)
            VALUES (?, ?, 0, 0, 0, 100.0)
            """,
            (today, open_count),
        )
        return

    pnls = [float(r[0]) for r in closed_rows]
    r_mults = [float(r[1]) for r in closed_rows if r[1] is not None]

    wins = sum(1 for p in pnls if p > 0)
    losses = total_closed - wins
    win_rate = round((wins / total_closed) * 100.0, 1) if total_closed > 0 else 0.0
    avg_pnl = round(sum(pnls) / len(pnls), 2)
    avg_r = round(sum(r_mults) / len(r_mults), 2) if r_mults else None
    total_pnl = round(sum(pnls), 2)

    # Max drawdown from cumulative P&L curve
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    max_dd = round(max_dd, 2)

    # Sharpe-like ratio
    if len(pnls) > 1:
        mean_p = sum(pnls) / len(pnls)
        var_p = sum((p - mean_p) ** 2 for p in pnls) / (len(pnls) - 1)
        std_p = math.sqrt(var_p) if var_p > 0 else 0
        sharpe = round(mean_p / std_p, 2) if std_p > 0 else None
    else:
        sharpe = None

    # Profit factor
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

    best_trade = round(max(pnls), 2)
    worst_trade = round(min(pnls), 2)

    # Equity index: start 100, compound each trade's pnl
    equity = 100.0
    for p in pnls:
        equity *= (1.0 + p / 100.0)
    equity = round(equity, 2)

    conn.execute(
        """
        INSERT OR REPLACE INTO paper_portfolio_snapshots
            (snapshot_date, open_trades, closed_trades_total, wins, losses,
             win_rate_pct, avg_pnl_pct, avg_r_multiple, total_pnl_pct,
             max_drawdown_pct, sharpe_ratio, profit_factor, equity_index,
             best_trade_pct, worst_trade_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (today, open_count, total_closed, wins, losses,
         win_rate, avg_pnl, avg_r, total_pnl,
         max_dd, sharpe, profit_factor, equity, best_trade, worst_trade),
    )


# ── Summary / Metrics ───────────────────────────────────────────────

def paper_trade_summary(
    conn: sqlite3.Connection,
    *,
    window_days: int = 180,
) -> dict[str, Any]:
    """Return comprehensive paper trading performance metrics."""
    ensure_paper_trade_schema(conn)

    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    # Overall stats
    all_closed = conn.execute(
        """
        SELECT pnl_pct, r_multiple, direction, setup_family, setup_tier, exit_reason
        FROM paper_trades
        WHERE status != 'open' AND pnl_pct IS NOT NULL AND entry_date >= ?
        ORDER BY exit_date
        """,
        (cutoff,),
    ).fetchall()

    open_trades = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
    ).fetchone()[0]

    total = len(all_closed)
    if total == 0:
        return {
            "overall": {"total_trades": 0, "open_count": open_trades},
            "by_direction": {},
            "by_family": {},
            "by_tier": {},
            "by_exit_reason": {},
            "equity_curve": [],
            "recent_trades": _recent_trades(conn, limit=20),
        }

    pnls = [float(r[0]) for r in all_closed]
    r_mults = [float(r[1]) for r in all_closed if r[1] is not None]
    wins = sum(1 for p in pnls if p > 0)

    overall = {
        "total_trades": total,
        "open_count": open_trades,
        "wins": wins,
        "losses": total - wins,
        "win_rate_pct": round(wins / total * 100, 1),
        "avg_pnl_pct": round(sum(pnls) / total, 2),
        "avg_r_multiple": round(sum(r_mults) / len(r_mults), 2) if r_mults else None,
        "total_pnl_pct": round(sum(pnls), 2),
        "best_trade_pct": round(max(pnls), 2),
        "worst_trade_pct": round(min(pnls), 2),
    }

    # By direction
    by_direction: dict[str, dict[str, Any]] = {}
    for direction in ("long", "short"):
        d_rows = [r for r in all_closed if r[2] == direction]
        if d_rows:
            d_pnls = [float(r[0]) for r in d_rows]
            d_wins = sum(1 for p in d_pnls if p > 0)
            by_direction[direction] = {
                "total": len(d_rows),
                "wins": d_wins,
                "win_rate_pct": round(d_wins / len(d_rows) * 100, 1),
                "avg_pnl_pct": round(sum(d_pnls) / len(d_pnls), 2),
            }

    # By family
    by_family: dict[str, dict[str, Any]] = {}
    families = set(r[3] for r in all_closed if r[3])
    for fam in sorted(families):
        f_rows = [r for r in all_closed if r[3] == fam]
        f_pnls = [float(r[0]) for r in f_rows]
        f_wins = sum(1 for p in f_pnls if p > 0)
        by_family[fam] = {
            "total": len(f_rows),
            "wins": f_wins,
            "win_rate_pct": round(f_wins / len(f_rows) * 100, 1),
            "avg_pnl_pct": round(sum(f_pnls) / len(f_pnls), 2),
        }

    # By tier
    by_tier: dict[str, dict[str, Any]] = {}
    tiers = set(r[4] for r in all_closed if r[4])
    for tier in sorted(tiers):
        t_rows = [r for r in all_closed if r[4] == tier]
        t_pnls = [float(r[0]) for r in t_rows]
        t_wins = sum(1 for p in t_pnls if p > 0)
        by_tier[tier] = {
            "total": len(t_rows),
            "wins": t_wins,
            "win_rate_pct": round(t_wins / len(t_rows) * 100, 1),
            "avg_pnl_pct": round(sum(t_pnls) / len(t_pnls), 2),
        }

    # By exit reason
    by_exit_reason: dict[str, int] = {}
    for r in all_closed:
        reason = r[5] or "unknown"
        by_exit_reason[reason] = by_exit_reason.get(reason, 0) + 1

    # Equity curve from snapshots
    eq_rows = conn.execute(
        """
        SELECT snapshot_date, equity_index, open_trades, closed_trades_total
        FROM paper_portfolio_snapshots
        WHERE snapshot_date >= ?
        ORDER BY snapshot_date
        """,
        (cutoff,),
    ).fetchall()
    equity_curve = [
        {
            "date": r[0],
            "equity_index": r[1],
            "open_trades": r[2],
            "closed_total": r[3],
        }
        for r in eq_rows
    ]

    return {
        "overall": overall,
        "by_direction": by_direction,
        "by_family": by_family,
        "by_tier": by_tier,
        "by_exit_reason": by_exit_reason,
        "equity_curve": equity_curve,
        "recent_trades": _recent_trades(conn, limit=20),
    }


def _recent_trades(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict[str, Any]]:
    """Return most recent paper trades."""
    rows = conn.execute(
        """
        SELECT id, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, exit_price, exit_date,
               status, pnl_pct, r_multiple, unrealized_pnl_pct,
               setup_family, setup_tier, score, exit_reason
        FROM paper_trades
        ORDER BY COALESCE(exit_date, entry_date) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    keys = [
        "id", "ticker", "direction", "entry_price", "entry_date",
        "target_price", "stop_loss", "exit_price", "exit_date",
        "status", "pnl_pct", "r_multiple", "unrealized_pnl_pct",
        "setup_family", "setup_tier", "score", "exit_reason",
    ]
    return [dict(zip(keys, r)) for r in rows]


# ── Manual Close ────────────────────────────────────────────────────

def manually_close_trade(
    conn: sqlite3.Connection,
    *,
    trade_id: int,
    exit_price: float | None = None,
    exit_reason: str = "manual_close",
) -> dict[str, Any]:
    """Manually close an open paper trade."""
    row = conn.execute(
        "SELECT ticker, direction, entry_price, stop_loss, status FROM paper_trades WHERE id = ?",
        (trade_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Paper trade {trade_id} not found")
    ticker, direction, entry_price, stop_loss, status = row
    if status != "open":
        raise ValueError(f"Paper trade {trade_id} is already {status}")

    if exit_price is None:
        price_row = conn.execute(
            "SELECT CAST(close AS REAL) FROM price_daily WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        if not price_row or price_row[0] is None:
            raise ValueError(f"No price data for {ticker} to close trade")
        exit_price = float(price_row[0])

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    _close_trade(conn, trade_id, exit_price, today, exit_reason,
                 direction, entry_price, stop_loss)
    conn.commit()

    pnl = round(_compute_pnl(direction, entry_price, exit_price), 2)
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "direction": direction,
        "exit_price": exit_price,
        "pnl_pct": pnl,
        "status": "closed",
    }


# ── Trade Listing ───────────────────────────────────────────────────

def list_paper_trades(
    conn: sqlite3.Connection,
    *,
    status: str = "all",
    ticker: str | None = None,
    direction: str | None = None,
    family: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List paper trades with optional filters."""
    ensure_paper_trade_schema(conn)

    clauses: list[str] = []
    params: list[Any] = []

    if status != "all":
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper().strip())
    if direction:
        clauses.append("direction = ?")
        params.append(direction)
    if family:
        clauses.append("setup_family = ?")
        params.append(family)
    if from_date:
        clauses.append("entry_date >= ?")
        params.append(from_date)
    if to_date:
        clauses.append("entry_date <= ?")
        params.append(to_date)

    where = " AND ".join(clauses) if clauses else "1=1"
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT id, report_date, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, exit_price, exit_date, exit_reason,
               status, current_price, unrealized_pnl_pct, pnl_pct, r_multiple,
               setup_family, setup_tier, score, signal_bias, actionability,
               observation, action_text, risk_note, debate_agreement_score,
               high_water_mark, low_water_mark
        FROM paper_trades
        WHERE {where}
        ORDER BY entry_date DESC, id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    keys = [
        "id", "report_date", "ticker", "direction", "entry_price", "entry_date",
        "target_price", "stop_loss", "exit_price", "exit_date", "exit_reason",
        "status", "current_price", "unrealized_pnl_pct", "pnl_pct", "r_multiple",
        "setup_family", "setup_tier", "score", "signal_bias", "actionability",
        "observation", "action_text", "risk_note", "debate_agreement_score",
        "high_water_mark", "low_water_mark",
    ]
    return [dict(zip(keys, r)) for r in rows]

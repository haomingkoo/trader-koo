"""Daily morning summary push for the Telegram group.

Assembles a market briefing from:
- Latest daily report JSON (setups, market data, earnings)
- ``price_daily`` table (SPY / QQQ / VIX latest close + change)
- ``crypto_bars`` table (BTC, ETH overnight moves)
- ``price_daily`` for GLD (gold) and UUP (dollar) overnight moves
- ``paper_trades`` table (open positions with P&L + summary stats)
- Earnings calendar data embedded in the report
- Economic calendar events for today (CPI, FOMC, etc.)
- Active Hyperliquid counter-trade signals

Public API
----------
``generate_morning_summary(db_path, report_dir) -> str``
``send_morning_summary(db_path, report_dir) -> bool``
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from pathlib import Path
from typing import Any

from trader_koo.notifications.telegram import is_configured, send_message

LOG = logging.getLogger("trader_koo.notifications.morning_summary")

# SGT is UTC+8 — the user wants the summary at 08:00 SGT = 00:00 UTC
_SGT = dt.timezone(dt.timedelta(hours=8))

# US market opens at 09:30 ET.  At 08:00 SGT (00:00 UTC) that is 19:00 ET
# the previous day during EST, or 20:00 ET during EDT.  The *next* open is
# ~13.5 h away (EST) or ~12.5 h (EDT).  We calculate dynamically below.
_ET = dt.timezone(dt.timedelta(hours=-5))  # EST baseline; DST handled inline


# ---------------------------------------------------------------------------
# Data fetching helpers (all from real DB / report)
# ---------------------------------------------------------------------------

def _fetch_index_snapshot(
    conn: sqlite3.Connection,
    ticker: str,
) -> dict[str, Any] | None:
    """Latest close + 1-day change for a ticker from ``price_daily``."""
    rows = conn.execute(
        """
        SELECT date, CAST(close AS REAL) AS close
        FROM price_daily
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT 2
        """,
        (ticker,),
    ).fetchall()
    if not rows:
        return None
    latest = float(rows[0]["close"])
    prev = float(rows[1]["close"]) if len(rows) >= 2 else latest
    change_pct = ((latest / prev) - 1.0) * 100.0 if prev > 0 else 0.0
    return {
        "ticker": ticker,
        "date": str(rows[0]["date"]),
        "close": round(latest, 2),
        "change_pct": round(change_pct, 2),
    }


def _vix_label(vix_close: float) -> str:
    """Translate VIX close to a human-readable volatility label."""
    if vix_close < 15:
        return "Low vol"
    if vix_close < 20:
        return "Moderate"
    if vix_close < 25:
        return "Elevated"
    if vix_close < 30:
        return "High"
    return "Extreme"


def _fear_greed_label(score: int | float) -> str:
    """Map a 0-100 fear/greed score to a label."""
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


def _fetch_fear_greed(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Compute fear/greed index from the DB."""
    try:
        from trader_koo.structure.fear_greed import compute_fear_greed_index

        result = compute_fear_greed_index(conn)
        if isinstance(result, dict) and result.get("composite_score") is not None:
            return result
    except Exception as exc:
        LOG.debug("Fear/greed computation failed (non-fatal): %s", exc)
    return None


def _extract_top_setups(
    report: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Pull top setups from the report's ``signals.setup_quality_top``."""
    signals = report.get("signals")
    if not isinstance(signals, dict):
        return []
    rows = signals.get("setup_quality_top")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        tier = str(row.get("setup_tier") or "?")
        bias = str(row.get("signal_bias") or "neutral").capitalize()
        # Key level: prefer support for bullish, resistance for bearish
        level_str = ""
        if bias.lower() == "bullish" and row.get("support_level"):
            level_str = f"Support ${row['support_level']}"
        elif bias.lower() == "bearish" and row.get("resistance_level"):
            level_str = f"Resistance ${row['resistance_level']}"
        elif row.get("support_level"):
            level_str = f"Support ${row['support_level']}"
        elif row.get("resistance_level"):
            level_str = f"Resistance ${row['resistance_level']}"
        # Fallback to observation snippet
        if not level_str:
            obs = str(row.get("observation") or "")
            if len(obs) > 40:
                obs = obs[:37] + "..."
            level_str = obs if obs else "See report"
        out.append({
            "ticker": ticker,
            "tier": tier,
            "bias": bias,
            "level": level_str,
        })
    return out


def _fetch_paper_trade_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Fetch paper-trade summary stats from the DB."""
    stats: dict[str, Any] = {
        "open_count": 0,
        "closed_today": 0,
        "win_rate_pct": 0.0,
        "portfolio_value": 1_000_000.0,
    }
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
        ).fetchone()
        stats["open_count"] = int(row[0]) if row else 0
    except Exception:
        pass

    today_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status != 'open' AND exit_date = ?",
            (today_str,),
        ).fetchone()
        stats["closed_today"] = int(row[0]) if row else 0
    except Exception:
        pass

    try:
        closed = conn.execute(
            "SELECT pnl_pct, position_size_pct FROM paper_trades WHERE status != 'open' AND pnl_pct IS NOT NULL"
        ).fetchall()
        if closed:
            total = len(closed)
            wins = sum(1 for r in closed if float(r[0]) > 0)
            stats["win_rate_pct"] = round((wins / total) * 100, 0) if total > 0 else 0.0

            # Portfolio value from equity curve
            starting_capital = 1_000_000.0
            realized = 0.0
            for r in closed:
                pnl_pct = float(r[0])
                pos_pct = float(r[1] or 8.0) if len(r) > 1 else 8.0
                position_dollars = starting_capital * (pos_pct / 100)
                realized += position_dollars * (pnl_pct / 100)
            # Include unrealized P&L from open trades
            unrealized = 0.0
            try:
                open_rows = conn.execute(
                    "SELECT unrealized_pnl_pct, position_size_pct FROM paper_trades WHERE status = 'open'"
                ).fetchall()
                for orow in open_rows:
                    u_pnl = float(orow[0] or 0)
                    pos_pct = float(orow[1] or 8.0)
                    position_dollars = starting_capital * (pos_pct / 100)
                    unrealized += position_dollars * (u_pnl / 100)
            except Exception:
                pass
            stats["portfolio_value"] = round(starting_capital + realized + unrealized, 0)
    except Exception:
        pass
    return stats


def _extract_earnings_today(
    report: dict[str, Any],
) -> dict[str, list[str]]:
    """Extract today's earnings from the report's earnings_catalysts."""
    result: dict[str, list[str]] = {"BMO": [], "AMC": []}
    signals = report.get("signals")
    if not isinstance(signals, dict):
        return result
    catalysts = signals.get("earnings_catalysts")
    if not isinstance(catalysts, dict):
        return result
    events = catalysts.get("events")
    if not isinstance(events, list):
        return result

    today_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("earnings_date") or "")[:10] != today_str:
            # Also check tomorrow (since this runs at 00:00 UTC = previous US day)
            # The "today" for US market is the next business day
            continue
        ticker = str(ev.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        session = str(ev.get("earnings_session") or "TBD").upper()
        if session == "BMO":
            result["BMO"].append(ticker)
        else:
            result["AMC"].append(ticker)

    # If nothing found for today, check the next trading day
    if not result["BMO"] and not result["AMC"]:
        tomorrow = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).strftime("%Y-%m-%d")
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("earnings_date") or "")[:10] != tomorrow:
                continue
            ticker = str(ev.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            session = str(ev.get("earnings_session") or "TBD").upper()
            if session == "BMO":
                result["BMO"].append(ticker)
            else:
                result["AMC"].append(ticker)

    return result


def _count_watched_tickers(report: dict[str, Any]) -> int:
    """Count the number of tickers in the setup watchlist."""
    signals = report.get("signals")
    if not isinstance(signals, dict):
        return 0
    top = signals.get("setup_quality_top")
    if isinstance(top, list):
        return len(top)
    return 0


def _hours_to_market_open() -> float:
    """Estimate hours until US market open (09:30 ET) from current UTC time."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    # US Eastern Time — approximate with fixed offset; actual DST logic not critical
    # for a rough "X hours" display
    et_offset = dt.timedelta(hours=-5)
    # Check if we're in DST (Mar second Sun – Nov first Sun) — simplified
    month = now_utc.month
    if 3 < month < 11:
        et_offset = dt.timedelta(hours=-4)
    elif month == 3 and now_utc.day >= 8:
        et_offset = dt.timedelta(hours=-4)
    elif month == 11 and now_utc.day < 7:
        et_offset = dt.timedelta(hours=-4)

    et_tz = dt.timezone(et_offset)
    now_et = now_utc.astimezone(et_tz)

    # Next market open: today 09:30 ET if before that, else tomorrow 09:30 ET
    market_open_today = now_et.replace(
        hour=9, minute=30, second=0, microsecond=0,
    )
    if now_et >= market_open_today:
        market_open_today += dt.timedelta(days=1)
    # Skip weekends
    while market_open_today.weekday() >= 5:
        market_open_today += dt.timedelta(days=1)

    delta = market_open_today - now_et
    return round(delta.total_seconds() / 3600, 1)


def _fmt_money(value: float) -> str:
    """Format a dollar value like $1,012,340."""
    if value >= 1_000_000:
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def _fetch_crypto_snapshot(
    conn: sqlite3.Connection,
    symbol: str,
) -> dict[str, Any] | None:
    """Latest close + 24h change for a crypto symbol from ``crypto_bars``.

    Uses the 1d interval if available, otherwise falls back to the latest
    two 1h bars to approximate a 24h change.
    """
    try:
        rows = conn.execute(
            """
            SELECT timestamp, close
            FROM crypto_bars
            WHERE symbol = ? AND interval = '1d'
            ORDER BY timestamp DESC
            LIMIT 2
            """,
            (symbol,),
        ).fetchall()
        if rows and len(rows) >= 2:
            latest = float(rows[0]["close"])
            prev = float(rows[1]["close"])
            change_pct = ((latest / prev) - 1.0) * 100.0 if prev > 0 else 0.0
            return {
                "symbol": symbol,
                "close": round(latest, 2),
                "change_pct": round(change_pct, 2),
            }
        # Fallback: use latest 1h bar close vs 24h-ago 1h bar
        rows = conn.execute(
            """
            SELECT timestamp, close
            FROM crypto_bars
            WHERE symbol = ? AND interval = '1h'
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchall()
        if not rows:
            return None
        latest_close = float(rows[0]["close"])
        latest_ts = str(rows[0]["timestamp"])
        # Find bar ~24h ago
        try:
            ts_dt = dt.datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
            ago_ts = (ts_dt - dt.timedelta(hours=24)).isoformat()
        except Exception:
            return {"symbol": symbol, "close": round(latest_close, 2), "change_pct": 0.0}
        old_rows = conn.execute(
            """
            SELECT close FROM crypto_bars
            WHERE symbol = ? AND interval = '1h' AND timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (symbol, ago_ts),
        ).fetchall()
        if old_rows:
            prev = float(old_rows[0]["close"])
            change_pct = ((latest_close / prev) - 1.0) * 100.0 if prev > 0 else 0.0
        else:
            change_pct = 0.0
        return {
            "symbol": symbol,
            "close": round(latest_close, 2),
            "change_pct": round(change_pct, 2),
        }
    except Exception as exc:
        LOG.debug("Crypto snapshot for %s failed: %s", symbol, exc)
        return None


def _fetch_open_positions(
    conn: sqlite3.Connection,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Fetch open paper trade positions with unrealized P&L."""
    positions: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            """
            SELECT ticker, direction, entry_price,
                   unrealized_pnl_pct, position_size_pct, entry_date
            FROM paper_trades
            WHERE status = 'open'
            ORDER BY entry_date DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        for row in rows:
            pnl = float(row["unrealized_pnl_pct"] or 0)
            positions.append({
                "ticker": str(row["ticker"]),
                "direction": str(row["direction"] or "long"),
                "entry_price": float(row["entry_price"] or 0),
                "pnl_pct": round(pnl, 2),
                "size_pct": float(row["position_size_pct"] or 8.0),
            })
    except Exception as exc:
        LOG.debug("Open positions fetch failed: %s", exc)
    return positions


def _fetch_economic_events_today() -> list[dict[str, Any]]:
    """Fetch today's (and tomorrow's) economic calendar events."""
    events: list[dict[str, Any]] = []
    try:
        from trader_koo.catalyst_data import fetch_economic_calendar

        # At 00:00 UTC, the upcoming US trading day is "today" UTC or
        # "tomorrow" UTC depending on weekday. Fetch a 2-day window.
        today_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        tomorrow_str = (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
        ).strftime("%Y-%m-%d")
        raw = fetch_economic_calendar(today_str, tomorrow_str)
        for ev in raw:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("impact", "")).lower() in ("high", "medium"):
                events.append({
                    "date": str(ev.get("date", ""))[:10],
                    "event": str(ev.get("event", "")),
                    "impact": str(ev.get("impact", "")),
                })
    except Exception as exc:
        LOG.debug("Economic calendar fetch failed: %s", exc)
    return events


def _fetch_counter_trade_signals(
    conn: sqlite3.Connection,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Fetch recent active counter-trade signals from Hyperliquid tracker."""
    signals: list[dict[str, Any]] = []
    try:
        columns = {
            str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
            for row in conn.execute("PRAGMA table_info(hyperliquid_counter_signals)")
        }
        if not columns:
            return []

        direction_expr = (
            "counter_side AS direction"
            if "counter_side" in columns
            else "direction"
            if "direction" in columns
            else "'unknown' AS direction"
        )
        entry_expr = (
            "entry_price"
            if "entry_price" in columns
            else "NULL AS entry_price"
        )
        wallet_expr = (
            "wallet_label"
            if "wallet_label" in columns
            else "'' AS wallet_label"
        )
        rows = conn.execute(
            f"""
            SELECT coin, {direction_expr}, {entry_expr}, signal_ts, {wallet_expr}
            FROM hyperliquid_counter_signals
            ORDER BY signal_ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        now_utc = dt.datetime.now(dt.timezone.utc)
        for row in rows:
            # Only include signals from the last 24 hours
            try:
                ts = dt.datetime.fromisoformat(
                    str(row["signal_ts"]).replace("Z", "+00:00")
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                if (now_utc - ts).total_seconds() > 86400:
                    continue
            except Exception:
                pass
            signals.append({
                "coin": str(row["coin"]),
                "direction": str(row["direction"]),
                "entry_price": float(row["entry_price"]) if row["entry_price"] is not None else None,
                "wallet": str(row["wallet_label"] or ""),
            })
    except Exception as exc:
        LOG.debug("Counter-trade signals fetch failed: %s", exc)
    return signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_morning_summary(
    db_path: Path,
    report_dir: Path,
) -> str:
    """Build the morning briefing Markdown message from live DB + report data.

    Returns the formatted Telegram message string.
    """
    from trader_koo.backend.services.report_loader import latest_daily_report_json

    now_sgt = dt.datetime.now(_SGT)
    date_label = now_sgt.strftime("%b %d, %Y")

    # Load latest report
    _, report = latest_daily_report_json(report_dir)
    report = report if isinstance(report, dict) else {}

    # Open DB connection
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Market snapshot
        spy = _fetch_index_snapshot(conn, "SPY")
        qqq = _fetch_index_snapshot(conn, "QQQ")
        vix = _fetch_index_snapshot(conn, "^VIX")

        # Overnight moves: crypto + macro proxies
        btc = _fetch_crypto_snapshot(conn, "BTC-USD")
        eth = _fetch_crypto_snapshot(conn, "ETH-USD")
        gold = _fetch_index_snapshot(conn, "GLD")
        dxy = _fetch_index_snapshot(conn, "UUP")

        # Fear/Greed
        fg = _fetch_fear_greed(conn)
        fg_score = int(fg.get("composite_score", 0)) if fg else None
        fg_label = _fear_greed_label(fg_score) if fg_score is not None else "N/A"

        # Top setups
        setups = _extract_top_setups(report, limit=5)

        # Paper trades: summary stats + individual open positions
        pt = _fetch_paper_trade_stats(conn)
        open_positions = _fetch_open_positions(conn, limit=8)

        # Earnings
        earnings = _extract_earnings_today(report)

        # Watched tickers
        watched = _count_watched_tickers(report)

        # Counter-trade signals (Hyperliquid)
        counter_signals = _fetch_counter_trade_signals(conn, limit=5)

    finally:
        conn.close()

    # Crypto derivatives (funding, L/S, F&G) — uses its own connection
    try:
        from trader_koo.crypto.derivatives import get_latest_derivatives_summary
        derivatives = get_latest_derivatives_summary(db_path)
    except Exception:
        derivatives = {"funding": {}, "ls_ratios": {}, "fng": None}

    # Economic calendar events (no DB needed)
    econ_events = _fetch_economic_events_today()

    # Hours to market open
    hours_to_open = _hours_to_market_open()

    # Build message
    lines: list[str] = []

    # Header
    lines.append(f"\u2600\ufe0f *Morning Briefing \u2014 {date_label}*")
    lines.append("")

    # Market Snapshot
    lines.append("\U0001f4ca *Market Snapshot*")
    spy_line = _format_index_line("SPY", spy)
    qqq_line = _format_index_line("QQQ", qqq)
    lines.append(f"{spy_line}  |  {qqq_line}")

    vix_line = f"VIX: {vix['close']}" if vix else "VIX: N/A"
    if vix:
        vix_line += f" ({_vix_label(vix['close'])})"
    fg_line = f"Fear/Greed: {fg_score} ({fg_label})" if fg_score is not None else "Fear/Greed: N/A"
    lines.append(f"{vix_line}  |  {fg_line}")
    lines.append("")

    # Overnight Moves (crypto + macro)
    overnight_parts: list[str] = []
    if btc:
        sign = "+" if btc["change_pct"] >= 0 else ""
        overnight_parts.append(f"BTC ${btc['close']:,.0f} ({sign}{btc['change_pct']}%)")
    if eth:
        sign = "+" if eth["change_pct"] >= 0 else ""
        overnight_parts.append(f"ETH ${eth['close']:,.0f} ({sign}{eth['change_pct']}%)")
    if gold:
        sign = "+" if gold["change_pct"] >= 0 else ""
        overnight_parts.append(f"Gold ${gold['close']} ({sign}{gold['change_pct']}%)")
    if dxy:
        sign = "+" if dxy["change_pct"] >= 0 else ""
        overnight_parts.append(f"DXY ${dxy['close']} ({sign}{dxy['change_pct']}%)")
    if overnight_parts:
        lines.append("\U0001f30d *Overnight Moves*")
        # Two items per line for readability
        for i in range(0, len(overnight_parts), 2):
            pair = overnight_parts[i : i + 2]
            lines.append("  |  ".join(pair))
        lines.append("")

    # Crypto Derivatives
    deriv_parts: list[str] = []
    if derivatives.get("fng"):
        fng_val = derivatives["fng"]["value"]
        fng_cls = derivatives["fng"]["classification"]
        deriv_parts.append(f"F&G: {fng_val} ({fng_cls})")
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        fr = derivatives.get("funding", {}).get(sym)
        ls = derivatives.get("ls_ratios", {}).get(sym)
        if fr:
            label = sym.split("-")[0]
            fr_str = f"{label} FR: {fr['rate_pct']:+.4f}%"
            if ls:
                fr_str += f" | L/S: {ls['long_pct']}/{ls['short_pct']}"
            deriv_parts.append(fr_str)
    if deriv_parts:
        lines.append("\U0001f4b0 *Crypto Derivatives*")
        for i in range(0, len(deriv_parts), 2):
            pair = deriv_parts[i : i + 2]
            lines.append("  |  ".join(pair))
        lines.append("")

    # Top Setups
    if setups:
        lines.append(f"\U0001f3af *Top {len(setups)} Setups Today*")
        for idx, s in enumerate(setups, 1):
            lines.append(
                f"{idx}. {s['ticker']} \u2014 Tier {s['tier']} | {s['bias']} | {s['level']}"
            )
        lines.append("")

    # Paper Trades: summary + open positions
    lines.append("\U0001f4c8 *Paper Trades*")
    lines.append(
        f"Open: {pt['open_count']} | Closed today: {pt['closed_today']}"
    )
    lines.append(
        f"Win rate: {int(pt['win_rate_pct'])}% | Equity: {_fmt_money(pt['portfolio_value'])}"
    )
    if open_positions:
        lines.append("")
        lines.append("_Open positions:_")
        for pos in open_positions:
            pnl_sign = "+" if pos["pnl_pct"] >= 0 else ""
            direction_arrow = "\u2191" if pos["direction"] == "long" else "\u2193"
            lines.append(
                f"  {direction_arrow} {pos['ticker']} @ ${pos['entry_price']:,.2f}"
                f" ({pnl_sign}{pos['pnl_pct']}%)"
            )
    lines.append("")

    # Earnings Today
    bmo = earnings.get("BMO", [])
    amc = earnings.get("AMC", [])
    if bmo or amc:
        lines.append("\U0001f4c5 *Earnings Today*")
        parts: list[str] = []
        if bmo:
            parts.append(f"PRE: {', '.join(bmo[:8])}")
        if amc:
            parts.append(f"AFT: {', '.join(amc[:8])}")
        lines.append("  |  ".join(parts))
        lines.append("")

    # Economic Calendar
    if econ_events:
        lines.append("\U0001f4cb *Economic Calendar*")
        for ev in econ_events[:6]:
            impact_icon = "\U0001f534" if ev["impact"].lower() == "high" else "\U0001f7e0"
            lines.append(f"  {impact_icon} {ev['event']} ({ev['date'][:10]})")
        lines.append("")

    # Counter-Trade Signals (Hyperliquid)
    if counter_signals:
        lines.append("\u2694\ufe0f *Active Counter-Trade Signals*")
        for sig in counter_signals:
            direction_icon = "\U0001f7e2" if sig["direction"] == "long" else "\U0001f534"
            line = f"  {direction_icon} {sig['coin']} {sig['direction'].upper()}"
            if sig.get("entry_price") is not None:
                line += f" @ ${sig['entry_price']:,.2f}"
            if sig.get("wallet"):
                line += f" ({sig['wallet']})"
            lines.append(line)
        lines.append("")

    # Alerts watching
    if watched > 0:
        lines.append(f"\U0001f514 *Alerts watching: {watched} tickers*")
        lines.append("")

    # Time to open
    lines.append(f"_US market opens in {hours_to_open} hours_")

    return "\n".join(lines)


def _format_index_line(
    label: str,
    data: dict[str, Any] | None,
) -> str:
    """Format a single index line like ``SPY: $585.20 (+0.3%)``."""
    if not data:
        return f"{label}: N/A"
    sign = "+" if data["change_pct"] >= 0 else ""
    return f"{label}: ${data['close']} ({sign}{data['change_pct']}%)"


def send_morning_summary(
    db_path: Path,
    report_dir: Path,
) -> bool:
    """Generate and send the morning summary to Telegram.

    Returns True on success, False on failure (logged, never raises).
    """
    if not is_configured():
        LOG.warning(
            "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing). "
            "Skipping morning summary."
        )
        return False

    try:
        message = generate_morning_summary(db_path, report_dir)
    except Exception as exc:
        LOG.error("Failed to generate morning summary: %s", exc, exc_info=True)
        return False

    try:
        send_message(message)
        LOG.info("Morning summary sent to Telegram")
        return True
    except Exception as exc:
        LOG.error("Failed to send morning summary to Telegram: %s", exc, exc_info=True)
        return False

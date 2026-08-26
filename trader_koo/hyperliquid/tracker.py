"""Hyperliquid whale tracker - monitor and counter-trade tracked wallets.

Polls tracked wallets for position changes. Stores snapshots in SQLite
for historical analysis. Generates counter-trade signals when tracked
traders open large positions (inverse direction, scaled by their size).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import sqlite3
import urllib.request
from dataclasses import asdict, dataclass
from html import escape
from typing import Any

from trader_koo.hyperliquid.wallets import get_tracked_wallets

LOG = logging.getLogger(__name__)

_TELEGRAM_REQUEST_TIMEOUT_SECONDS = 10
_TELEGRAM_REENTRY_LOOKBACK_HOURS = 24
_HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
_HYPERLIQUID_FILL_PAGE_LIMIT = 2_000


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
    mark_price_source: str = "unknown"
    notional_source: str = "unknown"
    data_warnings: tuple[str, ...] = ()


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


def _as_finite_float(value: Any) -> float | None:
    """Parse API numeric strings without converting invalid values to real data."""
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _position_size(position: dict[str, Any]) -> float:
    raw_size = position.get("szi")
    if isinstance(raw_size, dict):
        return _as_finite_float(raw_size.get("base")) or 0.0
    return _as_finite_float(raw_size) or 0.0


def _derive_position_prices(
    position: dict[str, Any],
    abs_size: float,
) -> tuple[float, float, str, str, tuple[str, ...]]:
    """Return mark/notional values plus provenance for Hyperliquid positions."""
    warnings: list[str] = []
    direct_mark = _as_finite_float(position.get("markPx"))
    position_value = _as_finite_float(position.get("positionValue"))

    if direct_mark is not None and direct_mark > 0:
        mark_price = direct_mark
        mark_source = "markPx"
    elif position_value is not None and position_value > 0 and abs_size > 0:
        mark_price = position_value / abs_size
        mark_source = "positionValue"
    else:
        mark_price = 0.0
        mark_source = "missing"
        warnings.append("missing_mark_price")

    if position_value is not None and position_value >= 0:
        notional_usd = abs(position_value)
        notional_source = "positionValue"
    elif mark_price > 0 and abs_size > 0:
        notional_usd = abs_size * mark_price
        notional_source = f"{mark_source}_times_size"
        warnings.append("missing_position_value")
    else:
        notional_usd = 0.0
        notional_source = "missing"
        warnings.append("missing_position_value")

    return mark_price, notional_usd, mark_source, notional_source, tuple(warnings)


def fetch_wallet_state(
    wallet_address: str,
    wallet_label: str = "",
) -> WalletSnapshot | None:
    """Fetch current positions and account state for a wallet."""
    try:
        info = _get_info_client()
        state = info.user_state(wallet_address)

        margin = state.get("marginSummary", {})
        account_value = _as_finite_float(margin.get("accountValue")) or 0.0
        total_margin = _as_finite_float(margin.get("totalMarginUsed")) or 0.0
        margin_ratio = total_margin / account_value if account_value > 0 else 0

        positions: list[WalletPosition] = []
        for asset_pos in state.get("assetPositions", []):
            p = asset_pos.get("position", {})
            sz = _position_size(p)
            if sz == 0:
                continue

            lev = p.get("leverage", {})
            entry_px = _as_finite_float(p.get("entryPx")) or 0.0
            abs_size = abs(sz)
            mark_px, notional_usd, mark_source, notional_source, data_warnings = (
                _derive_position_prices(p, abs_size)
            )
            liq_px = _as_finite_float(p.get("liquidationPx"))

            positions.append(WalletPosition(
                wallet_label=wallet_label,
                wallet_address=wallet_address,
                coin=p.get("coin", ""),
                side="long" if sz > 0 else "short",
                size=abs_size,
                entry_price=entry_px,
                mark_price=mark_px,
                unrealized_pnl=_as_finite_float(p.get("unrealizedPnl")) or 0.0,
                leverage_type=str(lev.get("type", "cross")),
                leverage_value=int(_as_finite_float(lev.get("value")) or 1),
                notional_usd=notional_usd,
                liquidation_price=liq_px,
                mark_price_source=mark_source,
                notional_source=notional_source,
                data_warnings=data_warnings,
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

    Returns the provider portfolio series plus a capped execution summary.
    Fill-derived win rates stay unavailable when the provider page is truncated.
    """
    import time as _time

    portfolio = fetch_wallet_portfolio_history(wallet_address, lookback_days)
    try:
        info = _get_info_client()
        start_ms = int((_time.time() - lookback_days * 86400) * 1000)
        fills = info.user_fills_by_time(wallet_address, start_ms)

        if not fills:
            return {
                "fill_count": 0,
                "lookback_days": lookback_days,
                "stats": {},
                "by_coin": {},
                "execution_coverage": {
                    "complete": True,
                    "start": None,
                    "end": None,
                    "reason": None,
                },
                "portfolio": portfolio,
            }

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

        fill_times = sorted(
            timestamp
            for timestamp in (int(f.get("time") or 0) for f in fills)
            if timestamp > 0
        )
        execution_complete = len(fills) < _HYPERLIQUID_FILL_PAGE_LIMIT
        return {
            "fill_count": len(fills),
            "lookback_days": lookback_days,
            "stats": {
                "total_pnl": round(total_pnl, 2),
                "total_fees": round(fees, 2),
                "net_pnl": round(total_pnl - fees, 2),
                "wins": wins,
                "losses": losses,
                "win_rate_pct": (
                    round(wins / (wins + losses) * 100, 1)
                    if execution_complete and (wins + losses) > 0
                    else None
                ),
                "liquidations": liqs,
            },
            "by_coin": {
                coin: {
                    "pnl": round(data["pnl"], 2),
                    "fills": data["fills"],
                    "win_rate_pct": (
                        round(data["wins"] / (data["wins"] + data["losses"]) * 100, 1)
                        if execution_complete and (data["wins"] + data["losses"]) > 0
                        else None
                    ),
                }
                for coin, data in sorted(by_coin.items(), key=lambda x: x[1]["pnl"])
            },
            "execution_coverage": {
                "complete": execution_complete,
                "start": _millis_to_iso(fill_times[0]) if fill_times else None,
                "end": _millis_to_iso(fill_times[-1]) if fill_times else None,
                "reason": None if execution_complete else "hyperliquid_2000_execution_page_cap",
            },
            "portfolio": portfolio,
        }
    except Exception as exc:
        LOG.warning("Failed to fetch wallet history for %s: %s", wallet_address, exc)
        return {
            "fill_count": 0,
            "lookback_days": lookback_days,
            "stats": {},
            "by_coin": {},
            "execution_coverage": {
                "complete": False,
                "start": None,
                "end": None,
                "reason": "execution_history_unavailable",
            },
            "portfolio": portfolio,
        }


def _millis_to_iso(value: int) -> str:
    return dt.datetime.fromtimestamp(value / 1000, tz=dt.timezone.utc).isoformat()


def _parse_portfolio_history(
    payload: Any,
    lookback_days: int,
    *,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Convert provider portfolio history into one truthful point per UTC day."""
    period = "week" if lookback_days <= 7 else "month" if lookback_days <= 30 else "allTime"
    periods = dict(payload) if isinstance(payload, list) else {}
    selected = periods.get(period)
    if not isinstance(selected, dict):
        return {"available": False, "source": "hyperliquid_portfolio", "period": period, "daily": []}

    accounts = {
        int(row[0]): float(row[1])
        for row in (selected.get("accountValueHistory") or [])
        if isinstance(row, list) and len(row) >= 2
    }
    pnl = {
        int(row[0]): float(row[1])
        for row in (selected.get("pnlHistory") or [])
        if isinstance(row, list) and len(row) >= 2
    }
    now = (now_utc or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    cutoff_ms = int((now - dt.timedelta(days=lookback_days)).timestamp() * 1000)
    daily_last: dict[str, tuple[int, float, float | None]] = {}
    for timestamp, account_value in accounts.items():
        if timestamp < cutoff_ms:
            continue
        date = dt.datetime.fromtimestamp(timestamp / 1000, tz=dt.timezone.utc).date().isoformat()
        candidate = (timestamp, account_value, pnl.get(timestamp))
        if date not in daily_last or timestamp > daily_last[date][0]:
            daily_last[date] = candidate

    points: list[dict[str, Any]] = []
    previous_pnl: float | None = None
    for date, (timestamp, account_value, period_pnl) in sorted(daily_last.items()):
        daily_change = (
            period_pnl - previous_pnl
            if period_pnl is not None and previous_pnl is not None
            else None
        )
        points.append({
            "date": date,
            "timestamp": _millis_to_iso(timestamp),
            "account_value": round(account_value, 2),
            "period_pnl": round(period_pnl, 2) if period_pnl is not None else None,
            "daily_pnl_change": round(daily_change, 2) if daily_change is not None else None,
        })
        if period_pnl is not None:
            previous_pnl = period_pnl

    first = points[0] if points else None
    last = points[-1] if points else None
    pnl_points = [point["period_pnl"] for point in points if point["period_pnl"] is not None]
    return {
        "available": bool(points),
        "source": "hyperliquid_portfolio",
        "period": period,
        "coverage_start": first["timestamp"] if first else None,
        "coverage_end": last["timestamp"] if last else None,
        "account_value": last["account_value"] if last else None,
        "account_value_change": (
            round(last["account_value"] - first["account_value"], 2)
            if first and last else None
        ),
        "period_pnl_change": (
            round(pnl_points[-1] - pnl_points[0], 2)
            if len(pnl_points) >= 2 else None
        ),
        "daily": points,
    }


def fetch_wallet_portfolio_history(wallet_address: str, lookback_days: int) -> dict[str, Any]:
    """Fetch provider account-value and PnL history without inferring it from fills."""
    request = urllib.request.Request(
        _HYPERLIQUID_INFO_URL,
        data=json.dumps({"type": "portfolio", "user": wallet_address}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "trader-koo/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return _parse_portfolio_history(payload, lookback_days)
    except Exception as exc:
        LOG.warning("Failed to fetch wallet portfolio history for %s: %s", wallet_address, exc)
        return {
            "available": False,
            "source": "hyperliquid_portfolio",
            "period": None,
            "daily": [],
            "error": type(exc).__name__,
        }


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
        LIMIT 500
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


# Minimum notional to promote a signal to COUNTER (configurable via env).
# Study-derived default: only very large positions showed counter-trade edge.
_MIN_COUNTER_NOTIONAL_USD = float(
    os.getenv("TRADER_KOO_HL_MIN_COUNTER_NOTIONAL_USD", "25000000")
)

# Coins where he consistently wins — do NOT counter-trade these (configurable).
# BTC: 94.7% WR over 19 cycles (+$489K). He's skilled at BTC.
# Set env var to comma-separated list: "BTC,SOL" or "" to disable.
_SKIP_COUNTER_COINS: frozenset[str] = frozenset(
    c.strip() for c in os.getenv("TRADER_KOO_HL_SKIP_COUNTER_COINS", "BTC").split(",")
    if c.strip()
)
_RELOAD_SIGNAL_BOOST = max(0, int(os.getenv("TRADER_KOO_HL_RELOAD_SIGNAL_BOOST", "3")))
_RELOAD_LOOKBACK_HOURS = max(
    1.0,
    float(os.getenv("TRADER_KOO_HL_RELOAD_LOOKBACK_HOURS", "72")),
)
_CROWD_RATIO_THRESHOLD = max(
    1.1,
    float(os.getenv("TRADER_KOO_HL_CROWD_RATIO_THRESHOLD", "1.8")),
)
_CROWD_FUNDING_THRESHOLD = max(
    0.0,
    float(os.getenv("TRADER_KOO_HL_CROWD_FUNDING_THRESHOLD", "0.0003")),
)


def get_counter_signal_config() -> dict[str, Any]:
    """Expose live signal thresholds from the backend single source of truth."""
    return {
        "min_counter_notional_usd": _MIN_COUNTER_NOTIONAL_USD,
        "skip_counter_coins": sorted(_SKIP_COUNTER_COINS),
        "reload_signal_boost": _RELOAD_SIGNAL_BOOST,
        "reload_lookback_hours": _RELOAD_LOOKBACK_HOURS,
        "crowd_ratio_threshold": _CROWD_RATIO_THRESHOLD,
        "crowd_funding_threshold": _CROWD_FUNDING_THRESHOLD,
    }


def _format_millions(value: float) -> str:
    return f"${value / 1_000_000:.0f}M"


def _html(value: Any) -> str:
    """Escape external text before inserting it into Telegram HTML."""
    return escape(str(value), quote=False)


_DERIVATIVE_SYMBOLS: dict[str, str] = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
}


def _parse_iso_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _recent_reload_context(
    conn: sqlite3.Connection | None,
    wallet_label: str,
    as_of_ts: str,
) -> dict[str, Any] | None:
    """Return the latest reload event if it is still within the boost window."""
    if conn is None:
        return None

    as_of = _parse_iso_ts(as_of_ts) or dt.datetime.now(dt.timezone.utc)
    try:
        row = conn.execute(
            """
            SELECT account_value, position_count, detected_ts
            FROM hyperliquid_reload_events
            WHERE wallet_label = ?
            ORDER BY detected_ts DESC
            LIMIT 1
            """,
            (wallet_label,),
        ).fetchone()
    except sqlite3.Error:
        return None

    if not row:
        return None

    detected_at = _parse_iso_ts(row[2])
    if detected_at is None:
        return None

    hours_since = max(0.0, (as_of - detected_at).total_seconds() / 3600)
    if hours_since > _RELOAD_LOOKBACK_HOURS:
        return None

    return {
        "detected_ts": detected_at.isoformat(),
        "hours_since": round(hours_since, 1),
        "account_value": float(row[0] or 0),
        "position_count": int(row[1] or 0),
        "score_boost": _RELOAD_SIGNAL_BOOST,
    }


def _latest_market_crowding_context(
    conn: sqlite3.Connection | None,
    coin: str,
    their_side: str,
) -> dict[str, Any] | None:
    """Build free market crowding context from stored Binance derivatives data."""
    if conn is None:
        return None

    symbol = _DERIVATIVE_SYMBOLS.get(coin.upper())
    if not symbol:
        return None

    funding_rate: float | None = None
    ratio: float | None = None
    long_pct: float | None = None
    short_pct: float | None = None

    try:
        funding_row = conn.execute(
            """
            SELECT funding_rate
            FROM crypto_funding_rates
            WHERE symbol = ?
            ORDER BY snapshot_ts DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if funding_row:
            funding_rate = float(funding_row[0])

        ratio_row = conn.execute(
            """
            SELECT long_account, short_account, long_short_ratio
            FROM crypto_long_short_ratio
            WHERE symbol = ?
            ORDER BY snapshot_ts DESC
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if ratio_row:
            long_pct = float(ratio_row[0]) * 100
            short_pct = float(ratio_row[1]) * 100
            ratio = float(ratio_row[2])
    except sqlite3.Error:
        return None

    if funding_rate is None and ratio is None:
        return None

    score_boost = 0
    notes: list[str] = []
    crowd_side = "neutral"

    if funding_rate is not None and abs(funding_rate) >= _CROWD_FUNDING_THRESHOLD:
        if funding_rate > 0:
            crowd_side = "long"
            if their_side == "long":
                score_boost += 1
            notes.append(f"funding {funding_rate * 100:+.4f}% longs crowded")
        else:
            crowd_side = "short"
            if their_side == "short":
                score_boost += 1
            notes.append(f"funding {funding_rate * 100:+.4f}% shorts crowded")

    if ratio is not None:
        if ratio >= _CROWD_RATIO_THRESHOLD:
            crowd_side = "long"
            if their_side == "long":
                score_boost += 1
            if long_pct is not None and short_pct is not None:
                notes.append(f"Binance top traders {long_pct:.0f}/{short_pct:.0f} long")
            else:
                notes.append(f"Binance long/short ratio {ratio:.2f}")
        elif ratio <= 1.0 / _CROWD_RATIO_THRESHOLD:
            crowd_side = "short"
            if their_side == "short":
                score_boost += 1
            if long_pct is not None and short_pct is not None:
                notes.append(f"Binance top traders {long_pct:.0f}/{short_pct:.0f} short")
            else:
                notes.append(f"Binance long/short ratio {ratio:.2f}")

    score_boost = min(score_boost, 2)
    if not notes:
        return {
            "symbol": symbol,
            "funding_rate_pct": round(funding_rate * 100, 4) if funding_rate is not None else None,
            "long_short_ratio": round(ratio, 2) if ratio is not None else None,
            "long_pct": round(long_pct, 1) if long_pct is not None else None,
            "short_pct": round(short_pct, 1) if short_pct is not None else None,
            "crowd_side": crowd_side,
            "aligns_with_counter": False,
            "score_boost": 0,
            "summary": "No strong crowding signal",
        }

    counter_side = "short" if their_side == "long" else "long"
    return {
        "symbol": symbol,
        "funding_rate_pct": round(funding_rate * 100, 4) if funding_rate is not None else None,
        "long_short_ratio": round(ratio, 2) if ratio is not None else None,
        "long_pct": round(long_pct, 1) if long_pct is not None else None,
        "short_pct": round(short_pct, 1) if short_pct is not None else None,
        "crowd_side": crowd_side,
        "aligns_with_counter": score_boost > 0,
        "score_boost": score_boost,
        "summary": "; ".join(notes[:2]),
        "counter_side": counter_side,
    }


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
    - Notional gate: only COUNTER for positions above configured threshold
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

    recent_reload = _recent_reload_context(conn, snapshot.wallet_label, snapshot.timestamp)
    signals: list[dict[str, Any]] = []
    for pos in snapshot.positions:
        counter_side = "short" if pos.side == "long" else "long"
        market_context = _latest_market_crowding_context(conn, pos.coin, pos.side)

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
                "wallet_context": {"recent_reload": recent_reload},
                "market_context": market_context,
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

        if recent_reload and _RELOAD_SIGNAL_BOOST > 0:
            score += _RELOAD_SIGNAL_BOOST
            reasons.append(
                f"post-reload {recent_reload['hours_since']:.0f}h after wipe"
            )

        if market_context and market_context.get("score_boost", 0) > 0:
            score += int(market_context["score_boost"])
            summary = str(market_context.get("summary") or "").strip()
            if summary:
                reasons.append(summary)

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
            reasons.append(
                f"notional ${pos.notional_usd / 1e6:.1f}M < "
                f"{_format_millions(_MIN_COUNTER_NOTIONAL_USD)} gate"
            )
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
            "wallet_context": {"recent_reload": recent_reload},
            "market_context": market_context,
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
    signal_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(hyperliquid_counter_signals)").fetchall()
    }
    for name, ddl in {
        "action": "TEXT",
        "score": "REAL",
        "reasons_json": "TEXT",
    }.items():
        if name not in signal_columns:
            conn.execute(f"ALTER TABLE hyperliquid_counter_signals ADD COLUMN {name} {ddl}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hl_signals_coin_ts "
        "ON hyperliquid_counter_signals(coin, signal_ts DESC)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hyperliquid_reload_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_label TEXT NOT NULL,
            wallet_address TEXT NOT NULL,
            account_value REAL,
            position_count INTEGER NOT NULL,
            detected_ts TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hl_reload_wallet_ts "
        "ON hyperliquid_reload_events(wallet_label, detected_ts DESC)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hyperliquid_alert_state (
            wallet_label TEXT NOT NULL,
            coin TEXT NOT NULL,
            signal_action TEXT NOT NULL,
            counter_side TEXT NOT NULL,
            updated_ts TEXT NOT NULL,
            PRIMARY KEY (wallet_label, coin)
        )
    """)
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
                 their_leverage, their_notional_usd, confidence, reasoning, signal_ts,
                 action, score, reasons_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig["wallet_label"], sig["coin"], sig["counter_side"],
                sig["their_side"], sig["their_size"], sig["their_leverage"],
                sig["their_notional_usd"], sig["confidence"],
                sig["reasoning"], sig["timestamp"],
                sig.get("action"),
                sig.get("score"),
                json.dumps(sig.get("reasons") or []),
            ),
        )
        inserted += 1
    conn.commit()
    return inserted


def seed_default_wallets(conn: sqlite3.Connection) -> None:
    """Insert default tracked wallets if not already present."""
    ensure_hyperliquid_schema(conn)
    for label, address in get_tracked_wallets().items():
        conn.execute(
            """
            INSERT INTO hyperliquid_wallets (label, address)
            VALUES (?, ?)
            ON CONFLICT(label) DO UPDATE SET
                address = excluded.address
            WHERE COALESCE(hyperliquid_wallets.address, '') != COALESCE(excluded.address, '')
            """,
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


def _recent_closed_position_context(
    conn: sqlite3.Connection,
    snapshot: WalletSnapshot,
    coin: str,
) -> dict[str, Any] | None:
    """Return the last position before a recent flat interval for ``coin``."""
    try:
        current_dt = dt.datetime.fromisoformat(snapshot.timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    cutoff = current_dt - dt.timedelta(hours=_TELEGRAM_REENTRY_LOOKBACK_HOURS)
    rows = conn.execute(
        """
        SELECT positions_json, snapshot_ts
        FROM hyperliquid_snapshots
        WHERE wallet_label = ?
          AND datetime(snapshot_ts) < datetime(?)
          AND datetime(snapshot_ts) >= datetime(?)
        ORDER BY datetime(snapshot_ts) DESC
        """,
        (snapshot.wallet_label, snapshot.timestamp, cutoff.isoformat()),
    ).fetchall()

    flat_since: dt.datetime | None = None
    for positions_json, snapshot_ts in rows:
        try:
            positions = json.loads(positions_json or "[]")
            row_dt = dt.datetime.fromisoformat(str(snapshot_ts).replace("Z", "+00:00"))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        previous = next((p for p in positions if p.get("coin") == coin), None)
        if previous is not None:
            if flat_since is None:
                return None
            return {
                "side": str(previous.get("side") or "?"),
                "size": float(previous.get("size") or 0),
                "flat_minutes": max(0, round((current_dt - flat_since).total_seconds() / 60)),
            }
        flat_since = row_dt
    return None


def _format_elapsed_minutes(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h" if remaining_minutes == 0 else f"{hours}h {remaining_minutes}m"


_TELEGRAM_LIFECYCLE_CHANGE_TYPES = frozenset({
    "new",
    "closed",
    "flipped",
    "partial_liq",
})


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
                    if liq_dist < 2:  # critical zone only — 2-5% is likely voluntary
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
            "HL snapshot: %s | $%s acct | %d positions | margin ratio %.2f",
            label, f"{snapshot.account_value:,.0f}", len(snapshot.positions), snapshot.margin_ratio,
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
        _send_telegram_signal_alert(conn, snapshot, signals, changes)

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
        "LIQUIDATION DETECTED: %s went from $%s (%d positions) to $%s (0 positions)",
        label, f"{float(prev_value):,.0f}", len(prev_positions), f"{snapshot.account_value:,.0f}",
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

    conn.execute(
        """
        INSERT INTO hyperliquid_reload_events
            (wallet_label, wallet_address, account_value, position_count, detected_ts)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            label,
            snapshot.wallet_address,
            snapshot.account_value,
            len(snapshot.positions),
            snapshot.timestamp,
        ),
    )
    conn.commit()

    LOG.info(
        "RELOAD DETECTED: %s back with $%s and %d positions",
        label,
        f"{snapshot.account_value:,.0f}",
        len(snapshot.positions),
    )

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
        f"<b>RELOAD: {_html(snapshot.wallet_label)}</b>",
        f"Back with ${snapshot.account_value:,.0f} | {leverage:.0f}x leverage",
        "",
    ]

    for p in snapshot.positions:
        lines.append(
            f"  {_html(p.coin)} {_html(p.side.upper())} "
            f"${p.notional_usd:,.0f} at {p.leverage_value}x"
        )

    lines.append("")
    lines.append("Watch for research signals; no automatic trade.")

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
        f"<b>LIQUIDATED: {_html(label)}</b>",
        f"Account went from ${prev_value:,.0f} to $0",
        f"Last seen: {_html(prev_ts)}",
        "",
    ]

    for p in prev_positions[:5]:
        coin = p.get("coin", "?")
        side = p.get("side", "?").upper()
        notional = float(p.get("notional_usd", 0))
        leverage = p.get("leverage_value", "?")
        liq_px = p.get("liquidation_price")
        entry_px = float(p.get("entry_price", 0))
        lines.append(
            f"  {_html(coin)} {_html(side)} ${notional:,.0f} "
            f"at {_html(leverage)}x (entry ${entry_px:,.2f})"
        )
        if liq_px:
            lines.append(f"  Liq price was ${float(liq_px):,.2f}")

    lines.append("")
    lines.append("One liquidation event observed; log it as evidence, not proof.")

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


def _signal_alert_states(
    signals: list[dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    """Return the semantic signal state that controls Telegram transitions."""
    return {
        str(signal["coin"]): (
            str(signal.get("action") or "").upper(),
            str(signal.get("counter_side") or "").upper(),
        )
        for signal in signals
        if signal.get("coin")
    }


def _load_signal_alert_states(
    conn: sqlite3.Connection,
    wallet_label: str,
) -> dict[str, tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT coin, signal_action, counter_side
        FROM hyperliquid_alert_state
        WHERE wallet_label = ?
        """,
        (wallet_label,),
    ).fetchall()
    return {coin: (action, counter_side) for coin, action, counter_side in rows}


def _store_signal_alert_states(
    conn: sqlite3.Connection,
    snapshot: WalletSnapshot,
    states: dict[str, tuple[str, str]],
) -> None:
    """Atomically replace one wallet's last observed semantic signal state."""
    conn.execute(
        "DELETE FROM hyperliquid_alert_state WHERE wallet_label = ?",
        (snapshot.wallet_label,),
    )
    conn.executemany(
        """
        INSERT INTO hyperliquid_alert_state
            (wallet_label, coin, signal_action, counter_side, updated_ts)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (snapshot.wallet_label, coin, action, counter_side, snapshot.timestamp)
            for coin, (action, counter_side) in states.items()
        ],
    )
    conn.commit()


def _send_telegram_signal_alert(
    conn: sqlite3.Connection,
    snapshot: WalletSnapshot,
    signals: list[dict[str, Any]],
    changes: list[PositionChange] | None = None,
) -> bool:
    """Send Telegram only for lifecycle events or new COUNTER states.

    Position resizing and ordinary partial closes remain available in stored
    snapshots and the web UI. Signal alert state is stored in SQLite so the
    decision survives process restarts. If state persistence fails, the
    exception is allowed to stop the send rather than falling back in memory.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return False

    lifecycle_changes = [
        change
        for change in (changes or [])
        if change.change_type in _TELEGRAM_LIFECYCLE_CHANGE_TYPES
    ]
    current_states = _signal_alert_states(signals)
    previous_states = _load_signal_alert_states(conn, snapshot.wallet_label)

    critical_signals = [
        signal
        for signal in signals
        if str(signal.get("action") or "").upper() == "COUNTER"
        and previous_states.get(str(signal.get("coin")))
        != current_states.get(str(signal.get("coin")))
    ]
    if not lifecycle_changes and not critical_signals:
        _store_signal_alert_states(conn, snapshot, current_states)
        return False

    total_notional = sum(p.notional_usd for p in snapshot.positions)
    acct_leverage = total_notional / snapshot.account_value if snapshot.account_value > 0 else 0

    sig_by_coin = {s["coin"]: s for s in signals}
    affected_coins = {
        *(change.coin for change in lifecycle_changes),
        *(str(signal["coin"]) for signal in critical_signals),
    }

    lines = [f"<b>{_html(snapshot.wallet_label)}</b>"]
    lines.append(
        f"Account ${snapshot.account_value:,.0f} | {acct_leverage:.0f}x leverage"
        f" | {len(snapshot.positions)} positions"
    )
    if snapshot.positions and total_notional > 0:
        top_position = max(snapshot.positions, key=lambda position: position.notional_usd)
        top_share = top_position.notional_usd / total_notional * 100
        lines.append(
            f"Exposure ${total_notional:,.0f} gross | {_html(top_position.coin)} {top_share:.0f}%"
        )
    lines.append("")

    # Position changes section
    if lifecycle_changes:
        for ch in lifecycle_changes:
            _CHANGE_EMOJI = {
                "new": "\U0001f7e2",       # green circle
                "closed": "\u274c",         # red X
                "partial_liq": "\U0001f4a5",    # explosion — suspected partial liquidation
                "flipped": "\U0001f504",    # arrows
            }
            emoji = _CHANGE_EMOJI.get(ch.change_type, "\u2022")

            if ch.change_type == "closed":
                prev_size = ch.prev_size or 0.0
                lines.append(
                    f"{emoji} <b>{_html(ch.coin)}</b> CLOSED "
                    f"(was {_html(ch.prev_side or '?')} {prev_size:,.2f})"
                )
            elif ch.change_type == "partial_liq":
                prev_size = ch.prev_size or 0.0
                curr_size = ch.curr_size or 0.0
                delta_pct = ch.size_delta_pct or 0.0
                lines.append(
                    f"{emoji} <b>{_html(ch.coin)}</b> PARTIAL LIQ {delta_pct:+.0f}%"
                    f" ({prev_size:,.2f} \u2192 {curr_size:,.2f} {_html(ch.curr_side or '?')})"
                )
            elif ch.change_type == "new":
                curr_size = ch.curr_size or 0.0
                prior = _recent_closed_position_context(conn, snapshot, ch.coin)
                if prior is None:
                    lines.append(
                        f"{emoji} <b>{_html(ch.coin)}</b> NEW "
                        f"{_html((ch.curr_side or '?').upper())} {curr_size:,.2f}"
                    )
                else:
                    prior_size = float(prior["size"])
                    size_ratio = curr_size / prior_size if prior_size > 0 else None
                    ratio_text = f"; {size_ratio:.1f}x size" if size_ratio is not None else ""
                    lines.append(
                        f"{emoji} <b>{_html(ch.coin)}</b> REOPENED "
                        f"{_html((ch.curr_side or '?').upper())} {curr_size:,.2f} after "
                        f"{_format_elapsed_minutes(int(prior['flat_minutes']))} "
                        f"(was {_html(str(prior['side']).upper())} {prior_size:,.2f}{ratio_text})"
                    )
            elif ch.change_type == "flipped":
                curr_size = ch.curr_size or 0.0
                lines.append(
                    f"{emoji} <b>{_html(ch.coin)}</b> FLIPPED"
                    f" {_html(ch.prev_side or '?')} \u2192 {_html(ch.curr_side or '?')}"
                    f" ({curr_size:,.2f})"
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

    affected_positions = [
        position for position in snapshot.positions if position.coin in affected_coins
    ]
    if affected_positions:
        lines.append("<b>Affected positions:</b>")
        for pos in affected_positions:
            liq_info = ""
            if pos.liquidation_price:
                if pos.mark_price > 0:
                    if pos.side == "long":
                        dist = (pos.mark_price - pos.liquidation_price) / pos.mark_price * 100
                    else:
                        dist = (pos.liquidation_price - pos.mark_price) / pos.mark_price * 100
                    liq_info = f" | liq ${pos.liquidation_price:,.2f} ({dist:.1f}%)"
            lines.append(
                f"  {_html(pos.coin)} {_html(pos.side.upper())} ${pos.notional_usd:,.0f}"
                f" ({pos.leverage_value}x) uPnL ${pos.unrealized_pnl:+,.0f}{liq_info}"
            )
        lines.append("")

    # Counter signals for actionable research signals only
    if critical_signals:
        best = max(critical_signals, key=lambda s: s.get("score", 0))
        lines.append(
            "\U0001f6a8 <b>Hyperliquid research signal</b> "
            f"(score {_html(best.get('score', '?'))})"
        )
        for sig in critical_signals:
            reasons = sig.get("reasons", [])
            reasons_text = ", ".join(_html(reason) for reason in reasons)
            age_str = ""
            age_h = sig.get("position_age_hours")
            if age_h is not None and age_h > 1:
                if age_h >= 24:
                    age_str = f" [{age_h / 24:.0f}d held]"
                else:
                    age_str = f" [{age_h:.0f}h held]"
            lines.append(
                f"  {_html(str(sig['counter_side']).upper())} {_html(sig['coin'])}: "
                f"{reasons_text}{age_str}"
            )
        lines.append("")

    lines.append("<i>Research only. NFA. In-sample rule until paper validated.</i>")
    text = "\n".join(lines)

    try:
        import httpx

        response = httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=_TELEGRAM_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as exc:
        LOG.debug("Telegram whale alert failed: %s", exc)
        return False

    _store_signal_alert_states(conn, snapshot, current_states)
    return True

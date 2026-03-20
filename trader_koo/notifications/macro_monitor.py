"""Macro market alert system — yields, commodities, VIX, dollar.

Monitors macro instruments for significant daily moves and detects
risk-regime shifts (RISK_OFF / RISK_ON / MIXED).  Sends formatted
Telegram alerts when thresholds are breached or regime shifts occur.

Public API
----------
``check_macro_moves(db_path) -> list[dict]``
``detect_risk_regime(moves) -> dict``
``send_macro_alert(db_path) -> bool``
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import httpx

LOG = logging.getLogger("trader_koo.notifications.macro_monitor")

# ---------------------------------------------------------------------------
# Instrument configuration
# ---------------------------------------------------------------------------

MACRO_WATCH: dict[str, dict[str, Any]] = {
    # Yields
    "^TNX": {"name": "10Y Yield", "spike_threshold_pct": 2.0, "emoji": "\U0001f4c8"},
    "^TYX": {"name": "30Y Yield", "spike_threshold_pct": 2.0, "emoji": "\U0001f4c8"},
    "^IRX": {"name": "3M T-Bill", "spike_threshold_pct": 3.0, "emoji": "\U0001f4c8"},
    # Commodities
    "GLD": {"name": "Gold", "spike_threshold_pct": 2.0, "emoji": "\U0001f947"},
    "USO": {"name": "Oil", "spike_threshold_pct": 3.0, "emoji": "\U0001f6e2\ufe0f"},
    "SLV": {"name": "Silver", "spike_threshold_pct": 3.0, "emoji": "\U0001f4c8"},
    # Dollar
    "UUP": {"name": "USD Index", "spike_threshold_pct": 1.5, "emoji": "\U0001f4b5"},
    # Volatility
    "^VIX": {"name": "VIX", "spike_threshold_pct": 10.0, "emoji": "\U0001f4ca"},
}

# Yield tickers used for regime detection
_YIELD_TICKERS = {"^TNX", "^TYX", "^IRX"}
_SAFE_HAVEN_TICKERS = {"GLD", "SLV"}
_VIX_TICKER = "^VIX"

# Finnhub REST API
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_TIMEOUT_SEC = 10

# Alert cooldown: suppress macro alerts for 1 hour
MACRO_COOLDOWN_SEC = 3600

# Module-level cooldown tracker
_last_macro_alert_ts: float = 0.0


# ---------------------------------------------------------------------------
# Finnhub price fetching
# ---------------------------------------------------------------------------

def _get_finnhub_key() -> str:
    """Return the Finnhub API key from environment."""
    return os.getenv("FINNHUB_API_KEY", "").strip()


def _fetch_quote(ticker: str, api_key: str) -> float | None:
    """Fetch current price for *ticker* via Finnhub REST ``/quote``.

    Returns the current price (``c`` field) or ``None`` on failure.
    """
    if not api_key:
        LOG.warning("FINNHUB_API_KEY not set — cannot fetch quote for %s", ticker)
        return None

    try:
        with httpx.Client(timeout=FINNHUB_TIMEOUT_SEC) as client:
            resp = client.get(
                FINNHUB_QUOTE_URL,
                params={"symbol": ticker, "token": api_key},
            )
        if resp.status_code != 200:
            LOG.warning("Finnhub quote returned %d for %s", resp.status_code, ticker)
            return None

        data = resp.json()
        price = data.get("c")
        if price is None or price == 0:
            LOG.debug("Finnhub returned no price for %s: %s", ticker, data)
            return None
        return float(price)
    except httpx.HTTPError as exc:
        LOG.warning("Finnhub HTTP error for %s: %s", ticker, exc)
        return None
    except Exception as exc:
        LOG.warning("Finnhub quote fetch failed for %s: %s", ticker, exc)
        return None


def _get_prev_close(db_path: Path, ticker: str) -> float | None:
    """Retrieve yesterday's close from price_daily table.

    Falls back to the latest available row if today's row is missing.
    """
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT close FROM price_daily
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (ticker,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return float(row["close"])
    except Exception as exc:
        LOG.warning("Failed to get prev close for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def check_macro_moves(db_path: Path) -> list[dict[str, Any]]:
    """Compare current macro prices to previous close.

    For each instrument in MACRO_WATCH, fetches the current price via
    Finnhub and compares to the latest close in price_daily.  Returns
    a list of move dicts for instruments where the percent change
    exceeds the configured threshold.
    """
    api_key = _get_finnhub_key()
    if not api_key:
        LOG.warning("FINNHUB_API_KEY not set — macro monitor inactive")
        return []

    moves: list[dict[str, Any]] = []

    for ticker, config in MACRO_WATCH.items():
        prev_close = _get_prev_close(db_path, ticker)
        if prev_close is None or prev_close <= 0:
            LOG.debug("No prev close for %s — skipping", ticker)
            continue

        current = _fetch_quote(ticker, api_key)
        if current is None:
            continue

        change_pct = ((current - prev_close) / prev_close) * 100
        direction = "up" if change_pct > 0 else "down"
        threshold = config["spike_threshold_pct"]
        exceeded = abs(change_pct) >= threshold

        move: dict[str, Any] = {
            "ticker": ticker,
            "name": config["name"],
            "emoji": config["emoji"],
            "prev_close": round(prev_close, 4),
            "current": round(current, 4),
            "change_pct": round(change_pct, 2),
            "direction": direction,
            "threshold_pct": threshold,
            "exceeded": exceeded,
        }
        moves.append(move)

    LOG.info(
        "Macro monitor checked %d/%d instruments, %d exceeded threshold",
        len(moves),
        len(MACRO_WATCH),
        sum(1 for m in moves if m["exceeded"]),
    )
    return moves


def detect_risk_regime(moves: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the current macro environment as RISK_OFF, RISK_ON, or MIXED.

    Logic
    -----
    - RISK_OFF: yields spiking AND VIX up AND gold up (flight to safety)
    - RISK_ON:  yields dropping AND VIX down
    - MIXED:    otherwise

    Returns a dict with regime, confidence (0-100), and reasoning.
    """
    if not moves:
        return {
            "regime": "UNKNOWN",
            "confidence": 0,
            "reasoning": "No macro data available",
            "signals": [],
        }

    move_map = {m["ticker"]: m for m in moves}
    signals: list[str] = []

    # Yield direction (average across available yield tickers)
    yield_changes = [
        move_map[t]["change_pct"]
        for t in _YIELD_TICKERS
        if t in move_map
    ]
    yields_up = False
    yields_down = False
    if yield_changes:
        avg_yield_change = sum(yield_changes) / len(yield_changes)
        if avg_yield_change > 0.5:
            yields_up = True
            signals.append(f"Yields rising ({avg_yield_change:+.2f}%)")
        elif avg_yield_change < -0.5:
            yields_down = True
            signals.append(f"Yields falling ({avg_yield_change:+.2f}%)")

    # VIX
    vix_up = False
    vix_down = False
    if _VIX_TICKER in move_map:
        vix_change = move_map[_VIX_TICKER]["change_pct"]
        if vix_change > 3.0:
            vix_up = True
            signals.append(f"VIX surging ({vix_change:+.1f}%)")
        elif vix_change < -3.0:
            vix_down = True
            signals.append(f"VIX declining ({vix_change:+.1f}%)")

    # Safe haven (gold/silver)
    haven_changes = [
        move_map[t]["change_pct"]
        for t in _SAFE_HAVEN_TICKERS
        if t in move_map
    ]
    haven_up = False
    if haven_changes:
        avg_haven = sum(haven_changes) / len(haven_changes)
        if avg_haven > 0.5:
            haven_up = True
            signals.append(f"Safe havens bid ({avg_haven:+.2f}%)")
        elif avg_haven < -0.5:
            signals.append(f"Safe havens sold ({avg_haven:+.2f}%)")

    # Dollar
    dollar_up = False
    if "UUP" in move_map:
        uup_change = move_map["UUP"]["change_pct"]
        if uup_change > 0.3:
            dollar_up = True
            signals.append(f"Dollar strengthening ({uup_change:+.2f}%)")
        elif uup_change < -0.3:
            signals.append(f"Dollar weakening ({uup_change:+.2f}%)")

    # Regime classification
    risk_off_score = 0
    risk_on_score = 0

    if yields_up:
        risk_off_score += 1
    if yields_down:
        risk_on_score += 1
    if vix_up:
        risk_off_score += 2  # VIX spike is strong risk-off signal
    if vix_down:
        risk_on_score += 2
    if haven_up:
        risk_off_score += 1
    if dollar_up:
        risk_off_score += 1

    total_signals = risk_off_score + risk_on_score
    if total_signals == 0:
        regime = "MIXED"
        confidence = 20
        reasoning = "No strong directional signals"
    elif risk_off_score >= 3 and risk_off_score > risk_on_score:
        regime = "RISK_OFF"
        confidence = min(90, 40 + risk_off_score * 15)
        reasoning = "Multiple risk-off signals aligned"
    elif risk_on_score >= 3 and risk_on_score > risk_off_score:
        regime = "RISK_ON"
        confidence = min(90, 40 + risk_on_score * 15)
        reasoning = "Multiple risk-on signals aligned"
    elif risk_off_score > risk_on_score:
        regime = "RISK_OFF"
        confidence = min(60, 30 + risk_off_score * 10)
        reasoning = "Moderate risk-off tilt"
    elif risk_on_score > risk_off_score:
        regime = "RISK_ON"
        confidence = min(60, 30 + risk_on_score * 10)
        reasoning = "Moderate risk-on tilt"
    else:
        regime = "MIXED"
        confidence = 30
        reasoning = "Conflicting signals across asset classes"

    return {
        "regime": regime,
        "confidence": confidence,
        "reasoning": reasoning,
        "signals": signals,
        "risk_off_score": risk_off_score,
        "risk_on_score": risk_on_score,
    }


# ---------------------------------------------------------------------------
# Telegram alert formatting
# ---------------------------------------------------------------------------

def _format_macro_alert(
    moves: list[dict[str, Any]],
    regime: dict[str, Any],
) -> str:
    """Format macro moves and regime into a Telegram message."""
    regime_emoji = {
        "RISK_OFF": "\u26a0\ufe0f",
        "RISK_ON": "\u2705",
        "MIXED": "\U0001f504",
        "UNKNOWN": "\u2753",
    }

    header_emoji = regime_emoji.get(regime["regime"], "\U0001f30d")
    header = f"\U0001f30d *Macro Alert \u2014 {regime['regime']}*"
    if regime["confidence"] >= 50:
        header = f"{header_emoji} *Macro Alert \u2014 {regime['regime']}*"

    lines = [header, ""]

    # Sort moves: exceeded thresholds first, then by absolute change
    sorted_moves = sorted(
        moves,
        key=lambda m: (not m["exceeded"], -abs(m["change_pct"])),
    )

    for move in sorted_moves:
        sign = "+" if move["change_pct"] > 0 else ""
        flag = " \u26a0\ufe0f" if move["exceeded"] else ""

        # Format prices — yields display differently from dollar prices
        if move["ticker"] in _YIELD_TICKERS:
            price_fmt = f"{move['prev_close']:.3f} \u2192 {move['current']:.3f}"
        else:
            price_fmt = f"${move['prev_close']:,.2f} \u2192 ${move['current']:,.2f}"

        lines.append(
            f"{move['emoji']} {move['name']}: {price_fmt} "
            f"({sign}{move['change_pct']:.2f}%){flag}"
        )

    lines.append("")

    # Regime reasoning
    if regime["signals"]:
        for signal in regime["signals"]:
            lines.append(f"\u2022 {signal}")
        lines.append("")

    lines.append(f"\U0001f4a1 {regime['reasoning']}")
    if regime["confidence"] >= 50:
        lines.append(f"Confidence: {regime['confidence']}%")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_macro_alert(db_path: Path) -> bool:
    """Run macro checks and send Telegram alert if thresholds breached.

    Fires an alert when:
    - Any instrument exceeds its spike threshold, OR
    - A regime shift with confidence >= 50 is detected

    Returns True when an alert was sent.
    """
    global _last_macro_alert_ts  # noqa: PLW0603

    from trader_koo.notifications.telegram import is_configured, send_message

    if not is_configured():
        LOG.info("Telegram not configured — skipping macro alert")
        return False

    # Cooldown check
    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
    if now_ts - _last_macro_alert_ts < MACRO_COOLDOWN_SEC:
        LOG.debug("Macro alert on cooldown — skipping")
        return False

    moves = check_macro_moves(db_path)
    if not moves:
        LOG.debug("No macro data — skipping alert")
        return False

    regime = detect_risk_regime(moves)

    # Determine if alert is warranted
    any_exceeded = any(m["exceeded"] for m in moves)
    regime_shift = regime["regime"] in ("RISK_OFF", "RISK_ON") and regime["confidence"] >= 50

    if not any_exceeded and not regime_shift:
        LOG.debug(
            "No macro threshold breached and no regime shift — skipping alert"
        )
        return False

    msg = _format_macro_alert(moves, regime)
    sent = send_message(msg)
    if sent:
        _last_macro_alert_ts = now_ts
        LOG.info(
            "Macro alert sent: regime=%s confidence=%d exceeded=%d",
            regime["regime"],
            regime["confidence"],
            sum(1 for m in moves if m["exceeded"]),
        )
    else:
        LOG.warning("Failed to send macro alert via Telegram")

    return sent


def get_macro_live(db_path: Path) -> dict[str, Any]:
    """Return current prices + daily change for all MACRO_WATCH instruments.

    Used by the ``GET /api/macro-live`` endpoint.
    """
    api_key = _get_finnhub_key()
    instruments: list[dict[str, Any]] = []

    for ticker, config in MACRO_WATCH.items():
        prev_close = _get_prev_close(db_path, ticker)
        current: float | None = None
        change_pct: float | None = None

        if api_key:
            current = _fetch_quote(ticker, api_key)

        if current is not None and prev_close is not None and prev_close > 0:
            change_pct = round(((current - prev_close) / prev_close) * 100, 2)

        instruments.append({
            "ticker": ticker,
            "name": config["name"],
            "emoji": config["emoji"],
            "current": round(current, 4) if current is not None else None,
            "prev_close": round(prev_close, 4) if prev_close is not None else None,
            "change_pct": change_pct,
            "threshold_pct": config["spike_threshold_pct"],
            "exceeded": (
                abs(change_pct) >= config["spike_threshold_pct"]
                if change_pct is not None
                else False
            ),
        })

    # Also compute regime
    moves = [
        inst for inst in instruments
        if inst["current"] is not None and inst["prev_close"] is not None
    ]
    regime_input = [
        {
            "ticker": inst["ticker"],
            "name": inst["name"],
            "change_pct": inst["change_pct"],
            "direction": "up" if (inst["change_pct"] or 0) > 0 else "down",
        }
        for inst in moves
    ]
    regime = detect_risk_regime(regime_input)

    return {
        "ok": True,
        "instruments": instruments,
        "regime": regime,
        "checked_at": dt.datetime.now(dt.timezone.utc).replace(
            microsecond=0,
        ).isoformat(),
    }

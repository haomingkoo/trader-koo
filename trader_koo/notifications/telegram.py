"""Telegram Bot API integration for price alerts.

Sends formatted Markdown messages to a configured Telegram chat
via the Bot API ``sendMessage`` endpoint.  If ``TELEGRAM_BOT_TOKEN``
or ``TELEGRAM_CHAT_ID`` are missing the functions log a warning and
return ``False`` — they never crash the caller.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

LOG = logging.getLogger("trader_koo.notifications.telegram")

TELEGRAM_API_BASE = "https://api.telegram.org"
SEND_TIMEOUT_SEC = 15


def _get_credentials() -> tuple[str, str] | None:
    """Return (bot_token, chat_id) or None when either is missing."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        LOG.warning(
            "Telegram credentials missing — set TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID to enable notifications"
        )
        return None
    return token, chat_id


def is_configured() -> bool:
    """Return True when both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set."""
    return _get_credentials() is not None


def send_message(text: str, parse_mode: str = "Markdown") -> bool:
    """Convenience alias for ``send_telegram_message``."""
    return send_telegram_message(text, parse_mode=parse_mode)


# ------------------------------------------------------------------
# Low-level send
# ------------------------------------------------------------------


def send_telegram_message(
    text: str,
    parse_mode: str = "Markdown",
) -> bool:
    """Send *text* to the configured Telegram chat.

    Returns ``True`` on success, ``False`` on any failure (missing
    credentials, network error, non-200 response).
    """
    creds = _get_credentials()
    if creds is None:
        return False
    token, chat_id = creds

    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        with httpx.Client(timeout=SEND_TIMEOUT_SEC) as client:
            resp = client.post(url, json=payload)
        if resp.status_code != 200:
            LOG.error(
                "Telegram API returned %d: %s",
                resp.status_code,
                resp.text[:300],
            )
            return False
        LOG.info("Telegram message sent successfully")
        return True
    except httpx.HTTPError as exc:
        LOG.error("Telegram send failed (HTTP): %s", exc)
        return False
    except Exception as exc:
        LOG.error("Telegram send failed (unexpected): %s", exc)
        return False


# ------------------------------------------------------------------
# Rich price alert formatter
# ------------------------------------------------------------------

_ALERT_EMOJIS: dict[str, str] = {
    "approaching_support": "\U0001f7e2",       # green circle
    "approaching_resistance": "\U0001f534",     # red circle
    "breakout_above_resistance": "\U0001f680",  # rocket
    "breakdown_below_support": "\U0001f4a5",    # collision / explosion
}


def _alert_headline(level_type: str, alert_type: str) -> str:
    """Human-readable headline for the alert type."""
    headlines: dict[str, str] = {
        "approaching_support": "approaching support",
        "approaching_resistance": "approaching resistance",
        "breakout_above_resistance": "breakout above resistance",
        "breakdown_below_support": "breakdown below support",
    }
    return headlines.get(alert_type, f"near {level_type}")


def _action_hint(alert_type: str) -> str:
    """Contextual action text for each alert type."""
    hints: dict[str, str] = {
        "approaching_support": "Watch for bounce entry",
        "approaching_resistance": "Watch for rejection / exit",
        "breakout_above_resistance": "Monitor for continuation",
        "breakdown_below_support": "Monitor for breakdown follow-through",
    }
    return hints.get(alert_type, "Review price action")


def send_price_alert(
    ticker: str,
    current_price: float,
    level: float,
    level_type: str,
    setup_tier: str,
    bias: str,
    *,
    alert_type: str = "",
) -> bool:
    """Format and send a rich price alert to Telegram.

    Parameters
    ----------
    ticker:
        Symbol (e.g. ``"AAPL"``).
    current_price:
        Live tick price.
    level:
        Support or resistance level value.
    level_type:
        ``"support"`` or ``"resistance"``.
    setup_tier:
        Setup quality tier (``"A"``/``"B"``/``"C"``/``"D"``).
    bias:
        Signal bias (``"bullish"``/``"bearish"``/``"neutral"``).
    alert_type:
        One of ``approaching_support``, ``approaching_resistance``,
        ``breakout_above_resistance``, ``breakdown_below_support``.
        Auto-detected from *level_type* and price vs level if empty.

    Returns
    -------
    bool
        ``True`` when the message was sent successfully.
    """
    if not alert_type:
        if level_type == "support":
            alert_type = (
                "breakdown_below_support"
                if current_price < level
                else "approaching_support"
            )
        else:
            alert_type = (
                "breakout_above_resistance"
                if current_price > level
                else "approaching_resistance"
            )

    emoji = _ALERT_EMOJIS.get(alert_type, "\U0001f514")  # bell fallback
    headline = _alert_headline(level_type, alert_type)
    action = _action_hint(alert_type)
    bias_display = bias.capitalize() if bias else "Neutral"

    text = (
        f"{emoji} *{ticker}* {headline}\n"
        f"\n"
        f"\U0001f4b0 Price: ${current_price:,.2f}\n"
        f"\U0001f4cd Level: ${level:,.2f} ({level_type})\n"
        f"\U0001f4ca Setup: Tier {setup_tier} | {bias_display}\n"
        f"\u26a1 Action: {action}\n"
        f"\n"
        f"[View Chart \u2192]"
        f"(https://trader.kooexperience.com/chart?ticker={ticker})"
    )
    LOG.info(
        "Sending price alert: ticker=%s price=%.2f level=%.2f type=%s",
        ticker,
        current_price,
        level,
        alert_type,
    )
    return send_telegram_message(text)

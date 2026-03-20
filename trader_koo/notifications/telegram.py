"""Telegram Bot API integration for trader-koo notifications.

Sends messages via the Telegram Bot API using ``urllib`` (no extra
dependency).  All config comes from environment variables:

- ``TELEGRAM_BOT_TOKEN``  — bot token from @BotFather
- ``TELEGRAM_CHAT_ID``    — target chat/group ID
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

LOG = logging.getLogger("trader_koo.notifications.telegram")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

_API_BASE = "https://api.telegram.org/bot"
_TIMEOUT_SEC = 15


def is_configured() -> bool:
    """Return True when both bot token and chat ID are set."""
    return bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


def send_message(
    text: str,
    *,
    parse_mode: str = "Markdown",
    disable_web_page_preview: bool = True,
    token: str | None = None,
    chat_id: str | None = None,
) -> dict[str, Any]:
    """Send a text message to the configured Telegram chat.

    Returns the parsed JSON response from the Telegram API on success.
    Raises ``RuntimeError`` on HTTP or API errors.
    """
    bot_token = token or TELEGRAM_BOT_TOKEN
    target_chat = chat_id or TELEGRAM_CHAT_ID
    if not bot_token or not target_chat:
        raise RuntimeError(
            "Telegram not configured: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required"
        )

    url = f"{_API_BASE}{bot_token}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        LOG.error("Telegram API HTTP %s: %s", exc.code, error_body)
        raise RuntimeError(f"Telegram API error {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        LOG.error("Telegram API connection error: %s", exc.reason)
        raise RuntimeError(f"Telegram connection error: {exc.reason}") from exc

    if not body.get("ok"):
        LOG.error("Telegram API returned ok=false: %s", body)
        raise RuntimeError(f"Telegram API error: {body}")

    LOG.info("Telegram message sent to chat %s (message_id=%s)", target_chat, body.get("result", {}).get("message_id"))
    return body

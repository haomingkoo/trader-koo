"""Formatting helpers shared by Telegram notification surfaces."""
from __future__ import annotations

from typing import Any


def telegram_markdown_safe(value: Any, *, max_len: int = 160) -> str:
    """Keep dynamic report text from breaking Telegram's Markdown parser."""
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())
    for ch in ("*", "_", "`"):
        text = text.replace(ch, "")
    text = text.replace("[", "(").replace("]", ")")
    if len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip() + "..."
    return text

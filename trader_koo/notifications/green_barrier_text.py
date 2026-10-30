"""Telegram text for Green Barrier hits: the completed-bar reading beside the bar still forming."""
from __future__ import annotations

from typing import Any

from trader_koo.notifications.formatting import telegram_markdown_safe as _md_safe

GREEN_BARRIER_SUMMARY_NOTE = (
    "Williams %R(14) at or below the configured threshold on the last completed bar. "
    'Repeated daily while active; "now" is the bar still forming.'
)


def _in_progress_value(hit: dict[str, Any]) -> float | None:
    value = hit.get("in_progress_value")
    return float(value) if isinstance(value, (int, float)) else None


def format_green_barrier_line(hit: dict[str, Any]) -> str:
    ticker = _md_safe(str(hit.get("ticker") or "?"))
    timeframe = _md_safe(str(hit.get("timeframe") or "?"))
    value = hit.get("value")
    value_text = f"{float(value):.1f}" if isinstance(value, (int, float)) else "N/A"
    asof = _md_safe(str(hit.get("asof") or "unknown"))
    line = f"  🟢 {ticker} {timeframe.title()} %R {value_text} at {asof} close"
    now_value = _in_progress_value(hit)
    if now_value is not None:
        now_asof = _md_safe(str(hit.get("in_progress_asof") or "unknown"))
        line += f" · now {now_value:.1f} ({now_asof})"
    return line


def green_barrier_caption(
    hit: dict[str, Any],
    *,
    ticker: str,
    timeframe: str,
    value: float,
    threshold: float,
) -> str:
    asof = _md_safe(str(hit.get("asof") or "unknown"))
    caption = (
        f"🟢 *{ticker} {timeframe.title()} Green Barrier Current Condition*\n"
        f"Williams %R(14) at {asof} close: *{value:.1f}* | Threshold: *{threshold:g}*\n"
    )
    now_value = _in_progress_value(hit)
    if now_value is not None:
        now_asof = _md_safe(str(hit.get("in_progress_asof") or "unknown"))
        caption += f"Now: *{now_value:.1f}* ({now_asof}, bar still forming)\n"
    return caption + "Research context only — not a buy signal."

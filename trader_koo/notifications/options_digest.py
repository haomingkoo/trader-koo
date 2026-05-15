"""Telegram digest for options premium proxy snapshots."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import sqlite3
from pathlib import Path
from typing import Any

from trader_koo.config import get_options_config, normalize_options_limit
from trader_koo.notifications.telegram import is_configured, send_message
from trader_koo.options_research import build_options_premium_proxy

LOG = logging.getLogger("trader_koo.notifications.options_digest")


@dataclass(frozen=True)
class OptionsDigest:
    """Structured digest result so sending logic never parses message text."""

    message: str
    has_data: bool


def _fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    sign = "-" if number < 0 else ""
    value_abs = abs(number)
    if value_abs >= 1_000_000_000:
        return f"{sign}${value_abs / 1_000_000_000:.2f}B"
    if value_abs >= 1_000_000:
        return f"{sign}${value_abs / 1_000_000:.2f}M"
    if value_abs >= 1_000:
        return f"{sign}${value_abs / 1_000:.1f}K"
    return f"{sign}${value_abs:.0f}"


def _bias_label(value: Any) -> str:
    text = str(value or "")
    if text == "call_premium_skew":
        return "Call skew"
    if text == "put_premium_skew":
        return "Put skew"
    if text == "balanced":
        return "Balanced"
    return "Unknown"


def _resolve_limit(limit: int | None) -> int:
    if limit is None:
        limit = get_options_config().digest.limit
    return normalize_options_limit(limit, name="options digest limit")


def build_options_digest(db_path: Path, *, limit: int | None = None) -> OptionsDigest:
    """Build a compact Telegram digest from ``options_iv`` snapshots."""
    resolved_limit = _resolve_limit(limit)
    if not db_path.exists():
        return OptionsDigest(
            message="*Options Premium Proxy*\nDatabase not available.",
            has_data=False,
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        payload = build_options_premium_proxy(conn, limit=resolved_limit)
    finally:
        conn.close()

    rows = payload.get("rows") if isinstance(payload, dict) else []
    if not rows:
        detail = str(payload.get("detail") or "No option-chain snapshots are available yet.")
        return OptionsDigest(
            message="*Options Premium Proxy*\n" + detail,
            has_data=False,
        )

    latest = str(payload.get("latest_snapshot_ts") or "")
    date_label = latest[:16].replace("T", " ") if latest else "latest"
    lines = [
        f"*Options Premium Proxy* ({date_label} UTC)",
        "_Estimated from option-chain volume/OI. Not live signed flow._",
        "",
    ]
    for idx, row in enumerate(rows[:resolved_limit], start=1):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        net_volume = row.get("net_volume_premium")
        net_oi = row.get("net_oi_premium")
        bias = _bias_label(row.get("premium_bias"))
        pc_oi = row.get("put_call_oi_ratio")
        pc_part = ""
        if isinstance(pc_oi, (int, float)):
            pc_part = f" | P/C OI {float(pc_oi):.2f}"
        lines.append(
            f"{idx}. {ticker} - {bias} | Vol net {_fmt_money(net_volume)} "
            f"| OI net {_fmt_money(net_oi)}{pc_part}"
        )

    return OptionsDigest(message="\n".join(lines), has_data=True)


def generate_options_digest(db_path: Path, *, limit: int | None = None) -> str:
    """Build a compact Telegram message from ``options_iv`` snapshots."""
    return build_options_digest(db_path, limit=limit).message


def send_options_digest(db_path: Path, *, limit: int | None = None) -> bool:
    """Send the latest options premium proxy digest to Telegram."""
    if not is_configured():
        LOG.info("Telegram not configured; skipping options digest")
        return False
    try:
        digest = build_options_digest(db_path, limit=limit)
        if not digest.has_data:
            LOG.info("Options digest skipped: no data")
            return False
        sent = send_message(digest.message)
        if sent:
            LOG.info("Options digest sent to Telegram")
        return bool(sent)
    except Exception as exc:
        LOG.warning("Options digest failed: %s", exc)
        return False

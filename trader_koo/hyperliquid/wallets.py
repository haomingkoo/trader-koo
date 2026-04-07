"""Tracked Hyperliquid wallet configuration."""
from __future__ import annotations

import json
import os
import re

DEFAULT_TRACKED_WALLETS: dict[str, str] = {
    "machibro": "0x020ca66c30bec2c4fe3861a94e4db4a498a35872",
    "james_wynn": "0x5078c2fbea2b2ad61bc840bc023e35fce56bedb6",
}

_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")


def _normalise_label(raw: str) -> str:
    label = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch for ch in label if ch.isalnum() or ch == "_")


def _is_valid_address(raw: str) -> bool:
    return bool(_ADDRESS_RE.fullmatch(raw.strip()))


def _parse_wallet_mapping(raw: str) -> dict[str, str]:
    text = raw.strip()
    if not text:
        return {}

    parsed: dict[str, str] = {}

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        items = payload.items()
    else:
        items = []
        for part in text.split(","):
            if "=" not in part:
                continue
            label, address = part.split("=", 1)
            items.append((label, address))

    for raw_label, raw_address in items:
        label = _normalise_label(str(raw_label))
        address = str(raw_address).strip()
        if not label or not _is_valid_address(address):
            continue
        parsed[label] = address.lower()

    return parsed


def get_tracked_wallets() -> dict[str, str]:
    """Return the merged tracked-wallet map.

    `TRADER_KOO_HL_TRACKED_WALLETS` accepts either JSON:
        {"james_wynn": "0xabc..."}

    or CSV:
        james_wynn=0xabc...,another=0xdef...

    Parsed wallets are merged on top of the built-in defaults so Machi stays
    available unless explicitly overridden with a different address.
    """
    raw = os.getenv("TRADER_KOO_HL_TRACKED_WALLETS", "")
    merged = dict(DEFAULT_TRACKED_WALLETS)
    merged.update(_parse_wallet_mapping(raw))
    return merged

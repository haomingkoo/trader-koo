"""Shared helpers, constants, and utilities used across admin sub-routers."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from trader_koo.audit import AuditLogger
from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.utils import clean_optional_url as _clean_optional_url

LOG = logging.getLogger("trader_koo.routers.admin")

PROJECT_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))

LOG_DIR = Path(os.getenv("TRADER_KOO_LOG_DIR", "/data/logs"))
RUN_LOG_PATH = LOG_DIR / "cron_daily.log"
LOG_PATHS: dict[str, Path] = {
    "cron": RUN_LOG_PATH,
    "update_market_db": LOG_DIR / "update_market_db.log",
    "yolo": LOG_DIR / "yolo_patterns.log",
    "api": LOG_DIR / "api.log",
}

ANALYTICS_ENABLED = str(
    os.getenv("TRADER_KOO_ANALYTICS_ENABLED", "1")
).strip().lower() in {"1", "true", "yes", "on"}

STATUS_APP_URL = _clean_optional_url(
    os.getenv("TRADER_KOO_APP_URL")
) or _clean_optional_url(os.getenv("TRADER_KOO_ALLOWED_ORIGIN"))
STATUS_BASE_URL = _clean_optional_url(os.getenv("TRADER_KOO_BASE_URL"))

# Shared mutable state for background threads
_yolo_seed_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_update_mode(mode: str | None) -> str | None:
    value = str(mode or "full").strip().lower()
    aliases = {
        "full": "full",
        "all": "full",
        "yolo": "yolo",
        "yolo_report": "yolo",
        "yolo+report": "yolo",
        "report": "report",
        "report_only": "report",
        "email": "report",
    }
    return aliases.get(value)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Failed to parse JSON file %s: %s", path.name, exc)
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _find_timeframe_summary(rows: Any, timeframe: str) -> dict[str, Any]:
    target = str(timeframe or "").strip().lower()
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(
            row.get("timeframe", "")
        ).strip().lower() == target:
            return row
    return {}


def get_audit_logger() -> AuditLogger:
    conn = sqlite3.connect(str(DB_PATH))
    return AuditLogger(conn)

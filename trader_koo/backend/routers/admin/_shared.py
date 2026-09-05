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
from trader_koo.backend.utils import normalize_update_mode as _normalize_update_mode

LOG = logging.getLogger("trader_koo.routers.admin")

PROJECT_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))

LOG_DIR = Path(os.getenv("TRADER_KOO_LOG_DIR", "/data/logs"))
RUN_LOG_PATH = LOG_DIR / "cron_daily.log"
LOG_PATHS: dict[str, Path] = {
    "cron": RUN_LOG_PATH,
    "update_market_db": LOG_DIR / "update_market_db.log",
    # daily_update.sh redirects run_yolo_patterns.py into cron_daily.log;
    # yolo_patterns.log was never written by anything.
    "yolo": RUN_LOG_PATH,
    "api": LOG_DIR / "api.log",
}

ANALYTICS_ENABLED = str(
    os.getenv("TRADER_KOO_ANALYTICS_ENABLED", "1")
).strip().lower() in {"1", "true", "yes", "on"}

# Shared mutable state for background threads
_yolo_seed_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_audit_logger() -> AuditLogger:
    conn = sqlite3.connect(str(DB_PATH))
    return AuditLogger(conn)

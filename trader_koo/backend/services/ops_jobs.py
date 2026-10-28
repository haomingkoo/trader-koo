"""Operational scheduler jobs: failure alerts and the daily database backup.

Registered by ``trader_koo.backend.services.scheduler``. Kept apart from it so
the alert cooldown and the backup job can be tested without the scheduler.
"""
from __future__ import annotations

import datetime as dt
import html
import logging
import time
from pathlib import Path
from typing import Callable

LOG = logging.getLogger("trader_koo.services.ops_jobs")

# Per alert kind, so a failing backup is not muted by a pipeline alert. The
# cooldown is in memory: a restart can repeat one alert, never suppress one.
OPS_ALERT_COOLDOWN_SEC = 6 * 3600
ALERT_DETAIL_MAX_CHARS = 1500
BYTES_PER_MB = 1_048_576
_last_ops_alert_ts: dict[str, float] = {}


def send_ops_alert(kind: str, title: str, detail: str) -> None:
    """Send one Telegram alert per *kind* per cooldown. Never raises.

    Callers have already logged the original failure; an alert problem is
    logged as a warning and must not replace or hide that error.
    """
    now_mono = time.monotonic()
    last = _last_ops_alert_ts.get(kind)
    if last is not None and now_mono - last <= OPS_ALERT_COOLDOWN_SEC:
        LOG.info("Ops alert %s suppressed by cooldown", kind)
        return
    try:
        from trader_koo.notifications import telegram

        if not telegram.is_configured():
            LOG.warning("Telegram not configured; %s alert not sent", kind)
            return
        body = html.escape(detail[:ALERT_DETAIL_MAX_CHARS])
        text = f"⚠️ <b>{html.escape(title)}</b>\n\n<code>{body}</code>"
        if telegram.send_message(text, parse_mode="HTML"):
            _last_ops_alert_ts[kind] = now_mono
        else:
            LOG.warning("Telegram rejected the %s alert", kind)
    except Exception as exc:
        LOG.warning("Failed to send %s alert: %s", kind, exc)


def run_daily_backup(db_path: Path, append_run_log: Callable[[str, str], None]) -> None:
    """Quick-check and compress *db_path* to a timestamped .gz backup.

    Skips while a maintenance intent blocks writers, so the scheduled copy
    cannot prune the named backup a maintenance run is still re-hashing.
    """
    from trader_koo.backend.services.maintenance import MaintenanceError, writers_blocked

    if not db_path.exists():
        LOG.warning("Skipping daily backup — DB not found at %s", db_path)
        append_run_log("BACKUP", f"Skipped — DB not found at {db_path}")
        send_ops_alert("backup_failed", "Backup failed", f"database not found at {db_path}")
        return
    try:
        blocked = writers_blocked(db_path)
    except MaintenanceError as exc:
        blocked = True
        LOG.error("Daily backup fail-closed on maintenance state: %s", exc.code)
    if blocked:
        LOG.warning("Skipping daily backup while maintenance blocks writers")
        append_run_log("BACKUP", "Skipped — maintenance intent blocks writers")
        return

    started = dt.datetime.now(dt.timezone.utc)
    append_run_log("BACKUP", "Starting daily SQLite backup")
    LOG.info("Starting daily SQLite backup (db=%s)", db_path)
    try:
        from trader_koo.scripts.backup_db import backup_database

        result = backup_database(db_path)
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        append_run_log(
            "BACKUP",
            (
                f"Backup completed: {result['backup_name']} "
                f"src={result['src_size_bytes']}B dest={result['dest_size_bytes']}B "
                f"ratio={result['compression_ratio_pct']}% "
                f"pruned={result['pruned_count']} sec={elapsed:.1f}"
            ),
        )
        LOG.info(
            "Daily backup completed OK (%s, %.1f MB -> %.1f MB, %.1fs)",
            result["backup_name"],
            result["src_size_bytes"] / BYTES_PER_MB,
            result["dest_size_bytes"] / BYTES_PER_MB,
            elapsed,
        )
    except Exception as exc:
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        append_run_log("BACKUP", f"Backup failed: {exc} sec={elapsed:.1f}")
        LOG.error("Daily backup failed (sec=%.1f): %s", elapsed, exc)
        send_ops_alert("backup_failed", "Backup failed", f"{type(exc).__name__}: {exc}")

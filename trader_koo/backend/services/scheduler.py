"""Scheduler setup and daily/weekly update runners.

Manages the APScheduler background jobs (daily update Mon--Fri,
weekly YOLO on Saturday) and the functions they invoke.  All
subprocess invocations are contained here.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

LOG = logging.getLogger("trader_koo.services.scheduler")

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DEFAULT_DB_PRIMARY = (PROJECT_DIR / "data" / "trader_koo.db").resolve()
DB_PATH = Path(os.getenv("TRADER_KOO_DB_PATH", str(DEFAULT_DB_PRIMARY)))

REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))


def _resolve_log_dir() -> Path:
    requested = Path(os.getenv("TRADER_KOO_LOG_DIR", "/data/logs"))
    try:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    except OSError:
        fallback = (PROJECT_DIR / "data" / "logs").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


LOG_DIR = _resolve_log_dir()
RUN_LOG_PATH = LOG_DIR / "cron_daily.log"


# ---------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------

from trader_koo.backend.utils import current_rss_mb as _current_rss_mb


def _fmt_mb(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

def _append_run_log(
    tag: str,
    message: str,
    *,
    log_dir: Path | None = None,
    run_log_path: Path | None = None,
) -> None:
    """Append a timestamped line to the cron run log."""
    stamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    line = f"{stamp} [{tag}] {message}\n"
    _log_dir = log_dir or LOG_DIR
    _run_log = run_log_path or RUN_LOG_PATH
    try:
        _log_dir.mkdir(parents=True, exist_ok=True)
        with _run_log.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        LOG.warning("Failed to append run log %s: %s", _run_log, exc)


# ---------------------------------------------------------------------------
# Mode normalisation
# ---------------------------------------------------------------------------

def _normalize_update_mode(mode: str | None) -> str | None:
    """Map mode aliases to canonical names (full | yolo | report)."""
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


# ---------------------------------------------------------------------------
# Daily update
# ---------------------------------------------------------------------------

def _tail_text_file(path: Path, lines: int = 60, max_bytes: int = 64_000) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]
    except Exception:
        return []


def _run_daily_update(mode: str = "full", source: str = "scheduler") -> None:
    """Execute ``daily_update.sh`` with the given *mode*.

    Called by the APScheduler job or manually from the admin trigger
    endpoint.
    """
    mode_norm = _normalize_update_mode(mode) or "full"
    script = SCRIPTS_DIR / "daily_update.sh"
    started = dt.datetime.now(dt.timezone.utc)
    rss_before = _current_rss_mb()
    _append_run_log(
        "SCHED",
        f"daily_update invoked source={source} mode={mode_norm} rss_before_mb={_fmt_mb(rss_before)}",
    )
    LOG.info(
        "Scheduler: starting daily_update.sh (source=%s, mode=%s, rss_before_mb=%s, run_log=%s)",
        source,
        mode_norm,
        _fmt_mb(rss_before),
        RUN_LOG_PATH,
    )
    env = os.environ.copy()
    env["TRADER_KOO_UPDATE_MODE"] = mode_norm
    result = subprocess.run(["bash", str(script)], capture_output=False, env=env)
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    rss_after = _current_rss_mb()
    delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    if result.returncode == 0:
        _append_run_log(
            "SCHED",
            (
                f"daily_update completed source={source} mode={mode_norm} rc={result.returncode} "
                f"sec={elapsed:.1f} rss_after_mb={_fmt_mb(rss_after)} rss_delta_mb={_fmt_mb(delta)}"
            ),
        )
        LOG.info(
            (
                "Scheduler: daily_update.sh completed OK "
                "(source=%s, mode=%s, rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s)"
            ),
            source,
            mode_norm,
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
        )
    else:
        _append_run_log(
            "SCHED",
            (
                f"daily_update failed source={source} mode={mode_norm} rc={result.returncode} "
                f"sec={elapsed:.1f} rss_after_mb={_fmt_mb(rss_after)} rss_delta_mb={_fmt_mb(delta)}"
            ),
        )
        LOG.error(
            (
                "Scheduler: daily_update.sh failed "
                "(source=%s, mode=%s, rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s, run_log=%s)"
            ),
            source,
            mode_norm,
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
            RUN_LOG_PATH,
        )
        tail = _tail_text_file(RUN_LOG_PATH, lines=25, max_bytes=100_000)
        if tail:
            LOG.error("Scheduler: recent cron tail:\n%s", "\n".join(tail))


# ---------------------------------------------------------------------------
# Weekly backup
# ---------------------------------------------------------------------------

def _run_weekly_backup() -> None:
    """Saturday job: compress SQLite DB to a timestamped .gz backup."""
    if not DB_PATH.exists():
        LOG.warning("Scheduler: skipping weekly backup — DB not found at %s", DB_PATH)
        _append_run_log("BACKUP", f"Skipped — DB not found at {DB_PATH}")
        return

    started = dt.datetime.now(dt.timezone.utc)
    _append_run_log("BACKUP", "Starting weekly SQLite backup")
    LOG.info("Scheduler: starting weekly SQLite backup (db=%s)", DB_PATH)

    try:
        from trader_koo.scripts.backup_db import backup_database

        result = backup_database(DB_PATH)
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        _append_run_log(
            "BACKUP",
            (
                f"Backup completed: {result['backup_name']} "
                f"src={result['src_size_bytes']}B dest={result['dest_size_bytes']}B "
                f"ratio={result['compression_ratio_pct']}% "
                f"pruned={result['pruned_count']} sec={elapsed:.1f}"
            ),
        )
        LOG.info(
            "Scheduler: weekly backup completed OK (%s, %.1f MB -> %.1f MB, %.1fs)",
            result["backup_name"],
            result["src_size_bytes"] / 1_048_576,
            result["dest_size_bytes"] / 1_048_576,
            elapsed,
        )
    except Exception as exc:
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        _append_run_log("BACKUP", f"Backup failed: {exc} sec={elapsed:.1f}")
        LOG.error("Scheduler: weekly backup failed (sec=%.1f): %s", elapsed, exc)


# ---------------------------------------------------------------------------
# Weekly YOLO
# ---------------------------------------------------------------------------

def _run_weekly_yolo() -> None:
    """Saturday job: run YOLO weekly pass + regenerate report."""
    script_yolo = SCRIPTS_DIR / "run_yolo_patterns.py"
    script_report = SCRIPTS_DIR / "generate_daily_report.py"
    run_log = RUN_LOG_PATH
    report_dir = REPORT_DIR

    started = dt.datetime.now(dt.timezone.utc)
    rss_before = _current_rss_mb()
    LOG.info(
        "Scheduler: starting weekly YOLO pass (rss_before_mb=%s, run_log=%s)",
        _fmt_mb(rss_before),
        run_log,
    )
    _append_run_log("WEEKLY", "Starting weekly YOLO pass")
    with run_log.open("a", encoding="utf-8") as log_file:
        result = subprocess.run(
            [
                sys.executable, str(script_yolo),
                "--db-path", str(DB_PATH),
                "--timeframe", "weekly",
                "--weekly-lookback-days", "730",
                "--sleep", "0.05",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    rss_after = _current_rss_mb()
    delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    if result.returncode == 0:
        LOG.info(
            "Scheduler: weekly YOLO completed OK (rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s)",
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
        )
        _append_run_log("WEEKLY", "Weekly YOLO completed OK")
    else:
        LOG.error(
            "Scheduler: weekly YOLO failed (rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s, run_log=%s)",
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
            run_log,
        )
        _append_run_log("WEEKLY", f"Weekly YOLO failed rc={result.returncode}")

    # Regenerate report so weekly patterns appear
    LOG.info("Scheduler: regenerating report after weekly YOLO")
    _append_run_log("WEEKLY", "Report regeneration starting")
    with run_log.open("a", encoding="utf-8") as log_file:
        report_result = subprocess.run(
            [
                sys.executable, str(script_report),
                "--db-path", str(DB_PATH),
                "--out-dir", str(report_dir),
                "--run-log", str(run_log),
                "--tail-lines", "120",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if report_result.returncode == 0:
        _append_run_log("WEEKLY", "Report regeneration completed")
    else:
        LOG.error(
            "Scheduler: report regeneration after weekly YOLO failed (rc=%d)",
            report_result.returncode,
        )
        _append_run_log("WEEKLY", f"Report regeneration failed rc={report_result.returncode}")


# ---------------------------------------------------------------------------
# Morning summary (Telegram push at 00:00 UTC = 08:00 SGT)
# ---------------------------------------------------------------------------

def _run_morning_summary() -> None:
    """Send the daily morning briefing to Telegram (Mon-Fri)."""
    from trader_koo.notifications.morning_summary import send_morning_summary

    _append_run_log("MORNING", "Morning summary job started")
    LOG.info("Scheduler: starting morning summary push")
    ok = send_morning_summary(DB_PATH, REPORT_DIR)
    if ok:
        _append_run_log("MORNING", "Morning summary sent OK")
        LOG.info("Scheduler: morning summary sent to Telegram")
    else:
        _append_run_log("MORNING", "Morning summary failed or skipped")
        LOG.warning("Scheduler: morning summary failed or was skipped (Telegram not configured?)")


# ---------------------------------------------------------------------------
# Market monitor (Polymarket snapshots + spike alerts)
# ---------------------------------------------------------------------------

def _run_polymarket_snapshot() -> None:
    """Hourly job: archive current Polymarket probabilities."""
    from trader_koo.notifications.market_monitor import snapshot_polymarket

    _append_run_log("MARKET_MONITOR", "Polymarket snapshot job started")
    LOG.info("Scheduler: starting Polymarket snapshot")
    try:
        count = snapshot_polymarket(DB_PATH)
        _append_run_log("MARKET_MONITOR", f"Polymarket snapshot saved {count} rows")
        LOG.info("Scheduler: Polymarket snapshot saved %d rows", count)
    except Exception as exc:
        _append_run_log("MARKET_MONITOR", f"Polymarket snapshot failed: {exc}")
        LOG.error("Scheduler: Polymarket snapshot failed: %s", exc)


def _run_spike_alerts() -> None:
    """Periodic job: detect spikes and send Telegram alerts."""
    from trader_koo.notifications.market_monitor import send_spike_alerts

    _append_run_log("MARKET_MONITOR", "Spike alert check started")
    LOG.info("Scheduler: starting spike alert check")
    try:
        alerts_sent = send_spike_alerts(DB_PATH, REPORT_DIR)
        _append_run_log("MARKET_MONITOR", f"Spike alerts: {alerts_sent} sent")
        LOG.info("Scheduler: spike alerts sent: %d", alerts_sent)
    except Exception as exc:
        _append_run_log("MARKET_MONITOR", f"Spike alert check failed: {exc}")
        LOG.error("Scheduler: spike alert check failed: %s", exc)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def create_scheduler() -> BackgroundScheduler:
    """Create and configure the APScheduler instance with daily + weekly jobs.

    Returns the scheduler (not started -- caller should call ``.start()``).
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = BackgroundScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 3600,
        },
    )
    scheduler.add_job(
        _run_daily_update,
        CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone="UTC"),
        id="daily_update",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_weekly_yolo,
        CronTrigger(hour=0, minute=30, day_of_week="sat", timezone="UTC"),
        id="weekly_yolo",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_weekly_backup,
        CronTrigger(hour=2, minute=0, day_of_week="sat", timezone="UTC"),
        id="weekly_backup",
        replace_existing=True,
    )

    # Morning summary — only register if Telegram is configured
    telegram_configured = bool(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    if telegram_configured:
        scheduler.add_job(
            _run_morning_summary,
            CronTrigger(hour=0, minute=0, day_of_week="mon-fri", timezone="UTC"),
            id="morning_summary",
            replace_existing=True,
        )
        LOG.info("Morning summary job registered: daily 00:00 UTC (08:00 SGT) Mon-Fri")
    else:
        LOG.info("TELEGRAM_BOT_TOKEN not set — morning summary job not registered")

    # Polymarket snapshot + spike detection — every 15 min, 24/7
    scheduler.add_job(
        _run_polymarket_snapshot,
        IntervalTrigger(minutes=15),
        id="polymarket_snapshot",
        replace_existing=True,
    )
    LOG.info("Polymarket snapshot job registered: every 15min (24/7)")

    # Spike alerts — every 15 min (only sends if Telegram configured)
    if telegram_configured:
        scheduler.add_job(
            _run_spike_alerts,
            IntervalTrigger(minutes=15),
            id="spike_alerts",
            replace_existing=True,
        )
        LOG.info("Spike alert job registered: every 15min")
    else:
        LOG.info("TELEGRAM_BOT_TOKEN not set — spike alert job not registered")

    return scheduler

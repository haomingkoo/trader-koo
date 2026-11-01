"""The cron run log (``cron_daily.log``): its location, writer and markers.

``daily_update.sh`` and the scheduler both write to this log. This module is
the one place that knows the marker strings the shell script writes, so the
live run path and the status inference read the log the same way.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from trader_koo.config import DEFAULT_LOG_DIR, ConfigError, env_path

LOG = logging.getLogger("trader_koo.services.run_log")


def _resolve_log_dir() -> Path:
    requested = env_path("TRADER_KOO_LOG_DIR", DEFAULT_LOG_DIR)
    try:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    except OSError as exc:
        raise ConfigError(f"TRADER_KOO_LOG_DIR is not writable: {requested}") from exc


LOG_DIR = _resolve_log_dir()
RUN_LOG_PATH = LOG_DIR / "cron_daily.log"


def append_run_log(
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
# daily_update.sh markers
# ---------------------------------------------------------------------------

START_MARKER = "[START] daily_update.sh"
DONE_MARKER = "[DONE]  daily_update.sh"
INGEST_DONE_MARKER = "[INGEST] done"
YOLO_START_MARKER = "[YOLO]  Starting"
YOLO_DONE_MARKER = "[YOLO]  Daily pattern detection done."
REPORT_START_MARKER = "[REPORT] Generating"
REPORT_DONE_MARKER = "[REPORT] Done."
YOLO_PROGRESS_MARKERS = (
    "[daily] Processing",
    "[weekly] Processing",
    "[daily ",
    "[weekly ",
)

_RC_RE = re.compile(r"\brc=(\d+)")


def last_marker_index(lines: list[str], *tokens: str) -> int:
    """Index of the last line containing any of *tokens*, or -1."""
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if any(token in line for token in tokens):
            return idx
    return -1


def latest_run_block(lines: list[str]) -> list[str]:
    """Lines from the newest ``[START]`` marker on, or all lines if none."""
    start = last_marker_index(lines, START_MARKER)
    return lines[start:] if start >= 0 else lines


def _stage_done_ok(block: list[str], marker: str) -> bool:
    idx = last_marker_index(block, marker)
    if idx < 0:
        return False
    match = _RC_RE.search(block[idx], block[idx].index(marker))
    return match is not None and int(match.group(1)) == 0


@dataclass(frozen=True)
class StageResults:
    ingest_ok: bool
    yolo_ok: bool
    report_ok: bool


def stage_results(
    mode: str,
    returncode: int,
    read_tail: Callable[[], list[str]],
) -> StageResults:
    """Which stages of one ``daily_update.sh`` run succeeded.

    The script exits 0 only when every enabled stage passed. On a failed full
    run the log is read (only the newest run's block) to keep the stages that
    did pass, in order: report counts only after YOLO, YOLO only after ingest.
    """
    run_ingest = mode == "full"
    run_yolo = mode in ("full", "yolo")
    run_report = mode in ("full", "yolo", "report")
    if returncode == 0:
        return StageResults(run_ingest, run_yolo, run_report)
    if not run_ingest:
        return StageResults(False, False, False)
    block = latest_run_block(read_tail())
    ingest_ok = _stage_done_ok(block, INGEST_DONE_MARKER)
    yolo_ok = ingest_ok and run_yolo and _stage_done_ok(block, YOLO_DONE_MARKER)
    report_ok = yolo_ok and run_report and _stage_done_ok(block, REPORT_DONE_MARKER)
    return StageResults(ingest_ok, yolo_ok, report_ok)

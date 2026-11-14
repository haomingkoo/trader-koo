"""Pipeline health conditions reported as ``/api/status`` operational warnings.

The central health watch (koo-control-center) alerts when ``/api/status``
reports ``ok`` false, so every condition here reaches Telegram and ntfy with
deduplication and a recovery message. Each warning is one readable string
that names the run, stage and failed stages.
"""
from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from typing import Any

from trader_koo.backend.services.market_data import parse_iso_utc
from trader_koo.report.utils import expected_price_date

# A full run normally takes about 25 minutes.
PIPELINE_STUCK_MIN = max(30, int(os.getenv("TRADER_KOO_PIPELINE_STUCK_MIN", "90")))
# Ingest starts at 22:00 UTC and can run late; on top of INGEST_DUE_LAG.
PRICE_BEHIND_GRACE = dt.timedelta(hours=2)
HEALTHY_RUN_STATUSES = frozenset({"ok", "skipped"})


def _minutes_since(ts: Any, now: dt.datetime) -> int | None:
    parsed = parse_iso_utc(ts)
    return None if parsed is None else max(0, int((now - parsed).total_seconds() // 60))


def _stuck_warning(active_run: Mapping[str, Any] | None, now: dt.datetime) -> str | None:
    if not active_run:
        return None
    run_min = _minutes_since(active_run.get("started_ts"), now)
    if run_min is not None and run_min <= PIPELINE_STUCK_MIN:
        return None
    stage_min = _minutes_since(active_run.get("stage_started_ts"), now)
    return (
        f"pipeline stuck: stage {active_run.get('stage') or 'unknown'} for "
        f"{'?' if stage_min is None else stage_min} min (run {active_run.get('run_id')} started "
        f"{'?' if run_min is None else run_min} min ago, limit {PIPELINE_STUCK_MIN})"
    )


def pipeline_health_warnings(
    pipeline_snap: Mapping[str, Any],
    *,
    price_date: str | None,
    now: dt.datetime,
) -> list[str]:
    """Warnings for a hung run, an unhealthy last run, and prices behind the expected session."""
    warnings: list[str] = []
    stuck = _stuck_warning(pipeline_snap.get("active_pipeline_run"), now)
    if stuck:
        warnings.append(stuck)

    # A live run supersedes the previous result; a stuck one does not.
    status = pipeline_snap.get("last_completed_status")
    if status and status not in HEALTHY_RUN_STATUSES and (stuck or not pipeline_snap.get("active")):
        detail = pipeline_snap.get("last_completed_error") or f"stage {pipeline_snap.get('last_completed_stage')}"
        warnings.append(
            f"pipeline last run {status}: {detail} (finished {pipeline_snap.get('last_completed_ts')})"
        )

    expected = expected_price_date(now - PRICE_BEHIND_GRACE).isoformat()
    if not price_date or str(price_date)[:10] < expected:
        warnings.append(f"price_daily behind expected session {expected} (latest {price_date})")
    return warnings

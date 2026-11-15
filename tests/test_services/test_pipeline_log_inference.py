"""What /api/status infers from the cron log tail: running stage, the line it read, and the last completion."""
from __future__ import annotations

import pytest

from trader_koo.backend.services.pipeline import (
    _infer_last_completed_event_from_tail,
    _infer_pipeline_from_log_tail,
)

START = "[START] daily_update.sh mode=full"
START_YOLO = "[START] daily_update.sh mode=yolo"
DONE = "[DONE]  daily_update.sh mode=full ingest=0 yolo=0 report=0"
YOLO_START = "[YOLO]  Starting daily pattern detection (180d)..."
YOLO_DONE = "[YOLO]  Daily pattern detection done. rc=0 sec=300"
DAILY_PROGRESS = "[daily 3/500] AAPL ok"
REPORT_START = "[REPORT] Generating daily report..."
REPORT_DONE = "[REPORT] Done. rc=0 sec=60"
WEEKLY_YOLO_START = "[WEEKLY] Starting weekly YOLO pass"
WEEKLY_PROGRESS = "[weekly] Processing SPY"
WEEKLY_YOLO_OK = "[WEEKLY] Weekly YOLO completed OK"
WEEKLY_YOLO_FAILED = "[WEEKLY] Weekly YOLO failed rc=1"
WEEKLY_REPORT_START = "[WEEKLY] Report regeneration starting"
WEEKLY_REPORT_OK = "[WEEKLY] Report regeneration completed"
WEEKLY_REPORT_FAILED = "[WEEKLY] Report regeneration failed rc=1"


def _state(tail: list[str]) -> tuple:
    out = _infer_pipeline_from_log_tail(tail)
    return out["active"], out["stage"], out["stage_line"], out["start_line"], out["done_line"]


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ([], (False, "unknown", None, None, None)),
        ([WEEKLY_REPORT_START], (True, "report", WEEKLY_REPORT_START, None, None)),
        ([WEEKLY_REPORT_START, WEEKLY_REPORT_OK], (False, "unknown", None, None, None)),
        ([WEEKLY_REPORT_START, WEEKLY_REPORT_FAILED], (False, "unknown", None, None, None)),
        ([WEEKLY_YOLO_START], (True, "yolo", WEEKLY_YOLO_START, None, None)),
        ([WEEKLY_YOLO_START, WEEKLY_PROGRESS], (True, "yolo", WEEKLY_PROGRESS, None, None)),
        ([WEEKLY_PROGRESS, WEEKLY_YOLO_START], (True, "yolo", WEEKLY_YOLO_START, None, None)),
        ([WEEKLY_YOLO_START, WEEKLY_YOLO_OK], (False, "unknown", None, None, None)),
        ([WEEKLY_YOLO_START, WEEKLY_YOLO_FAILED], (False, "unknown", None, None, None)),
        ([WEEKLY_YOLO_START, WEEKLY_REPORT_START, WEEKLY_REPORT_OK], (False, "unknown", None, None, None)),
        ([REPORT_START], (True, "report", REPORT_START, None, None)),
        ([REPORT_START, REPORT_DONE], (False, "unknown", None, None, None)),
        ([DAILY_PROGRESS], (True, "yolo", DAILY_PROGRESS, None, None)),
        ([DAILY_PROGRESS, YOLO_DONE], (False, "unknown", None, None, None)),
        ([DONE], (False, "unknown", None, None, DONE)),
    ],
)
def test_tail_without_a_start_marker_reads_weekly_and_orphaned_stages(tail, expected) -> None:
    assert _state(tail) == expected


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ([START], (True, "ingest", START, START, None)),
        ([START, DONE], (False, "idle", None, START, DONE)),
        ([START, YOLO_START], (True, "yolo", YOLO_START, START, None)),
        ([START, DAILY_PROGRESS], (True, "yolo", DAILY_PROGRESS, START, None)),
        ([START, YOLO_START, DAILY_PROGRESS], (True, "yolo", DAILY_PROGRESS, START, None)),
        ([START, YOLO_START, YOLO_DONE], (True, "ingest", START, START, None)),
        ([START, DAILY_PROGRESS, YOLO_DONE], (True, "ingest", START, START, None)),
        ([START, YOLO_START, "loading model"], (True, "yolo", YOLO_START, START, None)),
        ([START, YOLO_START, YOLO_DONE, REPORT_START], (True, "report", REPORT_START, START, None)),
        ([START, REPORT_START, REPORT_DONE], (True, "ingest", START, START, None)),
        # A new run after a finished one reads only the new run's block.
        ([START, REPORT_START, DONE, START_YOLO, YOLO_START], (True, "yolo", YOLO_START, START_YOLO, DONE)),
        ([START, DAILY_PROGRESS, YOLO_DONE, DONE, START_YOLO], (True, "ingest", START_YOLO, START_YOLO, DONE)),
    ],
)
def test_tail_with_a_start_marker_reads_the_newest_run(tail, expected) -> None:
    assert _state(tail) == expected


def test_marker_lines_keep_the_last_forty_marker_lines_only() -> None:
    markers = [START, "[YOLO] x", "[REPORT] y", "[WEEKLY] z", "job run_id=abc", DAILY_PROGRESS, DONE]
    tail = ["noise", *markers, "more noise"]

    assert _infer_pipeline_from_log_tail(tail)["marker_lines"] == markers

    many = [f"[REPORT] line {i}" for i in range(41)]
    assert _infer_pipeline_from_log_tail(many)["marker_lines"] == many[1:]


def test_last_completion_carries_its_line_and_whole_second_timestamp() -> None:
    line = f"2026-10-22T22:05:07.654321+00:00 {DONE}"

    assert _infer_last_completed_event_from_tail(["noise", line]) == {
        "stage": "daily_update",
        "status": "ok",
        "line": line,
        "ts": "2026-10-22T22:05:07+00:00",
    }


def test_last_completion_without_a_timestamp_has_no_ts() -> None:
    event = _infer_last_completed_event_from_tail([REPORT_DONE])

    assert event == {"stage": "report", "status": "ok", "line": REPORT_DONE, "ts": None}

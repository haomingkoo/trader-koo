"""run_yolo_patterns.py fails the run when too many tickers fail.

Before Nov 2026 it exited 0 unless every ticker failed, so a broken install
that fails most tickers looked like a successful stage. The threshold is the
same coverage rule /api/status and the report use, so the stage fails exactly
when the data it leaves behind would be judged stale.
"""
from __future__ import annotations

import pytest

from trader_koo.report.freshness import YOLO_MIN_DAILY_COVERAGE
from trader_koo.scripts.run_yolo_patterns import failure_verdict


@pytest.mark.parametrize(
    ("ok", "failed", "fails_run"),
    [
        (500, 0, False),
        (0, 0, False),  # nothing eligible (e.g. --only-new on a weekend)
        (0, 500, True),
        (1, 499, True),  # the six incident: a handful passed, the run still "succeeded"
        (450, 50, False),  # exactly at the coverage floor
        (449, 51, True),
    ],
)
def test_failure_verdict(ok: int, failed: int, fails_run: bool) -> None:
    message = failure_verdict(ok, failed, top_reasons=[("No module named 'six'", failed)])

    assert (message is not None) is fails_run


def test_failure_message_names_counts_threshold_and_reason() -> None:
    message = failure_verdict(10, 490, top_reasons=[("No module named 'six'", 490)])

    assert message is not None
    assert "490/500" in message
    assert f"{1 - YOLO_MIN_DAILY_COVERAGE:.0%}" in message
    assert "No module named 'six'" in message

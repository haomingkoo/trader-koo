from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tests.dark_deploy_harness import WORKFLOW, run_step, step


PREFLIGHT = "Refuse a restart during the nightly pipeline"
BIND = "Bind release identity without starting a second deployment"
ROLLBACK = "Roll back code and variables after any failed release check"


def _run(tmp_path: Path, clock: str, statuses: list[Any]):
    scenario = {
        "date": [{"match": "-u", "responses": [{"body": f"{clock}\n"}]}],
        "curl": [
            {"match": "/api/status", "responses": [{"body": body} for body in statuses]}
        ],
    }
    env = {
        "BASE_URL": "https://prod.example.invalid",
        "PIPELINE_WAIT_ATTEMPTS": "3",
        "PIPELINE_WAIT_INTERVAL_SEC": "0",
    }
    return run_step(tmp_path, PREFLIGHT, env=env, scenario=scenario)


def _status(active: Any, stage: str = "idle") -> dict[str, Any]:
    return {"ok": True, "pipeline_active": active, "pipeline_stage": stage}


def test_preflight_passes_outside_the_window_with_an_idle_pipeline(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, "2 1430", [_status(False)])

    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_refuses_the_weekday_nightly_window(tmp_path: Path) -> None:
    for clock in ("1 2115", "3 2200", "5 2314"):
        result, _, _ = _run(tmp_path, clock, [_status(False)])
        assert result.returncode != 0, clock
        assert "nightly" in result.stdout


def test_preflight_allows_the_window_hours_at_the_weekend(tmp_path: Path) -> None:
    for clock in ("6 2200", "7 2200", "5 2315", "1 2114"):
        result, _, _ = _run(tmp_path, clock, [_status(False)])
        assert result.returncode == 0, (clock, result.stdout + result.stderr)


def test_preflight_waits_for_an_active_pipeline_to_finish(tmp_path: Path) -> None:
    result, calls, _ = _run(
        tmp_path, "2 1430", [_status(True, "ingest"), _status(True, "report"), _status(False)]
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ingest" in result.stdout
    assert sum(1 for call in calls if call["tool"] == "sleep") == 2


def test_preflight_fails_when_the_pipeline_stays_active(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, "2 1430", [_status(True, "ingest")])

    assert result.returncode != 0
    assert "still active" in result.stdout


def test_preflight_fails_closed_without_a_boolean_pipeline_flag(tmp_path: Path) -> None:
    for body in ({"ok": True}, _status("false"), _status(None)):
        result, _, _ = _run(tmp_path, "2 1430", [body])
        assert result.returncode != 0, body


def test_preflight_runs_right_before_the_first_railway_change() -> None:
    names = [s["name"] for s in yaml.safe_load(WORKFLOW.read_text())["jobs"]["dark-deploy"]["steps"]]

    assert names.index(PREFLIGHT) + 1 == names.index(BIND)


def test_rollback_only_runs_after_railway_was_touched() -> None:
    bind = step(BIND)

    assert bind["id"] == "bind"
    assert bind["run"].startswith('echo "started=true" >> "$GITHUB_OUTPUT"')
    assert "steps.bind.outputs.started == 'true'" in step(ROLLBACK)["if"]

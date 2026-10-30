from __future__ import annotations

from pathlib import Path

import yaml

from tests.dark_deploy_harness import WORKFLOW, run_step, step


GUARD = "Refuse a release that does not descend from production"
CI_GATE = "Require a passing CI run for the release commit"
DRILL = "Fail on purpose to exercise the rollback drill"
ROLLBACK = "Roll back code and variables after any failed release check"
SHA = "c" * 40


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_dispatch_offers_only_the_rollback_drill() -> None:
    # PyYAML reads the bare `on` key as True.
    triggers = _workflow()[True]
    drill_input = triggers["workflow_dispatch"]["inputs"]["rollback_drill"]

    assert drill_input["type"] == "boolean"
    assert drill_input["default"] is False
    assert set(triggers) == {"workflow_run", "workflow_dispatch"}


def test_dispatch_requires_main_and_the_drill_input() -> None:
    job = _workflow()["jobs"]["dark-deploy"]

    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "inputs.rollback_drill" in job["if"]
    assert job["env"]["RELEASE_SHA"] == (
        "${{ github.event.workflow_run.head_sha || github.sha }}"
    )
    assert job["env"]["ROLLBACK_DRILL"] == (
        "${{ github.event_name == 'workflow_dispatch' }}"
    )
    assert step("Check out the exact CI commit")["with"]["ref"] == "${{ env.RELEASE_SHA }}"


def test_drill_fails_after_every_verification_and_before_rollback() -> None:
    names = [s["name"] for s in _workflow()["jobs"]["dark-deploy"]["steps"]]
    drill = step(DRILL)

    assert names.index("Verify production Chromium journeys") + 1 == names.index(DRILL)
    assert names.index(DRILL) + 1 == names.index(ROLLBACK)
    assert drill["if"] == "${{ env.ROLLBACK_DRILL == 'true' }}"
    assert drill["run"].rstrip().endswith("exit 1")
    assert "env.ROLLBACK_DRILL" not in step(ROLLBACK)["if"]


def _guard(tmp_path: Path, previous: str, drill: str):
    scenario = {
        "curl": [{"match": "/api/release", "responses": [{"body": {"git_sha": previous}}]}],
        "git": [
            {"match": "ls-remote", "responses": [{"body": f"{SHA}\trefs/heads/main\n"}]},
            {"match": "cat-file", "responses": [{"exit": 0}]},
            {"match": "merge-base --is-ancestor", "responses": [{"exit": 0}]},
        ]
    }
    env = {
        "RELEASE_SHA": SHA,
        "BASE_URL": "https://prod.example.invalid",
        "ROLLBACK_DRILL": drill,
    }
    return run_step(tmp_path, GUARD, env=env, scenario=scenario)


def test_drill_refuses_to_ship_undeployed_code(tmp_path: Path) -> None:
    result, _, _ = _guard(tmp_path, "a" * 40, "true")

    assert result.returncode != 0
    assert "rollback drill" in result.stdout


def test_drill_redeploys_the_running_release(tmp_path: Path) -> None:
    result, _, _ = _guard(tmp_path, SHA, "true")

    assert result.returncode == 0, result.stdout + result.stderr


def test_ci_gate_requires_a_successful_push_run_for_the_sha(tmp_path: Path) -> None:
    for total, expected in ((1, 0), (0, 1)):
        scenario = {"gh": [{"match": "actions/workflows/ci.yml/runs", "responses": [{"body": f"{total}\n"}]}]}
        result, calls, _ = run_step(
            tmp_path,
            CI_GATE,
            env={"RELEASE_SHA": SHA, "GITHUB_REPOSITORY": "owner/repo", "GH_TOKEN": "x"},
            scenario=scenario,
        )
        assert (result.returncode != 0) == bool(expected), result.stdout + result.stderr
        query = " ".join(calls[0]["args"])
        assert f"head_sha={SHA}" in query
        assert "event=push" in query and "status=success" in query and "branch=main" in query
    assert step(CI_GATE)["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert _workflow()["permissions"]["actions"] == "read"

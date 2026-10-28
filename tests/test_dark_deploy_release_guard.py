from __future__ import annotations

from pathlib import Path

import yaml

from tests.dark_deploy_harness import WORKFLOW, run_step, step


GUARD = "Refuse a stale or non-descendant release"
RELEASE_SHA = "c" * 40
PREVIOUS_SHA = "a" * 40


def _git(*, main_head: str = RELEASE_SHA, known: bool = True, ancestor: bool = True):
    return {
        "git": [
            {"match": "ls-remote", "responses": [{"body": f"{main_head}\trefs/heads/main\n"}]},
            {"match": "cat-file", "responses": [{"exit": 0 if known else 1}]},
            {"match": "merge-base --is-ancestor", "responses": [{"exit": 0 if ancestor else 1}]},
        ]
    }


def _run(tmp_path: Path, **kwargs):
    env = {"RELEASE_SHA": RELEASE_SHA, "PREVIOUS_SHA": PREVIOUS_SHA}
    return run_step(tmp_path, GUARD, env=env, scenario=_git(**kwargs))


def test_guard_accepts_the_main_head_descending_from_production(tmp_path: Path) -> None:
    result, calls, _ = _run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    merge_base = [call["args"] for call in calls if "merge-base" in call["args"]]
    assert merge_base == [["merge-base", "--is-ancestor", PREVIOUS_SHA, RELEASE_SHA]]


def test_guard_refuses_a_sha_that_is_no_longer_the_main_head(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, main_head="d" * 40)

    assert result.returncode != 0
    assert "not the current main head" in result.stdout


def test_guard_refuses_when_production_is_not_an_ancestor(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, ancestor=False)

    assert result.returncode != 0
    assert "not an ancestor" in result.stdout


def test_guard_refuses_an_unknown_production_sha(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, known=False)

    assert result.returncode != 0
    assert "not in the repository history" in result.stdout


def test_guard_runs_after_previous_state_and_before_any_railway_change() -> None:
    names = [s["name"] for s in yaml.safe_load(WORKFLOW.read_text())["jobs"]["dark-deploy"]["steps"]]

    assert names.index("Record previous verified deployment") < names.index(GUARD)
    assert names.index(GUARD) < names.index(
        "Bind release identity without starting a second deployment"
    )
    assert step(GUARD)["env"]["PREVIOUS_SHA"] == "${{ steps.previous.outputs.sha }}"


def test_checkout_has_the_history_for_the_ancestor_check() -> None:
    checkout = step("Check out the exact CI commit")

    assert checkout["with"]["fetch-depth"] == 0


def test_job_only_runs_for_pushes_to_this_repository_main() -> None:
    job_if = yaml.safe_load(WORKFLOW.read_text())["jobs"]["dark-deploy"]["if"]

    for condition in (
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
    ):
        assert condition in job_if

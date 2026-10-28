from __future__ import annotations

from pathlib import Path

import yaml

from tests.dark_deploy_harness import WORKFLOW, run_step, step


HEAD_GUARD = "Refuse a release that is not the main head"
GUARD = "Refuse a release that does not descend from production"
BACKUP = "Create and download a fresh consistent production backup"
RELEASE_SHA = "c" * 40
PREVIOUS_SHA = "a" * 40


def _git(
    *,
    main_head: str = RELEASE_SHA,
    known: bool = True,
    ancestor: bool = True,
    production_sha: str = PREVIOUS_SHA,
):
    return {
        "curl": [
            {"match": "/api/release", "responses": [{"body": {"git_sha": production_sha}}]}
        ],
        "git": [
            {"match": "ls-remote", "responses": [{"body": f"{main_head}\trefs/heads/main\n"}]},
            {"match": "cat-file", "responses": [{"exit": 0 if known else 1}]},
            {"match": "merge-base --is-ancestor", "responses": [{"exit": 0 if ancestor else 1}]},
        ]
    }


def _run(tmp_path: Path, name: str = GUARD, **kwargs):
    env = {"RELEASE_SHA": RELEASE_SHA, "BASE_URL": "https://prod.example.invalid"}
    return run_step(tmp_path, name, env=env, scenario=_git(**kwargs))


def test_guard_accepts_the_main_head_descending_from_production(tmp_path: Path) -> None:
    result, calls, outputs = _run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["sha"] == PREVIOUS_SHA
    merge_base = [call["args"] for call in calls if "merge-base" in call["args"]]
    assert merge_base == [["merge-base", "--is-ancestor", PREVIOUS_SHA, RELEASE_SHA]]


def test_guard_refuses_a_sha_that_is_no_longer_the_main_head(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, HEAD_GUARD, main_head="d" * 40)

    assert result.returncode != 0
    assert "not the current main head" in result.stdout


def test_head_guard_accepts_the_main_head(tmp_path: Path) -> None:
    result, calls, _ = _run(tmp_path, HEAD_GUARD)

    assert result.returncode == 0, result.stdout + result.stderr
    assert [call["args"] for call in calls] == [["ls-remote", "origin", "refs/heads/main"]]


def test_guard_refuses_when_production_is_not_an_ancestor(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, ancestor=False)

    assert result.returncode != 0
    assert "not an ancestor" in result.stdout


def test_guard_refuses_an_unknown_production_sha(tmp_path: Path) -> None:
    result, _, _ = _run(tmp_path, known=False)

    assert result.returncode != 0
    assert "not in the repository history" in result.stdout


def test_head_guard_runs_before_install_and_the_production_backup() -> None:
    names = [s["name"] for s in yaml.safe_load(WORKFLOW.read_text())["jobs"]["dark-deploy"]["steps"]]

    assert names.index("Require a passing CI run for the release commit") < names.index(HEAD_GUARD)
    assert names.index(HEAD_GUARD) < names.index("Install release tooling")
    assert names.index(HEAD_GUARD) < names.index(BACKUP)
    assert "PREVIOUS_SHA" not in step(HEAD_GUARD).get("env", {})


def test_guard_reads_production_and_runs_before_the_production_backup() -> None:
    # A refused run must not write a backup to the production volume.
    names = [s["name"] for s in yaml.safe_load(WORKFLOW.read_text())["jobs"]["dark-deploy"]["steps"]]

    assert names.index(HEAD_GUARD) < names.index(GUARD) < names.index(BACKUP)
    assert step(GUARD)["id"] == "production"
    assert "secrets." not in str(step(GUARD))
    assert "/api/release" in step(GUARD)["run"]
    assert "/api/release" not in step("Record previous verified deployment")["run"]


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

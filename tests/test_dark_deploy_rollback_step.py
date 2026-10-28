from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.dark_deploy_harness import ROOT, deployments, run_step, step


ROLLBACK = "Roll back code and variables after any failed release check"
PREVIOUS_SHA = "a" * 40
BASE_URL = "https://prod.example.invalid"


def _env(**overrides: str) -> dict[str, str]:
    env = {
        "RAILWAY_TOKEN": "TEST_TOKEN_NOT_SECRET",
        "RAILWAY_SERVICE": "svc",
        "RAILWAY_ENVIRONMENT": "env",
        "BASE_URL": BASE_URL,
        "PREVIOUS_DEPLOYMENT_ID": "previous",
        "PREVIOUS_SHA": PREVIOUS_SHA,
        "DARK_DEPLOYMENT_ID": "dark",
        "EXPECTED_CAMPAIGN_STATUS": "draft",
        "EXPECTED_CAMPAIGN_WRITE_STATE": "paused",
        "EXPECTED_PAPER_TRADE_ENABLED": "0",
        "ROLLBACK_POLL_ATTEMPTS": "6",
        "ROLLBACK_NEW_DEPLOYMENT_GRACE_POLLS": "2",
        "DEPLOYMENT_POLL_INTERVAL_SEC": "0",
    }
    env.update(overrides)
    return env


def _scenario(
    deployment_lists: list[Any],
    *,
    rollback_response: dict[str, Any] | None = None,
    release_sha: str = PREVIOUS_SHA,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "railway": [
            {
                "match": "deployment list",
                "responses": [{"body": body} for body in deployment_lists],
            },
            {
                "match": "variable list",
                "responses": [{"body": {"TRADER_KOO_PAPER_TRADE_ENABLED": "0"}}],
            },
        ],
        "curl": [
            {
                "match": "backboard.railway.com",
                "data_contains": "deploymentRollback",
                "responses": [
                    rollback_response or {"body": {"data": {"deploymentRollback": True}}}
                ],
            },
            {
                "match": "backboard.railway.com",
                "data_contains": "deploymentCancel",
                "responses": [{"body": {"data": {"deploymentCancel": True}}}],
            },
            {"match": "/api/health", "responses": [{"body": {"ok": True}}]},
            {
                "match": "/api/release",
                "responses": [{"body": {"ok": True, "git_sha": release_sha}}],
            },
            {
                "match": "/api/paper-trades/summary",
                "responses": [
                    {
                        "body": {
                            "ok": True,
                            "campaign_health": {"status": "draft", "write_state": "paused"},
                        }
                    }
                ],
            },
        ],
    }


def _graphql_bodies(calls: list[dict[str, Any]]) -> list[str]:
    return [call["data"] for call in calls if call["tool"] == "curl" and "data" in call]


def test_rollback_mutation_selects_the_scalar_boolean_result() -> None:
    query = (ROOT / "ops" / "railway-rollback.graphql").read_text(encoding="utf-8")

    assert "deploymentRollback(id: $id)" in query
    assert "{" not in query.split("deploymentRollback(id: $id)", 1)[1].replace("}", "")


def test_rollback_tracks_the_new_restored_deployment(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    building = deployments(("restored", "DEPLOYING"), ("dark", "FAILED"), ("previous", "REMOVED"))
    done = deployments(("restored", "SUCCESS"), ("dark", "FAILED"), ("previous", "REMOVED"))

    result, calls, outputs = run_step(
        tmp_path, ROLLBACK, env=_env(), scenario=_scenario([before, building, done])
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["id"] == "restored"
    assert outputs["observed_campaign_status"] == "draft"
    assert outputs["observed_write_state"] == "paused"
    assert outputs["paper_trade_enabled"] == "0"
    bodies = _graphql_bodies(calls)
    assert len(bodies) == 1 and '"id":"previous"' in bodies[0].replace(" ", "")
    assert '"status":"DEPLOYING"' in result.stdout.replace(" ", "")
    assert "TEST_TOKEN_NOT_SECRET" not in result.stdout + result.stderr


def test_rollback_prints_the_graphql_error_body_and_fails(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    rejected = {
        "status": 400,
        "body": {"errors": [{"message": "Field must not have a selection"}]},
    }

    result, calls, _ = run_step(
        tmp_path,
        ROLLBACK,
        env=_env(),
        scenario=_scenario([before], rollback_response=rejected),
    )

    assert result.returncode != 0
    assert "HTTP 400" in result.stdout
    assert "Field must not have a selection" in result.stdout
    assert not any("--fail" in call["args"] for call in calls if "data" in call)


def test_rollback_fails_when_the_mutation_returns_false(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = run_step(
        tmp_path,
        ROLLBACK,
        env=_env(),
        scenario=_scenario(
            [before], rollback_response={"body": {"data": {"deploymentRollback": False}}}
        ),
    )

    assert result.returncode != 0


def test_rollback_requires_the_public_release_sha_to_be_restored(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    done = deployments(("restored", "SUCCESS"), ("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = run_step(
        tmp_path,
        ROLLBACK,
        env=_env(),
        scenario=_scenario([before, done], release_sha="b" * 40),
    )

    assert result.returncode != 0


def test_rollback_does_not_depend_on_railway_commit_metadata() -> None:
    script = step(ROLLBACK)["run"]

    assert "commitHash" not in script
    assert ".data.deploymentRollback.id" not in script
    assert '"404"' not in script


def test_rollback_cancels_an_in_flight_dark_deployment_first(tmp_path: Path) -> None:
    before = deployments(("dark", "DEPLOYING"), ("previous", "SUCCESS"))
    done = deployments(("restored", "SUCCESS"), ("dark", "REMOVED"), ("previous", "REMOVED"))

    result, calls, _ = run_step(
        tmp_path, ROLLBACK, env=_env(), scenario=_scenario([before, done])
    )

    assert result.returncode == 0, result.stdout + result.stderr
    bodies = _graphql_bodies(calls)
    assert "deploymentCancel" in bodies[0] and '"dark"' in bodies[0]
    assert "deploymentRollback" in bodies[1]


def test_rollback_rejects_two_new_deployments(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    both = deployments(
        ("restored", "SUCCESS"), ("manual", "BUILDING"), ("dark", "FAILED"), ("previous", "REMOVED")
    )

    result, _, _ = run_step(tmp_path, ROLLBACK, env=_env(), scenario=_scenario([before, both]))

    assert result.returncode != 0
    assert "AMBIGUOUS" in result.stdout


def test_rollback_fails_when_the_restored_deployment_crashes(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    crashed = deployments(("restored", "CRASHED"), ("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = run_step(tmp_path, ROLLBACK, env=_env(), scenario=_scenario([before, crashed]))

    assert result.returncode != 0


def test_rollback_accepts_the_previous_deployment_when_no_new_one_appears(
    tmp_path: Path,
) -> None:
    unchanged = deployments(("dark", "FAILED"), ("previous", "SUCCESS"))

    result, _, outputs = run_step(
        tmp_path, ROLLBACK, env=_env(), scenario=_scenario([unchanged])
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["id"] == "previous"


def test_rollback_times_out_after_its_poll_budget(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    stuck = deployments(("restored", "DEPLOYING"), ("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = run_step(tmp_path, ROLLBACK, env=_env(), scenario=_scenario([before, stuck]))

    assert result.returncode != 0
    assert "did not reach SUCCESS" in result.stdout

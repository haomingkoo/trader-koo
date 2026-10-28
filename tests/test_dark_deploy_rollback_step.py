from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.dark_deploy_harness import ROOT, deployments, run_step


ROLLBACK = "Roll back code and variables after any failed release check"
PREVIOUS_SHA = "a" * 40
RELEASE_SHA = "c" * 40
DARK_UUID = "0f1e2d3c-4b5a-4968-8776-655443322110"
BASE_URL = "https://prod.example.invalid"


def _env(**overrides: str) -> dict[str, str]:
    env = {
        "RAILWAY_TOKEN": "TEST_TOKEN_NOT_SECRET",
        "RAILWAY_SERVICE": "svc",
        "RAILWAY_ENVIRONMENT": "env",
        "RELEASE_SHA": RELEASE_SHA,
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
    cancel_response: dict[str, Any] | None = None,
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
                "responses": [
                    cancel_response or {"body": {"data": {"deploymentCancel": True}}}
                ],
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


def _recorded(*rows: dict[str, Any]) -> dict[str, str]:
    """The deployment list the 'Record previous verified deployment' step saved."""
    return {"previous-deployments.json": json.dumps(list(rows))}


_RECORDED = _recorded(*deployments(("previous", "SUCCESS")))


def _run(
    tmp_path: Path,
    env: dict[str, str],
    scenario: dict[str, list[dict[str, Any]]],
    files: dict[str, str] | None = None,
):
    return run_step(
        tmp_path, ROLLBACK, env=env, scenario=scenario, files=files or _RECORDED
    )


def _row(deployment_id: str, status: str, message: str) -> dict[str, Any]:
    return {
        "id": deployment_id,
        "status": status,
        "createdAt": "2026-10-26T18:00:00Z",
        "meta": {"cliMessage": message},
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

    result, calls, outputs = _run(
        tmp_path, _env(), _scenario([before, building, done])
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

    result, calls, _ = _run(
        tmp_path, _env(), _scenario([before], rollback_response=rejected)
    )

    assert result.returncode != 0
    assert "HTTP 400" in result.stdout
    assert "Field must not have a selection" in result.stdout
    assert not any("--fail" in call["args"] for call in calls if "data" in call)


def test_rollback_fails_when_the_mutation_returns_false(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = _run(
        tmp_path,
        _env(),
        _scenario(
            [before], rollback_response={"body": {"data": {"deploymentRollback": False}}}
        ),
    )

    assert result.returncode != 0


def test_rollback_requires_the_public_release_sha_to_be_restored(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    done = deployments(("restored", "SUCCESS"), ("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = _run(
        tmp_path,
        _env(),
        _scenario([before, done], release_sha="b" * 40),
    )

    assert result.returncode != 0


def test_rollback_cancels_an_in_flight_dark_deployment_first(tmp_path: Path) -> None:
    before = deployments(("dark", "DEPLOYING"), ("previous", "SUCCESS"))
    done = deployments(("restored", "SUCCESS"), ("dark", "REMOVED"), ("previous", "REMOVED"))

    result, calls, _ = _run(
        tmp_path, _env(), _scenario([before, done])
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

    result, _, _ = _run(tmp_path, _env(), _scenario([before, both]))

    assert result.returncode != 0
    assert "AMBIGUOUS" in result.stdout


def test_rollback_fails_when_the_restored_deployment_crashes(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    crashed = deployments(("restored", "CRASHED"), ("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = _run(tmp_path, _env(), _scenario([before, crashed]))

    assert result.returncode != 0


def test_rollback_accepts_the_previous_deployment_when_no_new_one_appears(
    tmp_path: Path,
) -> None:
    unchanged = deployments(("dark", "FAILED"), ("previous", "SUCCESS"))

    result, _, outputs = _run(
        tmp_path, _env(), _scenario([unchanged])
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["id"] == "previous"


def test_rollback_times_out_after_its_poll_budget(tmp_path: Path) -> None:
    before = deployments(("dark", "FAILED"), ("previous", "REMOVED"))
    stuck = deployments(("restored", "DEPLOYING"), ("dark", "FAILED"), ("previous", "REMOVED"))

    result, _, _ = _run(tmp_path, _env(), _scenario([before, stuck]))

    assert result.returncode != 0
    assert "did not reach SUCCESS" in result.stdout


def test_rollback_finds_the_dark_id_in_the_upload_log_after_a_cancel(
    tmp_path: Path,
) -> None:
    # A cancelled upload step never writes its id output, but tee already
    # wrote the Build Logs line to disk.
    before = [_row(DARK_UUID, "BUILDING", "x"), *deployments(("previous", "SUCCESS"))]
    done = [
        _row("restored", "SUCCESS", "x"),
        _row(DARK_UUID, "REMOVED", "x"),
        *deployments(("previous", "REMOVED")),
    ]
    log = f"Build Logs: https://railway.com/project/p/service/s?id={DARK_UUID}&\n"

    result, calls, outputs = _run(
        tmp_path,
        _env(DARK_DEPLOYMENT_ID=""),
        _scenario([before, done]),
        files={**_RECORDED, "railway-up.log": log},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    bodies = _graphql_bodies(calls)
    assert "deploymentCancel" in bodies[0] and DARK_UUID in bodies[0]
    assert outputs["id"] == "restored"


def test_rollback_treats_an_unrecorded_release_upload_as_the_dark_deployment(
    tmp_path: Path,
) -> None:
    message = f"dark release {RELEASE_SHA}"
    before = [_row("upload", "BUILDING", message), *deployments(("previous", "SUCCESS"))]
    done = [
        _row("restored", "SUCCESS", "x"),
        _row("upload", "REMOVED", message),
        *deployments(("previous", "REMOVED")),
    ]

    result, calls, outputs = _run(
        tmp_path, _env(DARK_DEPLOYMENT_ID=""), _scenario([before, done])
    )

    assert result.returncode == 0, result.stdout + result.stderr
    bodies = _graphql_bodies(calls)
    assert "deploymentCancel" in bodies[0] and '"upload"' in bodies[0]
    assert outputs["id"] == "restored"


def test_rollback_fails_while_an_unrecorded_dark_deployment_keeps_building(
    tmp_path: Path,
) -> None:
    # The grace path would accept the previous deployment, which is still
    # SUCCESS only because the dark one has not passed its health check yet.
    message = f"dark release {RELEASE_SHA}"
    building = [_row("upload", "BUILDING", message), *deployments(("previous", "SUCCESS"))]

    result, _, outputs = _run(
        tmp_path,
        _env(DARK_DEPLOYMENT_ID=""),
        _scenario(
            [building],
            cancel_response={"body": {"data": {"deploymentCancel": False}}},
        ),
    )

    assert result.returncode != 0
    assert "still running" in result.stdout
    assert "upload" in result.stdout
    assert "observed_campaign_status" not in outputs


def test_rollback_fails_when_the_cancel_does_not_stop_the_dark_deployment(
    tmp_path: Path,
) -> None:
    before = deployments(("dark", "DEPLOYING"), ("previous", "SUCCESS"))
    restored = deployments(("restored", "SUCCESS"), ("dark", "DEPLOYING"), ("previous", "REMOVED"))

    result, _, _ = _run(
        tmp_path,
        _env(),
        _scenario(
            [before, restored],
            cancel_response={"status": 400, "body": {"errors": [{"message": "nope"}]}},
        ),
    )

    assert result.returncode != 0
    assert "still running" in result.stdout


def test_rollback_waits_for_a_healthy_dark_deployment_to_be_removed(
    tmp_path: Path,
) -> None:
    before = deployments(("dark", "SUCCESS"), ("previous", "REMOVED"))
    draining = deployments(("restored", "SUCCESS"), ("dark", "REMOVING"), ("previous", "REMOVED"))
    removed = deployments(("restored", "SUCCESS"), ("dark", "REMOVED"), ("previous", "REMOVED"))

    result, calls, outputs = _run(
        tmp_path, _env(), _scenario([before, draining, removed])
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["id"] == "restored"
    assert not any("deploymentCancel" in body for body in _graphql_bodies(calls))
    lists = [call for call in calls if "deployment list" in " ".join(call["args"])]
    assert len(lists) == 3
    assert sum(1 for call in calls if "/api/release" in " ".join(call["args"])) == 1


def test_a_drill_does_not_mistake_the_recorded_release_for_the_dark_one(
    tmp_path: Path,
) -> None:
    # A drill redeploys the running SHA, so the recorded deployment carries
    # the same release message as the dark upload.
    message = f"dark release {RELEASE_SHA}"
    unchanged = [_row("dark", "FAILED", message), _row("previous", "SUCCESS", message)]

    result, calls, outputs = _run(
        tmp_path,
        _env(DARK_DEPLOYMENT_ID=""),
        _scenario([unchanged], release_sha=PREVIOUS_SHA),
        files=_recorded(_row("previous", "SUCCESS", message)),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["id"] == "previous"
    assert not any("deploymentCancel" in body for body in _graphql_bodies(calls))

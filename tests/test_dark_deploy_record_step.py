from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.dark_deploy_harness import run_step


UPLOAD = "Upload dark release and wait for Railway health gate"
RECORD = "Record dark deployment"
RELEASE_SHA = "c" * 40
DARK_ID = "e2da9180-82d2-4184-8ccf-07b0c91646a7"
UP_OUTPUT = (
    "Indexing...\nUploading...\n"
    "  Build Logs: https://railway.com/project/p/service/s?id="
    f"{DARK_ID}&\nCI mode enabled\n"
)


def _env(**overrides: str) -> dict[str, str]:
    env = {
        "RAILWAY_TOKEN": "TEST_TOKEN_NOT_SECRET",
        "RAILWAY_PROJECT_ID": "project",
        "RAILWAY_SERVICE": "svc",
        "RAILWAY_ENVIRONMENT": "env",
        "RELEASE_SHA": RELEASE_SHA,
        "UPLOAD_DEPLOYMENT_ID": DARK_ID,
        "DEPLOYMENT_POLL_ATTEMPTS": "4",
        "DEPLOYMENT_POLL_INTERVAL_SEC": "0",
    }
    env.update(overrides)
    return env


def _row(deployment_id: str, status: str, message: str | None = None) -> dict[str, Any]:
    return {
        "id": deployment_id,
        "status": status,
        "createdAt": "2026-10-26T16:50:33Z",
        "meta": {"cliMessage": message or f"dark release {RELEASE_SHA}"},
    }


def _record(tmp_path: Path, lists: list[list[dict[str, Any]]], **env: str):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "previous-deployments.json").write_text(
        json.dumps([_row("previous", "SUCCESS", "dark release old")]), encoding="utf-8"
    )
    scenario = {
        "railway": [
            {"match": "deployment list", "responses": [{"body": rows} for rows in lists]}
        ]
    }
    return run_step(tmp_path, RECORD, env=_env(**env), scenario=scenario)


def test_upload_exposes_the_deployment_id_printed_by_railway_up(tmp_path: Path) -> None:
    scenario = {"railway": [{"match": "up --ci", "responses": [{"body": UP_OUTPUT}]}]}

    result, _, outputs = run_step(tmp_path, UPLOAD, env=_env(), scenario=scenario)

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["id"] == DARK_ID
    assert f"dark release {RELEASE_SHA}" in (tmp_path / "calls.jsonl").read_text()


def test_upload_keeps_the_id_when_railway_up_fails(tmp_path: Path) -> None:
    scenario = {
        "railway": [{"match": "up --ci", "responses": [{"body": UP_OUTPUT, "exit": 1}]}]
    }

    result, _, outputs = run_step(tmp_path, UPLOAD, env=_env(), scenario=scenario)

    assert result.returncode == 1
    assert outputs["id"] == DARK_ID


def test_record_waits_for_the_uploaded_deployment(tmp_path: Path) -> None:
    lists = [
        [_row(DARK_ID, "BUILDING"), _row("previous", "SUCCESS")],
        [_row(DARK_ID, "DEPLOYING"), _row("previous", "SUCCESS")],
        [_row(DARK_ID, "SUCCESS"), _row("previous", "REMOVED")],
    ]

    result, _, outputs = _record(tmp_path, lists)

    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["id"] == DARK_ID
    compact = result.stdout.replace(" ", "")
    assert '"status":"BUILDING"' in compact
    assert '"status":"DEPLOYING"' in compact
    assert f'"cliMessage":"darkrelease{RELEASE_SHA}"' in compact


def test_record_fails_without_an_id_from_railway_up(tmp_path: Path) -> None:
    result, _, _ = _record(tmp_path, [[_row(DARK_ID, "SUCCESS")]], UPLOAD_DEPLOYMENT_ID="")

    assert result.returncode != 0
    assert "no deployment id" in result.stdout


def test_record_rejects_a_deployment_that_existed_before_upload(tmp_path: Path) -> None:
    result, _, _ = _record(
        tmp_path, [[_row("previous", "SUCCESS")]], UPLOAD_DEPLOYMENT_ID="previous"
    )

    assert result.returncode != 0


def test_record_rejects_a_deployment_with_another_release_message(tmp_path: Path) -> None:
    result, _, _ = _record(tmp_path, [[_row(DARK_ID, "SUCCESS", "manual release")]])

    assert result.returncode != 0
    assert "manual release" in result.stdout


def test_record_fails_on_a_terminal_failure_status(tmp_path: Path) -> None:
    for status in ("FAILED", "CRASHED", "REMOVED", "SKIPPED"):
        result, _, _ = _record(tmp_path, [[_row(DARK_ID, status)]])
        assert result.returncode != 0, status
        assert status in result.stdout


def test_record_reports_the_last_state_when_the_budget_runs_out(tmp_path: Path) -> None:
    result, _, outputs = _record(tmp_path, [[_row(DARK_ID, "DEPLOYING")]])

    assert result.returncode != 0
    assert "did not reach SUCCESS within 4 polls" in result.stdout
    assert outputs["id"] == DARK_ID

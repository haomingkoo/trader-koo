from __future__ import annotations

import json
from pathlib import Path
import subprocess


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "dark-deploy.yml"


def test_dark_deploy_preserves_and_verifies_write_gate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert '"TRADER_KOO_PAPER_TRADE_ENABLED=0"' not in workflow
    assert "railway variable list --json" in workflow
    assert workflow.count(".TRADER_KOO_PAPER_TRADE_ENABLED") >= 4
    assert 'select(. == "0" or . == "1")' in workflow
    assert "steps.previous.outputs.paper_trade_enabled" in workflow
    assert "steps.previous.outputs.campaign_write_state" in workflow
    assert "--expected-campaign-write-state" in workflow
    assert "dark-deploy-manifest-v3" in workflow
    assert "previous_paper_trade_enabled" in workflow
    assert "deployed_paper_trade_enabled" in workflow
    assert "deployed_campaign_write_state" in workflow
    assert "deployed_campaign_status" in workflow
    assert "rollback_paper_trade_enabled" in workflow
    assert "deployment-verification.json" in workflow
    assert "jq -e '.ok == true' /tmp/previous-paper-summary.json" in workflow


def test_dark_deploy_does_not_publish_variable_inventory() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    upload_block = workflow.split("- name: Upload deployment evidence", 1)[1]

    assert "release-evidence/*.json" in upload_block
    assert "retention-days: 90" in upload_block
    assert "variables.json" not in upload_block
    assert "/tmp/previous-variables.json" in workflow
    assert "/tmp/deployed-variables.json" in workflow
    assert "/tmp/rollback-variables.json" in workflow


def test_dark_deploy_tracks_the_new_id_and_waits_for_its_terminal_status() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "UPLOAD_DEPLOYMENT_ID: ${{ steps.upload.outputs.id }}" in workflow
    assert '--message "dark release ${RELEASE_SHA}"' in workflow
    assert '--arg message "dark release ${RELEASE_SHA}"' in workflow
    assert "FAILED|CRASHED|REMOVED|REMOVING|SKIPPED) exit 1 ;;" in workflow
    assert "railway api" not in workflow
    assert "https://backboard.railway.com/graphql/v2" in workflow
    assert "Project-Access-Token: ${RAILWAY_TOKEN}" in workflow
    rollback_block = workflow.split(
        "- name: Roll back code and variables after any failed release check", 1
    )[1]
    assert "FAILED|CRASHED|REMOVED|REMOVING|SKIPPED)" in rollback_block


def test_dark_deploy_time_budgets_cover_health_check_and_rollback() -> None:
    import tomllib

    import yaml

    railway = tomllib.loads((WORKFLOW.parents[2] / "railway.toml").read_text(encoding="utf-8"))
    job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["dark-deploy"]
    steps = {step["name"]: step for step in job["steps"]}
    health_seconds = railway["deploy"]["healthcheckTimeout"]
    interval = int(job["env"]["DEPLOYMENT_POLL_INTERVAL_SEC"])
    deploy_poll_seconds = int(job["env"]["DEPLOYMENT_POLL_ATTEMPTS"]) * interval
    rollback_poll_seconds = int(job["env"]["ROLLBACK_POLL_ATTEMPTS"]) * interval
    install = steps["Install release tooling"]["timeout-minutes"]
    upload = steps["Upload dark release and wait for Railway health gate"]["timeout-minutes"]
    record = steps["Record dark deployment"]["timeout-minutes"]
    rollback_step = steps["Roll back code and variables after any failed release check"]
    rollback = rollback_step["timeout-minutes"]
    # Backup, migration replay, API checks, Chromium journeys, and manifests.
    unbounded_minutes = 15

    assert railway["deploy"]["drainingSeconds"] == 120
    assert 540 <= health_seconds <= 660
    assert deploy_poll_seconds >= health_seconds + 120
    assert rollback_poll_seconds >= health_seconds + 120
    assert record * 60 > deploy_poll_seconds
    assert rollback * 60 > rollback_poll_seconds
    assert install >= 20 and upload >= 15
    assert job["timeout-minutes"] >= install + upload + record + rollback + unbounded_minutes
    assert "failure() || cancelled()" in rollback_step["if"]


def test_railway_gate_json_shape_fails_closed() -> None:
    query = '.TRADER_KOO_PAPER_TRADE_ENABLED | select(. == "0" or . == "1")'
    for value in ("0", "1"):
        result = subprocess.run(
            ["jq", "-er", query],
            input=json.dumps({"TRADER_KOO_PAPER_TRADE_ENABLED": value}),
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == value

    for payload in ({}, {"TRADER_KOO_PAPER_TRADE_ENABLED": 0},
                    {"TRADER_KOO_PAPER_TRADE_ENABLED": "true"}):
        result = subprocess.run(
            ["jq", "-er", query], input=json.dumps(payload), text=True,
            capture_output=True, check=False,
        )
        assert result.returncode != 0


def test_every_workflow_action_is_pinned_to_a_full_commit_sha() -> None:
    import re

    workflows = sorted((WORKFLOW.parent).glob("*.yml"))
    assert workflows
    for path in workflows:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "uses:" not in line:
                continue
            reference = line.split("uses:", 1)[1].strip()
            assert re.fullmatch(
                r"[\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d+(\.\d+)*", reference
            ), f"{path.name}: {reference}"


def test_secrets_are_scoped_to_the_steps_that_use_them() -> None:
    import yaml

    job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["dark-deploy"]
    railway_steps = {
        "Verify exact Railway project-token scope",
        "Refuse a Railway GitHub autodeploy bypass",
        "Record previous verified deployment",
        "Bind release identity without starting a second deployment",
        "Upload dark release and wait for Railway health gate",
        "Record dark deployment",
        "Verify deployed Railway write gate",
        "Roll back code and variables after any failed release check",
    }
    admin_steps = {
        "Create and download a fresh consistent production backup",
        "Verify API contracts and exact commit",
        "Verify production Chromium journeys",
    }

    assert "secrets." not in json.dumps(job.get("env", {}))
    assert job["defaults"]["run"]["shell"] == "bash"
    for item in job["steps"]:
        env = json.dumps(item.get("env", {}))
        name = item["name"]
        assert ("secrets.RAILWAY_TOKEN" in env) == (name in railway_steps), name
        assert ("secrets.TRADER_KOO_API_KEY" in env) == (name in admin_steps), name
        assert "secrets." not in json.dumps(item.get("with", {})), name
        assert "secrets." not in item.get("run", ""), name
    steps = {item["name"]: item for item in job["steps"]}
    assert "npm ci --ignore-scripts" in steps["Install release tooling"]["run"]
    assert steps["Check out the exact CI commit"]["with"]["persist-credentials"] is False

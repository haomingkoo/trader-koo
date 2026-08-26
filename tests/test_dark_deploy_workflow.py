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
    assert "dark-deploy-manifest-v2" in workflow
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
    assert "variables.json" not in upload_block
    assert "/tmp/previous-variables.json" in workflow
    assert "/tmp/deployed-variables.json" in workflow
    assert "/tmp/rollback-variables.json" in workflow


def test_dark_deploy_tracks_the_new_id_and_waits_for_its_terminal_status() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "--slurpfile previous /tmp/previous-deployments.json" in workflow
    assert "map(.id) | index($id) | not" in workflow
    assert '--arg message "dark release ${RELEASE_SHA}"' in workflow
    assert "select(.meta.cliMessage == $message)" in workflow
    assert 'elif ($matches | length) > 1 then "AMBIGUOUS"' in workflow
    assert 'if [ "$status" = "SUCCESS" ]' in workflow
    assert (
        'if [ "$status" = "FAILED" ] || [ "$status" = "CRASHED" ] '
        '|| [ "$status" = "REMOVED" ]'
    ) in workflow
    assert "railway api" not in workflow
    assert "https://backboard.railway.com/graphql/v2" in workflow
    assert "Project-Access-Token: ${RAILWAY_TOKEN}" in workflow
    rollback_block = workflow.split(
        "- name: Roll back code and variables after any failed release check", 1
    )[1]
    assert '|| [ "$status" = "REMOVED" ]' in rollback_block


def test_dark_deploy_reserves_time_for_rollback() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    timeout_minutes = 45
    deploy_seconds = 90 * 5
    rollback_seconds = 120 * 5
    reserve_seconds = timeout_minutes * 60 - deploy_seconds - rollback_seconds

    assert "timeout-minutes: 45" in workflow
    assert 'DEPLOYMENT_POLL_ATTEMPTS: "90"' in workflow
    assert 'ROLLBACK_POLL_ATTEMPTS: "120"' in workflow
    assert reserve_seconds >= 25 * 60


def test_new_deployment_selector_does_not_reuse_previous_success(tmp_path: Path) -> None:
    previous = [
        {
            "id": "old-success",
            "status": "SUCCESS",
            "createdAt": "2026-08-26T07:40:00Z",
        }
    ]
    current = [
        {
            "id": "new-building",
            "status": "BUILDING",
            "createdAt": "2026-08-26T08:26:00Z",
        },
        *previous,
    ]
    query = """
        [.[]
          | select(.meta.cliMessage == $message)
          | select(.id as $id | ($previous[0] | map(.id) | index($id) | not))
        ] as $matches
        | if ($matches | length) == 1 then $matches[0].id
          elif ($matches | length) > 1 then "AMBIGUOUS"
          else ""
          end
    """
    previous_path = tmp_path / "previous.json"
    current_path = tmp_path / "current.json"
    previous_path.write_text(json.dumps(previous), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")
    result = subprocess.run(
        [
            "jq",
            "-r",
            "--slurpfile",
            "previous",
            str(previous_path),
            "--arg",
            "message",
            "dark release expected-sha",
            query,
            str(current_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == ""

    current[0]["meta"] = {"cliMessage": "dark release expected-sha"}
    current.append(
        {
            "id": "manual",
            "status": "BUILDING",
            "createdAt": "2026-08-26T08:27:00Z",
            "meta": {"cliMessage": "manual release"},
        }
    )
    current_path.write_text(json.dumps(current), encoding="utf-8")
    result = subprocess.run(result.args, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == "new-building"

    current.append(
        {
            "id": "duplicate-run-release",
            "status": "BUILDING",
            "createdAt": "2026-08-26T08:28:00Z",
            "meta": {"cliMessage": "dark release expected-sha"},
        }
    )
    current_path.write_text(json.dumps(current), encoding="utf-8")
    result = subprocess.run(result.args, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == "AMBIGUOUS"


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

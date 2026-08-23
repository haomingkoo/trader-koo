"""Fail-closed HTTP verification for a state-preserving Trader Koo release."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def _get(base_url: str, path: str, *, api_key: str | None = None) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.load(exc)
        except Exception:
            payload = None
        return exc.code, payload


def verify(
    base_url: str,
    expected_sha: str,
    api_key: str,
    expected_campaign_status: str = "draft",
) -> dict[str, Any]:
    release_status, release = _get(base_url, "/api/release")
    health_status, health = _get(base_url, "/api/health")
    status_code, status = _get(base_url, "/api/status")
    unauth_code, _ = _get(base_url, "/api/admin/agent-observability")
    auth_code, agents = _get(
        base_url, "/api/admin/agent-observability", api_key=api_key,
    )
    paper_status, paper = _get(base_url, "/api/paper-trades/summary")
    campaign = paper.get("campaign_health") or {}
    campaign_status = str(campaign.get("status") or "")
    campaign_status_matches = (
        campaign_status == "draft"
        if expected_campaign_status == "absent"
        else campaign_status == expected_campaign_status
    )
    contracts = {
        "release": release_status == 200 and release.get("git_sha") == expected_sha,
        "health": health_status == 200 and health.get("ok") is True,
        "sanitized_status": (
            status_code == 200
            and "db_name" not in status
            and "process" not in status
            and "git_sha" not in status.get("service_meta", {})
        ),
        "admin_rejects_missing_key": unauth_code in {401, 403},
        "admin_accepts_valid_key": auth_code == 200 and agents.get("ok") is True,
        "campaign_v2_status_preserved": (
            paper_status == 200
            and campaign.get("campaign_id") == "paper-v2"
            and campaign_status_matches
        ),
    }
    api_paths = {
        "report": "/api/daily-report",
        "chart": "/api/dashboard/SPY/quick?months=12",
        "experiment_results": "/api/research/experiments",
    }
    for name, path in api_paths.items():
        code, _ = _get(base_url, path)
        contracts[name] = code == 200
    failed = sorted(name for name, passed in contracts.items() if not passed)
    if failed:
        raise RuntimeError(f"deployment verification failed: {', '.join(failed)}")
    return {
        "ok": True,
        "expected_sha": expected_sha,
        "expected_campaign_status": expected_campaign_status,
        "observed_campaign_status": campaign_status,
        "contracts": contracts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--expected-campaign-status", default="draft")
    args = parser.parse_args()
    print(json.dumps(verify(
        args.base_url,
        args.expected_sha,
        args.api_key,
        args.expected_campaign_status,
    ), sort_keys=True))


if __name__ == "__main__":
    main()

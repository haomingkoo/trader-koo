"""Check production reachability, data, backup and release freshness from outside.

Every in-app monitor dies with the process, so a stopped container went
unnoticed for 47 hours. Run from GitHub Actions on a schedule. Exit 1 on any
problem so the failed run notifies the owner. Standard library only.

Checks, all fail-closed (a missing or unparseable field is a failure):

- ``/api/health`` is ok and the database exists;
- ``latest_data.price_date`` is not behind the expected trading session;
- ``freshness.report_fresh`` is true;
- ``backup.latest_backup_utc`` is within the daily cadence plus grace;
- the ``/api/release`` git_sha is on ``main`` and the oldest commit not yet
  deployed landed less than ``RELEASE_LAG_MAX`` ago (GitHub compare API);
- the latest attempted dark-deploy run (cancelled runs are unapproved, not
  attempts) did not conclude ``failure``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader_koo.report.utils import expected_price_date  # noqa: E402

# Covers GitHub's delayed cron and a slow nightly run on top of INGEST_DUE_LAG.
EXTRA_GRACE = dt.timedelta(hours=2)
# The scheduler backs up daily at 03:00 UTC; the grace covers a slow run and
# GitHub's delayed cron.
BACKUP_CADENCE = dt.timedelta(hours=24)
BACKUP_GRACE = dt.timedelta(hours=6)
BACKUP_MAX_AGE = BACKUP_CADENCE + BACKUP_GRACE
RELEASE_LAG_MAX = dt.timedelta(days=3)
DEPLOY_WORKFLOW = "dark-deploy.yml"
NOT_ATTEMPTED = frozenset({"cancelled", "skipped"})
GITHUB_API = "https://api.github.com"
FETCH_ATTEMPTS = 3
FETCH_TIMEOUT_SEC = 20
RETRY_SLEEP_SEC = 10
# The site firewall answers 403 to the default Python-urllib User-Agent.
USER_AGENT = "trader-koo-freshness-monitor/1.0 (+https://github.com/haomingkoo/trader-koo)"
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class FetchError(OSError):
    """A GET that failed; ``status`` is the HTTP status or None if unreachable."""

    def __init__(self, url: str, status: int | None, reason: str) -> None:
        self.url = url
        self.status = status
        self.reason = reason
        detail = f"HTTP {status}" if status is not None else f"unreachable ({reason})"
        super().__init__(f"GET {url} -> {detail}")

    def __str__(self) -> str:
        return self.args[0]


def fetch_json(
    url: str,
    timeout: int,
    attempts: int,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """GET *url* and return ``(http_status, json_object)``; raise ``FetchError``."""
    last: FetchError | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, **(headers or {})}
            )
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200))
                payload = json.load(resp)
            if not isinstance(payload, dict):
                raise ValueError("response is not a JSON object")
            return status, payload
        except urllib.error.HTTPError as exc:
            last = FetchError(url, exc.code, "HTTPError")
        except (OSError, ValueError) as exc:
            last = FetchError(url, None, type(exc).__name__)
        if attempt < attempts:
            time.sleep(RETRY_SLEEP_SEC)
    assert last is not None
    raise last


def _parse_utc(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _check_backup(status: dict[str, Any], now_utc: dt.datetime) -> str | None:
    backup = status.get("backup") if isinstance(status.get("backup"), dict) else {}
    taken = _parse_utc(backup.get("latest_backup_utc"))
    size = backup.get("latest_backup_size_bytes")
    if taken is None:
        return f"no parseable backup.latest_backup_utc ({backup.get('latest_backup_utc')!r})"
    if not isinstance(size, int) or size <= 0:
        return f"latest backup size is {size!r}"
    age = now_utc - taken
    if age > BACKUP_MAX_AGE:
        hours = age.total_seconds() / 3600
        limit = BACKUP_MAX_AGE.total_seconds() / 3600
        return f"latest backup {taken.isoformat()} is {hours:.1f}h old (limit {limit:.0f}h)"
    return None


def _check_release_lag(
    release: dict[str, Any], compare: dict[str, Any], now_utc: dt.datetime
) -> str | None:
    sha = str(release.get("git_sha") or "")
    state = compare.get("status")
    if state in {"identical", "behind"}:
        return None
    if state != "ahead":
        return f"deployed git_sha {sha[:8]} is not on main (compare status {state!r})"
    commits = compare.get("commits") or []
    # The compare API lists commits oldest first.
    oldest = ((commits[0].get("commit") or {}).get("committer") or {}) if commits else {}
    landed = _parse_utc(oldest.get("date"))
    if landed is None:
        return f"deployed git_sha {sha[:8]} is behind main but the oldest undeployed commit has no date"
    lag = now_utc - landed
    if lag > RELEASE_LAG_MAX:
        return (
            f"deployed git_sha {sha[:8]} is {compare.get('ahead_by')} commit(s) behind main; "
            f"oldest undeployed commit landed {lag.total_seconds() / 86400:.1f} days ago "
            f"(limit {RELEASE_LAG_MAX.days} days)"
        )
    return None


def _check_deploy_runs(deploy_runs: dict[str, Any]) -> str | None:
    runs = deploy_runs.get("workflow_runs")
    if not isinstance(runs, list):
        return "dark-deploy workflow runs unavailable"
    for run in runs:
        if run.get("status") != "completed" or run.get("conclusion") in NOT_ATTEMPTED:
            continue
        if run.get("conclusion") == "failure":
            return (
                f"latest attempted dark-deploy run for {str(run.get('head_sha') or '')[:8]} "
                f"concluded failure ({run.get('created_at')})"
            )
        return None
    return None


def evaluate(
    health: dict[str, Any],
    status: dict[str, Any],
    now_utc: dt.datetime,
    *,
    release: dict[str, Any],
    compare: dict[str, Any],
    deploy_runs: dict[str, Any],
    http_status: dict[str, int] | None = None,
) -> list[str]:
    codes = http_status or {}

    def where(source: str) -> str:
        code = codes.get(source)
        return f"{source} (HTTP {code})" if code is not None else source

    problems: list[str] = []
    if not (health.get("ok") and health.get("db_exists")):
        problems.append(f"{where('/api/health')}: not ok: {health}")

    price_date = (status.get("latest_data") or {}).get("price_date")
    if not price_date:
        problems.append(f"{where('/api/status')}: no latest_data.price_date")
    else:
        expected = expected_price_date(now_utc - EXTRA_GRACE)
        if str(price_date)[:10] < expected.isoformat():
            problems.append(
                f"{where('/api/status')}: price_date {price_date} is behind expected {expected.isoformat()}"
            )

    freshness = status.get("freshness") if isinstance(status.get("freshness"), dict) else {}
    if freshness.get("report_fresh") is not True:
        problems.append(
            f"{where('/api/status')}: report not fresh "
            f"(report_generated_ts={freshness.get('report_generated_ts')})"
        )

    backup_problem = _check_backup(status, now_utc)
    if backup_problem:
        problems.append(f"{where('/api/status')}: {backup_problem}")

    if not release.get("git_sha"):
        problems.append(f"{where('/api/release')}: no git_sha")
    else:
        lag_problem = _check_release_lag(release, compare, now_utc)
        if lag_problem:
            problems.append(f"{where('github compare')}: {lag_problem}")

    deploy_problem = _check_deploy_runs(deploy_runs)
    if deploy_problem:
        problems.append(f"{where('github dark-deploy runs')}: {deploy_problem}")
    return problems


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--repo", required=True, help="owner/name, for example $GITHUB_REPOSITORY")
    args = parser.parse_args(argv)
    base = args.base_url.strip().rstrip("/")
    if not base:
        print("FAIL: base URL is empty; set the TRADER_KOO_PRODUCTION_URL repository variable")
        return 1
    repo = args.repo.strip()
    if not REPO_RE.match(repo):
        print(f"FAIL: GitHub repository {repo!r} is not owner/name; pass --repo \"$GITHUB_REPOSITORY\"")
        return 1

    github_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        github_headers["Authorization"] = f"Bearer {token}"

    codes: dict[str, int] = {}
    try:
        codes["/api/health"], health = fetch_json(f"{base}/api/health", FETCH_TIMEOUT_SEC, FETCH_ATTEMPTS)
        codes["/api/status"], status = fetch_json(f"{base}/api/status", FETCH_TIMEOUT_SEC, FETCH_ATTEMPTS)
        codes["/api/release"], release = fetch_json(f"{base}/api/release", FETCH_TIMEOUT_SEC, FETCH_ATTEMPTS)
        compare: dict[str, Any] = {}
        sha = str(release.get("git_sha") or "")
        if sha:
            codes["github compare"], compare = fetch_json(
                f"{GITHUB_API}/repos/{repo}/compare/{sha}...main",
                FETCH_TIMEOUT_SEC,
                FETCH_ATTEMPTS,
                github_headers,
            )
        codes["github dark-deploy runs"], deploy_runs = fetch_json(
            f"{GITHUB_API}/repos/{repo}/actions/workflows/{DEPLOY_WORKFLOW}/runs?branch=main&per_page=30",
            FETCH_TIMEOUT_SEC,
            FETCH_ATTEMPTS,
            github_headers,
        )
    except FetchError as exc:
        if exc.status is None:
            print(f"FAIL: service unreachable or returned invalid JSON: {exc}")
        else:
            print(f"FAIL: {exc}")
        return 1

    problems = evaluate(
        health,
        status,
        _utc_now(),
        release=release,
        compare=compare,
        deploy_runs=deploy_runs,
        http_status=codes,
    )
    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        backup = status.get("backup") or {}
        print(
            f"OK: price_date={status['latest_data']['price_date']} "
            f"report_generated_ts={status['freshness'].get('report_generated_ts')} "
            f"latest_backup_utc={backup.get('latest_backup_utc')} "
            f"git_sha={str(release.get('git_sha'))[:8]}"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

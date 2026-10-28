"""External uptime and freshness check run from GitHub Actions."""
from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("check_prod_freshness", ROOT / "ops" / "check_prod_freshness.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

UTC = dt.timezone.utc
HEALTHY = {"ok": True, "db_exists": True}
RECORDED = json.loads((ROOT / "tests" / "fixtures" / "prod_freshness_2026_10_28.json").read_text())
RECORDED_NOW = dt.datetime.fromisoformat(RECORDED["now_utc"])


def _fresh_backup_status(now: dt.datetime = RECORDED_NOW) -> dict:
    status = copy.deepcopy(RECORDED["status"])
    status["backup"]["latest_backup_utc"] = (now - dt.timedelta(hours=7)).isoformat()
    return status


def _recorded(**overrides):
    inputs = {
        "health": copy.deepcopy(RECORDED["health"]),
        "status": _fresh_backup_status(),
        "now_utc": RECORDED_NOW,
        "release": copy.deepcopy(RECORDED["release"]),
        "compare": copy.deepcopy(RECORDED["compare_identical"]),
        "deploy_runs": copy.deepcopy(RECORDED["deploy_runs"]),
    }
    inputs.update(overrides)
    return inputs


def _evaluate(**overrides):
    inputs = _recorded(**overrides)
    return mod.evaluate(
        inputs["health"],
        inputs["status"],
        inputs["now_utc"],
        release=inputs["release"],
        compare=inputs["compare"],
        deploy_runs=inputs["deploy_runs"],
        http_status=inputs.get("http_status"),
    )


def _status(price_date: str | None, now: dt.datetime) -> dict:
    status = _fresh_backup_status(now)
    status["latest_data"]["price_date"] = price_date
    status["freshness"]["report_generated_ts"] = (now - dt.timedelta(hours=1)).isoformat()
    return status


def test_recorded_production_fails_only_on_backup_age():
    # On 2026-10-28 the newest backup was 34 hours old (weekly schedule plus
    # dark-deploy POSTs); every other recorded check passed.
    problems = _evaluate(status=copy.deepcopy(RECORDED["status"]))

    assert len(problems) == 1
    assert "backup" in problems[0] and "34.1h" in problems[0]


def test_recorded_production_with_a_daily_backup_is_healthy():
    assert _evaluate() == []


@pytest.mark.parametrize(
    ("health", "price_date", "now", "healthy"),
    [
        # Sunday, Friday bar present.
        (HEALTHY, "2026-10-23", dt.datetime(2026, 10, 25, 12, tzinfo=UTC), True),
        # Oct 22 08:00Z: Wednesday's 22:00Z ingest is overdue and only Tuesday is stored.
        (HEALTHY, "2026-10-20", dt.datetime(2026, 10, 22, 8, tzinfo=UTC), False),
        # The Oct 21-22 outage seen on Oct 23 morning.
        (HEALTHY, "2026-10-20", dt.datetime(2026, 10, 23, 8, tzinfo=UTC), False),
        # Thanksgiving: Wednesday's bar is current all day.
        (HEALTHY, "2026-11-25", dt.datetime(2026, 11, 26, 15, tzinfo=UTC), True),
        # Cron slot at 22:30Z on a trading day is inside the grace window.
        (HEALTHY, "2026-10-21", dt.datetime(2026, 10, 22, 22, 30, tzinfo=UTC), True),
        # Database missing.
        ({"ok": False, "db_exists": False}, "2026-10-23", dt.datetime(2026, 10, 25, 12, tzinfo=UTC), False),
        # No price date reported.
        (HEALTHY, None, dt.datetime(2026, 10, 25, 12, tzinfo=UTC), False),
    ],
)
def test_price_freshness(health, price_date, now, healthy):
    problems = _evaluate(health=health, status=_status(price_date, now), now_utc=now)

    assert (problems == []) is healthy


def test_missing_latest_data_block_fails():
    status = _status("2026-10-27", RECORDED_NOW)
    del status["latest_data"]

    assert any("price_date" in p for p in _evaluate(status=status))


def test_stale_report_fails():
    # Generated before Tuesday's close, so Tuesday's report never published.
    status = _fresh_backup_status()
    status["freshness"]["report_generated_ts"] = "2026-10-26T22:27:53Z"

    problems = _evaluate(status=status)

    assert len(problems) == 1
    assert "report" in problems[0] and "2026-10-26T22:27:53Z" in problems[0]
    assert "2026-10-27" in problems[0]


@pytest.mark.parametrize(
    "now",
    [
        # weekly_yolo publishes a Saturday 00:35Z report. The app's
        # report_fresh flag gives it 26 hours, so it reads false from Sunday
        # 02:35Z until the Monday nightly run.
        dt.datetime(2026, 10, 25, 12, tzinfo=UTC),
        dt.datetime(2026, 10, 26, 12, tzinfo=UTC),
    ],
)
def test_saturday_report_is_fresh_until_the_next_session_is_due(now):
    status = _fresh_backup_status(now)
    status["latest_data"]["price_date"] = "2026-10-23"
    status["freshness"]["report_generated_ts"] = "2026-10-24T00:35:00Z"
    status["freshness"]["report_fresh"] = False

    assert _evaluate(status=status, now_utc=now) == []


def test_saturday_report_goes_stale_once_monday_is_due():
    now = dt.datetime(2026, 10, 27, 3, tzinfo=UTC)
    status = _fresh_backup_status(now)
    status["latest_data"]["price_date"] = "2026-10-26"
    status["freshness"]["report_generated_ts"] = "2026-10-24T00:35:00Z"

    problems = _evaluate(status=status, now_utc=now)

    assert len(problems) == 1 and "report" in problems[0]


def test_missing_report_freshness_fails_closed():
    status = _fresh_backup_status()
    del status["freshness"]

    assert any("report" in p for p in _evaluate(status=status))


@pytest.mark.parametrize(
    ("age_hours", "healthy"),
    [(1, True), (29, True), (mod.BACKUP_MAX_AGE.total_seconds() / 3600 + 0.1, False), (24 * 7, False)],
)
def test_backup_age_is_bounded_by_cadence_plus_grace(age_hours, healthy):
    status = copy.deepcopy(RECORDED["status"])
    status["backup"]["latest_backup_utc"] = (RECORDED_NOW - dt.timedelta(hours=age_hours)).isoformat()

    problems = _evaluate(status=status)

    assert (problems == []) is healthy
    if not healthy:
        assert "backup" in problems[0]


@pytest.mark.parametrize(
    "backup",
    [None, {}, {"latest_backup_utc": None, "latest_backup_size_bytes": None}, {"latest_backup_utc": "garbage"}],
)
def test_missing_or_unparseable_backup_fails_closed(backup):
    status = copy.deepcopy(RECORDED["status"])
    if backup is None:
        del status["backup"]
    else:
        status["backup"] = backup

    assert any("backup" in p for p in _evaluate(status=status))


def test_zero_byte_backup_fails():
    status = copy.deepcopy(RECORDED["status"])
    status["backup"]["latest_backup_utc"] = RECORDED_NOW.isoformat()
    status["backup"]["latest_backup_size_bytes"] = 0

    assert any("backup" in p for p in _evaluate(status=status))


def test_release_lag_within_three_days_is_healthy():
    compare = copy.deepcopy(RECORDED["compare_ahead"])

    # Recorded: the undeployed commit landed 2026-10-26T22:33Z, 35 hours earlier.
    problems = _evaluate(compare=compare, status=_fresh_backup_status())

    assert problems == []


def test_release_lag_beyond_three_days_fails():
    compare = copy.deepcopy(RECORDED["compare_ahead"])
    later = RECORDED_NOW + dt.timedelta(days=2)
    status = _fresh_backup_status(later)
    status["latest_data"]["price_date"] = "2026-10-29"
    status["freshness"]["report_generated_ts"] = "2026-10-29T22:27:53Z"

    problems = _evaluate(compare=compare, now_utc=later, status=status)

    assert len(problems) == 1
    assert "223aa607" in problems[0] and "behind main" in problems[0]


@pytest.mark.parametrize("state", ["diverged", "behind", None])
def test_release_not_on_main_fails(state):
    compare = {"status": state, "commits": []} if state else {}

    assert any("not on main" in p for p in _evaluate(compare=compare))


def test_release_built_from_an_unmerged_descendant_of_main_fails():
    # compare/{deployed}...main is "behind" when the deployed commit sits on
    # top of main without being merged (recorded: 223aa607...a561085f).
    compare = {"status": "behind", "ahead_by": 0, "behind_by": 2, "commits": []}

    problems = _evaluate(compare=compare)

    assert len(problems) == 1
    assert "223aa607 is not on main" in problems[0]


def test_release_without_git_sha_fails():
    release = {"ok": True, "service": "trader_koo-api"}

    assert any("git_sha" in p for p in _evaluate(release=release))


def test_latest_attempted_dark_deploy_failure_fails():
    runs = copy.deepcopy(RECORDED["deploy_runs"])
    runs["workflow_runs"] = runs["workflow_runs"][3:]

    problems = _evaluate(deploy_runs=runs)

    assert len(problems) == 1
    assert "dark-deploy" in problems[0] and "ad0cee74" in problems[0]


@pytest.mark.parametrize("conclusion", ["timed_out", "startup_failure"])
def test_latest_attempted_dark_deploy_must_conclude_success(conclusion):
    runs = copy.deepcopy(RECORDED["deploy_runs"])
    runs["workflow_runs"].insert(
        0, {"status": "completed", "conclusion": conclusion, "head_sha": "e" * 40}
    )

    problems = _evaluate(deploy_runs=runs)

    assert len(problems) == 1
    assert "dark-deploy" in problems[0] and conclusion in problems[0]


def test_unapproved_cancelled_runs_do_not_hide_an_earlier_failure():
    runs = copy.deepcopy(RECORDED["deploy_runs"])
    runs["workflow_runs"] = runs["workflow_runs"][1:]

    assert any("dark-deploy" in p for p in _evaluate(deploy_runs=runs))


def test_in_progress_run_is_ignored_in_favour_of_the_last_completed_one():
    runs = copy.deepcopy(RECORDED["deploy_runs"])
    runs["workflow_runs"].insert(0, {"status": "waiting", "conclusion": None, "head_sha": "f" * 40})

    assert _evaluate(deploy_runs=runs) == []


def test_missing_workflow_runs_fails_closed():
    assert any("dark-deploy" in p for p in _evaluate(deploy_runs={}))


def test_fail_messages_carry_the_http_status():
    status = _fresh_backup_status()
    status["freshness"]["report_generated_ts"] = None

    problems = _evaluate(status=status, http_status={"/api/status": 200})

    assert len(problems) == 1
    assert problems[0].startswith("/api/status (HTTP 200): report not fresh")


def test_main_fails_closed_when_unreachable(monkeypatch, capsys):
    def boom(url, timeout, attempts, headers=None):
        raise mod.FetchError(url, None, "URLError")

    monkeypatch.setattr(mod, "fetch_json", boom)

    code = mod.main(["--base-url", "https://example.invalid", "--repo", "owner/repo"])

    assert code == 1
    assert "unreachable" in capsys.readouterr().out


def test_main_reports_the_http_status_of_a_failed_fetch(monkeypatch, capsys):
    def fake_fetch(url, timeout, attempts, headers=None):
        if url.endswith("/api/status"):
            raise mod.FetchError(url, 503, "HTTPError")
        return 200, {}

    monkeypatch.setattr(mod, "fetch_json", fake_fetch)

    code = mod.main(["--base-url", "https://example.invalid", "--repo", "owner/repo"])

    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL: GET https://example.invalid/api/status -> HTTP 503" in out


def test_main_runs_every_check_against_recorded_payloads(monkeypatch, capsys):
    status = _fresh_backup_status()
    responses = {
        "/api/health": RECORDED["health"],
        "/api/status": status,
        "/api/release": RECORDED["release"],
        "/compare/": RECORDED["compare_identical"],
        "/actions/workflows/dark-deploy.yml/runs": RECORDED["deploy_runs"],
    }
    seen: list[tuple[str, dict]] = []

    def fake_fetch(url, timeout, attempts, headers=None):
        seen.append((url, headers or {}))
        for marker, payload in responses.items():
            if marker in url:
                return 200, copy.deepcopy(payload)
        raise AssertionError(url)

    monkeypatch.setattr(mod, "fetch_json", fake_fetch)
    monkeypatch.setattr(mod, "_utc_now", lambda: RECORDED_NOW)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-printed")

    code = mod.main(["--base-url", "https://example.invalid", "--repo", "owner/repo"])

    out = capsys.readouterr().out
    assert code == 0, out
    assert out.startswith("OK:")
    assert "test-token-not-printed" not in out
    compare_url = next(url for url, _ in seen if "/compare/" in url)
    assert compare_url == (
        "https://api.github.com/repos/owner/repo/compare/"
        "223aa607d8d983d9742438a179ad65a64cb3a419...main"
    )
    github_headers = [h for url, h in seen if url.startswith("https://api.github.com")]
    assert github_headers and all(h["Authorization"] == "Bearer test-token-not-printed" for h in github_headers)


@pytest.mark.parametrize("url", ["", "   "])
def test_main_rejects_empty_base_url(monkeypatch, capsys, url):
    def boom(*args, **kwargs):
        raise AssertionError("must not fetch with an empty base URL")

    monkeypatch.setattr(mod, "fetch_json", boom)

    code = mod.main(["--base-url", url, "--repo", "owner/repo"])

    assert code == 1
    assert "TRADER_KOO_PRODUCTION_URL" in capsys.readouterr().out


@pytest.mark.parametrize("repo", ["", "not-a-repo"])
def test_main_rejects_missing_repo(monkeypatch, capsys, repo):
    monkeypatch.setattr(mod, "fetch_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    code = mod.main(["--base-url", "https://example.invalid", "--repo", repo])

    assert code == 1
    assert "repository" in capsys.readouterr().out


def test_workflow_runs_the_monitor_on_a_schedule():
    text = (ROOT / ".github" / "workflows" / "prod-freshness.yml").read_text(encoding="utf-8")

    assert "schedule:" in text
    assert "ops/check_prod_freshness.py" in text
    assert "vars.TRADER_KOO_PRODUCTION_URL" in text
    assert "contents: read" in text
    assert "actions: read" in text
    assert '--repo "$GITHUB_REPOSITORY"' in text
    assert "GITHUB_TOKEN: ${{ github.token }}" in text


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return b'{"ok": true}'


def test_fetch_sends_explicit_user_agent(monkeypatch):
    # The site firewall answers 403 to the default Python-urllib User-Agent.
    seen = {}

    def fake_urlopen(request, timeout):
        seen["ua"] = request.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    assert mod.fetch_json("https://example.invalid/api/health", 1, 1) == (200, {"ok": True})
    assert seen["ua"] == mod.USER_AGENT
    assert "Python-urllib" not in seen["ua"]


def test_fetch_raises_with_the_http_status(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 502, "Bad Gateway", {}, io.BytesIO(b""))

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod, "RETRY_SLEEP_SEC", 0)

    with pytest.raises(mod.FetchError) as caught:
        mod.fetch_json("https://example.invalid/api/status", 1, 2)

    assert caught.value.status == 502
    assert str(caught.value) == "GET https://example.invalid/api/status -> HTTP 502"

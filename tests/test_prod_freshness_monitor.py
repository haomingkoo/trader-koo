"""External uptime and freshness check run from GitHub Actions."""
from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("check_prod_freshness", ROOT / "ops" / "check_prod_freshness.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

UTC = dt.timezone.utc
HEALTHY = {"ok": True, "db_exists": True}


def _status(price_date: str | None) -> dict:
    return {"ok": True, "latest_data": {"price_date": price_date}}


@pytest.mark.parametrize(
    ("health", "status", "now", "healthy"),
    [
        # Sunday, Friday bar present.
        (HEALTHY, _status("2026-10-23"), dt.datetime(2026, 10, 25, 12, tzinfo=UTC), True),
        # Oct 22 08:00Z: Wednesday's 22:00Z ingest is overdue and only Tuesday is stored.
        (HEALTHY, _status("2026-10-20"), dt.datetime(2026, 10, 22, 8, tzinfo=UTC), False),
        # The Oct 21-22 outage seen on Oct 23 morning.
        (HEALTHY, _status("2026-10-20"), dt.datetime(2026, 10, 23, 8, tzinfo=UTC), False),
        # Thanksgiving: Wednesday's bar is current all day.
        (HEALTHY, _status("2026-11-25"), dt.datetime(2026, 11, 26, 15, tzinfo=UTC), True),
        # Cron slot at 22:30Z on a trading day is inside the grace window.
        (HEALTHY, _status("2026-10-21"), dt.datetime(2026, 10, 22, 22, 30, tzinfo=UTC), True),
        # Database missing.
        ({"ok": False, "db_exists": False}, _status("2026-10-23"), dt.datetime(2026, 10, 25, 12, tzinfo=UTC), False),
        # No price date reported.
        (HEALTHY, _status(None), dt.datetime(2026, 10, 25, 12, tzinfo=UTC), False),
        (HEALTHY, {"ok": True}, dt.datetime(2026, 10, 25, 12, tzinfo=UTC), False),
    ],
)
def test_evaluate(health, status, now, healthy):
    problems = mod.evaluate(health, status, now)

    assert (problems == []) is healthy


def test_main_fails_closed_when_unreachable(monkeypatch, capsys):
    def boom(url, timeout, attempts):
        raise OSError("connection refused")

    monkeypatch.setattr(mod, "fetch_json", boom)

    code = mod.main(["--base-url", "https://example.invalid"])

    assert code == 1
    assert "unreachable" in capsys.readouterr().out


@pytest.mark.parametrize("url", ["", "   "])
def test_main_rejects_empty_base_url(monkeypatch, capsys, url):
    def boom(*args, **kwargs):
        raise AssertionError("must not fetch with an empty base URL")

    monkeypatch.setattr(mod, "fetch_json", boom)

    code = mod.main(["--base-url", url])

    assert code == 1
    assert "TRADER_KOO_PRODUCTION_URL" in capsys.readouterr().out


def test_workflow_runs_the_monitor_on_a_schedule():
    text = (ROOT / ".github" / "workflows" / "prod-freshness.yml").read_text(encoding="utf-8")

    assert "schedule:" in text
    assert "ops/check_prod_freshness.py" in text
    assert "vars.TRADER_KOO_PRODUCTION_URL" in text
    assert "contents: read" in text


def test_fetch_sends_explicit_user_agent(monkeypatch):
    # The site firewall answers 403 to the default Python-urllib User-Agent.
    seen = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        seen["ua"] = request.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    assert mod.fetch_json("https://example.invalid/api/health", 1, 1) == {"ok": True}
    assert seen["ua"] == mod.USER_AGENT
    assert "Python-urllib" not in seen["ua"]

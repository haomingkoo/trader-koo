from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from trader_koo.notifications import health_check


@pytest.fixture(autouse=True)
def _reset_health_state():
    health_check._failure_counts.clear()
    health_check._last_alert_at.clear()
    yield
    health_check._failure_counts.clear()
    health_check._last_alert_at.clear()


def test_trader_koo_uses_lightweight_health_endpoint() -> None:
    trader = next(site for site in health_check.SITES if site["name"] == "Trader Koo")
    assert trader["url"] == "https://trader.kooexperience.com/api/health"


def test_alerts_after_two_failures_with_truthful_heading(monkeypatch) -> None:
    monkeypatch.setattr(
        health_check,
        "SITES",
        [{"name": "Trader Koo", "url": "https://example.test/api/health"}],
    )

    with (
        patch.object(
            health_check.httpx,
            "get",
            side_effect=httpx.ConnectTimeout("timed out"),
        ),
        patch.object(health_check, "send_message", return_value=True) as send_message,
    ):
        health_check.run_health_check(now_monotonic=100.0)
        send_message.assert_not_called()
        health_check.run_health_check(now_monotonic=200.0)

    send_message.assert_called_once()
    text = send_message.call_args.args[0]
    assert "HEALTH CHECK FAILED" in text
    assert "SITE DOWN" not in text
    assert send_message.call_args.kwargs["parse_mode"] == "HTML"


def test_one_timeout_reports_partial_scope_path_duration_and_five_healthy() -> None:
    results = [
        {
            "name": site["name"],
            "url": site["url"],
            "status": 200,
            "ok": True,
            "elapsed_ms": 50,
        }
        for site in health_check.SITES
    ]
    trader = next(result for result in results if result["name"] == "Trader Koo")
    trader.update(
        {
            "status": 0,
            "ok": False,
            "elapsed_ms": 15_000,
            "error": "timed out",
            "error_type": "ReadTimeout",
            "timed_out": True,
        }
    )
    health_check._failure_counts["Trader Koo"] = 2

    with (
        patch.object(health_check, "check_all_sites", return_value=results),
        patch.object(health_check, "send_message", return_value=True) as send_message,
    ):
        health_check.run_health_check(now_monotonic=200.0)

    text = send_message.call_args.args[0]
    assert "HEALTH CHECK PARTIAL FAILURE" in text
    assert "Affected: 1/6 checks (Trader Koo)" in text
    assert "https://trader.kooexperience.com/api/health" in text
    assert "timeout after 15.0s (limit 15s; 2 consecutive)" in text
    assert "Healthy: 5/6 checks" in text
    assert health_check._last_alert_at["Trader Koo"] == 200.0


def test_partial_timeout_recovery_clears_incident_without_recovery_spam(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        health_check,
        "SITES",
        [
            {"name": "Trader Koo", "url": "https://example.test/api/health"},
            *[
                {"name": f"Healthy {index}", "url": f"https://healthy-{index}.test"}
                for index in range(5)
            ],
        ],
    )
    healthy = SimpleNamespace(status_code=200)

    with (
        patch.object(
            health_check.httpx,
            "get",
            side_effect=[
                httpx.ReadTimeout("timed out"),
                *([healthy] * 5),
                httpx.ReadTimeout("timed out"),
                *([healthy] * 5),
                *([healthy] * 6),
            ],
        ),
        patch.object(health_check, "send_message", return_value=True) as send_message,
    ):
        health_check.run_health_check(now_monotonic=100.0)
        health_check.run_health_check(now_monotonic=200.0)
        health_check.run_health_check(now_monotonic=300.0)

    send_message.assert_called_once()
    assert health_check._failure_counts["Trader Koo"] == 0
    assert "Trader Koo" not in health_check._last_alert_at


def test_alert_scope_matches_detailed_subset_when_another_failure_is_new() -> None:
    alertable = {
        "name": "Trader Koo",
        "url": "https://trader.kooexperience.com/api/health",
        "status": 0,
        "ok": False,
        "elapsed_ms": 15_000,
        "timed_out": True,
    }
    new_failure = {
        "name": "Portfolio",
        "url": "https://kooexperience.com",
        "status": 500,
        "ok": False,
        "elapsed_ms": 100,
    }
    healthy = [
        {"name": f"Healthy {index}", "ok": True}
        for index in range(4)
    ]
    health_check._failure_counts["Trader Koo"] = 2

    with patch.object(health_check, "send_message", return_value=True) as send_message:
        health_check._send_health_alert(
            [alertable],
            [alertable, new_failure, *healthy],
        )

    text = send_message.call_args.args[0]
    assert "Affected: 1/6 checks (Trader Koo)" in text
    assert "Current failures: 2/6 checks (Trader Koo, Portfolio)" in text
    assert "<b>Portfolio</b>" not in text


def test_continuing_failure_uses_time_based_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(
        health_check,
        "SITES",
        [{"name": "Trader Koo", "url": "https://example.test/api/health"}],
    )

    with (
        patch.object(
            health_check.httpx,
            "get",
            side_effect=httpx.ConnectTimeout("timed out"),
        ),
        patch.object(health_check, "send_message", return_value=True) as send_message,
    ):
        health_check.run_health_check(now_monotonic=100.0)
        health_check.run_health_check(now_monotonic=200.0)
        health_check.run_health_check(now_monotonic=300.0)
        assert send_message.call_count == 1
        health_check.run_health_check(
            now_monotonic=200.0 + health_check.HEALTH_ALERT_COOLDOWN_SEC
        )

    assert send_message.call_count == 2


def test_recovery_resets_incident_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(
        health_check,
        "SITES",
        [{"name": "Trader Koo", "url": "https://example.test/api/health"}],
    )
    healthy = SimpleNamespace(status_code=200)

    with (
        patch.object(
            health_check.httpx,
            "get",
            side_effect=[
                httpx.ConnectTimeout("timed out"),
                httpx.ConnectTimeout("timed out"),
                healthy,
                httpx.ConnectTimeout("timed out"),
                httpx.ConnectTimeout("timed out"),
            ],
        ),
        patch.object(health_check, "send_message", return_value=True) as send_message,
    ):
        for now in (100.0, 200.0, 300.0, 400.0, 500.0):
            health_check.run_health_check(now_monotonic=now)

    assert send_message.call_count == 2


def test_failed_telegram_send_does_not_start_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(
        health_check,
        "SITES",
        [{"name": "Trader Koo", "url": "https://example.test/api/health"}],
    )

    with (
        patch.object(
            health_check.httpx,
            "get",
            side_effect=httpx.ConnectTimeout("timed out"),
        ),
        patch.object(
            health_check,
            "send_message",
            side_effect=[False, True],
        ) as send_message,
    ):
        health_check.run_health_check(now_monotonic=100.0)
        health_check.run_health_check(now_monotonic=200.0)
        health_check.run_health_check(now_monotonic=300.0)

    assert send_message.call_count == 2
    assert health_check._last_alert_at["Trader Koo"] == 300.0

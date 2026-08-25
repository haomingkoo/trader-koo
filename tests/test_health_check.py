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

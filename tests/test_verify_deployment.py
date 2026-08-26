from __future__ import annotations

import io

import pytest

from trader_koo.scripts import verify_deployment


def test_get_identifies_release_verifier_and_sends_admin_key(monkeypatch) -> None:
    captured = {}

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response(b'{"ok": true}')

    monkeypatch.setattr(verify_deployment.urllib.request, "urlopen", fake_urlopen)

    status, payload = verify_deployment._get(
        "https://example.invalid", "/api/health", api_key="test-key"
    )

    assert status == 200
    assert payload == {"ok": True}
    assert captured["request"].get_header("User-agent") == (
        "trader-koo-release-verifier/1"
    )
    assert captured["request"].get_header("X-api-key") == "test-key"


def test_verify_deployment_checks_exact_sha_auth_and_public_contracts(monkeypatch) -> None:
    responses = {
        "/api/release": (200, {"ok": True, "git_sha": "a" * 40}),
        "/api/health": (200, {"ok": True}),
        "/api/status": (200, {"ok": True, "service_meta": {"version": "0.2.0"}}),
        "/api/admin/agent-observability": (401, {"detail": "unauthorized"}),
        "/api/daily-report": (200, {"ok": True}),
        "/api/dashboard/SPY/quick?months=12": (200, {"ticker": "SPY"}),
        "/api/paper-trades/summary": (
            200,
            {"ok": True, "campaign_health": {
                "campaign_id": "paper-v2", "status": "draft", "write_state": "paused",
            }},
        ),
        "/api/research/experiments": (200, {"ok": True}),
    }

    def fake_get(_base_url, path, *, api_key=None):
        if path == "/api/admin/agent-observability" and api_key:
            return 200, {"ok": True}
        return responses[path]

    monkeypatch.setattr(verify_deployment, "_get", fake_get)
    result = verify_deployment.verify(
        "https://example.invalid", "a" * 40, "key", "draft", "paused"
    )
    assert result["ok"] is True
    assert all(result["contracts"].values())


def test_verify_deployment_rejects_wrong_release(monkeypatch) -> None:
    monkeypatch.setattr(
        verify_deployment,
        "_get",
        lambda *_args, **_kwargs: (200, {"ok": True, "git_sha": "wrong"}),
    )
    with pytest.raises(RuntimeError, match="release"):
        verify_deployment.verify(
            "https://example.invalid", "a" * 40, "key", "draft", "paused"
        )


def test_verify_deployment_preserves_active_campaign_on_later_releases(monkeypatch) -> None:
    def fake_get(_base_url, path, *, api_key=None):
        if path == "/api/release":
            return 200, {"ok": True, "git_sha": "a" * 40}
        if path == "/api/health":
            return 200, {"ok": True}
        if path == "/api/status":
            return 200, {"ok": True, "service_meta": {}}
        if path == "/api/admin/agent-observability":
            return (200, {"ok": True}) if api_key else (401, {})
        if path == "/api/paper-trades/summary":
            return 200, {"ok": True, "campaign_health": {
                "campaign_id": "paper-v2", "status": "active", "write_state": "enabled",
            }}
        return 200, {"ok": True}

    monkeypatch.setattr(verify_deployment, "_get", fake_get)
    result = verify_deployment.verify(
        "https://example.invalid", "a" * 40, "key", "active", "enabled"
    )
    assert result["observed_campaign_status"] == "active"
    assert result["observed_campaign_write_state"] == "enabled"

    with pytest.raises(RuntimeError, match="campaign_v2_status_preserved"):
        verify_deployment.verify(
            "https://example.invalid", "a" * 40, "key", "draft", "enabled"
        )

    with pytest.raises(RuntimeError, match="campaign_v2_write_state_preserved"):
        verify_deployment.verify(
            "https://example.invalid", "a" * 40, "key", "active", "paused"
        )

    def degraded_get(base_url, path, *, api_key=None):
        code, payload = fake_get(base_url, path, api_key=api_key)
        if path == "/api/paper-trades/summary":
            payload = {**payload, "ok": False}
        return code, payload

    monkeypatch.setattr(verify_deployment, "_get", degraded_get)
    with pytest.raises(RuntimeError, match="paper_summary"):
        verify_deployment.verify(
            "https://example.invalid", "a" * 40, "key", "active", "enabled"
        )


def test_verify_deployment_allows_only_absent_to_draft_bootstrap(monkeypatch) -> None:
    def fake_get(_base_url, path, *, api_key=None):
        if path == "/api/release":
            return 200, {"ok": True, "git_sha": "a" * 40}
        if path == "/api/health":
            return 200, {"ok": True}
        if path == "/api/status":
            return 200, {"ok": True, "service_meta": {}}
        if path == "/api/admin/agent-observability":
            return (200, {"ok": True}) if api_key else (401, {})
        if path == "/api/paper-trades/summary":
            return 200, {"ok": True, "campaign_health": {
                "campaign_id": "paper-v2", "status": "draft", "write_state": "paused",
            }}
        return 200, {"ok": True}

    monkeypatch.setattr(verify_deployment, "_get", fake_get)
    assert verify_deployment.verify(
        "https://example.invalid", "a" * 40, "key", "absent", "paused"
    )["ok"] is True


def test_verify_deployment_rejects_missing_or_unknown_write_state(monkeypatch) -> None:
    def fake_get(_base_url, path, *, api_key=None):
        if path == "/api/release":
            return 200, {"ok": True, "git_sha": "a" * 40}
        if path == "/api/health":
            return 200, {"ok": True}
        if path == "/api/status":
            return 200, {"ok": True, "service_meta": {}}
        if path == "/api/admin/agent-observability":
            return (200, {"ok": True}) if api_key else (401, {})
        if path == "/api/paper-trades/summary":
            return 200, {"campaign_health": {"campaign_id": "paper-v2", "status": "draft"}}
        return 200, {"ok": True}

    monkeypatch.setattr(verify_deployment, "_get", fake_get)
    with pytest.raises(TypeError):
        verify_deployment.verify(
            "https://example.invalid", "a" * 40, "key", "draft"
        )
    with pytest.raises(RuntimeError, match="campaign_v2_write_state_preserved"):
        verify_deployment.verify(
            "https://example.invalid", "a" * 40, "key", "draft", "paused"
        )
    with pytest.raises(ValueError, match="enabled or paused"):
        verify_deployment.verify(
            "https://example.invalid", "a" * 40, "key", "draft", "unknown"
        )

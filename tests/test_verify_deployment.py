from __future__ import annotations

import pytest

from trader_koo.scripts import verify_deployment


def test_verify_deployment_checks_exact_sha_auth_and_public_contracts(monkeypatch) -> None:
    responses = {
        "/api/release": (200, {"ok": True, "git_sha": "a" * 40}),
        "/api/health": (200, {"ok": True}),
        "/api/status": (200, {"ok": True, "service_meta": {"version": "0.2.0"}}),
        "/api/admin/agent-observability": (401, {"detail": "unauthorized"}),
        "/api/daily-report": (200, {"ok": True}),
        "/api/dashboard/SPY/quick?months=12": (200, {"ticker": "SPY"}),
        "/api/paper-trades/summary": (200, {"ok": True}),
        "/api/research/experiments": (200, {"ok": True}),
    }

    def fake_get(_base_url, path, *, api_key=None):
        if path == "/api/admin/agent-observability" and api_key:
            return 200, {"ok": True}
        return responses[path]

    monkeypatch.setattr(verify_deployment, "_get", fake_get)
    result = verify_deployment.verify("https://example.invalid", "a" * 40, "key")
    assert result["ok"] is True
    assert all(result["contracts"].values())


def test_verify_deployment_rejects_wrong_release(monkeypatch) -> None:
    monkeypatch.setattr(
        verify_deployment,
        "_get",
        lambda *_args, **_kwargs: (200, {"ok": True, "git_sha": "wrong"}),
    )
    with pytest.raises(RuntimeError, match="release"):
        verify_deployment.verify("https://example.invalid", "a" * 40, "key")

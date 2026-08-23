from __future__ import annotations

from trader_koo.backend.routers import system


def test_release_contract_exposes_only_non_secret_identity(monkeypatch) -> None:
    monkeypatch.setattr(system, "STATUS_GIT_SHA", "a" * 40)
    payload = system.release()

    assert payload == {
        "ok": True,
        "service": "trader_koo-api",
        "version": system.APP_VERSION,
        "git_sha": "a" * 40,
    }


def test_release_contract_fails_closed_without_commit(monkeypatch) -> None:
    monkeypatch.setattr(system, "STATUS_GIT_SHA", None)
    assert system.release()["ok"] is False

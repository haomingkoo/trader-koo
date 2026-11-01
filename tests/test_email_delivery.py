"""Observable contract of the single email delivery module.

Every caller (nightly send, admin resend, SMTP health, the daily-report CLI)
resolves config, validates it and sends through ``trader_koo.email_delivery``.
These tests drive that interface with a fake transport; no network is used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from trader_koo import email_delivery
from trader_koo.email_delivery import OutgoingEmail, load_email_config
from trader_koo.report import email_dispatch
from trader_koo.report_email import build_report_email_subject

_EMAIL_ENV = (
    "TRADER_KOO_EMAIL_TRANSPORT",
    "TRADER_KOO_SMTP_HOST",
    "TRADER_KOO_SMTP_PORT",
    "TRADER_KOO_SMTP_USER",
    "TRADER_KOO_SMTP_PASS",
    "TRADER_KOO_SMTP_FROM",
    "TRADER_KOO_SMTP_SECURITY",
    "TRADER_KOO_SMTP_TIMEOUT_SEC",
    "TRADER_KOO_RESEND_API_KEY",
    "TRADER_KOO_RESEND_FROM",
    "TRADER_KOO_RESEND_TIMEOUT_SEC",
    "TRADER_KOO_REPORT_EMAIL_TO",
    "TRADER_KOO_AUTO_EMAIL",
    "TRADER_KOO_EMAIL_MAX_RECIPIENTS",
)


class FakeTransport:
    name = "fake"

    def __init__(self) -> None:
        self.sent: list[tuple[str, OutgoingEmail]] = []

    def send(self, email: OutgoingEmail, recipient: str) -> None:
        self.sent.append((recipient, email))


@pytest.fixture(autouse=True)
def _clean_email_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _EMAIL_ENV:
        monkeypatch.delenv(key, raising=False)


def _smtp_env(monkeypatch: pytest.MonkeyPatch, **extra: str) -> None:
    monkeypatch.setenv("TRADER_KOO_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("TRADER_KOO_SMTP_FROM", "reports@example.com")
    monkeypatch.setenv("TRADER_KOO_REPORT_EMAIL_TO", "reader@example.com")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def _report() -> dict[str, Any]:
    return {"ok": True, "generated_ts": "2026-10-30T22:00:00+00:00", "meta": {}}


# A1: transport selection comes from env, identically for every caller.
@pytest.mark.parametrize(
    ("mode", "api_key", "expected"),
    [
        ("auto", "YOUR_KEY", "resend"),
        ("auto", "", "smtp"),
        ("smtp", "YOUR_KEY", "smtp"),
        ("resend", "", "resend"),
        ("bogus", "YOUR_KEY", "resend"),
    ],
)
def test_transport_selection_from_env(monkeypatch, mode, api_key, expected):
    monkeypatch.setenv("TRADER_KOO_EMAIL_TRANSPORT", mode)
    if api_key:
        monkeypatch.setenv("TRADER_KOO_RESEND_API_KEY", api_key)
    config = load_email_config()
    assert config.transport == expected
    assert email_delivery.transport_for(config).name == expected


# A2: SMTP user without SMTP_PASS is "not configured" in every path.
def test_missing_smtp_pass_is_not_configured_everywhere(monkeypatch, tmp_path):
    from trader_koo.backend.routers.admin import email_admin

    _smtp_env(monkeypatch, TRADER_KOO_SMTP_USER="mailer")
    config = load_email_config()
    assert "TRADER_KOO_SMTP_PASS" in config.missing()
    assert not config.is_configured

    health = email_admin.smtp_health()
    assert health["ok"] is False
    assert "TRADER_KOO_SMTP_PASS" in health["missing"]

    fake = FakeTransport()
    monkeypatch.setattr(email_admin, "transport_for", lambda config: fake)
    with pytest.raises(HTTPException) as admin_exc:
        email_admin.email_latest_report(to=None, include_markdown=True, attach_json=True)
    assert admin_exc.value.status_code == 500
    assert "TRADER_KOO_SMTP_PASS" in str(admin_exc.value.detail)

    with pytest.raises(RuntimeError, match="TRADER_KOO_SMTP_PASS"):
        email_dispatch.send_report_email(
            _report(), "# md", db_path=tmp_path / "e.db", transport=fake
        )
    assert fake.sent == []


# A3: recipients resolve the same way for nightly send and admin resend.
def test_recipient_resolution_identical_for_nightly_and_admin(monkeypatch, tmp_path):
    from trader_koo.backend.routers.admin import email_admin

    _smtp_env(
        monkeypatch,
        TRADER_KOO_REPORT_EMAIL_TO="B@Example.com, a@example.com ; not-an-email",
    )
    expected = ["b@example.com", "a@example.com"]
    assert load_email_config().recipients() == expected

    nightly = FakeTransport()
    email_dispatch.send_report_email(
        _report(), "# md", db_path=tmp_path / "e.db", transport=nightly
    )
    assert [r for r, _ in nightly.sent] == expected

    admin = FakeTransport()
    monkeypatch.setattr(email_admin, "transport_for", lambda config: admin)
    monkeypatch.setattr(
        email_admin,
        "latest_daily_report_json",
        lambda report_dir: (tmp_path / "daily_report_x.json", _report()),
    )
    email_admin.email_latest_report(to=None, include_markdown=False, attach_json=False)
    assert [r for r, _ in admin.sent] == expected


# A4: an explicit ?to= override replaces the env recipients.
def test_admin_to_override(monkeypatch, tmp_path):
    from trader_koo.backend.routers.admin import email_admin

    _smtp_env(monkeypatch)
    fake = FakeTransport()
    monkeypatch.setattr(email_admin, "transport_for", lambda config: fake)
    monkeypatch.setattr(
        email_admin,
        "latest_daily_report_json",
        lambda report_dir: (tmp_path / "daily_report_x.json", _report()),
    )
    out = email_admin.email_latest_report(
        to="Other@Example.com", include_markdown=False, attach_json=False
    )
    assert [r for r, _ in fake.sent] == ["other@example.com"]
    assert out["to"] == "other@example.com"


# A5: the fake transport receives the expected nightly message.
def test_nightly_message_contents(monkeypatch, tmp_path):
    _smtp_env(monkeypatch)
    fake = FakeTransport()
    summary = email_dispatch.send_report_email(
        _report(), "# Daily", db_path=tmp_path / "e.db", transport=fake
    )
    assert summary["sent_count"] == 1
    assert summary["transport"] == "smtp"
    recipient, message = fake.sent[0]
    assert recipient == "reader@example.com"
    assert message.subject == build_report_email_subject(_report())
    assert message.html
    assert [a.filename for a in message.attachments] == ["daily_report_2026-10-30.md"]
    assert message.attachments[0].data == b"# Daily"


# A6: smtp-health keeps its response shape and never returns the secrets.
def test_smtp_health_shape_and_no_secrets(monkeypatch):
    from trader_koo.backend.routers.admin import email_admin

    _smtp_env(
        monkeypatch,
        TRADER_KOO_SMTP_USER="mailer",
        TRADER_KOO_SMTP_PASS="YOUR_SMTP_PASS",
        TRADER_KOO_RESEND_API_KEY="YOUR_RESEND_KEY",
        TRADER_KOO_EMAIL_TRANSPORT="smtp",
    )
    health = email_admin.smtp_health()
    assert set(health) == {
        "ok", "auto_email_enabled", "transport", "email_max_recipients",
        "missing", "smtp", "resend",
    }
    assert set(health["smtp"]) == {
        "host", "port", "security", "timeout_sec", "from_email",
        "default_to", "has_user", "has_password",
    }
    assert set(health["resend"]) == {
        "has_api_key", "from_email", "default_to", "timeout_sec",
    }
    assert health["ok"] is True
    dumped = json.dumps(health)
    assert "YOUR_SMTP_PASS" not in dumped
    assert "YOUR_RESEND_KEY" not in dumped


# A7: email-latest-report keeps its response shape; a send error becomes 500.
def test_email_latest_report_shape_and_failure(monkeypatch, tmp_path):
    from trader_koo.backend.routers.admin import email_admin

    _smtp_env(monkeypatch)
    monkeypatch.setattr(
        email_admin,
        "latest_daily_report_json",
        lambda report_dir: (tmp_path / "daily_report_x.json", _report()),
    )
    fake = FakeTransport()
    monkeypatch.setattr(email_admin, "transport_for", lambda config: fake)
    out = email_admin.email_latest_report(to=None, include_markdown=True, attach_json=True)
    assert set(out) == {
        "ok", "transport", "to", "subject", "report_file",
        "smtp_host", "smtp_port", "smtp_security",
    }
    assert out["transport"] == "smtp" and out["to"] == "reader@example.com"
    assert [a.filename for a in fake.sent[0][1].attachments] == ["daily_report_x.json"]

    class Boom(FakeTransport):
        def send(self, email, recipient):
            raise RuntimeError("relay refused")

    monkeypatch.setattr(email_admin, "transport_for", lambda config: Boom())
    with pytest.raises(HTTPException) as exc:
        email_admin.email_latest_report(to=None, include_markdown=False, attach_json=False)
    assert exc.value.status_code == 500
    assert "relay refused" in str(exc.value.detail)


# A8: the SMTP adapter builds a MIME message and logs in with STARTTLS.
def test_smtp_adapter_sends_mime(monkeypatch):
    _smtp_env(
        monkeypatch,
        TRADER_KOO_SMTP_USER="mailer",
        TRADER_KOO_SMTP_PASS="YOUR_SMTP_PASS",
    )
    calls: list[Any] = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls.append(("connect", host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def ehlo(self):
            calls.append("ehlo")

        def starttls(self, context):
            calls.append("starttls")

        def login(self, user, password):
            calls.append(("login", user, password == "YOUR_SMTP_PASS"))

        def send_message(self, msg):
            calls.append(("send", msg))

    monkeypatch.setattr(email_delivery.smtplib, "SMTP", FakeSMTP)
    transport = email_delivery.transport_for(load_email_config())
    transport.send(
        OutgoingEmail(
            subject="S",
            text="T",
            html="<p>H</p>",
            attachments=(email_delivery.Attachment(b"x", "text", "markdown", "r.md"),),
        ),
        "reader@example.com",
    )
    assert calls[0] == ("connect", "smtp.example.com", 587, 30)
    assert "starttls" in calls
    assert ("login", "mailer", True) in calls
    msg = calls[-1][1]
    assert msg["To"] == "reader@example.com"
    assert msg["From"] == "reports@example.com"
    assert [p.get_filename() for p in msg.iter_attachments()] == ["r.md"]


# A9: the Resend adapter posts one recipient with text and html.
def test_resend_adapter_payload(monkeypatch):
    monkeypatch.setenv("TRADER_KOO_RESEND_API_KEY", "YOUR_RESEND_KEY")
    monkeypatch.setenv("TRADER_KOO_RESEND_FROM", "reports@example.com")
    captured: dict[str, Any] = {}

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["auth_ok"] = req.get_header("Authorization") == "Bearer YOUR_RESEND_KEY"
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return Resp()

    monkeypatch.setattr(email_delivery.urllib.request, "urlopen", fake_urlopen)
    config = load_email_config()
    assert config.transport == "resend"
    email_delivery.transport_for(config).send(
        OutgoingEmail(subject="S", text="T", html="<p>H</p>"), "reader@example.com"
    )
    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["auth_ok"] is True
    assert captured["payload"] == {
        "from": "reports@example.com",
        "to": ["reader@example.com"],
        "subject": "S",
        "text": "T",
        "html": "<p>H</p>",
    }


# A10: the drifted private copies are gone; callers share one module.
def test_duplicate_helpers_removed():
    from trader_koo.backend.routers import email as email_router
    from trader_koo.scripts import generate_daily_report

    for name in ("_smtp_settings", "_resend_settings", "_email_transport",
                 "_send_resend_email", "_send_smtp_email"):
        assert not hasattr(email_router, name)
    for name in ("_smtp_cfg", "_resend_cfg", "_email_transport",
                 "_send_resend_email", "_send_smtp_email"):
        assert not hasattr(email_dispatch, name)
        assert not hasattr(generate_daily_report, name)

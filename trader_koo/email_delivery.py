"""Email delivery: one config, one definition of "configured", two transports.

Every sender (nightly report, admin resend, SMTP health, daily-report CLI)
goes through this module:

    config = load_email_config()          # env read at call time
    config.missing()                      # [] when the transport is usable
    config.recipients(override)           # normalised recipient list
    transport_for(config).send(email, recipient)

SMTP and Resend are adapters behind ``EmailTransport``; tests pass a fake.
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Protocol

from trader_koo.email_subscribers import email_max_recipients, parse_recipients

RESEND_API_URL = "https://api.resend.com/emails"
DEFAULT_SMTP_PORT = 587
DEFAULT_TIMEOUT_SEC = 30
MIN_TIMEOUT_SEC = 5
HTTP_OK = 200
HTTP_REDIRECT_FLOOR = 300
ERROR_BODY_CHARS = 500


def _env_timeout(raw: str) -> int:
    try:
        return max(MIN_TIMEOUT_SEC, int(raw.strip()))
    except ValueError:
        return DEFAULT_TIMEOUT_SEC


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    user: str
    password: str
    from_email: str
    timeout_sec: int
    security: str


@dataclass(frozen=True)
class ResendSettings:
    api_key: str
    from_email: str
    timeout_sec: int


@dataclass(frozen=True)
class EmailConfig:
    transport: str
    smtp: SmtpSettings
    resend: ResendSettings
    default_to: str

    @property
    def from_email(self) -> str:
        return self.resend.from_email if self.transport == "resend" else self.smtp.from_email

    def missing(self) -> list[str]:
        """Env vars the selected transport still needs (recipients excluded)."""
        missing: list[str] = []
        if self.transport == "resend":
            if not self.resend.api_key:
                missing.append("TRADER_KOO_RESEND_API_KEY")
            if not self.resend.from_email:
                missing.append("TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
        else:
            if not self.smtp.host:
                missing.append("TRADER_KOO_SMTP_HOST")
            if not self.smtp.from_email:
                missing.append("TRADER_KOO_SMTP_FROM")
            if self.smtp.user and not self.smtp.password:
                missing.append("TRADER_KOO_SMTP_PASS")
        return missing

    @property
    def is_configured(self) -> bool:
        return not self.missing()

    def recipients(self, override: str | None = None) -> list[str]:
        """Valid, lower-cased, de-duplicated recipients, capped at the max.

        ``override`` (e.g. an admin ``?to=``) replaces TRADER_KOO_REPORT_EMAIL_TO.
        """
        raw = override if override and override.strip() else self.default_to
        return parse_recipients(raw)[: max(1, email_max_recipients())]


def load_email_config() -> EmailConfig:
    port_raw = os.getenv("TRADER_KOO_SMTP_PORT", "587").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = DEFAULT_SMTP_PORT
    security = os.getenv("TRADER_KOO_SMTP_SECURITY", "starttls").strip().lower()
    if security not in {"starttls", "ssl", "none"}:
        security = "starttls"
    smtp_timeout = os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30")
    smtp_from = os.getenv("TRADER_KOO_SMTP_FROM", "")
    smtp = SmtpSettings(
        host=os.getenv("TRADER_KOO_SMTP_HOST", "").strip(),
        port=port,
        user=os.getenv("TRADER_KOO_SMTP_USER", "").strip(),
        password=os.getenv("TRADER_KOO_SMTP_PASS", ""),
        from_email=smtp_from.strip(),
        timeout_sec=_env_timeout(smtp_timeout),
        security=security,
    )
    resend = ResendSettings(
        api_key=os.getenv("TRADER_KOO_RESEND_API_KEY", "").strip(),
        from_email=os.getenv("TRADER_KOO_RESEND_FROM", smtp_from).strip(),
        timeout_sec=_env_timeout(os.getenv("TRADER_KOO_RESEND_TIMEOUT_SEC", smtp_timeout)),
    )
    mode = os.getenv("TRADER_KOO_EMAIL_TRANSPORT", "auto").strip().lower()
    if mode not in {"smtp", "resend"}:
        mode = "resend" if resend.api_key else "smtp"
    return EmailConfig(
        transport=mode,
        smtp=smtp,
        resend=resend,
        default_to=os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
    )


@dataclass(frozen=True)
class Attachment:
    data: bytes
    maintype: str
    subtype: str
    filename: str


@dataclass(frozen=True)
class OutgoingEmail:
    subject: str
    text: str
    html: str = ""
    attachments: tuple[Attachment, ...] = field(default_factory=tuple)


class EmailTransport(Protocol):
    name: str

    def send(self, email: OutgoingEmail, recipient: str) -> None: ...


class SmtpTransport:
    name = "smtp"

    def __init__(self, settings: SmtpSettings) -> None:
        self._s = settings

    def _message(self, email: OutgoingEmail, recipient: str) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = email.subject
        msg["From"] = self._s.from_email
        msg["To"] = recipient
        msg.set_content(email.text)
        if email.html:
            msg.add_alternative(email.html, subtype="html")
        for att in email.attachments:
            msg.add_attachment(
                att.data, maintype=att.maintype, subtype=att.subtype, filename=att.filename
            )
        return msg

    def send(self, email: OutgoingEmail, recipient: str) -> None:
        s = self._s
        msg = self._message(email, recipient)
        if s.security == "ssl":
            with smtplib.SMTP_SSL(
                s.host, s.port, timeout=s.timeout_sec, context=ssl.create_default_context()
            ) as server:
                if s.user:
                    server.login(s.user, s.password)
                server.send_message(msg)
            return
        with smtplib.SMTP(s.host, s.port, timeout=s.timeout_sec) as server:
            server.ehlo()
            if s.security == "starttls":
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
            if s.user:
                server.login(s.user, s.password)
            server.send_message(msg)


class ResendTransport:
    """Resend HTTP API. Attachments are not sent (unchanged from before)."""

    name = "resend"

    def __init__(self, settings: ResendSettings) -> None:
        self._s = settings

    def send(self, email: OutgoingEmail, recipient: str) -> None:
        payload: dict[str, object] = {
            "from": self._s.from_email,
            "to": [recipient],
            "subject": email.subject,
            "text": email.text,
        }
        if email.html:
            payload["html"] = email.html
        req = urllib.request.Request(
            RESEND_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._s.api_key}",
                "Content-Type": "application/json",
                "User-Agent": os.getenv("TRADER_KOO_EMAIL_USER_AGENT", "trader-koo/1.0"),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._s.timeout_sec) as resp:
                status = int(getattr(resp, "status", HTTP_OK))
                body = resp.read().decode("utf-8", errors="replace")
            if status >= HTTP_REDIRECT_FLOOR:
                raise RuntimeError(f"Resend API failed status={status} body={body[:ERROR_BODY_CHARS]}")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Resend API HTTP {exc.code}: {err_body[:ERROR_BODY_CHARS]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Resend connect failed: {exc.reason}") from exc


def transport_for(config: EmailConfig) -> EmailTransport:
    if config.transport == "resend":
        return ResendTransport(config.resend)
    return SmtpTransport(config.smtp)

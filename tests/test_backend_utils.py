"""Unit tests for shared backend utilities."""

from types import SimpleNamespace
from unittest.mock import Mock

from trader_koo.backend.utils import client_ip


def _mock_request(host: str | None, headers: dict[str, str] | None = None) -> Mock:
    request = Mock()
    request.headers = headers or {}
    request.client = SimpleNamespace(host=host) if host is not None else None
    return request


def test_client_ip_ignores_spoofed_forwarded_headers_by_default(monkeypatch):
    monkeypatch.delenv("TRADER_KOO_TRUST_PROXY_HEADERS", raising=False)
    monkeypatch.delenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", raising=False)

    request = _mock_request(
        "8.8.8.8",
        headers={"x-forwarded-for": "1.2.3.4", "x-real-ip": "5.6.7.8"},
    )

    assert client_ip(request) == "8.8.8.8"


def test_client_ip_trusts_forwarded_headers_from_private_proxy(monkeypatch):
    monkeypatch.delenv("TRADER_KOO_TRUST_PROXY_HEADERS", raising=False)
    monkeypatch.delenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", raising=False)

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": "198.51.100.23, 10.0.0.5"},
    )

    assert client_ip(request) == "198.51.100.23"


def test_client_ip_can_explicitly_trust_proxy_headers(monkeypatch):
    monkeypatch.setenv("TRADER_KOO_TRUST_PROXY_HEADERS", "1")

    request = _mock_request(
        "testclient",
        headers={"x-real-ip": "203.0.113.10"},
    )

    assert client_ip(request) == "203.0.113.10"

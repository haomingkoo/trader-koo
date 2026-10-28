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


# Production chain (checked read-only): client -> Cloudflare -> Railway edge ->
# app, and the *.up.railway.app origin answers without Cloudflare. Only the
# right end of the forwarded chain is written by proxies we run behind, so the
# client is the rightmost hop that is not a trusted proxy. Client addresses
# below are global because Python counts the documentation ranges as private.
_CLOUDFLARE_TEST_CIDR = "173.245.48.0/20"
_CLOUDFLARE_HOP = "173.245.48.10"


def _clear_proxy_env(monkeypatch) -> None:
    monkeypatch.delenv("TRADER_KOO_TRUST_PROXY_HEADERS", raising=False)
    monkeypatch.delenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", raising=False)


def test_client_ip_ignores_spoofed_leftmost_forwarded_entry(monkeypatch):
    _clear_proxy_env(monkeypatch)

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": "6.6.6.6, 9.9.9.9"},
    )

    assert client_ip(request) == "9.9.9.9"


def test_rotating_spoofed_forwarded_entries_keep_one_client_key(monkeypatch):
    _clear_proxy_env(monkeypatch)

    seen = {
        client_ip(
            _mock_request(
                "10.0.0.5",
                headers={"x-forwarded-for": f"4.4.4.{n}, 9.9.9.9"},
            )
        )
        for n in range(1, 6)
    }

    assert seen == {"9.9.9.9"}


def test_client_ip_skips_configured_proxy_hops_from_the_right(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", _CLOUDFLARE_TEST_CIDR)

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": f"6.6.6.6, 9.9.9.9, {_CLOUDFLARE_HOP}"},
    )

    assert client_ip(request) == "9.9.9.9"


def test_edge_real_ip_is_the_hop_when_origin_is_hit_directly(monkeypatch):
    """A direct origin hit cannot hide behind a forged X-Forwarded-For."""
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", _CLOUDFLARE_TEST_CIDR)

    request = _mock_request(
        "10.0.0.5",
        headers={
            "x-forwarded-for": f"9.9.9.9, {_CLOUDFLARE_HOP}",
            "x-real-ip": "7.7.7.7",
        },
    )

    assert client_ip(request) == "7.7.7.7"


def test_edge_real_ip_from_trusted_proxy_falls_through_to_forwarded_chain(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", _CLOUDFLARE_TEST_CIDR)

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": "9.9.9.9", "x-real-ip": _CLOUDFLARE_HOP},
    )

    assert client_ip(request) == "9.9.9.9"


def test_cloudflare_cidrs_let_a_worker_forge_the_client_while_origin_is_open(monkeypatch):
    """Why SECURITY.md forbids Cloudflare's ranges while the Railway origin is open.

    A Worker fetching *.up.railway.app egresses from 2a06:98c0:3600::103 (inside
    Cloudflare's 2a06:98c0::/29) and sets X-Forwarded-For itself. Trusting that
    range turns the forged entry into the client key.
    """
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", "2a06:98c0::/29")

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": "6.6.6.6, 2a06:98c0:3600::103"},
    )

    assert client_ip(request) == "6.6.6.6"


def test_worker_hop_is_the_client_without_cloudflare_cidrs(monkeypatch):
    _clear_proxy_env(monkeypatch)

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": "6.6.6.6, 2a06:98c0:3600::103"},
    )

    assert client_ip(request) == "2a06:98c0:3600::103"


def test_unconfigured_cloudflare_hop_is_treated_as_the_client(monkeypatch):
    """Without the CIDR list the key is coarse, but it cannot be spoofed."""
    _clear_proxy_env(monkeypatch)

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": "9.9.9.9", "x-real-ip": _CLOUDFLARE_HOP},
    )

    assert client_ip(request) == _CLOUDFLARE_HOP


def test_cf_connecting_ip_is_never_trusted(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", _CLOUDFLARE_TEST_CIDR)

    request = _mock_request(
        "10.0.0.5",
        headers={
            "cf-connecting-ip": "6.6.6.6",
            "x-forwarded-for": "9.9.9.9",
        },
    )

    assert client_ip(request) == "9.9.9.9"


def test_all_private_chain_returns_the_original_hop(monkeypatch):
    _clear_proxy_env(monkeypatch)

    request = _mock_request(
        "10.0.0.5",
        headers={"x-forwarded-for": "192.168.1.1, 10.0.0.9"},
    )

    assert client_ip(request) == "192.168.1.1"

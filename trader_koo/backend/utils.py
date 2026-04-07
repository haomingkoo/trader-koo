"""Shared utility functions for the trader_koo backend.

Extracted to eliminate duplication across routers and services.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import resource
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Request

LOG = logging.getLogger(__name__)
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _direct_client_host(request: Request) -> str | None:
    if request.client and request.client.host:
        return str(request.client.host).strip() or None
    return None


def _parse_ip(value: Any) -> ipaddress._BaseAddress | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return ipaddress.ip_address(raw)
    except ValueError:
        return None


def _proxy_header_trust_mode() -> str:
    raw = str(os.getenv("TRADER_KOO_TRUST_PROXY_HEADERS", "auto") or "auto").strip().lower()
    if raw in _TRUTHY:
        return "always"
    if raw in _FALSY:
        return "never"
    return "auto"


@lru_cache(maxsize=16)
def _parse_trusted_proxy_networks(raw: str) -> tuple[ipaddress._BaseNetwork, ...]:
    networks: list[ipaddress._BaseNetwork] = []
    for part in (segment.strip() for segment in str(raw or "").split(",")):
        if not part:
            continue
        try:
            networks.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            LOG.warning("Ignoring invalid TRADER_KOO_TRUSTED_PROXY_CIDRS entry: %s", part)
    return tuple(networks)


def _trusted_proxy_networks() -> tuple[ipaddress._BaseNetwork, ...]:
    return _parse_trusted_proxy_networks(os.getenv("TRADER_KOO_TRUSTED_PROXY_CIDRS", ""))


def _trust_forwarded_headers(request: Request) -> bool:
    mode = _proxy_header_trust_mode()
    if mode == "always":
        return True
    if mode == "never":
        return False

    peer = _parse_ip(_direct_client_host(request))
    if peer is None:
        return False
    if peer.is_loopback or peer.is_private or peer.is_link_local:
        return True
    return any(peer in network for network in _trusted_proxy_networks())


def _forwarded_client_ip(request: Request) -> str | None:
    if not _trust_forwarded_headers(request):
        return None

    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        for candidate in (part.strip() for part in xff.split(",")):
            parsed = _parse_ip(candidate)
            if parsed is not None:
                return str(parsed)

    real_ip = _parse_ip(request.headers.get("x-real-ip", ""))
    if real_ip is not None:
        return str(real_ip)

    return None


def client_ip(request: Request) -> str:
    """Extract the client IP from a request with trusted proxy handling."""
    forwarded = _forwarded_client_ip(request)
    if forwarded:
        return forwarded

    direct_host = _direct_client_host(request)
    direct_ip = _parse_ip(direct_host)
    if direct_ip is not None:
        return str(direct_ip)
    if direct_host:
        return direct_host
    return "-"


def resolve_child_filename(base_dir: Path, target_name: str) -> tuple[str, Path]:
    """Validate a file name and keep the resolved path inside ``base_dir``."""
    candidate = str(target_name or "").strip()
    if not candidate:
        raise ValueError("Filename is required.")
    if any(sep in candidate for sep in ("/", "\\")):
        raise ValueError("Filename must not include directory components.")

    candidate_path = Path(candidate)
    if candidate_path.is_absolute() or candidate_path.name != candidate or candidate in {".", ".."}:
        raise ValueError("Filename must be a plain file name without directories.")

    safe_name = candidate_path.name
    dest_root = base_dir.resolve()
    dest_path = (dest_root / safe_name).resolve()
    try:
        dest_path.relative_to(dest_root)
    except ValueError as exc:
        raise ValueError("Filename resolves outside the target directory.") from exc
    return safe_name, dest_path


def clean_optional_url(value: Any) -> str | None:
    """Normalise an optional URL env var: strip, drop wildcards, rstrip '/'."""
    raw = str(value or "").strip()
    if not raw or raw == "*":
        return None
    if raw.startswith(("http://", "https://")):
        return raw.rstrip("/")
    return raw


def current_rss_mb() -> float | None:
    """Current RSS in MB (Linux /proc preferred, macOS ru_maxrss fallback)."""
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
        except Exception:
            pass
    try:
        rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            rss_kb = rss_kb / 1024.0
        return rss_kb / 1024.0
    except Exception:
        return None

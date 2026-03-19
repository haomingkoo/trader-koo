"""Shared utility functions for the trader_koo backend.

Extracted to eliminate duplication across routers and services.
"""
from __future__ import annotations

import resource
import sys
from pathlib import Path
from typing import Any

from fastapi import Request


def client_ip(request: Request) -> str:
    """Extract the client IP from a request (X-Forwarded-For aware)."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "-"


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

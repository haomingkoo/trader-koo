"""Framework-native authentication for the complete admin router."""

from __future__ import annotations

import datetime as dt
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import HTTPException, Request, Security
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader

from trader_koo.backend.utils import client_ip as _client_ip

LOG = logging.getLogger(__name__)
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

AuditRecorder = Callable[..., None]


def _env_bool(name: str, default: bool) -> bool:
    value = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AdminAuthConfig:
    api_key: str
    username: str = "admin"
    strict_api_key: bool = True
    development_mode: bool = False
    failure_window_sec: int = 300
    max_failures: int = 20
    block_sec: int = 600

    @classmethod
    def from_env(cls) -> "AdminAuthConfig":
        return cls(
            api_key=os.getenv("TRADER_KOO_API_KEY", ""),
            username=str(os.getenv("TRADER_KOO_ADMIN_USERNAME", "admin") or "admin").strip()
            or "admin",
            strict_api_key=_env_bool("ADMIN_STRICT_API_KEY", True),
            development_mode=_env_bool("TRADER_KOO_DEVELOPMENT_MODE", False),
            failure_window_sec=max(
                30, int(os.getenv("TRADER_KOO_ADMIN_AUTH_WINDOW_SEC", "300"))
            ),
            max_failures=max(
                3, int(os.getenv("TRADER_KOO_ADMIN_AUTH_MAX_FAILS", "20"))
            ),
            block_sec=max(
                30, int(os.getenv("TRADER_KOO_ADMIN_AUTH_BLOCK_SEC", "600"))
            ),
        )


class AdminAuthenticator:
    """Authenticate admin requests behind one router dependency."""

    def __init__(
        self,
        config: AdminAuthConfig,
        *,
        audit_recorder: AuditRecorder | None = None,
    ) -> None:
        self.config = config
        self._audit_recorder = audit_recorder
        self._lock = threading.Lock()
        self._failures: dict[str, dict[str, float]] = {}

    def authenticate(self, request: Request, provided_key: str | None) -> dict[str, str]:
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
        ip_address = _client_ip(request)
        user_agent = request.headers.get("user-agent", "-")
        blocked, retry_after = self._blocked(ip_address, now_ts)
        if blocked:
            raise HTTPException(
                status_code=429,
                detail="Too many unauthorized attempts. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )

        if not self.config.api_key:
            if self.config.strict_api_key or not self.config.development_mode:
                detail = (
                    "Admin API key is not configured on server."
                    if self.config.strict_api_key
                    else "Admin API key required. Set TRADER_KOO_DEVELOPMENT_MODE=1 for local dev."
                )
                raise HTTPException(status_code=503, detail=detail)
            identity = {"username": "local-dev", "mode": "open-admin"}
            request.state.admin_identity = identity
            self._record_audit(
                success=True,
                username="local-dev",
                ip_address=ip_address,
                user_agent=user_agent,
                auth_method="local_dev",
            )
            return identity

        if not secrets.compare_digest(provided_key or "", self.config.api_key):
            blocked_now, retry_after, fail_count = self._record_failure(
                ip_address, now_ts
            )
            LOG.warning(
                "Unauthorized admin request method=%s path=%s client_ip=%s "
                "fail_count=%s blocked=%s",
                request.method,
                request.url.path,
                ip_address,
                fail_count,
                blocked_now,
            )
            self._record_audit(
                success=False,
                username=None,
                ip_address=ip_address,
                user_agent=user_agent,
                auth_method="api_key",
                reason="invalid_api_key",
            )
            if blocked_now:
                raise HTTPException(
                    status_code=429,
                    detail="Too many unauthorized attempts. Try again later.",
                    headers={"Retry-After": str(retry_after)},
                )
            raise HTTPException(status_code=401, detail="Unauthorized")

        self._clear(ip_address)
        identity = {
            "username": self.config.username,
            "mode": "api_key",
            "user_id": self.config.username,
        }
        request.state.admin_identity = identity
        self._record_audit(
            success=True,
            username=self.config.username,
            ip_address=ip_address,
            user_agent=user_agent,
            auth_method="api_key",
        )
        return identity

    def _prune(self, now_ts: float) -> None:
        max_age = max(self.config.failure_window_sec, self.config.block_sec) * 3
        for ip_address, entry in list(self._failures.items()):
            if now_ts - float(entry.get("updated_ts", 0.0)) > max_age:
                self._failures.pop(ip_address, None)

    def _blocked(self, ip_address: str, now_ts: float) -> tuple[bool, int]:
        with self._lock:
            self._prune(now_ts)
            entry = self._failures.get(ip_address)
            if not entry:
                return False, 0
            blocked_until = float(entry.get("blocked_until", 0.0))
            if blocked_until > now_ts:
                return True, max(1, int(blocked_until - now_ts))
            return False, 0

    def _record_failure(
        self, ip_address: str, now_ts: float
    ) -> tuple[bool, int, int]:
        with self._lock:
            self._prune(now_ts)
            entry = self._failures.get(ip_address) or {}
            window_start = float(entry.get("window_start", now_ts))
            if now_ts - window_start > self.config.failure_window_sec:
                window_start = now_ts
                count = 0
            else:
                count = int(entry.get("count", 0))
            count += 1
            blocked = count >= self.config.max_failures
            blocked_until = (
                now_ts + self.config.block_sec
                if blocked
                else float(entry.get("blocked_until", 0.0))
            )
            self._failures[ip_address] = {
                "window_start": window_start,
                "count": float(count),
                "blocked_until": blocked_until,
                "updated_ts": now_ts,
            }
            retry_after = max(1, int(blocked_until - now_ts)) if blocked else 0
            return blocked, retry_after, count

    def _clear(self, ip_address: str) -> None:
        with self._lock:
            self._failures.pop(ip_address, None)

    def _record_audit(self, **payload: Any) -> None:
        if self._audit_recorder is None:
            return
        try:
            self._audit_recorder(**payload)
        except Exception as exc:  # pragma: no cover - logging must not deny access
            LOG.warning("Failed to record admin authentication attempt: %s", exc)


def require_admin(
    request: Request,
    provided_key: str | None = Security(_API_KEY_HEADER),
) -> dict[str, str]:
    """FastAPI dependency applied once to the complete admin router."""
    authenticator = getattr(request.app.state, "admin_authenticator", None)
    if not isinstance(authenticator, AdminAuthenticator):
        raise HTTPException(
            status_code=503,
            detail="Admin authentication is not initialized.",
        )
    return authenticator.authenticate(request, provided_key)


def route_uses_admin_dependency(route: APIRoute) -> bool:
    """Return whether an APIRoute's resolved dependency tree enforces admin auth."""
    pending = [route.dependant]
    while pending:
        dependant = pending.pop()
        if dependant.call is require_admin:
            return True
        pending.extend(dependant.dependencies)
    return False


def admin_route_inventory(app: Any) -> list[dict[str, Any]]:
    """Describe the runtime admin surface from FastAPI's resolved routes."""
    rows: list[dict[str, Any]] = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or not (
            route.path == "/api/admin" or route.path.startswith("/api/admin/")
        ):
            continue
        protected = route_uses_admin_dependency(route)
        for method in sorted(route.methods or set()):
            rows.append(
                {
                    "method": method,
                    "path": route.path,
                    "has_auth": protected,
                    "key": f"{method}:{route.path}",
                }
            )
    return sorted(rows, key=lambda row: str(row["key"]))

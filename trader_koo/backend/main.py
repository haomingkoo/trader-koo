"""trader_koo API — slim app factory.

All endpoint logic lives in ``trader_koo.backend.routers.*``.
This module is responsible for:
  - FastAPI app creation and lifespan
  - Middleware registration (CORS, auth, audit, rate limiting, error sanitisation)
  - Static file serving for v1 frontend (``GET /``) and v2 mount point (``/v2``)
  - Router inclusion
  - Scheduler setup
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import secrets
import sqlite3
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Path bootstrapping (ensure trader_koo package is importable)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Core library imports
# ---------------------------------------------------------------------------

from trader_koo.config import validate_config, ConfigError
from trader_koo.middleware.cors import RestrictiveCORSMiddleware
from trader_koo.middleware.auth import (
    auto_register_admin_endpoints,
    verify_all_admin_endpoints_protected,
    require_admin_auth,
)
from trader_koo.security.logging_filter import install_secret_redaction_filter
from trader_koo.security.error_middleware import ErrorSanitizationMiddleware
from trader_koo.audit import AuditLogger, ensure_audit_schema
from trader_koo.audit.middleware import AuditMiddleware, log_auth_attempt
from trader_koo.audit.export import schedule_daily_export
from trader_koo.email_subscribers import ensure_subscriber_schema
from trader_koo.paper_trades import ensure_paper_trade_schema
from trader_koo.ratelimit.integration import initialize_rate_limiting

# ---------------------------------------------------------------------------
# Service-layer imports
# ---------------------------------------------------------------------------

from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.services.scheduler import create_scheduler
from trader_koo.backend.services.pipeline import (
    reconcile_stale_running_runs,
    queue_post_ingest_resume,
)
from trader_koo.crypto.service import start_crypto_feed, stop_crypto_feed
from trader_koo.crypto.storage import ensure_crypto_schema
from trader_koo.streaming.service import start_equity_feed, stop_equity_feed

# ---------------------------------------------------------------------------
# Router imports
# ---------------------------------------------------------------------------

from trader_koo.backend.routers.system import router as system_router
from trader_koo.backend.routers.dashboard import router as dashboard_router
from trader_koo.backend.routers.report import router as report_router
from trader_koo.backend.routers.opportunities import router as opportunities_router
from trader_koo.backend.routers.paper_trades import router as paper_trades_router
from trader_koo.backend.routers.email import router as email_router
from trader_koo.backend.routers.usage import router as usage_router
from trader_koo.backend.routers.admin import router as admin_router
from trader_koo.backend.routers.crypto import router as crypto_router
from trader_koo.backend.routers.data_sync import router as data_sync_router
from trader_koo.backend.routers.streaming import router as streaming_router

# Usage module helpers needed at startup
from trader_koo.backend.routers.usage import (
    ensure_analytics_schema,
    ensure_feedback_schema,
    prune_analytics_sessions,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_INDEX = (PROJECT_DIR / "frontend" / "index.html").resolve()
DIST_V2 = (PROJECT_DIR / ".." / "dist-v2").resolve()
API_KEY = os.getenv("TRADER_KOO_API_KEY", "")
ADMIN_USER = str(os.getenv("TRADER_KOO_ADMIN_USERNAME", "admin") or "admin").strip() or "admin"
ADMIN_STRICT_API_KEY = str(os.getenv("ADMIN_STRICT_API_KEY", "1")).strip().lower() in {
    "1", "true", "yes", "on",
}
ADMIN_AUTH_WINDOW_SEC = max(30, int(os.getenv("TRADER_KOO_ADMIN_AUTH_WINDOW_SEC", "300")))
ADMIN_AUTH_MAX_FAILS = max(3, int(os.getenv("TRADER_KOO_ADMIN_AUTH_MAX_FAILS", "20")))
ADMIN_AUTH_BLOCK_SEC = max(30, int(os.getenv("TRADER_KOO_ADMIN_AUTH_BLOCK_SEC", "600")))
ADMIN_API_PREFIX = "/api/admin/"

_ALLOWED_ORIGIN = os.getenv("TRADER_KOO_ALLOWED_ORIGIN", "https://trader.kooexperience.com")


def _resolve_log_dir() -> Path:
    requested = Path(os.getenv("TRADER_KOO_LOG_DIR", "/data/logs"))
    try:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    except OSError:
        fallback = (PROJECT_DIR / "data" / "logs").resolve()
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


LOG_DIR = _resolve_log_dir()
API_LOG_PATH = LOG_DIR / "api.log"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG = logging.getLogger("trader_koo.api")
ROOT_LOGGER = logging.getLogger()
_log_level = os.getenv("TRADER_KOO_LOG_LEVEL", "INFO").upper()
_log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
if not ROOT_LOGGER.handlers:
    logging.basicConfig(level=_log_level, format=_log_format)

install_secret_redaction_filter()

if not any(
    isinstance(h, RotatingFileHandler) and Path(getattr(h, "baseFilename", "")) == API_LOG_PATH
    for h in ROOT_LOGGER.handlers
):
    try:
        file_handler = RotatingFileHandler(API_LOG_PATH, maxBytes=10_000_000, backupCount=3)
        file_handler.setFormatter(logging.Formatter(_log_format))
        file_handler.setLevel(getattr(logging, _log_level, logging.INFO))
        ROOT_LOGGER.addHandler(file_handler)
    except Exception as exc:
        LOG.warning("Failed to attach rotating file logger at %s: %s", API_LOG_PATH, exc)

# ---------------------------------------------------------------------------
# In-memory admin auth rate-limit state
# ---------------------------------------------------------------------------

import threading

_ADMIN_AUTH_LOCK = threading.Lock()
_ADMIN_AUTH_STATE: dict[str, dict[str, float]] = {}


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "-"


def _prune_admin_auth_state(now_ts: float) -> None:
    max_age = max(ADMIN_AUTH_WINDOW_SEC, ADMIN_AUTH_BLOCK_SEC) * 3
    stale_keys = [
        ip
        for ip, entry in _ADMIN_AUTH_STATE.items()
        if now_ts - float(entry.get("updated_ts", 0.0)) > max_age
    ]
    for ip in stale_keys:
        _ADMIN_AUTH_STATE.pop(ip, None)


def _admin_auth_blocked(client_ip: str, now_ts: float) -> tuple[bool, int]:
    with _ADMIN_AUTH_LOCK:
        _prune_admin_auth_state(now_ts)
        entry = _ADMIN_AUTH_STATE.get(client_ip)
        if not entry:
            return False, 0
        blocked_until = float(entry.get("blocked_until", 0.0))
        if blocked_until > now_ts:
            return True, max(1, int(blocked_until - now_ts))
        return False, 0


def _admin_auth_record_failure(client_ip: str, now_ts: float) -> tuple[bool, int, int]:
    with _ADMIN_AUTH_LOCK:
        _prune_admin_auth_state(now_ts)
        entry = _ADMIN_AUTH_STATE.get(client_ip) or {}
        window_start = float(entry.get("window_start", now_ts))
        if now_ts - window_start > ADMIN_AUTH_WINDOW_SEC:
            window_start = now_ts
            count = 0
        else:
            count = int(entry.get("count", 0))
        count += 1
        blocked_until = float(entry.get("blocked_until", 0.0))
        blocked = False
        if count >= ADMIN_AUTH_MAX_FAILS:
            blocked = True
            blocked_until = now_ts + ADMIN_AUTH_BLOCK_SEC
        _ADMIN_AUTH_STATE[client_ip] = {
            "window_start": window_start,
            "count": float(count),
            "blocked_until": blocked_until,
            "updated_ts": now_ts,
        }
        retry_after = max(1, int(blocked_until - now_ts)) if blocked else 0
        return blocked, retry_after, count


def _admin_auth_clear(client_ip: str) -> None:
    with _ADMIN_AUTH_LOCK:
        _ADMIN_AUTH_STATE.pop(client_ip, None)


def _get_audit_logger() -> AuditLogger:
    conn = sqlite3.connect(str(DB_PATH))
    return AuditLogger(conn)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

_scheduler = create_scheduler()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Validate configuration at startup
    try:
        config = validate_config()
        LOG.info("Configuration validation passed")
        _app.state.config = config
    except ConfigError as exc:
        LOG.error("Configuration validation failed: %s", str(exc))
        raise

    # Store scheduler in app state so routers can access it
    _app.state.scheduler = _scheduler

    # Register all admin endpoints and verify authentication
    auto_register_admin_endpoints(_app)
    all_protected, unprotected = verify_all_admin_endpoints_protected()
    if not all_protected:
        error_msg = (
            f"FATAL: {len(unprotected)} admin endpoint(s) lack authentication. "
            f"All /api/admin/* endpoints must have @require_admin_auth decorator. "
            f"Unprotected endpoints: {', '.join(unprotected)}"
        )
        LOG.error(error_msg)
        if config.admin_strict_api_key:
            raise RuntimeError(error_msg)
        else:
            LOG.warning(
                "Running in development mode (ADMIN_STRICT_API_KEY=0). "
                "Unprotected admin endpoints are allowed but not recommended."
            )
    else:
        LOG.info("All admin endpoints are properly protected with authentication")

    # Initialize database schemas (skip if DB doesn't exist yet)
    if DB_PATH.exists():
        LOG.info("Database found, initializing schemas...")
        ensure_analytics_schema()
        ensure_feedback_schema()
        prune_analytics_sessions()
        ensure_subscriber_schema(DB_PATH)
        conn = sqlite3.connect(str(DB_PATH))
        try:
            ensure_audit_schema(conn)
            LOG.info("Audit logging schema initialized")
            ensure_paper_trade_schema(conn)
            LOG.info("Paper trade schema initialized")
            ensure_crypto_schema(conn)
            LOG.info("Crypto bars schema initialized")
        finally:
            conn.close()
    else:
        LOG.warning(
            "Database not found at %s - skipping schema initialization (will be created by start.sh)",
            DB_PATH,
        )

    # Initialize rate limiting
    rate_limiter = initialize_rate_limiting(_app)
    LOG.info("Rate limiting initialized")

    # Skip reconciliation and resume if DB doesn't exist
    if DB_PATH.exists():
        reconcile = reconcile_stale_running_runs()
        if reconcile.get("reconciled"):
            LOG.warning(
                "Startup recovered %s stale ingest run(s): %s",
                reconcile.get("reconciled"),
                ",".join(reconcile.get("run_ids", [])),
            )
        resume = queue_post_ingest_resume(_scheduler, source="startup_resume")
        if resume.get("scheduled"):
            LOG.warning(
                "Startup queued post-ingest resume job_id=%s mode=%s reason=%s",
                resume.get("job_id"),
                resume.get("mode"),
                resume.get("reason"),
            )

    # Start real-time crypto feed (Binance WebSocket, daemon thread)
    # Pass DB path so bars are persisted and history is restored on restart
    crypto_db = str(DB_PATH) if DB_PATH.exists() else None
    try:
        start_crypto_feed(db_path_str=crypto_db)
        LOG.info("Crypto feed started (Binance WS for BTC/ETH/SOL/XRP/DOGE)")
    except Exception as exc:
        LOG.warning("Failed to start crypto feed: %s — continuing without it", exc)

    # Start real-time equity feed (Finnhub WebSocket, daemon thread)
    finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
    if finnhub_api_key:
        try:
            start_equity_feed(api_key=finnhub_api_key)
            LOG.info("Equity feed started (Finnhub WS for SPY/QQQ + on-demand)")
        except Exception as exc:
            LOG.warning("Failed to start equity feed: %s — continuing without it", exc)
    else:
        LOG.warning(
            "FINNHUB_API_KEY not set — equity streaming disabled. "
            "Set the env var to enable real-time equity data."
        )

    _scheduler.start()
    LOG.info("Scheduler started -- daily_update: 22:00 UTC Mon-Fri | weekly_yolo: 00:30 UTC Sat")
    LOG.info("Application startup complete - ready to serve requests")

    # Prefetch sentiment data in background so first user request is fast
    def _prefetch_sentiment() -> None:
        try:
            from trader_koo.structure.fear_greed import compute_fear_greed_index
            from trader_koo.backend.services.database import get_conn

            conn = get_conn()
            try:
                compute_fear_greed_index(conn)
                LOG.info("Sentiment prefetch complete")
            finally:
                conn.close()
        except Exception as exc:
            LOG.debug("Sentiment prefetch failed (non-fatal): %s", exc)

    threading.Thread(target=_prefetch_sentiment, daemon=True, name="sentiment-prefetch").start()

    yield
    stop_equity_feed()
    stop_crypto_feed()
    _scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="trader_koo API",
    version="0.2.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
if os.getenv("TRADER_KOO_DOCS_ENABLED", "0") == "1":
    app.docs_url = "/docs"
    app.redoc_url = "/redoc"
    app.openapi_url = "/openapi.json"

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

cors_origins_env = os.getenv("TRADER_KOO_CORS_ORIGINS", "")
if cors_origins_env:
    from trader_koo.config import Config

    temp_config = Config()
    app.add_middleware(
        RestrictiveCORSMiddleware,
        allowed_origins=temp_config.cors_allowed_origins,
        development_mode=temp_config.development_mode,
    )
else:
    LOG.warning(
        "Using legacy TRADER_KOO_ALLOWED_ORIGIN for CORS. "
        "Please migrate to TRADER_KOO_CORS_ORIGINS for restrictive CORS policy."
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_ALLOWED_ORIGIN] if _ALLOWED_ORIGIN != "*" else ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*", "X-API-Key"],
    )

# Error sanitization middleware
app.add_middleware(ErrorSanitizationMiddleware)

# Audit logging middleware
app.add_middleware(AuditMiddleware, db_path=DB_PATH)

# Rate limiting middleware
from trader_koo.ratelimit.middleware import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware)


# ---------------------------------------------------------------------------
# Admin auth middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Require a valid X-API-Key on /api/admin/* routes."""
    path = request.url.path
    if path.startswith(ADMIN_API_PREFIX):
        client_ip = _client_ip(request)
        user_agent = request.headers.get("user-agent", "-")

        blocked, retry_after = _admin_auth_blocked(
            client_ip, dt.datetime.now(dt.timezone.utc).timestamp()
        )
        if blocked:
            LOG.warning(
                "Admin auth throttled method=%s path=%s client_ip=%s retry_after_sec=%s",
                request.method, path, client_ip, retry_after,
            )
            return JSONResponse(
                {"detail": "Too many unauthorized attempts. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        if not API_KEY:
            if ADMIN_STRICT_API_KEY:
                LOG.error(
                    "Admin endpoint denied because TRADER_KOO_API_KEY is not configured (path=%s, client_ip=%s)",
                    path, client_ip,
                )
                return JSONResponse(
                    {"detail": "Admin API key is not configured on server."},
                    status_code=503,
                )
            request.state.admin_identity = {"username": "local-dev", "mode": "open-admin"}
            try:
                audit_logger = _get_audit_logger()
                log_auth_attempt(
                    audit_logger,
                    success=True,
                    username="local-dev",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    auth_method="local_dev",
                )
                audit_logger.conn.close()
            except Exception as exc:
                LOG.warning("Failed to log auth attempt: %s", exc)
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(provided, API_KEY):
            blocked_now, retry_after, fail_count = _admin_auth_record_failure(
                client_ip, dt.datetime.now(dt.timezone.utc).timestamp()
            )
            ua = request.headers.get("user-agent", "-")
            referer = request.headers.get("referer", "-")
            LOG.warning(
                "Unauthorized request blocked method=%s path=%s client_ip=%s "
                "fail_count=%s blocked=%s user_agent=%s referer=%s",
                request.method, path, client_ip, fail_count, blocked_now, ua, referer,
            )
            try:
                audit_logger = _get_audit_logger()
                log_auth_attempt(
                    audit_logger,
                    success=False,
                    username=None,
                    ip_address=client_ip,
                    user_agent=ua,
                    auth_method="api_key",
                    reason="invalid_api_key",
                )
                audit_logger.conn.close()
            except Exception as exc:
                LOG.warning("Failed to log auth attempt: %s", exc)
            if blocked_now:
                return JSONResponse(
                    {"detail": "Too many unauthorized attempts. Try again later."},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

        _admin_auth_clear(client_ip)
        username = ADMIN_USER or "admin"
        request.state.admin_identity = {"username": username, "mode": "api_key", "user_id": username}
        try:
            audit_logger = _get_audit_logger()
            log_auth_attempt(
                audit_logger,
                success=True,
                username=username,
                ip_address=client_ip,
                user_agent=user_agent,
                auth_method="api_key",
            )
            audit_logger.conn.close()
        except Exception as exc:
            LOG.warning("Failed to log auth attempt: %s", exc)

    return await call_next(request)


# ---------------------------------------------------------------------------
# Root: serve v2 React app at / (promoted from /v2)
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> Any:
    """Serve v2 React dashboard at root. v1 is retired."""
    v2_index = DIST_V2 / "index.html" if DIST_V2.exists() else None
    if v2_index and v2_index.is_file():
        return FileResponse(str(v2_index), headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    return {"ok": True, "message": "trader_koo API is up", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

app.include_router(system_router)
app.include_router(dashboard_router)
app.include_router(report_router)
app.include_router(opportunities_router)
app.include_router(paper_trades_router)
app.include_router(email_router)
app.include_router(usage_router)
app.include_router(admin_router)
app.include_router(data_sync_router)
app.include_router(crypto_router)
app.include_router(streaming_router)

# ---------------------------------------------------------------------------
# v2 React frontend (served from dist-v2/)
# ---------------------------------------------------------------------------

# SPA routes at root level (v2 promoted to /)
_SPA_ROUTES = {
    "report", "vix", "earnings", "chart", "crypto",
    "opportunities", "paper-trades", "markets",
}

# Ensure .js files are served with correct MIME type (some Linux systems lack this)
import mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

if DIST_V2.exists() and DIST_V2.is_dir():
    _root_v2_index = DIST_V2 / "index.html"
    _root_v2_assets = DIST_V2 / "assets"
    _root_shell_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    if _root_v2_index.is_file():
        # Only intercept KNOWN SPA routes at root level.
        # Do NOT use a wildcard catch-all — it breaks /v2/assets, /api, etc.
        for _route in _SPA_ROUTES:
            def _make_handler(_r: str = _route):
                def _handler() -> Any:
                    return FileResponse(str(_root_v2_index), headers=_root_shell_headers)
                _handler.__name__ = f"spa_{_r.replace('-', '_')}"
                return _handler
            app.get(f"/{_route}", include_in_schema=False)(_make_handler())
            app.get(f"/{_route}/{{rest:path}}", include_in_schema=False)(_make_handler())


# Legacy /v2 mount (backward compatibility)
    _v2_index = DIST_V2 / "index.html"
    _v2_assets = DIST_V2 / "assets"
    _v2_shell_headers = {
        # Never let the SPA shell stick around across deploys: old index.html files
        # point at hashed lazy-route chunks that disappear on the next build.
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    # Only mount hashed assets when the build output is fully present.
    if _v2_assets.is_dir():
        app.mount("/v2/assets", StaticFiles(directory=str(_v2_assets)), name="v2-assets")

    if _v2_index.is_file():
        @app.get("/v2", include_in_schema=False)
        @app.get("/v2/", include_in_schema=False)
        def v2_index() -> Any:
            return FileResponse(str(_v2_index), headers=_v2_shell_headers)

        # Catch-all for SPA routing — any /v2/* path that isn't an asset serves index.html
        @app.get("/v2/{rest_of_path:path}", include_in_schema=False)
        def v2_spa_fallback(rest_of_path: str = "") -> Any:
            # Serve static files if they exist (favicon, etc.)
            static_file = DIST_V2 / rest_of_path
            if rest_of_path and static_file.is_file():
                return FileResponse(str(static_file))
            return FileResponse(str(_v2_index), headers=_v2_shell_headers)

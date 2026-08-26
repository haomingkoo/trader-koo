"""trader_koo API — slim app factory.

All endpoint logic lives in ``trader_koo.backend.routers.*``.
This module is responsible for:
  - FastAPI app creation and lifespan
  - Middleware registration (CORS, auth, audit, rate limiting, error sanitisation)
  - Static file serving for React frontend at ``/``
  - Router inclusion
  - Scheduler setup
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import sqlite3
import sys
import threading
import secrets
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator
from trader_koo.security.logging_filter import install_secret_redaction_filter
from trader_koo.security.error_middleware import ErrorSanitizationMiddleware
from trader_koo.audit import AuditLogger, ensure_audit_schema
from trader_koo.audit.middleware import AuditMiddleware, log_auth_attempt
from trader_koo.email_subscribers import ensure_subscriber_schema
from trader_koo.paper_trades import ensure_paper_trade_schema
from trader_koo.ratelimit.integration import initialize_rate_limiting

# ---------------------------------------------------------------------------
# Service-layer imports
# ---------------------------------------------------------------------------

from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.frontend_routes import SPA_ROUTE_PATHS
from trader_koo.backend.services.scheduler import create_scheduler
from trader_koo.backend.services.maintenance import (
    acquire_lease,
    writers_blocked,
    MaintenanceError,
    MaintenanceInterlockMiddleware,
)
from trader_koo.backend.services.pipeline import (
    AUTO_RESUME_INTERRUPTED_PIPELINE,
    determine_resume_mode,
    ensure_pipeline_runs_schema,
    finish_pipeline_run,
    read_interrupted_pipeline_run,
    reconcile_stale_running_runs,
    queue_post_ingest_resume,
    queue_reserved_pipeline_run,
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
from trader_koo.backend.routers.options import router as options_router
from trader_koo.backend.routers.paper_trades import router as paper_trades_router
from trader_koo.backend.routers.email import router as email_router
from trader_koo.backend.routers.usage import router as usage_router
from trader_koo.backend.routers.admin import router as admin_router
from trader_koo.backend.routers.crypto import router as crypto_router
from trader_koo.backend.routers.streaming import router as streaming_router
from trader_koo.backend.routers.alerts import router as alerts_router
try:
    from trader_koo.hyperliquid.routes import router as hyperliquid_router
except ImportError:
    hyperliquid_router = None  # SDK not installed

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
DIST_DIR = (PROJECT_DIR / ".." / "dist-v2").resolve()
_ALLOWED_ORIGIN = os.getenv("TRADER_KOO_ALLOWED_ORIGIN", "https://trader.kooexperience.com")


class ImmutableStaticFiles(StaticFiles):
    """Serve content-hashed frontend assets with long-lived browser caching."""

    async def get_response(self, path: str, scope: dict[str, Any]) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


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

# Suppress HTTP client loggers — they log full URLs including API tokens
for _http_logger_name in ("httpx", "httpcore", "urllib3", "requests"):
    logging.getLogger(_http_logger_name).setLevel(logging.WARNING)

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

def _record_admin_auth_attempt(**payload: Any) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        log_auth_attempt(AuditLogger(conn), **payload)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

_scheduler = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _scheduler
    boot_id = secrets.token_hex(16)
    _app.state.boot_id = boot_id
    try:
        maintenance_boot = writers_blocked(DB_PATH)
    except MaintenanceError as exc:
        maintenance_boot = True
        _app.state.maintenance_state_error = exc.code
    if maintenance_boot:
        _app.state.maintenance_mode = True
        LOG.warning("Maintenance-only boot: normal routes and every writer remain stopped")
        yield
        return
    writer_lease = acquire_lease(DB_PATH, exclusive=False)
    _app.state.writer_lease = writer_lease
    os.environ["TRADER_KOO_WRITER_LEASE_FD"] = str(writer_lease.fileno())
    try:
        intent_won_race = writers_blocked(DB_PATH)
    except MaintenanceError as exc:
        intent_won_race = True
        _app.state.maintenance_state_error = exc.code
    if intent_won_race:
        writer_lease.close()
        os.environ.pop("TRADER_KOO_WRITER_LEASE_FD", None)
        _app.state.maintenance_mode = True
        LOG.warning("Maintenance intent won startup race; entering maintenance-only boot")
        yield
        return
    _app.state.maintenance_mode = False
    _scheduler = create_scheduler()
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
            try:
                from trader_koo.report.calibration_pulse import ensure_calibration_schema
                ensure_calibration_schema(conn)
                LOG.info("Calibration state schema initialized")
            except Exception as exc:
                LOG.warning("Calibration state schema init failed (non-fatal): %s", exc)
            ensure_crypto_schema(conn)
            LOG.info("Crypto bars schema initialized")
            ensure_pipeline_runs_schema(conn)
            LOG.info("Pipeline runs schema initialized")
            from trader_koo.notifications.market_monitor import ensure_polymarket_schema
            ensure_polymarket_schema(conn)
            LOG.info("Polymarket snapshots schema initialized")
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

        reserved = queue_reserved_pipeline_run(_scheduler)
        if reserved.get("scheduled"):
            LOG.warning(
                "Startup requeued reserved pipeline run_id=%s job_id=%s mode=%s",
                reserved.get("run_id"),
                reserved.get("job_id"),
                reserved.get("mode"),
            )

        # DB-based pipeline resume: check for interrupted pipeline_runs
        interrupted = read_interrupted_pipeline_run()
        if interrupted:
            interrupted_id = str(interrupted.get("run_id") or "")
            interrupted_mode = str(interrupted.get("mode") or "")
            interrupted_stage = str(interrupted.get("stage") or "")
            LOG.warning(
                "Startup detected interrupted pipeline run_id=%s mode=%s stage=%s",
                interrupted_id,
                interrupted_mode,
                interrupted_stage,
            )
            # Mark the interrupted run as failed
            try:
                finish_pipeline_run(
                    interrupted_id,
                    status="interrupted",
                    error_message="killed_by_restart",
                )
            except Exception as exc:
                LOG.warning("Failed to mark interrupted pipeline run: %s", exc)

            # Determine resume mode and schedule it
            resume_mode = determine_resume_mode(interrupted)
            if resume_mode and not AUTO_RESUME_INTERRUPTED_PIPELINE:
                LOG.warning(
                    "Startup auto-resume disabled; not queueing interrupted pipeline "
                    "run_id=%s mode=%s stage=%s resume_mode=%s",
                    interrupted_id,
                    interrupted_mode,
                    interrupted_stage,
                    resume_mode,
                )
            elif resume_mode:
                from trader_koo.backend.services.scheduler import _run_daily_update
                resume_source = f"startup_resume:interrupted:{interrupted_id}"
                job_id = f"resume_pipeline_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}"
                _scheduler.add_job(
                    _run_daily_update,
                    trigger="date",
                    run_date=dt.datetime.now(dt.timezone.utc),
                    id=job_id,
                    kwargs={"mode": resume_mode, "source": resume_source},
                )
                LOG.warning(
                    "Startup queued pipeline resume job_id=%s mode=%s from interrupted run_id=%s stage=%s",
                    job_id,
                    resume_mode,
                    interrupted_id,
                    interrupted_stage,
                )
            else:
                LOG.info(
                    "Startup: interrupted pipeline run_id=%s does not need resume (stage=%s)",
                    interrupted_id,
                    interrupted_stage,
                )

        # Legacy log-based post-ingest resume (kept for backward compat)
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
    LOG.info(
        "Scheduler started -- daily_update: 22:00 UTC Mon-Fri | weekly_yolo: 00:30 UTC Sat"
        " | weekly_backup: 02:00 UTC Sat | morning_summary: 00:00 UTC Mon-Fri (if TELEGRAM_BOT_TOKEN set)"
    )
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

    # Start Telegram price alert engine (optional — requires credentials)
    # Uses Finnhub REST polling (not WebSocket) to preserve WS slots for dashboard
    _alert_task: asyncio.Task | None = None  # type: ignore[type-arg]
    _bot_task: asyncio.Task | None = None  # type: ignore[type-arg]
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if telegram_token and telegram_chat:
        report_dir = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))

        # Alert engine (outbound price alerts)
        try:
            from trader_koo.notifications.alert_engine import AlertEngine

            alert_engine = AlertEngine(
                db_path=DB_PATH,
                report_dir=report_dir,
                finnhub_api_key=finnhub_api_key,
            )
            _app.state.alert_engine = alert_engine
            _alert_task = asyncio.create_task(alert_engine.run())
            LOG.info("Telegram alert engine started (REST polling, top 10 setups)")
        except Exception as exc:
            LOG.warning("Failed to start Telegram alert engine: %s — continuing without it", exc)
            alert_engine = None  # type: ignore[assignment]

        # Bot command handler (inbound commands from Telegram)
        try:
            from trader_koo.notifications.bot_commands import TelegramCommandHandler

            bot_handler = TelegramCommandHandler(
                bot_token=telegram_token,
                chat_id=telegram_chat,
                db_path=DB_PATH,
                report_dir=report_dir,
                finnhub_api_key=finnhub_api_key,
                alert_engine=getattr(_app.state, "alert_engine", None),
            )
            _app.state.bot_handler = bot_handler
            _bot_task = asyncio.create_task(bot_handler.run())
            LOG.info("Telegram bot command handler started (polling getUpdates)")
        except Exception as exc:
            LOG.warning("Failed to start Telegram bot handler: %s — continuing without it", exc)
    else:
        LOG.info(
            "Telegram credentials not set — alert engine and bot handler disabled. "
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable."
        )

    yield

    # Shutdown Telegram tasks
    if _alert_task is not None:
        try:
            alert_engine = getattr(_app.state, "alert_engine", None)
            if alert_engine is not None:
                alert_engine.stop()
            _alert_task.cancel()
        except Exception as exc:
            LOG.debug("Alert engine shutdown: %s", exc)
    if _bot_task is not None:
        try:
            bot_handler = getattr(_app.state, "bot_handler", None)
            if bot_handler is not None:
                bot_handler.stop()
            _bot_task.cancel()
        except Exception as exc:
            LOG.debug("Bot handler shutdown: %s", exc)

    cancelled_tasks = [task for task in (_alert_task, _bot_task) if task is not None]
    if cancelled_tasks:
        await asyncio.gather(*cancelled_tasks, return_exceptions=True)

    stop_equity_feed()
    stop_crypto_feed()
    # Do not release the whole-process writer lease while a scheduler job may
    # still be using the database.  Platform force-kill remains fail-closed:
    # inherited writer children retain the lease until they exit.
    _scheduler.shutdown(wait=True)
    # Intentionally retain the shared lease until OS process exit. Admin
    # endpoints can own daemon writer threads that are not part of APScheduler;
    # closing here would create a gap before those threads are terminated.
    # Maintenance therefore requires a real process restart, not an in-process
    # lifespan recycle.


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
app.state.admin_authenticator = AdminAuthenticator(
    AdminAuthConfig.from_env(),
    audit_recorder=_record_admin_auth_attempt,
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
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "Accept"],
    )

# Error sanitization middleware
app.add_middleware(ErrorSanitizationMiddleware)

# Audit logging middleware
app.add_middleware(AuditMiddleware, db_path=DB_PATH)

# Rate limiting middleware
from trader_koo.ratelimit.middleware import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware)
app.add_middleware(MaintenanceInterlockMiddleware, db_path=DB_PATH)


# ---------------------------------------------------------------------------
# Root: serve React app at /
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> Any:
    """Serve React dashboard at root."""
    index = DIST_DIR / "index.html" if DIST_DIR.exists() else None
    if index and index.is_file():
        return FileResponse(str(index), headers={
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
app.include_router(options_router)
app.include_router(paper_trades_router)
app.include_router(email_router)
app.include_router(usage_router)
app.include_router(admin_router)
app.include_router(crypto_router)
app.include_router(streaming_router)
app.include_router(alerts_router)
if hyperliquid_router is not None:
    app.include_router(hyperliquid_router)

# ---------------------------------------------------------------------------
# Logos static mount — serve cached company logos from /data/logos
# ---------------------------------------------------------------------------

_LOGOS_DIR = Path(os.getenv("TRADER_KOO_LOGOS_DIR", "/data/logos"))
_LOGOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/logos", StaticFiles(directory=str(_LOGOS_DIR)), name="logos")
LOG.info("Logos static mount: %s", _LOGOS_DIR)

# ---------------------------------------------------------------------------
# React frontend (served from dist-v2/)
# ---------------------------------------------------------------------------

# SPA routes at root level
_SPA_ROUTES = SPA_ROUTE_PATHS

# Ensure .js files are served with correct MIME type (some Linux systems lack this)
import mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")

LOG.info("DIST_DIR path: %s exists=%s", DIST_DIR, DIST_DIR.exists())
if DIST_DIR.exists() and DIST_DIR.is_dir():
    _root_index = DIST_DIR / "index.html"
    _root_assets = DIST_DIR / "assets"
    LOG.info("DIST_DIR assets dir: %s exists=%s", _root_assets, _root_assets.is_dir())
    _root_shell_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    if _root_assets.is_dir():
        app.mount("/assets", ImmutableStaticFiles(directory=str(_root_assets)), name="root-assets")

    if _root_index.is_file():
        # Only intercept KNOWN SPA routes at root level.
        # Do NOT use a wildcard catch-all — it breaks /assets, /api, etc.
        def _spa_shell() -> Any:
            return FileResponse(str(_root_index), headers=_root_shell_headers)

        for _route in _SPA_ROUTES:
            app.get(f"/{_route}", include_in_schema=False)(_spa_shell)
            app.get(f"/{_route}/{{rest:path}}", include_in_schema=False)(_spa_shell)


    # Serve favicon and other root-level static files from dist-v2/
    for _static_file in ("favicon.svg", "favicon.ico", "robots.txt", "sitemap.xml"):
        _static_path = DIST_DIR / _static_file
        if _static_path.is_file():
            app.add_api_route(
                f"/{_static_file}",
                (lambda _p=str(_static_path): FileResponse(_p)),
                include_in_schema=False,
            )

    # Legacy /v2 routes removed — React app is now served at root.

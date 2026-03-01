from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import resource
import secrets
import smtplib
import sqlite3
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from email.message import EmailMessage
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trader_koo.data.schema import ensure_ohlcv_schema
from trader_koo.cv.compare import HybridCVCompareConfig, compare_hybrid_vs_cv
from trader_koo.cv.proxy_patterns import CVProxyConfig, detect_cv_proxy_patterns
from trader_koo.features.candle_patterns import CandlePatternConfig, detect_candlestick_patterns
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.catalyst_data import (
    build_earnings_calendar_payload,
    get_ticker_earnings_markers,
)
from trader_koo.email_chart_preview import (
    build_email_chart_preview_png,
    verify_chart_preview_signature,
)
from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)
from trader_koo.scripts.generate_daily_report import (
    _describe_setup as _report_describe_setup,
    _score_setup_from_confluence as _report_score_setup_from_confluence,
    _yolo_age_factor as _report_yolo_age_factor,
    _yolo_recency_label as _report_yolo_recency_label,
)
from trader_koo.structure.gaps import GapConfig, detect_gaps, select_gaps_for_display
from trader_koo.structure.hybrid_patterns import HybridPatternConfig, score_hybrid_patterns
from trader_koo.structure.levels import LevelConfig, add_fallback_levels, build_levels_from_pivots, select_target_levels
from trader_koo.structure.patterns import PatternConfig, detect_patterns
from trader_koo.structure.trendlines import TrendlineConfig, detect_trendlines


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PRIMARY = (PROJECT_DIR / "data" / "trader_koo.db").resolve()
DB_PATH = Path(os.getenv("TRADER_KOO_DB_PATH", str(DEFAULT_DB_PRIMARY)))
FRONTEND_INDEX = (PROJECT_DIR / "frontend" / "index.html").resolve()
API_KEY = os.getenv("TRADER_KOO_API_KEY", "")  # empty = auth disabled (local dev)
ADMIN_USER = str(os.getenv("TRADER_KOO_ADMIN_USERNAME", "admin") or "admin").strip() or "admin"
PROCESS_START_UTC = dt.datetime.now(dt.timezone.utc)


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _clean_optional_url(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw == "*":
        return None
    if raw.startswith(("http://", "https://")):
        return raw.rstrip("/")
    return raw


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
RUN_LOG_PATH = LOG_DIR / "cron_daily.log"
API_LOG_PATH = LOG_DIR / "api.log"
LOG_PATHS: dict[str, Path] = {
    "cron": RUN_LOG_PATH,
    "update_market_db": LOG_DIR / "update_market_db.log",
    "yolo": LOG_DIR / "yolo_patterns.log",
    "api": API_LOG_PATH,
}
STATUS_CACHE_TTL_SEC = max(0, int(os.getenv("TRADER_KOO_STATUS_CACHE_SEC", "20")))
PIPELINE_STALE_SEC = max(60, int(os.getenv("TRADER_KOO_PIPELINE_STALE_SEC", "1200")))
INGEST_RUNNING_STALE_MIN = max(10, int(os.getenv("TRADER_KOO_INGEST_RUNNING_STALE_MIN", "75")))
AUTO_RESUME_POST_INGEST = _as_bool(os.getenv("TRADER_KOO_AUTO_RESUME_POST_INGEST", "1"))
AUTO_RESUME_MAX_AGE_HOURS = max(1, int(os.getenv("TRADER_KOO_AUTO_RESUME_MAX_AGE_HOURS", "18")))
AUTO_RESUME_MAX_RETRIES = max(0, int(os.getenv("TRADER_KOO_AUTO_RESUME_MAX_RETRIES", "2")))
ADMIN_STRICT_API_KEY = _as_bool(os.getenv("TRADER_KOO_ADMIN_STRICT_KEY", "1"))
ADMIN_AUTH_WINDOW_SEC = max(30, int(os.getenv("TRADER_KOO_ADMIN_AUTH_WINDOW_SEC", "300")))
ADMIN_AUTH_MAX_FAILS = max(3, int(os.getenv("TRADER_KOO_ADMIN_AUTH_MAX_FAILS", "20")))
ADMIN_AUTH_BLOCK_SEC = max(30, int(os.getenv("TRADER_KOO_ADMIN_AUTH_BLOCK_SEC", "600")))
EXPOSE_STATUS_INTERNAL = _as_bool(os.getenv("TRADER_KOO_EXPOSE_STATUS_INTERNAL", "0"))
CONTROL_CENTER_CONTRACT_VERSION = "2026-03-01"
STATUS_BASE_URL = _clean_optional_url(os.getenv("TRADER_KOO_BASE_URL"))
STATUS_APP_URL = _clean_optional_url(os.getenv("TRADER_KOO_APP_URL")) or _clean_optional_url(os.getenv("TRADER_KOO_ALLOWED_ORIGIN"))
STATUS_REPO_URL = _clean_optional_url(os.getenv("TRADER_KOO_REPO_URL"))
_STATUS_CACHE_LOCK = threading.Lock()
_STATUS_CACHE_AT: dt.datetime | None = None
_STATUS_CACHE_PAYLOAD: dict[str, Any] | None = None
_ADMIN_AUTH_LOCK = threading.Lock()
_ADMIN_AUTH_STATE: dict[str, dict[str, float]] = {}
_MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")
try:
    MARKET_TZ = ZoneInfo(_MARKET_TZ_NAME)
except Exception:
    MARKET_TZ = dt.timezone.utc
MARKET_CLOSE_HOUR = min(23, max(0, int(os.getenv("TRADER_KOO_MARKET_CLOSE_HOUR", "16"))))
ANALYTICS_ENABLED = _as_bool(os.getenv("TRADER_KOO_ANALYTICS_ENABLED", "1"))
ANALYTICS_MAX_SESSION_AGE_DAYS = max(7, int(os.getenv("TRADER_KOO_ANALYTICS_MAX_SESSION_AGE_DAYS", "180")))
LOG = logging.getLogger("trader_koo.api")
ROOT_LOGGER = logging.getLogger()
log_level = os.getenv("TRADER_KOO_LOG_LEVEL", "INFO").upper()
log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
if not ROOT_LOGGER.handlers:
    logging.basicConfig(level=log_level, format=log_format)
if not any(
    isinstance(h, RotatingFileHandler) and Path(getattr(h, "baseFilename", "")) == API_LOG_PATH
    for h in ROOT_LOGGER.handlers
):
    try:
        file_handler = RotatingFileHandler(API_LOG_PATH, maxBytes=10_000_000, backupCount=3)
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        ROOT_LOGGER.addHandler(file_handler)
    except Exception as exc:
        logging.getLogger("trader_koo.api").warning(
            "Failed to attach rotating file logger at %s: %s", API_LOG_PATH, exc
        )

FEATURE_CFG = FeatureConfig()
LEVEL_CFG = LevelConfig()
GAP_CFG = GapConfig()
TREND_CFG = TrendlineConfig()
PATTERN_CFG = PatternConfig()
CANDLE_CFG = CandlePatternConfig()
HYBRID_PATTERN_CFG = HybridPatternConfig()
CV_PROXY_CFG = CVProxyConfig()
HYBRID_CV_CMP_CFG = HybridCVCompareConfig()

SCRIPTS_DIR = PROJECT_DIR / "scripts"
REPORT_DIR = Path(os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))


def _current_rss_mb() -> float | None:
    # Linux /proc gives current RSS; fallback to ru_maxrss when /proc is unavailable.
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


def _max_rss_mb() -> float | None:
    try:
        rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            rss_kb = rss_kb / 1024.0
        return rss_kb / 1024.0
    except Exception:
        return None


def _fmt_mb(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def _append_run_log(tag: str, message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    line = f"{stamp} [{tag}] {message}\n"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        LOG.warning("Failed to append run log %s: %s", RUN_LOG_PATH, exc)


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
    # Bound in-memory state for failed auth tracking.
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


def _normalize_update_mode(mode: str | None) -> str | None:
    value = str(mode or "full").strip().lower()
    aliases = {
        "full": "full",
        "all": "full",
        "yolo": "yolo",
        "yolo_report": "yolo",
        "yolo+report": "yolo",
        "report": "report",
        "report_only": "report",
        "email": "report",
    }
    return aliases.get(value)


def _run_daily_update(mode: str = "full", source: str = "scheduler") -> None:
    mode_norm = _normalize_update_mode(mode) or "full"
    script = SCRIPTS_DIR / "daily_update.sh"
    started = dt.datetime.now(dt.timezone.utc)
    rss_before = _current_rss_mb()
    _append_run_log(
        "SCHED",
        f"daily_update invoked source={source} mode={mode_norm} rss_before_mb={_fmt_mb(rss_before)}",
    )
    LOG.info(
        "Scheduler: starting daily_update.sh (source=%s, mode=%s, rss_before_mb=%s, run_log=%s)",
        source,
        mode_norm,
        _fmt_mb(rss_before),
        RUN_LOG_PATH,
    )
    env = os.environ.copy()
    env["TRADER_KOO_UPDATE_MODE"] = mode_norm
    result = subprocess.run(["bash", str(script)], capture_output=False, env=env)
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    rss_after = _current_rss_mb()
    delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    if result.returncode == 0:
        _append_run_log(
            "SCHED",
            (
                f"daily_update completed source={source} mode={mode_norm} rc={result.returncode} "
                f"sec={elapsed:.1f} rss_after_mb={_fmt_mb(rss_after)} rss_delta_mb={_fmt_mb(delta)}"
            ),
        )
        LOG.info(
            (
                "Scheduler: daily_update.sh completed OK "
                "(source=%s, mode=%s, rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s)"
            ),
            source,
            mode_norm,
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
        )
    else:
        _append_run_log(
            "SCHED",
            (
                f"daily_update failed source={source} mode={mode_norm} rc={result.returncode} "
                f"sec={elapsed:.1f} rss_after_mb={_fmt_mb(rss_after)} rss_delta_mb={_fmt_mb(delta)}"
            ),
        )
        LOG.error(
            (
                "Scheduler: daily_update.sh failed "
                "(source=%s, mode=%s, rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s, run_log=%s)"
            ),
            source,
            mode_norm,
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
            RUN_LOG_PATH,
        )
        tail = _tail_text_file(RUN_LOG_PATH, lines=25, max_bytes=100_000)
        if tail:
            LOG.error("Scheduler: recent cron tail:\n%s", "\n".join(tail))


def _run_weekly_yolo() -> None:
    """Saturday job: run YOLO weekly pass + regenerate report."""
    script_yolo = SCRIPTS_DIR / "run_yolo_patterns.py"
    script_report = SCRIPTS_DIR / "generate_daily_report.py"
    run_log = RUN_LOG_PATH
    report_dir = REPORT_DIR

    started = dt.datetime.now(dt.timezone.utc)
    rss_before = _current_rss_mb()
    LOG.info(
        "Scheduler: starting weekly YOLO pass (rss_before_mb=%s, run_log=%s)",
        _fmt_mb(rss_before),
        run_log,
    )
    _append_run_log("WEEKLY", "Starting weekly YOLO pass")
    with run_log.open("a", encoding="utf-8") as log_file:
        result = subprocess.run(
            [
                sys.executable, str(script_yolo),
                "--db-path", str(DB_PATH),
                "--timeframe", "weekly",
                "--weekly-lookback-days", "730",
                "--sleep", "0.05",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    rss_after = _current_rss_mb()
    delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    if result.returncode == 0:
        LOG.info(
            "Scheduler: weekly YOLO completed OK (rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s)",
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
        )
        _append_run_log("WEEKLY", "Weekly YOLO completed OK")
    else:
        LOG.error(
            "Scheduler: weekly YOLO failed (rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s, run_log=%s)",
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
            run_log,
        )
        _append_run_log("WEEKLY", f"Weekly YOLO failed rc={result.returncode}")

    # Regenerate report so weekly patterns appear
    LOG.info("Scheduler: regenerating report after weekly YOLO")
    _append_run_log("WEEKLY", "Report regeneration starting")
    with run_log.open("a", encoding="utf-8") as log_file:
        report_result = subprocess.run(
            [
                sys.executable, str(script_report),
                "--db-path", str(DB_PATH),
                "--out-dir", str(report_dir),
                "--run-log", str(run_log),
                "--tail-lines", "120",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if report_result.returncode == 0:
        _append_run_log("WEEKLY", "Report regeneration completed")
    else:
        LOG.error("Scheduler: report regeneration after weekly YOLO failed (rc=%d)", report_result.returncode)
        _append_run_log("WEEKLY", f"Report regeneration failed rc={report_result.returncode}")


_scheduler = BackgroundScheduler(
    timezone="UTC",
    job_defaults={
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 3600,
    },
)
_scheduler.add_job(
    _run_daily_update,
    CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone="UTC"),
    id="daily_update",
    replace_existing=True,
)
_scheduler.add_job(
    _run_weekly_yolo,
    CronTrigger(hour=0, minute=30, day_of_week="sat", timezone="UTC"),
    id="weekly_yolo",
    replace_existing=True,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_analytics_schema()
    prune_analytics_sessions()
    reconcile = _reconcile_stale_running_runs()
    if reconcile.get("reconciled"):
        LOG.warning(
            "Startup recovered %s stale ingest run(s): %s",
            reconcile.get("reconciled"),
            ",".join(reconcile.get("run_ids", [])),
        )
    _scheduler.start()
    LOG.info("Scheduler started — daily_update: 22:00 UTC Mon–Fri | weekly_yolo: 00:30 UTC Sat")
    resume = _queue_post_ingest_resume(source="startup_resume")
    if resume.get("scheduled"):
        LOG.warning(
            "Startup queued post-ingest resume job_id=%s mode=%s reason=%s",
            resume.get("job_id"),
            resume.get("mode"),
            resume.get("reason"),
        )
    yield
    _scheduler.shutdown(wait=False)


_ALLOWED_ORIGIN = os.getenv("TRADER_KOO_ALLOWED_ORIGIN", "*")
ADMIN_API_PREFIX = "/api/admin/"

app = FastAPI(
    title="trader_koo API",
    version="0.2.0",
    docs_url=None,   # disable /docs in production; set TRADER_KOO_DOCS_ENABLED=1 locally
    redoc_url=None,
    lifespan=lifespan,
)
if os.getenv("TRADER_KOO_DOCS_ENABLED", "0") == "1":
    app.docs_url = "/docs"
    app.redoc_url = "/redoc"
    app.openapi_url = "/openapi.json"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_ALLOWED_ORIGIN] if _ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Require a valid X-API-Key on /api/admin/* routes."""
    path = request.url.path
    if path.startswith(ADMIN_API_PREFIX):
        client_ip = _client_ip(request)
        blocked, retry_after = _admin_auth_blocked(client_ip, dt.datetime.now(dt.timezone.utc).timestamp())
        if blocked:
            LOG.warning(
                "Admin auth throttled method=%s path=%s client_ip=%s retry_after_sec=%s",
                request.method,
                path,
                client_ip,
                retry_after,
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
                    path,
                    client_ip,
                )
                return JSONResponse(
                    {"detail": "Admin API key is not configured on server."},
                    status_code=503,
                )
            request.state.admin_identity = {"username": "local-dev", "mode": "open-admin"}
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(provided, API_KEY):
            blocked_now, retry_after, fail_count = _admin_auth_record_failure(
                client_ip, dt.datetime.now(dt.timezone.utc).timestamp()
            )
            ua = request.headers.get("user-agent", "-")
            referer = request.headers.get("referer", "-")
            LOG.warning(
                "Unauthorized request blocked method=%s path=%s client_ip=%s fail_count=%s blocked=%s user_agent=%s referer=%s",
                request.method,
                path,
                client_ip,
                fail_count,
                blocked_now,
                ua,
                referer,
            )
            if blocked_now:
                return JSONResponse(
                    {"detail": "Too many unauthorized attempts. Try again later."},
                    status_code=429,
                        headers={"Retry-After": str(retry_after)},
                    )
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        _admin_auth_clear(client_ip)
        request.state.admin_identity = {"username": ADMIN_USER or "admin", "mode": "api_key"}
    return await call_next(request)


@app.get("/", include_in_schema=False)
def root() -> Any:
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    return {"ok": True, "message": "trader_koo API is up", "docs": "/docs"}


def parse_iso_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_log_line_ts_utc(line: str | None) -> dt.datetime | None:
    """Parse a UTC timestamp from a cron log line prefix."""
    if not line:
        return None
    text = str(line).strip()
    if not text:
        return None

    # ISO-style prefixes, e.g.:
    # 2026-02-25T20:19:18+0000 [YOLO] ...
    # 2026-02-25T20:19:18Z [YOLO] ...
    m_iso = re.match(r"^(\d{4}-\d{2}-\d{2}T[0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2}))\b", text)
    if m_iso:
        ts = m_iso.group(1)
        if re.match(r".*[+-]\d{4}$", ts):
            ts = f"{ts[:-5]}{ts[-5:-2]}:{ts[-2:]}"
        return parse_iso_utc(ts)

    # Python logging prefixes, e.g.:
    # 2026-02-25 21:05:19,307 | INFO | ...
    m_py = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,(\d{1,6}))?\b", text)
    if m_py:
        base = m_py.group(1)
        frac = (m_py.group(2) or "0")[:6]
        micro = int(frac.ljust(6, "0"))
        try:
            parsed = dt.datetime.strptime(base, "%Y-%m-%d %H:%M:%S").replace(
                microsecond=micro,
                tzinfo=dt.timezone.utc,
            )
            return parsed
        except ValueError:
            return None
    return None


def hours_since(ts: str | None, now: dt.datetime) -> float | None:
    parsed = parse_iso_utc(ts)
    if parsed is None:
        return None
    return (now - parsed).total_seconds() / 3600.0


def days_since(date_str: str | None, now: dt.datetime) -> float | None:
    if not date_str:
        return None
    try:
        market_date = dt.date.fromisoformat(str(date_str).strip()[:10])
    except ValueError:
        return None
    market_close = dt.datetime.combine(market_date, dt.time(hour=MARKET_CLOSE_HOUR), tzinfo=MARKET_TZ)
    now_market = now.astimezone(MARKET_TZ)
    age_days = (now_market - market_close).total_seconds() / 86400.0
    return max(0.0, age_days)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_analytics_schema() -> None:
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_usage_sessions (
                session_id TEXT PRIMARY KEY,
                visitor_id TEXT NOT NULL,
                started_ts TEXT,
                last_seen_ts TEXT,
                active_ms INTEGER NOT NULL DEFAULT 0,
                page_views_total INTEGER NOT NULL DEFAULT 0,
                guide_views INTEGER NOT NULL DEFAULT 0,
                report_views INTEGER NOT NULL DEFAULT 0,
                earnings_views INTEGER NOT NULL DEFAULT 0,
                chart_views INTEGER NOT NULL DEFAULT 0,
                opportunities_views INTEGER NOT NULL DEFAULT 0,
                chart_loads INTEGER NOT NULL DEFAULT 0,
                last_tab TEXT,
                last_ticker TEXT,
                market TEXT,
                path TEXT,
                tz TEXT,
                created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ui_usage_sessions_visitor ON ui_usage_sessions(visitor_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ui_usage_sessions_last_seen ON ui_usage_sessions(last_seen_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ui_usage_sessions_last_ticker ON ui_usage_sessions(last_ticker)"
        )
        conn.commit()
    finally:
        conn.close()


def prune_analytics_sessions() -> None:
    if not DB_PATH.exists():
        return
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=ANALYTICS_MAX_SESSION_AGE_DAYS)).isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            "DELETE FROM ui_usage_sessions WHERE COALESCE(last_seen_ts, started_ts, created_ts) < ?",
            (cutoff,),
        )
        conn.commit()
    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"DB not found at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _clamp_int(value: Any, *, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return minimum
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _clean_session_text(value: Any, *, max_len: int = 120) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:max_len]


def _upsert_usage_session(conn: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    if not table_exists(conn, "ui_usage_sessions"):
        ensure_analytics_schema()
    session_id = _clean_session_text(payload.get("session_id"), max_len=64)
    visitor_id = _clean_session_text(payload.get("visitor_id"), max_len=64)
    if not session_id or not visitor_id:
        raise HTTPException(status_code=400, detail="session_id and visitor_id are required")
    started_ts = _clean_session_text(payload.get("started_ts"), max_len=40) or _utc_now_iso()
    last_seen_ts = _clean_session_text(payload.get("last_seen_ts"), max_len=40) or _utc_now_iso()
    active_ms = _clamp_int(payload.get("active_ms"), maximum=31_536_000_000)
    page_views_total = _clamp_int(payload.get("page_views_total"), maximum=1_000_000)
    guide_views = _clamp_int(payload.get("guide_views"), maximum=1_000_000)
    report_views = _clamp_int(payload.get("report_views"), maximum=1_000_000)
    earnings_views = _clamp_int(payload.get("earnings_views"), maximum=1_000_000)
    chart_views = _clamp_int(payload.get("chart_views"), maximum=1_000_000)
    opportunities_views = _clamp_int(payload.get("opportunities_views"), maximum=1_000_000)
    chart_loads = _clamp_int(payload.get("chart_loads"), maximum=1_000_000)
    last_tab = _clean_session_text(payload.get("last_tab"), max_len=32)
    last_ticker = _clean_session_text(payload.get("last_ticker"), max_len=24)
    market = _clean_session_text(payload.get("market"), max_len=24)
    path = _clean_session_text(payload.get("path"), max_len=200)
    tz = _clean_session_text(payload.get("tz"), max_len=64)

    conn.execute(
        """
        INSERT INTO ui_usage_sessions (
            session_id,
            visitor_id,
            started_ts,
            last_seen_ts,
            active_ms,
            page_views_total,
            guide_views,
            report_views,
            earnings_views,
            chart_views,
            opportunities_views,
            chart_loads,
            last_tab,
            last_ticker,
            market,
            path,
            tz,
            created_ts,
            updated_ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            visitor_id = excluded.visitor_id,
            started_ts = COALESCE(ui_usage_sessions.started_ts, excluded.started_ts),
            last_seen_ts = excluded.last_seen_ts,
            active_ms = MAX(ui_usage_sessions.active_ms, excluded.active_ms),
            page_views_total = MAX(ui_usage_sessions.page_views_total, excluded.page_views_total),
            guide_views = MAX(ui_usage_sessions.guide_views, excluded.guide_views),
            report_views = MAX(ui_usage_sessions.report_views, excluded.report_views),
            earnings_views = MAX(ui_usage_sessions.earnings_views, excluded.earnings_views),
            chart_views = MAX(ui_usage_sessions.chart_views, excluded.chart_views),
            opportunities_views = MAX(ui_usage_sessions.opportunities_views, excluded.opportunities_views),
            chart_loads = MAX(ui_usage_sessions.chart_loads, excluded.chart_loads),
            last_tab = COALESCE(excluded.last_tab, ui_usage_sessions.last_tab),
            last_ticker = COALESCE(excluded.last_ticker, ui_usage_sessions.last_ticker),
            market = COALESCE(excluded.market, ui_usage_sessions.market),
            path = COALESCE(excluded.path, ui_usage_sessions.path),
            tz = COALESCE(excluded.tz, ui_usage_sessions.tz),
            updated_ts = excluded.updated_ts
        """,
        (
            session_id,
            visitor_id,
            started_ts,
            last_seen_ts,
            active_ms,
            page_views_total,
            guide_views,
            report_views,
            earnings_views,
            chart_views,
            opportunities_views,
            chart_loads,
            last_tab,
            last_ticker,
            market,
            path,
            tz,
            _utc_now_iso(),
            _utc_now_iso(),
        ),
    )
    conn.commit()
    return {
        "ok": True,
        "session_id": session_id,
        "visitor_id": visitor_id,
        "active_ms": active_ms,
        "page_views_total": page_views_total,
        "chart_loads": chart_loads,
    }


def _usage_summary(conn: sqlite3.Connection, days: int = 7, limit: int = 10) -> dict[str, Any]:
    days = max(1, min(365, int(days)))
    limit = max(1, min(100, int(limit)))
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS sessions,
            COUNT(DISTINCT visitor_id) AS visitors,
            COALESCE(SUM(active_ms), 0) AS active_ms_total,
            COALESCE(AVG(active_ms), 0) AS active_ms_avg,
            COALESCE(SUM(page_views_total), 0) AS page_views_total,
            COALESCE(SUM(chart_loads), 0) AS chart_loads_total
        FROM ui_usage_sessions
        WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
        """,
        (cutoff,),
    ).fetchone()
    daily_rows = conn.execute(
        """
        SELECT
            substr(COALESCE(last_seen_ts, started_ts, created_ts), 1, 10) AS day,
            COUNT(*) AS sessions,
            COUNT(DISTINCT visitor_id) AS visitors,
            COALESCE(SUM(active_ms), 0) AS active_ms_total,
            COALESCE(SUM(page_views_total), 0) AS page_views_total,
            COALESCE(SUM(chart_loads), 0) AS chart_loads_total
        FROM ui_usage_sessions
        WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
        GROUP BY day
        ORDER BY day DESC
        LIMIT ?
        """,
        (cutoff, limit),
    ).fetchall()
    tab_totals = []
    tab_columns = {
        "guide": "guide_views",
        "report": "report_views",
        "earnings": "earnings_views",
        "chart": "chart_views",
        "opportunities": "opportunities_views",
    }
    for tab, column in tab_columns.items():
        row = conn.execute(
            f"""
            SELECT
                COALESCE(SUM({column}), 0) AS views,
                COUNT(CASE WHEN {column} > 0 THEN 1 END) AS sessions
            FROM ui_usage_sessions
            WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
            """,
            (cutoff,),
        ).fetchone()
        tab_totals.append(
            {
                "tab": tab,
                "views": int(row["views"] or 0),
                "sessions": int(row["sessions"] or 0),
            }
        )
    tab_totals.sort(key=lambda row: (-row["views"], row["tab"]))
    top_tickers = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                last_ticker AS ticker,
                COUNT(*) AS sessions,
                COALESCE(SUM(chart_loads), 0) AS chart_loads,
                COALESCE(SUM(active_ms), 0) AS active_ms_total
            FROM ui_usage_sessions
            WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
              AND last_ticker IS NOT NULL AND last_ticker != ''
            GROUP BY last_ticker
            ORDER BY chart_loads DESC, sessions DESC, ticker ASC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    ]
    recent_sessions = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
                session_id,
                visitor_id,
                started_ts,
                last_seen_ts,
                active_ms,
                page_views_total,
                chart_loads,
                last_tab,
                last_ticker
            FROM ui_usage_sessions
            WHERE COALESCE(last_seen_ts, started_ts, created_ts) >= ?
            ORDER BY COALESCE(last_seen_ts, started_ts, created_ts) DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    ]
    return {
        "ok": True,
        "days": days,
        "cutoff_ts": cutoff,
        "totals": {
            "sessions": int(totals["sessions"] or 0),
            "visitors": int(totals["visitors"] or 0),
            "active_hours_total": round(float(totals["active_ms_total"] or 0) / 3_600_000.0, 2),
            "avg_active_min_per_session": round(float(totals["active_ms_avg"] or 0) / 60_000.0, 2),
            "page_views_total": int(totals["page_views_total"] or 0),
            "chart_loads_total": int(totals["chart_loads_total"] or 0),
        },
        "top_tabs": tab_totals[:limit],
        "top_tickers": [
            {
                **row,
                "active_hours_total": round(float(row.get("active_ms_total") or 0) / 3_600_000.0, 2),
            }
            for row in top_tickers
        ],
        "daily": [
            {
                "day": row["day"],
                "sessions": int(row["sessions"] or 0),
                "visitors": int(row["visitors"] or 0),
                "active_hours_total": round(float(row["active_ms_total"] or 0) / 3_600_000.0, 2),
                "page_views_total": int(row["page_views_total"] or 0),
                "chart_loads_total": int(row["chart_loads_total"] or 0),
            }
            for row in daily_rows
        ],
        "recent_sessions": [
            {
                **row,
                "active_min": round(float(row.get("active_ms") or 0) / 60_000.0, 2),
            }
            for row in recent_sessions
        ],
    }


def get_latest_fundamentals(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT *
        FROM finviz_fundamentals
        WHERE ticker = ?
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """,
        (ticker,),
    ).fetchone()
    return dict(row) if row is not None else {}


def get_latest_options_summary(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    latest = conn.execute(
        "SELECT snapshot_ts FROM options_iv WHERE ticker = ? ORDER BY snapshot_ts DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if latest is None:
        return {}
    snap = latest["snapshot_ts"]
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS contracts,
            AVG(implied_vol) AS avg_iv,
            SUM(CASE WHEN option_type='call' THEN open_interest ELSE 0 END) AS call_oi,
            SUM(CASE WHEN option_type='put' THEN open_interest ELSE 0 END) AS put_oi
        FROM options_iv
        WHERE ticker = ? AND snapshot_ts = ?
        """,
        (ticker, snap),
    ).fetchone()
    if row is None:
        return {}

    call_oi = float(row["call_oi"] or 0.0)
    put_oi = float(row["put_oi"] or 0.0)
    put_call = (put_oi / call_oi) if call_oi > 0 else None
    return {
        "snapshot_ts": snap,
        "contracts": int(row["contracts"] or 0),
        "avg_iv": float(row["avg_iv"]) if row["avg_iv"] is not None else None,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "put_call_oi_ratio": float(put_call) if put_call is not None else None,
    }


def _parse_iso_date(value: Any) -> dt.date | None:
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


def _yolo_match_tolerance_days(timeframe: str) -> int:
    return 35 if str(timeframe or "").strip().lower() == "weekly" else 14


def _yolo_snapshot_matches(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if str(anchor.get("ticker") or "") != str(candidate.get("ticker") or ""):
        return False
    if str(anchor.get("timeframe") or "") != str(candidate.get("timeframe") or ""):
        return False
    if str(anchor.get("pattern") or "") != str(candidate.get("pattern") or ""):
        return False
    tolerance = _yolo_match_tolerance_days(str(anchor.get("timeframe") or ""))
    anchor_x0 = _parse_iso_date(anchor.get("x0_date"))
    anchor_x1 = _parse_iso_date(anchor.get("x1_date"))
    cand_x0 = _parse_iso_date(candidate.get("x0_date"))
    cand_x1 = _parse_iso_date(candidate.get("x1_date"))
    if anchor_x0 is not None and cand_x0 is not None and abs((anchor_x0 - cand_x0).days) <= tolerance:
        return True
    if anchor_x1 is not None and cand_x1 is not None and abs((anchor_x1 - cand_x1).days) <= tolerance:
        return True
    return False


def _yolo_streak_for_asofs(seen_asofs: set[str], asof_dates_desc: list[str], latest_asof: str | None) -> int:
    if not latest_asof:
        return 0
    streak = 0
    started = False
    for asof in asof_dates_desc:
        if not started:
            if asof != latest_asof:
                continue
            started = True
        if asof in seen_asofs:
            streak += 1
        else:
            break
    return streak


def _yolo_priority_score(item: dict[str, Any], *, active_now: bool) -> float:
    age_factor = float(_report_yolo_age_factor(item.get("age_days"), item.get("timeframe")) or 0.0)
    streak = min(6, int(item.get("current_streak") or 0))
    confidence = float(item.get("confidence") or 0.0)
    timeframe_bonus = 3.0 if str(item.get("timeframe") or "").strip().lower() == "daily" else 0.0
    active_bonus = 8.0 if active_now else 0.0
    return round((age_factor * 100.0) + (streak * 6.0) + (confidence * 10.0) + timeframe_bonus + active_bonus, 1)


def _yolo_signal_role(item: dict[str, Any], *, active_now: bool) -> str:
    age_factor = float(_report_yolo_age_factor(item.get("age_days"), item.get("timeframe")) or 0.0)
    streak = int(item.get("current_streak") or 0)
    if not active_now:
        return "historical_context"
    if age_factor >= 0.8 and (streak >= 2 or float(item.get("confidence") or 0.0) >= 0.6):
        return "primary"
    if age_factor >= 0.5 or streak >= 2:
        return "secondary"
    if age_factor > 0.0:
        return "recent_context"
    return "stale_context"


def _yolo_role_rank(role: str) -> int:
    return {
        "primary": 4,
        "secondary": 3,
        "recent_context": 2,
        "stale_context": 1,
        "historical_context": 0,
    }.get(str(role or ""), 0)


def get_yolo_patterns(conn: sqlite3.Connection, ticker: str) -> list[dict[str, Any]]:
    if not table_exists(conn, "yolo_patterns"):
        return []
    ticker_key = str(ticker or "").upper().strip()
    if not ticker_key:
        return []

    history_rows = conn.execute(
        """
        SELECT ticker, timeframe, pattern, x0_date, x1_date, as_of_date
        FROM yolo_patterns
        WHERE ticker = ?
          AND as_of_date IS NOT NULL
        ORDER BY as_of_date DESC
        """,
        (ticker_key,),
    ).fetchall()
    history_payload = [
        {
            "ticker": str(row["ticker"] or ""),
            "timeframe": str(row["timeframe"] or ""),
            "pattern": str(row["pattern"] or ""),
            "x0_date": row["x0_date"],
            "x1_date": row["x1_date"],
            "as_of_date": row["as_of_date"],
        }
        for row in history_rows
    ]
    asof_dates_desc: dict[str, list[str]] = {"daily": [], "weekly": []}
    for timeframe_key in ("daily", "weekly"):
        dates = {
            str(row.get("as_of_date") or "")
            for row in history_payload
            if str(row.get("timeframe") or "").strip().lower() == timeframe_key
            and str(row.get("as_of_date") or "").strip()
        }
        asof_dates_desc[timeframe_key] = sorted(dates, reverse=True)

    rows = conn.execute(
        """
        SELECT p.timeframe, p.pattern, p.confidence, p.x0_date, p.x1_date, p.y0, p.y1, p.lookback_days, p.as_of_date, p.detected_ts
        FROM yolo_patterns p
        JOIN (
            SELECT timeframe, MAX(as_of_date) AS latest_asof
            FROM yolo_patterns
            WHERE ticker = ?
            GROUP BY timeframe
        ) cur
          ON p.timeframe = cur.timeframe
         AND p.as_of_date = cur.latest_asof
        WHERE p.ticker = ?
        ORDER BY p.timeframe, p.confidence DESC
        """,
        (ticker_key, ticker_key),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["ticker"] = ticker_key
        age_days = None
        as_of_date = str(item.get("as_of_date") or "").strip()
        x1_date = str(item.get("x1_date") or "").strip()
        if len(as_of_date) >= 10 and len(x1_date) >= 10:
            try:
                asof_dt = dt.date.fromisoformat(as_of_date[:10])
                x1_dt = dt.date.fromisoformat(x1_date[:10])
                age_days = max(0, (asof_dt - x1_dt).days)
            except Exception:
                age_days = None
        item["age_days"] = age_days
        seen_asofs: set[str] = set()
        timeframe_key = str(item.get("timeframe") or "").strip().lower()
        for hist in history_payload:
            if _yolo_snapshot_matches(item, hist):
                hist_asof = str(hist.get("as_of_date") or "").strip()
                if hist_asof:
                    seen_asofs.add(hist_asof)
        if seen_asofs:
            seen_sorted = sorted(seen_asofs)
            item["first_seen_asof"] = seen_sorted[0]
            item["last_seen_asof"] = seen_sorted[-1]
            item["snapshots_seen"] = len(seen_asofs)
            item["current_streak"] = _yolo_streak_for_asofs(
                seen_asofs,
                asof_dates_desc.get(timeframe_key, []),
                as_of_date or None,
            )
        else:
            item["first_seen_asof"] = as_of_date or None
            item["last_seen_asof"] = as_of_date or None
            item["snapshots_seen"] = 1 if as_of_date else 0
            item["current_streak"] = 1 if as_of_date else 0
        item["yolo_recency"] = _report_yolo_recency_label(age_days, timeframe_key)
        item["signal_role"] = _yolo_signal_role(item, active_now=True)
        item["priority_score"] = _yolo_priority_score(item, active_now=True)
        out.append(item)
    out.sort(
        key=lambda item: (
            _yolo_role_rank(str(item.get("signal_role") or "")),
            float(item.get("priority_score") or 0.0),
            int(item.get("current_streak") or 0),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return out


def get_yolo_audit(conn: sqlite3.Connection, ticker: str, limit: int = 12) -> list[dict[str, Any]]:
    if not table_exists(conn, "yolo_patterns"):
        return []
    ticker_key = str(ticker or "").upper().strip()
    if not ticker_key:
        return []

    rows = conn.execute(
        """
        SELECT ticker, timeframe, pattern, confidence, x0_date, x1_date, as_of_date, detected_ts
        FROM yolo_patterns
        WHERE ticker = ?
          AND as_of_date IS NOT NULL
        ORDER BY timeframe, as_of_date DESC, confidence DESC
        """,
        (ticker_key,),
    ).fetchall()
    if not rows:
        return []

    history_payload = [dict(row) for row in rows]
    asof_dates_desc: dict[str, list[str]] = {"daily": [], "weekly": []}
    latest_asof_by_timeframe: dict[str, str | None] = {"daily": None, "weekly": None}
    for timeframe_key in ("daily", "weekly"):
        dates = {
            str(row.get("as_of_date") or "")
            for row in history_payload
            if str(row.get("timeframe") or "").strip().lower() == timeframe_key
            and str(row.get("as_of_date") or "").strip()
        }
        ordered_dates = sorted(dates, reverse=True)
        asof_dates_desc[timeframe_key] = ordered_dates
        latest_asof_by_timeframe[timeframe_key] = ordered_dates[0] if ordered_dates else None

    groups: list[dict[str, Any]] = []
    for row in history_payload:
        matched = None
        for group in groups:
            if str(group.get("timeframe") or "") != str(row.get("timeframe") or ""):
                continue
            if str(group.get("pattern") or "") != str(row.get("pattern") or ""):
                continue
            if _yolo_snapshot_matches(group["anchor"], row):
                matched = group
                break
        if matched is None:
            groups.append(
                {
                    "timeframe": str(row.get("timeframe") or ""),
                    "pattern": str(row.get("pattern") or ""),
                    "anchor": row,
                    "rows": [row],
                }
            )
        else:
            matched["rows"].append(row)

    audit_rows: list[dict[str, Any]] = []
    for group in groups:
        grouped_rows = list(group.get("rows") or [])
        if not grouped_rows:
            continue
        grouped_rows.sort(
            key=lambda item: (
                str(item.get("as_of_date") or ""),
                float(item.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        latest = grouped_rows[0]
        timeframe_key = str(latest.get("timeframe") or "").strip().lower()
        seen_asofs = {
            str(item.get("as_of_date") or "").strip()
            for item in grouped_rows
            if str(item.get("as_of_date") or "").strip()
        }
        seen_sorted = sorted(seen_asofs)
        first_seen = seen_sorted[0] if seen_sorted else None
        last_seen = seen_sorted[-1] if seen_sorted else None
        age_days = None
        if last_seen and str(latest.get("x1_date") or "").strip():
            last_seen_dt = _parse_iso_date(last_seen)
            x1_dt = _parse_iso_date(latest.get("x1_date"))
            if last_seen_dt is not None and x1_dt is not None:
                age_days = max(0, (last_seen_dt - x1_dt).days)
        active_now = bool(last_seen and last_seen == latest_asof_by_timeframe.get(timeframe_key))
        streak = _yolo_streak_for_asofs(seen_asofs, asof_dates_desc.get(timeframe_key, []), last_seen)
        audit_item = {
            "ticker": ticker_key,
            "timeframe": latest.get("timeframe"),
            "pattern": latest.get("pattern"),
            "confidence": round(float(latest.get("confidence") or 0.0), 3),
            "x0_date": latest.get("x0_date"),
            "x1_date": latest.get("x1_date"),
            "age_days": age_days,
            "yolo_recency": _report_yolo_recency_label(age_days, timeframe_key),
            "first_seen_asof": first_seen,
            "last_seen_asof": last_seen,
            "snapshots_seen": len(seen_asofs),
            "current_streak": streak,
            "active_now": active_now,
        }
        audit_item["signal_role"] = _yolo_signal_role(audit_item, active_now=active_now)
        audit_item["priority_score"] = _yolo_priority_score(audit_item, active_now=active_now)
        audit_rows.append(audit_item)

    audit_rows.sort(
        key=lambda item: (
            1 if bool(item.get("active_now")) else 0,
            _yolo_role_rank(str(item.get("signal_role") or "")),
            float(item.get("priority_score") or 0.0),
            str(item.get("last_seen_asof") or ""),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return audit_rows[: max(1, int(limit))]


def _build_chart_technical_context(model: pd.DataFrame, levels: pd.DataFrame) -> dict[str, Any]:
    closes = pd.to_numeric(model.get("close"), errors="coerce")
    highs = pd.to_numeric(model.get("high"), errors="coerce")
    lows = pd.to_numeric(model.get("low"), errors="coerce")
    volumes = pd.to_numeric(model.get("volume"), errors="coerce").fillna(0.0)
    if closes.empty:
        return {}

    close_now = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else close_now
    high_now = float(highs.iloc[-1])
    low_now = float(lows.iloc[-1])
    ma20 = float(pd.to_numeric(model.get("ma20"), errors="coerce").iloc[-1]) if "ma20" in model.columns and pd.notna(model["ma20"].iloc[-1]) else None
    ma50 = float(pd.to_numeric(model.get("ma50"), errors="coerce").iloc[-1]) if "ma50" in model.columns and pd.notna(model["ma50"].iloc[-1]) else None
    ma100 = float(pd.to_numeric(model.get("ma100"), errors="coerce").iloc[-1]) if "ma100" in model.columns and pd.notna(model["ma100"].iloc[-1]) else None
    ma200 = float(pd.to_numeric(model.get("ma200"), errors="coerce").iloc[-1]) if "ma200" in model.columns and pd.notna(model["ma200"].iloc[-1]) else None
    prev_ma20 = float(pd.to_numeric(model.get("ma20"), errors="coerce").iloc[-2]) if "ma20" in model.columns and len(model) >= 2 and pd.notna(model["ma20"].iloc[-2]) else None
    prev_ma50 = float(pd.to_numeric(model.get("ma50"), errors="coerce").iloc[-2]) if "ma50" in model.columns and len(model) >= 2 and pd.notna(model["ma50"].iloc[-2]) else None
    prev_ma200 = float(pd.to_numeric(model.get("ma200"), errors="coerce").iloc[-2]) if "ma200" in model.columns and len(model) >= 2 and pd.notna(model["ma200"].iloc[-2]) else None
    recent_high_20 = float(highs.tail(20).max()) if len(highs) >= 20 else None
    recent_low_20 = float(lows.tail(20).min()) if len(lows) >= 20 else None
    avg_volume_20 = float(volumes.tail(20).mean()) if len(volumes) >= 20 else None
    volume_ratio_20 = (float(volumes.iloc[-1]) / avg_volume_20) if avg_volume_20 and avg_volume_20 > 0 else None
    pct_vs_ma20 = ((close_now / ma20) - 1.0) * 100.0 if ma20 and ma20 > 0 else None
    pct_vs_ma50 = ((close_now / ma50) - 1.0) * 100.0 if ma50 and ma50 > 0 else None
    pct_from_20d_high = (
        ((recent_high_20 - close_now) / recent_high_20) * 100.0
        if recent_high_20 and recent_high_20 > 0
        else None
    )
    pct_from_20d_low = (
        ((close_now - recent_low_20) / recent_low_20) * 100.0
        if recent_low_20 and recent_low_20 > 0
        else None
    )
    recent_range_pct_10 = (
        ((float(highs.tail(10).max()) - float(lows.tail(10).min())) / close_now) * 100.0
        if len(highs) >= 10 and close_now > 0
        else None
    )
    recent_range_pct_20 = (
        ((float(highs.tail(20).max()) - float(lows.tail(20).min())) / close_now) * 100.0
        if len(highs) >= 20 and close_now > 0
        else None
    )

    trend_state = "mixed"
    if ma20 is not None and ma50 is not None and close_now > ma20 > ma50:
        trend_state = "uptrend"
    elif ma20 is not None and ma50 is not None and close_now < ma20 < ma50:
        trend_state = "downtrend"

    ma_signal = None
    if prev_ma20 is not None and prev_ma50 is not None and ma20 is not None and ma50 is not None:
        if prev_ma20 >= prev_ma50 and ma20 < ma50:
            ma_signal = "bearish_20_50_cross"
        elif prev_ma20 <= prev_ma50 and ma20 > ma50:
            ma_signal = "bullish_20_50_cross"
        elif ma20 < ma50:
            ma_signal = "20_below_50"
        elif ma20 > ma50:
            ma_signal = "20_above_50"

    ma_major_signal = None
    if prev_ma50 is not None and prev_ma200 is not None and ma50 is not None and ma200 is not None:
        if prev_ma50 >= prev_ma200 and ma50 < ma200:
            ma_major_signal = "death_cross"
        elif prev_ma50 <= prev_ma200 and ma50 > ma200:
            ma_major_signal = "golden_cross"
        elif ma50 < ma200:
            ma_major_signal = "50_below_200"
        elif ma50 > ma200:
            ma_major_signal = "50_above_200"

    ma_reclaim_state = None
    if prev_ma20 is not None and ma20 is not None:
        if prev_close <= prev_ma20 and close_now > ma20:
            ma_reclaim_state = "reclaimed_ma20"
        elif prev_close >= prev_ma20 and close_now < ma20:
            ma_reclaim_state = "lost_ma20"
    if prev_ma50 is not None and ma50 is not None:
        if prev_close <= prev_ma50 and close_now > ma50:
            ma_reclaim_state = "reclaimed_ma50"
        elif prev_close >= prev_ma50 and close_now < ma50:
            ma_reclaim_state = "lost_ma50"

    recent_gap_state = None
    recent_gap_days = None
    if len(model) >= 2:
        gap_rows = model[["high", "low"]].copy()
        gap_rows["high"] = pd.to_numeric(gap_rows["high"], errors="coerce")
        gap_rows["low"] = pd.to_numeric(gap_rows["low"], errors="coerce")
        start_idx = max(1, len(gap_rows) - 4)
        for idx in range(len(gap_rows) - 1, start_idx - 1, -1):
            prev_high = float(gap_rows.iloc[idx - 1]["high"])
            prev_low = float(gap_rows.iloc[idx - 1]["low"])
            bar_high = float(gap_rows.iloc[idx]["high"])
            bar_low = float(gap_rows.iloc[idx]["low"])
            if bar_low > prev_high:
                recent_gap_state = "bull_gap"
                recent_gap_days = len(gap_rows) - 1 - idx
                break
            if bar_high < prev_low:
                recent_gap_state = "bear_gap"
                recent_gap_days = len(gap_rows) - 1 - idx
                break

    level_context = "mid_range"
    if isinstance(pct_from_20d_low, (int, float)) and float(pct_from_20d_low) <= 2.5:
        level_context = "at_support"
    elif isinstance(pct_from_20d_high, (int, float)) and float(pct_from_20d_high) <= 2.5:
        level_context = "at_resistance"

    stretch_state = "normal"
    if isinstance(pct_vs_ma20, (int, float)):
        if float(pct_vs_ma20) >= 8.0:
            stretch_state = "extended_up"
        elif float(pct_vs_ma20) <= -8.0:
            stretch_state = "extended_down"

    support_level = None
    support_zone_low = None
    support_zone_high = None
    support_tier = None
    support_touches = None
    resistance_level = None
    resistance_zone_low = None
    resistance_zone_high = None
    resistance_tier = None
    resistance_touches = None
    pct_to_support = None
    pct_to_resistance = None
    range_position = None
    breakout_state = "none"
    level_event = "none"
    structure_state = "normal"

    def _pick_level(side: str) -> dict[str, Any] | None:
        if levels is None or levels.empty:
            return None
        pool = levels[levels["type"] == side].copy()
        if pool.empty:
            return None
        if side == "support":
            preferred = pool[pool["level"] <= close_now].sort_values("level", ascending=False)
        else:
            preferred = pool[pool["level"] >= close_now].sort_values("level", ascending=True)
        if preferred.empty:
            preferred = pool.sort_values(["dist", "touches", "recency_score"], ascending=[True, False, False])
        return preferred.iloc[0].to_dict() if not preferred.empty else None

    support = _pick_level("support")
    resistance = _pick_level("resistance")
    if support:
        support_level = round(float(support.get("level") or 0.0), 2)
        support_zone_low = round(float(support.get("zone_low") or 0.0), 2)
        support_zone_high = round(float(support.get("zone_high") or 0.0), 2)
        support_tier = str(support.get("tier") or "")
        support_touches = int(support.get("touches") or 0)
        if support_level > 0:
            pct_to_support = round(((close_now - support_level) / support_level) * 100.0, 2)
    if resistance:
        resistance_level = round(float(resistance.get("level") or 0.0), 2)
        resistance_zone_low = round(float(resistance.get("zone_low") or 0.0), 2)
        resistance_zone_high = round(float(resistance.get("zone_high") or 0.0), 2)
        resistance_tier = str(resistance.get("tier") or "")
        resistance_touches = int(resistance.get("touches") or 0)
        if resistance_level > 0:
            pct_to_resistance = round(((resistance_level - close_now) / resistance_level) * 100.0, 2)
    if (
        isinstance(support_level, (int, float))
        and isinstance(resistance_level, (int, float))
        and resistance_level > support_level
    ):
        range_position = round((close_now - support_level) / max(resistance_level - support_level, 0.01), 3)
    if isinstance(support_zone_low, (int, float)) and close_now < float(support_zone_low):
        level_context = "below_support"
    elif isinstance(resistance_zone_high, (int, float)) and close_now > float(resistance_zone_high):
        level_context = "above_resistance"
    elif isinstance(support_zone_high, (int, float)) and close_now <= float(support_zone_high):
        level_context = "at_support"
    elif isinstance(resistance_zone_low, (int, float)) and close_now >= float(resistance_zone_low):
        level_context = "at_resistance"
    elif isinstance(range_position, (int, float)):
        if float(range_position) <= 0.35:
            level_context = "closer_support"
        elif float(range_position) >= 0.65:
            level_context = "closer_resistance"
        else:
            level_context = "mid_range"

    if isinstance(resistance_zone_high, (int, float)):
        rzh = float(resistance_zone_high)
        if close_now > rzh:
            breakout_state = "breakout_up"
        elif high_now > rzh and close_now <= rzh:
            breakout_state = "failed_breakout_up"
    if isinstance(support_zone_low, (int, float)):
        szl = float(support_zone_low)
        if close_now < szl:
            breakout_state = "breakout_down"
        elif low_now < szl and close_now >= szl and breakout_state == "none":
            breakout_state = "failed_breakdown_down"
    if breakout_state == "breakout_up":
        level_event = "resistance_breakout"
    elif breakout_state == "breakout_down":
        level_event = "support_breakdown"
    elif breakout_state == "failed_breakout_up":
        level_event = "resistance_reject"
    elif breakout_state == "failed_breakdown_down":
        level_event = "support_reclaim"

    if (
        isinstance(recent_range_pct_10, (int, float))
        and isinstance(recent_range_pct_20, (int, float))
        and float(recent_range_pct_10) <= 7.0
        and float(recent_range_pct_20) <= 12.0
    ):
        if (
            (isinstance(range_position, (int, float)) and float(range_position) >= 0.58)
            or (isinstance(resistance_touches, int) and resistance_touches >= 2)
        ):
            structure_state = "tight_consolidation_high"
        elif (
            (isinstance(range_position, (int, float)) and float(range_position) <= 0.42)
            or (isinstance(support_touches, int) and support_touches >= 2)
        ):
            structure_state = "tight_consolidation_low"
        else:
            structure_state = "tight_consolidation_mid"

    if isinstance(pct_vs_ma20, (int, float)) and trend_state == "uptrend" and float(pct_vs_ma20) >= 7.5:
        if breakout_state == "breakout_up" or (
            isinstance(pct_from_20d_high, (int, float)) and float(pct_from_20d_high) <= 1.5
        ):
            structure_state = "parabolic_up"
    elif isinstance(pct_vs_ma20, (int, float)) and trend_state == "downtrend" and float(pct_vs_ma20) <= -7.5:
        if breakout_state == "breakout_down" or (
            isinstance(pct_from_20d_low, (int, float)) and float(pct_from_20d_low) <= 1.5
        ):
            structure_state = "parabolic_down"

    returns = closes.pct_change().dropna().tail(20)
    realized_vol_20 = None
    if len(returns) >= 5:
        rv = float(returns.std(ddof=0) * (252.0 ** 0.5) * 100.0)
        realized_vol_20 = round(rv, 2) if rv > 0 else None
    bb_width_20 = None
    if len(closes) >= 20:
        win = closes.tail(20)
        mean_20 = float(win.mean())
        sd_20 = float(win.std(ddof=0))
        if mean_20 > 0 and sd_20 >= 0:
            bb_width_20 = round(((4.0 * sd_20) / mean_20) * 100.0, 2)

    atr_pct_14 = None
    if "atr_pct" in model.columns and pd.notna(model["atr_pct"].iloc[-1]):
        atr_pct_14 = round(float(model["atr_pct"].iloc[-1]), 2)

    return {
        "close": round(close_now, 2),
        "ma20": round(ma20, 2) if ma20 is not None else None,
        "ma50": round(ma50, 2) if ma50 is not None else None,
        "ma100": round(ma100, 2) if ma100 is not None else None,
        "ma200": round(ma200, 2) if ma200 is not None else None,
        "avg_volume_20": round(avg_volume_20, 2) if avg_volume_20 is not None else None,
        "volume_ratio_20": round(volume_ratio_20, 2) if volume_ratio_20 is not None else None,
        "pct_vs_ma20": round(pct_vs_ma20, 2) if pct_vs_ma20 is not None else None,
        "pct_vs_ma50": round(pct_vs_ma50, 2) if pct_vs_ma50 is not None else None,
        "pct_from_20d_high": round(pct_from_20d_high, 2) if pct_from_20d_high is not None else None,
        "pct_from_20d_low": round(pct_from_20d_low, 2) if pct_from_20d_low is not None else None,
        "recent_range_pct_10": round(recent_range_pct_10, 2) if recent_range_pct_10 is not None else None,
        "recent_range_pct_20": round(recent_range_pct_20, 2) if recent_range_pct_20 is not None else None,
        "trend_state": trend_state,
        "ma_signal": ma_signal,
        "ma_major_signal": ma_major_signal,
        "ma_reclaim_state": ma_reclaim_state,
        "level_context": level_context,
        "stretch_state": stretch_state,
        "breakout_state": breakout_state,
        "level_event": level_event,
        "structure_state": structure_state,
        "recent_gap_state": recent_gap_state,
        "recent_gap_days": recent_gap_days,
        "support_level": support_level,
        "support_zone_low": support_zone_low,
        "support_zone_high": support_zone_high,
        "support_tier": support_tier,
        "support_touches": support_touches,
        "resistance_level": resistance_level,
        "resistance_zone_low": resistance_zone_low,
        "resistance_zone_high": resistance_zone_high,
        "resistance_tier": resistance_tier,
        "resistance_touches": resistance_touches,
        "pct_to_support": pct_to_support,
        "pct_to_resistance": pct_to_resistance,
        "range_position": range_position,
        "atr_pct_14": atr_pct_14,
        "realized_vol_20": realized_vol_20,
        "bb_width_20": bb_width_20,
    }


def _pick_chart_candle_signal(candle_patterns: pd.DataFrame, asof_date: str) -> dict[str, Any]:
    if candle_patterns is None or candle_patterns.empty or not asof_date:
        return {}
    rows = candle_patterns.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest = rows[rows["date"] == asof_date].copy()
    if latest.empty:
        return {}
    latest["confidence"] = pd.to_numeric(latest.get("confidence"), errors="coerce").fillna(0.0)
    latest = latest.sort_values("confidence", ascending=False)
    top = latest.iloc[0].to_dict()
    return {
        "candle_pattern": top.get("pattern"),
        "candle_bias": top.get("bias") or "neutral",
        "candle_confidence": round(float(top.get("confidence") or 0.0), 2) if top.get("confidence") is not None else None,
    }


def _build_chart_commentary_payload(
    *,
    ticker: str,
    fund: dict[str, Any],
    model: pd.DataFrame,
    levels: pd.DataFrame,
    candle_patterns: pd.DataFrame,
    yolo_patterns: list[dict[str, Any]],
    yolo_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    if model.empty:
        return {}
    asof_date = str(pd.to_datetime(model["date"].iloc[-1], errors="coerce").strftime("%Y-%m-%d"))
    close_now = float(pd.to_numeric(model["close"], errors="coerce").iloc[-1])
    prev_close = float(pd.to_numeric(model["close"], errors="coerce").iloc[-2]) if len(model) >= 2 else close_now
    pct_change = (((close_now - prev_close) / prev_close) * 100.0) if prev_close > 0 else 0.0
    tech = _build_chart_technical_context(model, levels)
    candle = _pick_chart_candle_signal(candle_patterns, asof_date)
    primary_yolo = yolo_patterns[0] if yolo_patterns else None
    raw_json = None
    sector = "Unknown"
    industry = None
    if fund.get("raw_json"):
        try:
            raw_json = json.loads(str(fund.get("raw_json") or ""))
        except Exception:
            raw_json = None
    if isinstance(raw_json, dict):
        sector = str(raw_json.get("Sector") or raw_json.get("sector") or "Unknown").strip() or "Unknown"
        industry = str(raw_json.get("Industry") or raw_json.get("industry") or "").strip() or None

    row: dict[str, Any] = {
        "ticker": ticker,
        "score": 0.0,
        "confluence_score": 0.0,
        "setup_tier": "D",
        "sector": sector,
        "industry": industry,
        "pct_change": round(pct_change, 2),
        "discount_pct": fund.get("discount_pct"),
        "peg": fund.get("peg"),
        "near_52w_high": False,
        "near_52w_low": False,
        "yolo_pattern": primary_yolo.get("pattern") if primary_yolo else None,
        "yolo_confidence": primary_yolo.get("confidence") if primary_yolo else None,
        "yolo_age_days": primary_yolo.get("age_days") if primary_yolo else None,
        "yolo_timeframe": primary_yolo.get("timeframe") if primary_yolo else None,
        "yolo_first_seen_asof": primary_yolo.get("first_seen_asof") if primary_yolo else None,
        "yolo_last_seen_asof": primary_yolo.get("last_seen_asof") if primary_yolo else None,
        "yolo_snapshots_seen": primary_yolo.get("snapshots_seen") if primary_yolo else None,
        "yolo_current_streak": primary_yolo.get("current_streak") if primary_yolo else None,
        "yolo_signal_role": primary_yolo.get("signal_role") if primary_yolo else None,
        "candle_pattern": candle.get("candle_pattern"),
        "candle_bias": candle.get("candle_bias") or "neutral",
        "candle_confidence": candle.get("candle_confidence"),
    }
    row.update(tech)
    row.update(_report_score_setup_from_confluence(row))
    row.update(_report_describe_setup(row))

    primary_audit = yolo_audit[0] if yolo_audit else None
    current_active = [item for item in yolo_audit if bool(item.get("active_now"))]
    fresh_active = [item for item in current_active if str(item.get("signal_role") or "") in {"primary", "secondary"}]
    commentary_summary = {
        "latest_actionable_yolo": primary_yolo,
        "recent_persisting_yolo": fresh_active[:3],
        "historical_yolo_context": [item for item in yolo_audit if not bool(item.get("active_now"))][:3],
        "primary_audit_row": primary_audit,
    }
    return {
        "ticker": ticker,
        "asof": asof_date,
        "score": row.get("score"),
        "setup_tier": row.get("setup_tier"),
        "signal_bias": row.get("signal_bias"),
        "setup_family": row.get("setup_family"),
        "observation": row.get("observation"),
        "actionability": row.get("actionability"),
        "action": row.get("action"),
        "risk_note": row.get("risk_note"),
        "technical_read": row.get("technical_read"),
        "primary_yolo_role": primary_yolo.get("signal_role") if primary_yolo else "none",
        "primary_yolo_recency": primary_yolo.get("yolo_recency") if primary_yolo else "none",
        "commentary_context": {
            "ticker": ticker,
            "asof": asof_date,
            "sector": sector,
            "industry": industry,
            "latest_close": close_now,
            "pct_change": round(pct_change, 2),
            "fundamentals": {
                "discount_pct": fund.get("discount_pct"),
                "peg": fund.get("peg"),
            },
            "technical": tech,
            "candle": candle,
            "yolo": commentary_summary,
        },
        "llm_ready_prompt": (
            "Summarize the chart in plain English using the freshest active YOLO pattern first, "
            "then recent persistent YOLO context, then any stale historical context. "
            "Explain trend, level location, candle confirmation, breakout/compression state, "
            "and whether the setup is actionable now or only on confirmation."
        ),
    }


def _tail_text_file(path: Path, lines: int = 60, max_bytes: int = 64_000) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]
    except Exception:
        return []


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_daily_report_json(report_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    latest = report_dir / "daily_report_latest.json"
    payload = _load_json_file(latest)
    if payload is not None:
        return latest, payload
    candidates = sorted(
        [p for p in report_dir.glob("daily_report_*.json") if p.name != "daily_report_latest.json"],
        key=lambda p: p.name,
        reverse=True,
    )
    for p in candidates:
        payload = _load_json_file(p)
        if payload is not None:
            return p, payload
    return None, None


def _daily_report_history(report_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = sorted(
        [p for p in report_dir.glob("daily_report_*.json") if p.name != "daily_report_latest.json"],
        key=lambda p: p.name,
        reverse=True,
    )[: max(1, limit)]
    for p in files:
        try:
            st = p.stat()
            out.append(
                {
                    "file": p.name,
                    "path": str(p),
                    "size_bytes": st.st_size,
                    "modified_ts": dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat(),
                }
            )
        except OSError:
            continue
    return out


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _find_timeframe_summary(rows: Any, timeframe: str) -> dict[str, Any]:
    target = str(timeframe or "").strip().lower()
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and str(row.get("timeframe", "")).strip().lower() == target:
            return row
    return {}


def _read_latest_ingest_run() -> dict[str, Any] | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if not table_exists(conn, "ingest_runs"):
            return None
        row = conn.execute(
            """
            SELECT
                run_id, started_ts, finished_ts, status, tickers_total, tickers_ok, tickers_failed, error_message
            FROM ingest_runs
            ORDER BY started_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        ticker_status_count: int | None = None
        if table_exists(conn, "ingest_ticker_status"):
            ts_row = conn.execute(
                "SELECT COUNT(*) AS c FROM ingest_ticker_status WHERE run_id = ?",
                (out["run_id"],),
            ).fetchone()
            ticker_status_count = int(ts_row["c"] or 0) if ts_row else 0

        if ticker_status_count is not None:
            out["tickers_processed"] = ticker_status_count
        elif out.get("status") in {"ok", "failed"}:
            completed = int(out.get("tickers_ok") or 0) + int(out.get("tickers_failed") or 0)
            out["tickers_processed"] = completed or int(out.get("tickers_total") or 0)
        return out
    except Exception:
        return None
    finally:
        conn.close()


def _count_post_ingest_resume_attempts(run_id: str, *, source: str = "startup_resume") -> int:
    if not run_id:
        return 0
    tail = _tail_text_file(RUN_LOG_PATH, lines=5000, max_bytes=1_000_000)
    needle_source = f"source={source}"
    needle_run = f"run_id={run_id}"
    return sum(
        1
        for line in tail
        if "[RESUME]" in line and "queued mode=yolo" in line and needle_source in line and needle_run in line
    )


def _post_ingest_resume_candidate(
    *,
    latest_run: dict[str, Any] | None = None,
    pipeline_active: bool | None = None,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any] | None:
    if not AUTO_RESUME_POST_INGEST:
        return None

    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    latest_run = latest_run or _read_latest_ingest_run()
    if not latest_run or str(latest_run.get("status") or "").lower() != "ok":
        return None

    run_finished_ts = parse_iso_utc(latest_run.get("finished_ts"))
    if run_finished_ts is None:
        return None
    run_age_sec = max(0.0, (now_utc - run_finished_ts).total_seconds())
    if run_age_sec > (AUTO_RESUME_MAX_AGE_HOURS * 3600):
        return None

    if pipeline_active:
        return None

    _, latest_payload = _latest_daily_report_json(REPORT_DIR)
    generated_ts = parse_iso_utc((latest_payload or {}).get("generated_ts")) if latest_payload else None
    if generated_ts is not None and generated_ts >= (run_finished_ts - dt.timedelta(seconds=60)):
        return None

    reason = (
        "report_missing_after_completed_ingest"
        if generated_ts is None
        else "report_stale_after_completed_ingest"
    )
    run_id = str(latest_run.get("run_id") or "")
    resume_attempts = _count_post_ingest_resume_attempts(run_id, source="startup_resume")
    eligible = resume_attempts < AUTO_RESUME_MAX_RETRIES
    return {
        "mode": "yolo",
        "reason": reason,
        "eligible": eligible,
        "resume_attempts": resume_attempts,
        "max_retries": AUTO_RESUME_MAX_RETRIES,
        "blocked_reason": None if eligible else "max_retries_reached",
        "latest_run": latest_run,
        "run_finished_ts": run_finished_ts.replace(microsecond=0).isoformat(),
        "report_generated_ts": generated_ts.replace(microsecond=0).isoformat() if generated_ts else None,
        "run_age_sec": round(run_age_sec, 1),
        "max_age_hours": AUTO_RESUME_MAX_AGE_HOURS,
    }


def _queue_post_ingest_resume(source: str = "startup") -> dict[str, Any]:
    now_utc = dt.datetime.now(dt.timezone.utc)
    pipeline = _pipeline_status_snapshot(log_lines=120)
    candidate = _post_ingest_resume_candidate(
        latest_run=pipeline.get("latest_run"),
        pipeline_active=bool(pipeline.get("active")),
        now_utc=now_utc,
    )
    if candidate is None:
        return {"scheduled": False}
    if not candidate.get("eligible", True):
        LOG.warning(
            "Post-ingest resume not queued: mode=%s reason=%s run_id=%s retries=%s/%s",
            candidate.get("mode"),
            candidate.get("reason"),
            ((candidate.get("latest_run") or {}).get("run_id") or "-"),
            candidate.get("resume_attempts"),
            candidate.get("max_retries"),
        )
        return {"scheduled": False, **candidate}

    job_id = f"resume_daily_update_{now_utc.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(3)}"
    _scheduler.add_job(
        _run_daily_update,
        trigger="date",
        run_date=now_utc,
        id=job_id,
        kwargs={"mode": str(candidate.get("mode") or "yolo"), "source": source},
    )
    _append_run_log(
        "RESUME",
        (
            f"queued mode={candidate.get('mode')} source={source} "
            f"reason={candidate.get('reason')} run_id={((candidate.get('latest_run') or {}).get('run_id') or '-')}"
        ),
    )
    LOG.warning(
        "Queued post-ingest resume mode=%s source=%s reason=%s run_id=%s",
        candidate.get("mode"),
        source,
        candidate.get("reason"),
        ((candidate.get("latest_run") or {}).get("run_id") or "-"),
    )
    return {"scheduled": True, "job_id": job_id, **candidate}


def _reconcile_stale_running_runs() -> dict[str, Any]:
    """Mark orphaned ingest_runs(status=running) as failed after stale threshold."""
    out: dict[str, Any] = {
        "checked": 0,
        "reconciled": 0,
        "run_ids": [],
    }
    if not DB_PATH.exists():
        return out

    now_utc = dt.datetime.now(dt.timezone.utc)
    stale_after_sec = INGEST_RUNNING_STALE_MIN * 60
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if not table_exists(conn, "ingest_runs"):
            return out

        rows = conn.execute(
            """
            SELECT run_id, started_ts, tickers_total, tickers_ok, tickers_failed
            FROM ingest_runs
            WHERE status = 'running'
            ORDER BY started_ts ASC
            """
        ).fetchall()
        out["checked"] = len(rows)
        has_ticker_status = table_exists(conn, "ingest_ticker_status")

        for row in rows:
            run_id = str(row["run_id"])
            started_ts = parse_iso_utc(row["started_ts"])
            if started_ts is None:
                # If started_ts is malformed, treat it as stale/orphaned.
                run_age_sec = float(stale_after_sec + 1)
            else:
                run_age_sec = max(0.0, (now_utc - started_ts).total_seconds())
            if run_age_sec <= stale_after_sec:
                continue

            tickers_total = int(row["tickers_total"] or 0)
            tickers_ok = int(row["tickers_ok"] or 0)
            tickers_failed = int(row["tickers_failed"] or 0)
            processed = tickers_ok + tickers_failed
            if has_ticker_status:
                agg = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_count,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
                        COUNT(*) AS processed_count
                    FROM ingest_ticker_status
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if agg is not None:
                    tickers_ok = int(agg["ok_count"] or 0)
                    tickers_failed = int(agg["failed_count"] or 0)
                    processed = int(agg["processed_count"] or 0)

            # Keep totals consistent even when some symbols were never reached.
            unresolved = max(0, tickers_total - processed)
            tickers_failed += unresolved
            if tickers_ok + tickers_failed > tickers_total and tickers_total > 0:
                tickers_failed = max(0, tickers_total - tickers_ok)

            age_min = run_age_sec / 60.0
            finished_ts = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            reconcile_reason = (
                f"run_orphaned_after_restart: auto-failed after stale running "
                f"({age_min:.1f}m > {INGEST_RUNNING_STALE_MIN}m)"
            )
            updated = conn.execute(
                """
                UPDATE ingest_runs
                SET
                    finished_ts = ?,
                    status = 'failed',
                    tickers_ok = ?,
                    tickers_failed = ?,
                    error_message = COALESCE(NULLIF(error_message, ''), ?)
                WHERE run_id = ? AND status = 'running'
                """,
                (finished_ts, tickers_ok, tickers_failed, reconcile_reason, run_id),
            ).rowcount
            if not updated:
                continue

            out["reconciled"] += 1
            out["run_ids"].append(run_id)
            LOG.warning(
                "Reconciled stale running ingest run_id=%s age_min=%.1f total=%s processed=%s",
                run_id,
                age_min,
                tickers_total,
                processed,
            )
            _append_run_log(
                "RECOVER",
                (
                    f"run_id={run_id} stale-running auto-failed "
                    f"age_min={age_min:.1f} total={tickers_total} processed={processed}"
                ),
            )

        if out["reconciled"] > 0:
            conn.commit()
            with _STATUS_CACHE_LOCK:
                global _STATUS_CACHE_AT, _STATUS_CACHE_PAYLOAD
                _STATUS_CACHE_AT = None
                _STATUS_CACHE_PAYLOAD = None
    except Exception:
        LOG.exception("Failed to reconcile stale running ingest runs")
    finally:
        conn.close()
    return out


def _infer_pipeline_from_log_tail(tail: list[str]) -> dict[str, Any]:
    def _last_idx(lines: list[str], token: str) -> int:
        for idx in range(len(lines) - 1, -1, -1):
            if token in lines[idx]:
                return idx
        return -1

    def _last_idx_any(lines: list[str], tokens: list[str]) -> int:
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            for token in tokens:
                if token in line:
                    return idx
        return -1

    start_idx = _last_idx(tail, "[START] daily_update.sh")
    done_idx = _last_idx(tail, "[DONE]  daily_update.sh")
    report_start_idx = _last_idx(tail, "[REPORT] Generating")
    report_done_idx = _last_idx(tail, "[REPORT] Done.")
    yolo_start_idx = _last_idx(tail, "[YOLO]  Starting")
    yolo_done_idx = _last_idx(tail, "[YOLO]  Daily pattern detection done.")
    weekly_start_idx = _last_idx(tail, "[WEEKLY] Starting weekly YOLO pass")
    weekly_report_start_idx = _last_idx(tail, "[WEEKLY] Report regeneration starting")
    weekly_yolo_done_idx = _last_idx(tail, "[WEEKLY] Weekly YOLO completed OK")
    weekly_yolo_failed_idx = _last_idx(tail, "[WEEKLY] Weekly YOLO failed")
    weekly_report_done_idx = _last_idx(tail, "[WEEKLY] Report regeneration completed")
    weekly_report_failed_idx = _last_idx(tail, "[WEEKLY] Report regeneration failed")
    yolo_progress_idx = _last_idx_any(
        tail,
        [
            "[daily] Processing",
            "[weekly] Processing",
            "[daily ",
            "[weekly ",
        ],
    )
    markers = [
        ln
        for ln in tail
        if (
            "[START] daily_update.sh" in ln
            or "[YOLO]" in ln
            or "[daily] Processing" in ln
            or "[weekly] Processing" in ln
            or "[daily " in ln
            or "[weekly " in ln
            or "[REPORT]" in ln
            or "[DONE]  daily_update.sh" in ln
            or "[WEEKLY]" in ln
            or "run_id=" in ln
        )
    ]
    out: dict[str, Any] = {
        "active": False,
        "stage": "unknown" if start_idx < 0 else "idle",
        "stage_line": None,
        "start_line": tail[start_idx] if start_idx >= 0 else None,
        "done_line": tail[done_idx] if done_idx >= 0 else None,
        "marker_lines": markers[-40:],
    }
    if start_idx < 0:
        weekly_report_terminal_idx = max(weekly_report_done_idx, weekly_report_failed_idx)
        weekly_yolo_terminal_idx = max(weekly_yolo_done_idx, weekly_yolo_failed_idx, weekly_report_terminal_idx)

        # Fallback: infer active phase from recent markers even if the explicit
        # [START] line is outside the retained tail window.
        if weekly_report_start_idx >= 0 and weekly_report_start_idx > weekly_report_terminal_idx:
            out["active"] = True
            out["stage"] = "report"
            out["stage_line"] = tail[weekly_report_start_idx]
            return out
        if weekly_start_idx >= 0 and weekly_start_idx > weekly_yolo_terminal_idx:
            out["active"] = True
            out["stage"] = "yolo"
            out["stage_line"] = tail[yolo_progress_idx] if yolo_progress_idx > weekly_start_idx else tail[weekly_start_idx]
            return out
        if report_start_idx >= 0 and report_start_idx > report_done_idx:
            out["active"] = True
            out["stage"] = "report"
            out["stage_line"] = tail[report_start_idx]
            return out
        if yolo_progress_idx >= 0 and yolo_progress_idx > yolo_done_idx:
            out["active"] = True
            out["stage"] = "yolo"
            out["stage_line"] = tail[yolo_progress_idx]
            return out
        return out
    if done_idx > start_idx:
        return out

    block = tail[start_idx:]
    out["active"] = True

    report_start = _last_idx(block, "[REPORT] Generating")
    report_done = _last_idx(block, "[REPORT] Done.")
    yolo_start = _last_idx(block, "[YOLO]  Starting")
    yolo_done = _last_idx(block, "[YOLO]  Daily pattern detection done.")
    yolo_progress = _last_idx_any(
        block,
        [
            "[daily] Processing",
            "[weekly] Processing",
            "[daily ",
            "[weekly ",
        ],
    )

    if report_start >= 0 and report_start > report_done:
        out["stage"] = "report"
        out["stage_line"] = block[report_start]
    elif (yolo_start >= 0 and yolo_start > yolo_done) or (yolo_progress >= 0 and yolo_progress > yolo_done):
        out["stage"] = "yolo"
        out["stage_line"] = block[yolo_progress] if yolo_progress >= 0 else block[yolo_start]
    else:
        out["stage"] = "ingest"
        out["stage_line"] = block[0] if block else None
    return out


def _infer_last_completed_event_from_tail(tail: list[str]) -> dict[str, Any] | None:
    candidates = [
        ("weekly_report", "ok", "[WEEKLY] Report regeneration completed"),
        ("weekly_report", "failed", "[WEEKLY] Report regeneration failed"),
        ("weekly_yolo", "ok", "[WEEKLY] Weekly YOLO completed OK"),
        ("weekly_yolo", "failed", "[WEEKLY] Weekly YOLO failed"),
        ("report", "ok", "[REPORT] Done."),
        ("report", "failed", "[REPORT] Failed to generate report"),
        ("daily_update", "ok", "[DONE]  daily_update.sh"),
        ("ingest", "failed", "[ERROR] ingest failed rc="),
    ]
    best_idx = -1
    best: dict[str, Any] | None = None
    for stage, status, token in candidates:
        for idx in range(len(tail) - 1, -1, -1):
            line = tail[idx]
            if token in line and idx > best_idx:
                ts = parse_log_line_ts_utc(line)
                best_idx = idx
                best = {
                    "stage": stage,
                    "status": status,
                    "line": line,
                    "ts": ts.replace(microsecond=0).isoformat() if ts else None,
                }
                break
    return best


def _pipeline_status_snapshot(log_lines: int = 160) -> dict[str, Any]:
    tail = _tail_text_file(RUN_LOG_PATH, lines=log_lines, max_bytes=256_000)
    inferred = _infer_pipeline_from_log_tail(tail)
    last_completed = _infer_last_completed_event_from_tail(tail)
    latest_run = _read_latest_ingest_run()
    stage = inferred.get("stage", "unknown")
    active = bool(inferred.get("active"))
    stage_line = inferred.get("stage_line")
    stage_line_ts = parse_log_line_ts_utc(stage_line)
    stage_age_sec: float | None = None
    stale_inference = False
    now_utc = dt.datetime.now(dt.timezone.utc)
    if stage_line_ts is not None:
        stage_age_sec = max(0.0, (now_utc - stage_line_ts).total_seconds())
    running_age_sec: float | None = None
    running_stale = False
    if latest_run and latest_run.get("status") == "running":
        started = parse_iso_utc(latest_run.get("started_ts"))
        if started is not None:
            running_age_sec = max(0.0, (now_utc - started).total_seconds())
            running_stale = running_age_sec > (INGEST_RUNNING_STALE_MIN * 60)
    if latest_run and latest_run.get("status") == "running" and not running_stale:
        active = True
        stage = "ingest"
    elif latest_run and latest_run.get("status") == "running" and running_stale:
        stale_inference = True
        active = False
        stage = "stale_running"
    elif active and stage_age_sec is not None and stage_age_sec > PIPELINE_STALE_SEC:
        # Prevent stale log tails from pinning pipeline_active=True forever.
        # Keep active only when DB still says a run is truly running.
        if not (latest_run and latest_run.get("status") == "running"):
            stale_inference = True
            active = False
            stage = "idle"
    if not active and stage != "stale_running":
        if last_completed is not None:
            stage = "idle"
        stage_line = None
        stage_line_ts = None
        stage_age_sec = None
    return {
        "run_log_path": str(RUN_LOG_PATH),
        "active": active,
        "stage": stage,
        "latest_run": latest_run,
        "markers": inferred.get("marker_lines", []),
        "stage_line": stage_line,
        "stage_line_ts": stage_line_ts.replace(microsecond=0).isoformat() if stage_line_ts else None,
        "stage_age_sec": round(stage_age_sec, 1) if stage_age_sec is not None else None,
        "stale_timeout_sec": PIPELINE_STALE_SEC,
        "stale_inference": stale_inference,
        "running_age_sec": round(running_age_sec, 1) if running_age_sec is not None else None,
        "running_stale_min": INGEST_RUNNING_STALE_MIN,
        "running_stale": running_stale,
        "last_completed_stage": last_completed.get("stage") if last_completed else None,
        "last_completed_status": last_completed.get("status") if last_completed else None,
        "last_completed_line": last_completed.get("line") if last_completed else None,
        "last_completed_ts": last_completed.get("ts") if last_completed else None,
        "tail": tail[-60:],
    }


def _smtp_settings() -> dict[str, Any]:
    port_raw = os.getenv("TRADER_KOO_SMTP_PORT", "587").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 587

    timeout_raw = os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30").strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30

    security = os.getenv("TRADER_KOO_SMTP_SECURITY", "starttls").strip().lower()
    if security not in {"starttls", "ssl", "none"}:
        security = "starttls"

    return {
        "host": os.getenv("TRADER_KOO_SMTP_HOST", "").strip(),
        "port": port,
        "user": os.getenv("TRADER_KOO_SMTP_USER", "").strip(),
        "password": os.getenv("TRADER_KOO_SMTP_PASS", ""),
        "from_email": os.getenv("TRADER_KOO_SMTP_FROM", "").strip(),
        "default_to": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
        "security": security,
    }


def _resend_settings() -> dict[str, Any]:
    timeout_raw = os.getenv("TRADER_KOO_RESEND_TIMEOUT_SEC", os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30")).strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30
    return {
        "api_key": os.getenv("TRADER_KOO_RESEND_API_KEY", "").strip(),
        "from_email": os.getenv("TRADER_KOO_RESEND_FROM", os.getenv("TRADER_KOO_SMTP_FROM", "")).strip(),
        "default_to": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
    }


def _email_transport() -> str:
    raw = os.getenv("TRADER_KOO_EMAIL_TRANSPORT", "auto").strip().lower()
    if raw not in {"auto", "smtp", "resend"}:
        raw = "auto"
    if raw == "auto":
        resend = _resend_settings()
        return "resend" if resend.get("api_key") else "smtp"
    return raw


def _send_resend_email(
    subject: str,
    text: str,
    recipient: str,
    resend: dict[str, Any],
    *,
    html_body: str | None = None,
) -> None:
    user_agent = os.getenv("TRADER_KOO_EMAIL_USER_AGENT", "trader-koo/1.0")
    payload = {
        "from": resend["from_email"],
        "to": [recipient],
        "subject": subject,
        "text": text,
    }
    if html_body:
        payload["html"] = html_body
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {resend['api_key']}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(resend["timeout_sec"])) as resp:
            status = int(getattr(resp, "status", 200))
            body = resp.read().decode("utf-8", errors="replace")
        if status >= 300:
            raise RuntimeError(f"Resend API failed status={status} body={body[:500]}")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Resend API HTTP {exc.code}: {err_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Resend connect failed: {exc.reason}") from exc


def _send_smtp_email(message: EmailMessage, smtp: dict[str, Any]) -> None:
    host = smtp["host"]
    port = int(smtp["port"])
    timeout_sec = int(smtp["timeout_sec"])
    security = str(smtp["security"])
    user = str(smtp.get("user") or "")
    password = str(smtp.get("password") or "")

    if security == "ssl":
        with smtplib.SMTP_SSL(host, port, timeout=timeout_sec, context=ssl.create_default_context()) as server:
            if user:
                server.login(user, password)
            server.send_message(message)
        return

    with smtplib.SMTP(host, port, timeout=timeout_sec) as server:
        server.ehlo()
        if security == "starttls":
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        if user:
            server.login(user, password)
        server.send_message(message)


def get_yolo_status(conn: sqlite3.Connection) -> dict[str, Any]:
    out: dict[str, Any] = {
        "table_exists": table_exists(conn, "yolo_patterns"),
        "events_table_exists": table_exists(conn, "yolo_run_events"),
        "universe_tickers": 0,
        "summary": {},
        "timeframes": [],
        "latest_run": None,
        "latest_non_ok_events": [],
    }
    if not out["table_exists"]:
        return out

    universe_row = conn.execute("SELECT COUNT(DISTINCT ticker) AS c FROM price_daily").fetchone()
    out["universe_tickers"] = int(universe_row["c"] or 0) if universe_row else 0

    summary = conn.execute(
        """
        SELECT
            COUNT(*) AS rows_total,
            COUNT(DISTINCT ticker) AS tickers_total,
            MAX(detected_ts) AS latest_detected_ts,
            MAX(as_of_date) AS latest_asof_date
        FROM yolo_patterns
        """
    ).fetchone()
    out["summary"] = dict(summary) if summary is not None else {}

    tf_rows = conn.execute(
        """
        SELECT
            timeframe,
            COUNT(*) AS rows_total,
            COUNT(DISTINCT ticker) AS tickers_total,
            MAX(detected_ts) AS latest_detected_ts,
            MAX(as_of_date) AS latest_asof_date,
            AVG(confidence) AS avg_confidence
        FROM yolo_patterns
        GROUP BY timeframe
        ORDER BY timeframe
        """
    ).fetchall()
    out["timeframes"] = [dict(r) for r in tf_rows]

    if out["events_table_exists"]:
        latest_run = conn.execute(
            """
            SELECT run_id, MAX(created_ts) AS latest_ts
            FROM yolo_run_events
            GROUP BY run_id
            ORDER BY latest_ts DESC
            LIMIT 1
            """
        ).fetchone()
        if latest_run is not None:
            run_id = latest_run["run_id"]
            run_stats = conn.execute(
                """
                SELECT
                    run_id,
                    MIN(created_ts) AS started_ts,
                    MAX(created_ts) AS latest_ts,
                    COUNT(*) AS events_total,
                    SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok_count,
                    SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skipped_count,
                    SUM(CASE WHEN status='timeout' THEN 1 ELSE 0 END) AS timeout_count,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count
                FROM yolo_run_events
                WHERE run_id = ?
                GROUP BY run_id
                """,
                (run_id,),
            ).fetchone()
            if run_stats is not None:
                out["latest_run"] = dict(run_stats)

            events = conn.execute(
                """
                SELECT run_id, timeframe, ticker, status, reason, elapsed_sec, bars, detections, as_of_date, created_ts
                FROM yolo_run_events
                WHERE run_id = ? AND status != 'ok'
                ORDER BY created_ts DESC
                LIMIT 100
                """,
                (run_id,),
            ).fetchall()
            out["latest_non_ok_events"] = [dict(r) for r in events]
    return out


def get_price_df(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume
        FROM price_daily
        WHERE ticker = ?
        ORDER BY date
        """,
        conn,
        params=(ticker,),
    )
    return ensure_ohlcv_schema(df)


def _serialize_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.fillna("").to_dict(orient="records")


def build_pattern_overlays(
    patterns: pd.DataFrame,
    hybrid_patterns: pd.DataFrame,
    cv_proxy_patterns: pd.DataFrame,
    max_rows: int = 10,
) -> pd.DataFrame:
    cols = [
        "source",
        "class_name",
        "status",
        "confidence",
        "start_date",
        "end_date",
        "x0_date",
        "x1_date",
        "y0",
        "y1",
        "y0b",
        "y1b",
        "notes",
    ]
    empty = pd.DataFrame(columns=cols)

    hp = hybrid_patterns.copy() if hybrid_patterns is not None else pd.DataFrame()
    if not hp.empty:
        hp["hybrid_confidence"] = pd.to_numeric(hp.get("hybrid_confidence"), errors="coerce")
        hp = hp.sort_values("hybrid_confidence", ascending=False).drop_duplicates(subset=["pattern"])
        hp = hp[["pattern", "hybrid_confidence"]]
    else:
        hp = pd.DataFrame(columns=["pattern", "hybrid_confidence"])

    rule_df = patterns.copy() if patterns is not None else pd.DataFrame()
    if not rule_df.empty:
        rule_df = rule_df.merge(hp, on="pattern", how="left")
        base_conf = pd.to_numeric(rule_df.get("confidence"), errors="coerce")
        hybrid_conf = pd.to_numeric(rule_df.get("hybrid_confidence"), errors="coerce")
        rule_df["confidence"] = hybrid_conf.where(hybrid_conf.notna(), base_conf)
        rule_df["source"] = rule_df["hybrid_confidence"].apply(
            lambda v: "hybrid_rule" if pd.notna(v) else "rule"
        )
        rule_df = rule_df.rename(columns={"pattern": "class_name"})
        rule_df = rule_df[
            [
                "source",
                "class_name",
                "status",
                "confidence",
                "start_date",
                "end_date",
                "x0_date",
                "x1_date",
                "y0",
                "y1",
                "y0b",
                "y1b",
                "notes",
            ]
        ]
    else:
        rule_df = empty.copy()

    cv_df = cv_proxy_patterns.copy() if cv_proxy_patterns is not None else pd.DataFrame()
    if not cv_df.empty:
        cv_df["source"] = "cv_proxy"
        cv_df["confidence"] = pd.to_numeric(cv_df.get("cv_confidence"), errors="coerce")
        cv_df = cv_df.rename(columns={"pattern": "class_name"})
        cv_df = cv_df[
            [
                "source",
                "class_name",
                "status",
                "confidence",
                "start_date",
                "end_date",
                "x0_date",
                "x1_date",
                "y0",
                "y1",
                "y0b",
                "y1b",
                "notes",
            ]
        ]
    else:
        cv_df = empty.copy()

    out = pd.concat([rule_df, cv_df], ignore_index=True)
    if out.empty:
        return empty

    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    for c in ["y0", "y1", "y0b", "y1b"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["x0_date", "x1_date", "y0", "y1", "y0b", "y1b"])
    out = out[(out["x0_date"].astype(str).str.len() > 0) & (out["x1_date"].astype(str).str.len() > 0)]
    if out.empty:
        return empty

    out = out.sort_values("confidence", ascending=False)
    out = out.drop_duplicates(subset=["source", "class_name", "start_date", "end_date"]).head(max_rows)
    out["confidence"] = out["confidence"].round(2)
    return out[cols].reset_index(drop=True)


def build_dashboard_payload(conn: sqlite3.Connection, ticker: str, months: int) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    fund = get_latest_fundamentals(conn, ticker)
    prices = get_price_df(conn, ticker)
    if prices.empty:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

    max_date = prices["date"].max()
    if months <= 0:
        calc_cutoff = prices["date"].min()
        view_cutoff = prices["date"].min()
    else:
        calc_cutoff = max_date - pd.DateOffset(months=max(6, months * 2))
        view_cutoff = max_date - pd.DateOffset(months=max(1, months))

    model_prices = prices[prices["date"] >= calc_cutoff].reset_index(drop=True)
    model = add_basic_features(model_prices, FEATURE_CFG)
    model = compute_pivots(model, left=3, right=3)

    last_close = float(model["close"].iloc[-1])
    levels_raw = build_levels_from_pivots(model, LEVEL_CFG)
    levels = select_target_levels(levels_raw, last_close, LEVEL_CFG)
    levels = add_fallback_levels(model, levels, last_close, LEVEL_CFG)

    gaps = select_gaps_for_display(
        detect_gaps(model),
        last_close=last_close,
        asof=max_date,
        cfg=GAP_CFG,
    )
    trendlines = detect_trendlines(model, last_close=last_close, cfg=TREND_CFG)
    patterns = detect_patterns(model, cfg=PATTERN_CFG)
    candle_patterns = detect_candlestick_patterns(model, cfg=CANDLE_CFG)
    hybrid_patterns = score_hybrid_patterns(model, patterns, candle_patterns, HYBRID_PATTERN_CFG)
    cv_proxy_patterns = detect_cv_proxy_patterns(model, cfg=CV_PROXY_CFG)
    hybrid_cv_compare = compare_hybrid_vs_cv(hybrid_patterns, cv_proxy_patterns, HYBRID_CV_CMP_CFG)
    pattern_overlays = build_pattern_overlays(
        patterns=patterns,
        hybrid_patterns=hybrid_patterns,
        cv_proxy_patterns=cv_proxy_patterns,
        max_rows=10,
    )

    chart_rows = model[model["date"] >= view_cutoff].copy()
    for col in ["date"]:
        chart_rows[col] = pd.to_datetime(chart_rows[col], errors="coerce").dt.strftime("%Y-%m-%d")

    chart_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma20",
        "ma50",
        "ma100",
        "ma200",
        "atr",
        "atr_pct",
    ]
    chart_rows = chart_rows[[c for c in chart_cols if c in chart_rows.columns]].copy()
    market_date = dt.datetime.now(MARKET_TZ).date()
    earnings_markers = get_ticker_earnings_markers(
        conn,
        ticker=ticker,
        market_date=market_date,
        forward_days=120,
        max_markers=3,
    )
    yolo_patterns = get_yolo_patterns(conn, ticker)
    yolo_audit = get_yolo_audit(conn, ticker, limit=14)
    chart_commentary = _build_chart_commentary_payload(
        ticker=ticker,
        fund=fund,
        model=model,
        levels=levels,
        candle_patterns=candle_patterns,
        yolo_patterns=yolo_patterns,
        yolo_audit=yolo_audit,
    )

    return {
        "ticker": ticker,
        "asof": chart_rows["date"].iloc[-1],
        "fundamentals": fund,
        "options_summary": get_latest_options_summary(conn, ticker),
        "chart": _serialize_df(chart_rows),
        "levels": _serialize_df(levels),
        "gaps": _serialize_df(gaps),
        "trendlines": _serialize_df(trendlines),
        "patterns": _serialize_df(patterns),
        "candlestick_patterns": _serialize_df(candle_patterns),
        "hybrid_patterns": _serialize_df(hybrid_patterns),
        "cv_proxy_patterns": _serialize_df(cv_proxy_patterns),
        "hybrid_cv_compare": _serialize_df(hybrid_cv_compare),
        "pattern_overlays": _serialize_df(pattern_overlays),
        "yolo_patterns": yolo_patterns,
        "yolo_audit": yolo_audit,
        "chart_commentary": chart_commentary,
        "earnings_markers": earnings_markers,
        "meta": {
            "schema": ["date", "open", "high", "low", "close", "volume"],
            "config": {
                "level": LEVEL_CFG.__dict__,
                "gap": GAP_CFG.__dict__,
                "trendline": TREND_CFG.__dict__,
                "pattern": PATTERN_CFG.__dict__,
                "candlestick_pattern": CANDLE_CFG.__dict__,
                "hybrid_pattern": HYBRID_PATTERN_CFG.__dict__,
                "cv_proxy_pattern": CV_PROXY_CFG.__dict__,
                "hybrid_cv_compare": HYBRID_CV_CMP_CFG.__dict__,
            },
        },
    }


def select_fund_snapshot(conn: sqlite3.Connection, min_complete_tickers: int = 400) -> tuple[str | None, int]:
    latest = conn.execute(
        """
        SELECT snapshot_ts, COUNT(DISTINCT ticker) AS c
        FROM finviz_fundamentals
        GROUP BY snapshot_ts
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """
    ).fetchone()
    if latest is None:
        return None, 0

    latest_snap = latest["snapshot_ts"]
    latest_count = int(latest["c"] or 0)
    if latest_count >= min_complete_tickers:
        return latest_snap, latest_count

    # Latest snapshot can be partial if ingest was interrupted.
    latest_complete = conn.execute(
        """
        SELECT snapshot_ts, COUNT(DISTINCT ticker) AS c
        FROM finviz_fundamentals
        GROUP BY snapshot_ts
        HAVING COUNT(DISTINCT ticker) >= ?
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """,
        (min_complete_tickers,),
    ).fetchone()
    if latest_complete is not None:
        return latest_complete["snapshot_ts"], int(latest_complete["c"] or 0)

    return latest_snap, latest_count


@app.get("/api/config", include_in_schema=False)
def config() -> dict[str, Any]:
    """Public client config — never expose secrets."""
    return {
        "auth": {
            "admin_api_key_required": bool(API_KEY),
            "admin_api_key_header": "X-API-Key",
        },
    }

@app.get("/api/health")
def health() -> dict[str, Any]:
    db_exists = DB_PATH.exists()
    payload = {"ok": db_exists, "db_exists": db_exists}
    if EXPOSE_STATUS_INTERNAL:
        payload["db_path"] = str(DB_PATH)
    return payload


@app.post("/api/usage/session", include_in_schema=False)
async def usage_session(request: Request) -> dict[str, Any]:
    if not ANALYTICS_ENABLED:
        return {"ok": True, "analytics_enabled": False}
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid analytics payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Analytics payload must be an object")
    conn = get_conn()
    try:
        result = _upsert_usage_session(conn, payload)
    finally:
        conn.close()
    return {
        **result,
        "analytics_enabled": True,
    }


@app.post("/api/admin/trigger-update")
def trigger_update(mode: str = Query(default="full")) -> dict[str, Any]:
    """
    Trigger daily_update.sh immediately.

    Modes:
    - full: ingest + yolo + report
    - yolo: skip ingest, run yolo + report
    - report: skip ingest/yolo, run report (+ optional email)
    """
    mode_norm = _normalize_update_mode(mode)
    if mode_norm is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid mode. Use one of: full, yolo, report "
                "(aliases: yolo_report, yolo+report, report_only)."
            ),
        )

    reconcile = _reconcile_stale_running_runs()
    pipeline = _pipeline_status_snapshot(log_lines=120)
    if pipeline["active"]:
        stage = pipeline.get("stage", "unknown")
        return {
            "ok": False,
            "message": f"daily_update already running (stage={stage}, requested_mode={mode_norm})",
            "stage": stage,
            "requested_mode": mode_norm,
            "latest_run": pipeline.get("latest_run"),
            "run_log_path": pipeline.get("run_log_path"),
            "reconciled_stale_runs": reconcile.get("reconciled", 0),
        }

    now_utc = dt.datetime.now(dt.timezone.utc)
    manual_job_id = f"manual_daily_update_{now_utc.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(3)}"
    _scheduler.add_job(
        _run_daily_update,
        trigger="date",
        run_date=now_utc,
        id=manual_job_id,
        kwargs={"mode": mode_norm, "source": "admin"},
    )

    mode_message = {
        "full": "full pipeline (ingest + yolo + report)",
        "yolo": "yolo + report (ingest skipped)",
        "report": "report only (ingest + yolo skipped)",
    }
    return {
        "ok": True,
        "message": (
            f"daily_update triggered ({mode_message.get(mode_norm, mode_norm)}) "
            "— check /data/logs/cron_daily.log"
        ),
        "stage": "queued",
        "mode": mode_norm,
        "job_id": manual_job_id,
        "run_log_path": str(RUN_LOG_PATH),
        "reconciled_stale_runs": reconcile.get("reconciled", 0),
    }


_yolo_seed_thread: threading.Thread | None = None


@app.post("/api/admin/run-yolo-seed")
def run_yolo_seed(timeframe: str = "both") -> dict[str, Any]:
    """Trigger full YOLO seed for all tickers in background (no --only-new)."""
    global _yolo_seed_thread
    timeframe_norm = str(timeframe or "both").strip().lower() or "both"
    if timeframe_norm not in {"daily", "weekly", "both"}:
        raise HTTPException(status_code=400, detail="Invalid timeframe. Use one of: daily, weekly, both")
    if _yolo_seed_thread and _yolo_seed_thread.is_alive():
        return {"ok": False, "message": "Seed already running — wait for it to finish"}
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_yolo_patterns.py"
    cmd = [
        sys.executable, str(script),
        "--db-path", str(DB_PATH),
        "--timeframe", timeframe_norm,
        "--lookback-days", "180",
        "--weekly-lookback-days", "730",
        "--sleep", "0.05",
    ]

    def _run() -> None:
        subprocess.run(cmd, capture_output=False)

    _yolo_seed_thread = threading.Thread(target=_run, daemon=True)
    _yolo_seed_thread.start()
    return {
        "ok": True,
        "message": f"YOLO seed started (timeframe={timeframe_norm}) — tail /data/logs/yolo_patterns.log or check Railway logs",
        "timeframe": timeframe_norm,
    }


@app.get("/api/admin/yolo-status")
def yolo_status(log_lines: int = Query(default=40, ge=0, le=400)) -> dict[str, Any]:
    """Return YOLO runner status + DB summary + recent log tail."""
    log_path = Path(os.getenv("TRADER_KOO_YOLO_LOG_PATH", "/data/logs/yolo_patterns.log"))
    conn = get_conn()
    try:
        db_status = get_yolo_status(conn)
    finally:
        conn.close()
    thread_alive = bool(_yolo_seed_thread and _yolo_seed_thread.is_alive())
    return {
        "ok": True,
        "thread_running": thread_alive,
        "log_path": str(log_path),
        "log_tail": _tail_text_file(log_path, lines=log_lines),
        "db": db_status,
    }


@app.get("/api/admin/yolo-events")
def yolo_events(
    limit: int = Query(default=200, ge=1, le=1000),
    run_id: str = Query(default=""),
    status: str = Query(default="", pattern="^(|ok|skipped|timeout|failed)$"),
) -> dict[str, Any]:
    """Return persisted per-ticker YOLO run events for diagnostics."""
    conn = get_conn()
    try:
        if not table_exists(conn, "yolo_run_events"):
            return {"ok": False, "events_table_exists": False, "rows": []}

        clauses: list[str] = []
        params: list[Any] = []
        if run_id.strip():
            clauses.append("run_id = ?")
            params.append(run_id.strip())
        if status.strip():
            clauses.append("status = ?")
            params.append(status.strip())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT run_id, timeframe, ticker, status, reason, elapsed_sec, bars, detections, as_of_date, created_ts
            FROM yolo_run_events
            {where_sql}
            ORDER BY created_ts DESC
            LIMIT ?
        """
        params.append(int(limit))
        rows = conn.execute(sql, tuple(params)).fetchall()

        latest_run_row = conn.execute(
            """
            SELECT run_id
            FROM yolo_run_events
            GROUP BY run_id
            ORDER BY MAX(created_ts) DESC
            LIMIT 1
            """
        ).fetchone()
        return {
            "ok": True,
            "events_table_exists": True,
            "latest_run_id": latest_run_row["run_id"] if latest_run_row else None,
            "count": len(rows),
            "rows": [dict(r) for r in rows],
        }
    finally:
        conn.close()


@app.get("/api/admin/report-stability")
def report_stability(limit: int = Query(default=60, ge=1, le=365)) -> dict[str, Any]:
    """Summarize recent report JSON files to diagnose YOLO/report stability over time."""
    report_dir = REPORT_DIR
    files = sorted(
        [p for p in report_dir.glob("daily_report_*.json") if p.name != "daily_report_latest.json"],
        key=lambda p: p.name,
        reverse=True,
    )
    scan_files = files[: max(1, limit)]
    rows: list[dict[str, Any]] = []

    for p in scan_files:
        modified_ts: str | None = None
        try:
            st = p.stat()
            modified_ts = (
                dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
        except OSError:
            pass

        payload = _load_json_file(p)
        if not isinstance(payload, dict):
            rows.append(
                {
                    "file": p.name,
                    "generated_ts": None,
                    "modified_ts": modified_ts,
                    "ok": False,
                    "parse_error": True,
                    "yolo_present": False,
                    "yolo_rows_total": 0,
                    "yolo_tickers_total": 0,
                }
            )
            continue

        yolo = payload.get("yolo") if isinstance(payload.get("yolo"), dict) else {}
        summary = yolo.get("summary") if isinstance(yolo.get("summary"), dict) else {}
        timeframes = yolo.get("timeframes") if isinstance(yolo.get("timeframes"), list) else []
        tf_daily = _find_timeframe_summary(timeframes, "daily")
        tf_weekly = _find_timeframe_summary(timeframes, "weekly")
        delta_legacy = yolo.get("delta") if isinstance(yolo.get("delta"), dict) else {}
        delta_daily = yolo.get("delta_daily") if isinstance(yolo.get("delta_daily"), dict) else delta_legacy
        delta_weekly = yolo.get("delta_weekly") if isinstance(yolo.get("delta_weekly"), dict) else {}
        counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
        freshness = payload.get("freshness") if isinstance(payload.get("freshness"), dict) else {}
        warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}

        yolo_rows_total = _to_int(summary.get("rows_total"), _to_int(counts.get("yolo_rows"), 0))
        yolo_tickers_total = _to_int(summary.get("tickers_with_patterns"), 0)
        yolo_daily_tickers = _to_int(tf_daily.get("tickers_with_patterns"), 0)
        yolo_weekly_tickers = _to_int(tf_weekly.get("tickers_with_patterns"), 0)
        yolo_present = yolo_rows_total > 0
        yolo_age_hours = _to_float(freshness.get("yolo_age_hours"))
        report_kind = str(meta.get("report_kind") or "daily").strip().lower()
        if report_kind not in {"daily", "weekly"}:
            report_kind = "daily"

        rows.append(
            {
                "file": p.name,
                "generated_ts": payload.get("generated_ts") or modified_ts,
                "modified_ts": modified_ts,
                "ok": bool(payload.get("ok", False)),
                "parse_error": False,
                "report_kind": report_kind,
                "warnings_count": len(warnings),
                "has_yolo_data_stale_warning": "yolo_data_stale" in {str(w) for w in warnings},
                "yolo_present": yolo_present,
                "yolo_rows_total": yolo_rows_total,
                "yolo_tickers_total": yolo_tickers_total,
                "yolo_daily_tickers": yolo_daily_tickers,
                "yolo_weekly_tickers": yolo_weekly_tickers,
                "yolo_latest_detected_ts": summary.get("latest_detected_ts"),
                "yolo_latest_asof_date": summary.get("latest_asof_date"),
                "yolo_age_hours": yolo_age_hours,
                "delta_daily_new": _to_int(delta_daily.get("new_count"), 0),
                "delta_daily_lost": _to_int(delta_daily.get("lost_count"), 0),
                "delta_weekly_new": _to_int(delta_weekly.get("new_count"), 0),
                "delta_weekly_lost": _to_int(delta_weekly.get("lost_count"), 0),
            }
        )

    parsed_rows = [r for r in rows if not r.get("parse_error")]
    yolo_present_reports = sum(1 for r in parsed_rows if r.get("yolo_present"))
    yolo_missing_reports = sum(1 for r in parsed_rows if not r.get("yolo_present"))
    yolo_presence_rate_pct = (
        round((100.0 * yolo_present_reports) / len(parsed_rows), 2) if parsed_rows else None
    )

    # newest-missing streak over newest->older ordering
    newest_missing_streak = 0
    for r in parsed_rows:
        if r.get("yolo_present"):
            break
        newest_missing_streak += 1

    # longest missing streak over chronological (older->newer) ordering
    longest_missing_streak = 0
    current_streak = 0
    for r in reversed(parsed_rows):
        if r.get("yolo_present"):
            current_streak = 0
            continue
        current_streak += 1
        if current_streak > longest_missing_streak:
            longest_missing_streak = current_streak

    yolo_rows_vals = [_to_float(r.get("yolo_rows_total")) for r in parsed_rows]
    yolo_tickers_vals = [_to_float(r.get("yolo_tickers_total")) for r in parsed_rows]
    yolo_daily_vals = [_to_float(r.get("yolo_daily_tickers")) for r in parsed_rows]
    yolo_weekly_vals = [_to_float(r.get("yolo_weekly_tickers")) for r in parsed_rows]
    yolo_age_vals = [
        _to_float(r.get("yolo_age_hours"))
        for r in parsed_rows
        if _to_float(r.get("yolo_age_hours")) is not None
    ]
    delta_daily_new_vals = [_to_float(r.get("delta_daily_new")) for r in parsed_rows]
    delta_daily_lost_vals = [_to_float(r.get("delta_daily_lost")) for r in parsed_rows]
    delta_weekly_new_vals = [_to_float(r.get("delta_weekly_new")) for r in parsed_rows]
    delta_weekly_lost_vals = [_to_float(r.get("delta_weekly_lost")) for r in parsed_rows]

    def _compact(values: list[float | None]) -> list[float]:
        return [float(v) for v in values if isinstance(v, (int, float))]

    newest_generated_ts = next((r.get("generated_ts") for r in parsed_rows if r.get("generated_ts")), None)
    oldest_generated_ts = next(
        (r.get("generated_ts") for r in reversed(parsed_rows) if r.get("generated_ts")), None
    )

    missing_examples: list[dict[str, Any]] = []
    for r in rows:
        if r.get("parse_error"):
            missing_examples.append(
                {
                    "file": r.get("file"),
                    "generated_ts": r.get("generated_ts"),
                    "reason": "parse_error",
                }
            )
        elif not r.get("yolo_present"):
            missing_examples.append(
                {
                    "file": r.get("file"),
                    "generated_ts": r.get("generated_ts"),
                    "reason": "no_yolo_rows",
                }
            )
        if len(missing_examples) >= 10:
            break

    summary = {
        "files_total": len(files),
        "reports_scanned": len(rows),
        "parsed_reports": len(parsed_rows),
        "parse_error_reports": len(rows) - len(parsed_rows),
        "yolo_present_reports": yolo_present_reports,
        "yolo_missing_reports": yolo_missing_reports,
        "yolo_presence_rate_pct": yolo_presence_rate_pct,
        "newest_missing_streak": newest_missing_streak,
        "longest_missing_streak": longest_missing_streak,
        "newest_generated_ts": newest_generated_ts,
        "oldest_generated_ts": oldest_generated_ts,
        "avg_yolo_rows_total": _avg(_compact(yolo_rows_vals)),
        "avg_yolo_tickers_total": _avg(_compact(yolo_tickers_vals)),
        "avg_yolo_daily_tickers": _avg(_compact(yolo_daily_vals)),
        "avg_yolo_weekly_tickers": _avg(_compact(yolo_weekly_vals)),
        "avg_yolo_age_hours": _avg(_compact(yolo_age_vals)),
        "avg_delta_daily_new": _avg(_compact(delta_daily_new_vals)),
        "avg_delta_daily_lost": _avg(_compact(delta_daily_lost_vals)),
        "avg_delta_weekly_new": _avg(_compact(delta_weekly_new_vals)),
        "avg_delta_weekly_lost": _avg(_compact(delta_weekly_lost_vals)),
    }

    return {
        "ok": True,
        "report_dir": str(report_dir),
        "sample_limit": int(limit),
        "summary": summary,
        "missing_examples": missing_examples,
        "rows": rows,
    }


@app.get("/api/admin/usage-summary")
def usage_summary(
    days: int = Query(default=7, ge=1, le=365),
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    if not ANALYTICS_ENABLED:
        return {"ok": True, "analytics_enabled": False, "detail": "Analytics collection is disabled."}
    conn = get_conn()
    try:
        summary = _usage_summary(conn, days=days, limit=limit)
    finally:
        conn.close()
    summary["analytics_enabled"] = True
    return summary


@app.get("/api/admin/pipeline-status")
def pipeline_status(log_lines: int = Query(default=120, ge=20, le=1000)) -> dict[str, Any]:
    """Return current pipeline phase inferred from run logs + latest ingest run."""
    snap = _pipeline_status_snapshot(log_lines=log_lines)
    resume_candidate = _post_ingest_resume_candidate(
        latest_run=snap.get("latest_run"),
        pipeline_active=bool(snap.get("active")),
    )
    return {
        "ok": True,
        **snap,
        "post_ingest_resume": resume_candidate,
    }


def _daily_report_response(
    *,
    limit: int,
    include_markdown: bool,
    include_internal_paths: bool,
    include_admin_log_hints: bool,
) -> dict[str, Any]:
    """Build daily report payload for admin/public APIs."""
    report_dir = REPORT_DIR
    latest_path, latest_payload = _latest_daily_report_json(report_dir)
    pipeline = _pipeline_status_snapshot(log_lines=120)
    detail: str | None = None
    log_hint = "/api/admin/logs?name=cron" if include_admin_log_hints else "server logs"
    if latest_payload is None:
        detail = "No report file found yet."
    elif pipeline.get("active"):
        detail = (
            "daily_update is still running"
            f" (stage={pipeline.get('stage', 'unknown')}); "
            "generated_ts will advance after report stage completes."
        )
    else:
        latest_run = pipeline.get("latest_run") or {}
        run_finished_ts = parse_iso_utc(latest_run.get("finished_ts")) if latest_run else None
        generated_ts = parse_iso_utc((latest_payload or {}).get("generated_ts"))
        if run_finished_ts is not None:
            if generated_ts is None:
                detail = (
                    "Latest ingest run finished, but latest report JSON has no generated_ts. "
                    f"Check {log_hint} for [REPORT] errors."
                )
            elif generated_ts < (run_finished_ts - dt.timedelta(seconds=60)):
                detail = (
                    "Latest ingest run finished at "
                    f"{run_finished_ts.replace(microsecond=0).isoformat()}, "
                    "but report generated_ts is still "
                    f"{generated_ts.replace(microsecond=0).isoformat()}. "
                    f"Report output is stale; check {log_hint} for [REPORT] errors."
                )
        email_block = latest_payload.get("email", {}) if isinstance(latest_payload, dict) else {}
        if detail is None and isinstance(email_block, dict):
            attempted = bool(email_block.get("attempted"))
            sent = bool(email_block.get("sent"))
            if attempted and not sent:
                error_msg = str(email_block.get("error") or "unknown SMTP error")
                detail = f"Report generated, but email delivery failed: {error_msg}"
    latest_md_path = report_dir / "daily_report_latest.md"
    md_text = ""
    if include_markdown and latest_md_path.exists():
        try:
            md_text = latest_md_path.read_text(encoding="utf-8")
        except Exception:
            md_text = ""

    history = _daily_report_history(report_dir, limit=limit)
    if not include_internal_paths:
        for row in history:
            row.pop("path", None)

    payload = {
        "ok": latest_payload is not None,
        "latest": latest_payload or {},
        "history": history,
        "detail": detail,
        "pipeline": {
            "active": pipeline.get("active"),
            "stage": pipeline.get("stage"),
            "latest_run": pipeline.get("latest_run"),
            "run_log_path": pipeline.get("run_log_path"),
        },
        "latest_markdown": md_text,
    }
    if include_internal_paths:
        payload["report_dir"] = str(report_dir)
        payload["latest_file"] = str(latest_path) if latest_path else None
    else:
        payload["pipeline"] = {
            "active": pipeline.get("active"),
            "stage": pipeline.get("stage"),
        }
    return payload


@app.get("/api/daily-report")
def public_daily_report(limit: int = Query(default=20, ge=1, le=200), include_markdown: bool = Query(default=False)) -> dict[str, Any]:
    """Return latest generated daily report for UI without admin auth."""
    return _daily_report_response(
        limit=limit,
        include_markdown=include_markdown,
        include_internal_paths=False,
        include_admin_log_hints=False,
    )


@app.get("/api/earnings-calendar")
def earnings_calendar(
    days: int = Query(default=21, ge=1, le=90),
    limit: int = Query(default=200, ge=1, le=1000),
    tickers: str | None = Query(default=None),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        _, latest_report = _latest_daily_report_json(REPORT_DIR)
        market_date = dt.datetime.now(ZoneInfo(_MARKET_TZ_NAME)).date()
        if isinstance(latest_report, dict):
            session = latest_report.get("market_session") or {}
            raw_market_date = str(session.get("market_date") or "").strip()
            try:
                if raw_market_date:
                    market_date = dt.date.fromisoformat(raw_market_date)
            except ValueError:
                pass
        requested = {
            str(token or "").strip().upper()
            for token in str(tickers or "").split(",")
            if str(token or "").strip()
        }
        setup_lookup = {}
        if isinstance(latest_report, dict):
            setup_lookup = (((latest_report.get("signals") or {}).get("setup_quality_lookup")) or {})
        payload = build_earnings_calendar_payload(
            conn,
            market_date=market_date,
            days=days,
            limit=limit,
            tickers=requested,
            setup_map=setup_lookup,
        )
        payload["report_generated_ts"] = (latest_report or {}).get("generated_ts") if isinstance(latest_report, dict) else None
        return payload
    finally:
        conn.close()


@app.get("/api/email/chart-preview")
def email_chart_preview(
    ticker: str = Query(..., min_length=1, max_length=16),
    timeframe: str = Query(default="daily"),
    report_ts: str | None = Query(default=None),
    exp: int = Query(...),
    sig: str = Query(..., min_length=32, max_length=128),
) -> Response:
    if not verify_chart_preview_signature(
        ticker=ticker,
        timeframe=timeframe,
        report_ts=report_ts,
        exp=exp,
        sig=sig,
    ):
        raise HTTPException(status_code=403, detail="Invalid chart preview signature")
    conn = get_conn()
    try:
        png = build_email_chart_preview_png(
            conn,
            ticker=ticker,
            timeframe=timeframe,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        conn.close()
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/api/admin/daily-report")
def daily_report(limit: int = Query(default=20, ge=1, le=200), include_markdown: bool = Query(default=False)) -> dict[str, Any]:
    """Return latest generated daily report and recent report files."""
    return _daily_report_response(
        limit=limit,
        include_markdown=include_markdown,
        include_internal_paths=True,
        include_admin_log_hints=True,
    )


@app.get("/api/admin/logs")
def admin_logs(
    name: str = Query(default="cron", pattern="^(cron|update_market_db|yolo|api)$"),
    lines: int = Query(default=80, ge=1, le=800),
) -> dict[str, Any]:
    """Return log tail for one known service log file."""
    path = LOG_PATHS[name]
    return {
        "ok": path.exists(),
        "name": name,
        "path": str(path),
        "tail": _tail_text_file(path, lines=lines, max_bytes=256_000),
    }


@app.get("/api/admin/smtp-health")
def smtp_health() -> dict[str, Any]:
    """Return email delivery config health (without secrets)."""
    smtp = _smtp_settings()
    resend = _resend_settings()
    transport = _email_transport()
    auto_email = str(os.getenv("TRADER_KOO_AUTO_EMAIL", "")).strip().lower() in {"1", "true", "yes"}
    missing: list[str] = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append("TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
        if auto_email and not resend["default_to"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
        if auto_email and not smtp["default_to"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
        if smtp["user"] and not smtp["password"]:
            missing.append("TRADER_KOO_SMTP_PASS")
    return {
        "ok": len(missing) == 0,
        "auto_email_enabled": auto_email,
        "transport": transport,
        "missing": missing,
        "smtp": {
            "host": smtp["host"],
            "port": smtp["port"],
            "security": smtp["security"],
            "timeout_sec": smtp["timeout_sec"],
            "from_email": smtp["from_email"],
            "default_to": smtp["default_to"],
            "has_user": bool(smtp["user"]),
            "has_password": bool(smtp["password"]),
        },
        "resend": {
            "has_api_key": bool(resend["api_key"]),
            "from_email": resend["from_email"],
            "default_to": resend["default_to"],
            "timeout_sec": resend["timeout_sec"],
        },
    }


@app.post("/api/admin/email-latest-report")
def email_latest_report(
    to: str | None = Query(default=None),
    include_markdown: bool = Query(default=True),
    attach_json: bool = Query(default=True),
) -> dict[str, Any]:
    """Send the latest daily report by email via configured transport."""
    smtp = _smtp_settings()
    resend = _resend_settings()
    transport = _email_transport()
    default_to = resend["default_to"] if transport == "resend" else smtp["default_to"]
    recipient = (to or default_to or "").strip()

    missing: list[str] = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append("TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
        if smtp["user"] and not smtp["password"]:
            missing.append("TRADER_KOO_SMTP_PASS")
    if not recipient:
        missing.append("TRADER_KOO_REPORT_EMAIL_TO (or use ?to=...)")
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing email config: {', '.join(missing)}")

    report_dir = REPORT_DIR
    latest_path, latest_payload = _latest_daily_report_json(report_dir)
    if latest_payload is None:
        raise HTTPException(status_code=404, detail=f"No report found in {report_dir}")

    latest_md_path = report_dir / "daily_report_latest.md"
    md_text = ""
    if latest_md_path.exists():
        try:
            md_text = latest_md_path.read_text(encoding="utf-8")
        except Exception:
            md_text = ""

    generated = str(latest_payload.get("generated_ts") or latest_payload.get("generated_at_utc") or latest_payload.get("snapshot_ts") or "").strip()
    if not generated:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    subject = build_report_email_subject(latest_payload)
    text_body, html_body = build_report_email_bodies(
        latest_payload,
        md_text if include_markdown else "",
        app_url=report_email_app_url(),
    )

    from_header = resend["from_email"] if transport == "resend" else smtp["from_email"]
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_header
    message["To"] = recipient
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    if attach_json:
        filename = latest_path.name if latest_path is not None else "daily_report_latest.json"
        json_bytes = json.dumps(latest_payload, indent=2).encode("utf-8")
        message.add_attachment(json_bytes, maintype="application", subtype="json", filename=filename)

    if include_markdown and md_text:
        message.add_attachment(
            md_text.encode("utf-8"),
            maintype="text",
            subtype="markdown",
            filename="daily_report_latest.md",
        )

    try:
        if transport == "resend":
            _send_resend_email(
                subject=subject,
                text=text_body,
                recipient=recipient,
                resend=resend,
                html_body=html_body,
            )
        else:
            _send_smtp_email(message, smtp)
    except Exception as exc:
        LOG.exception("Failed to send daily report email (transport=%s)", transport)
        raise HTTPException(status_code=500, detail=f"Email send failed: {exc}") from exc

    return {
        "ok": True,
        "transport": transport,
        "to": recipient,
        "subject": subject,
        "report_file": str(latest_path) if latest_path else None,
        "smtp_host": smtp["host"],
        "smtp_port": smtp["port"],
        "smtp_security": smtp["security"],
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    global _STATUS_CACHE_AT, _STATUS_CACHE_PAYLOAD
    now = dt.datetime.now(dt.timezone.utc)
    if STATUS_CACHE_TTL_SEC > 0:
        with _STATUS_CACHE_LOCK:
            cached_at = _STATUS_CACHE_AT
            cached_payload = _STATUS_CACHE_PAYLOAD
        if (
            cached_at is not None
            and cached_payload is not None
            and (now - cached_at).total_seconds() < STATUS_CACHE_TTL_SEC
        ):
            return dict(cached_payload)

    rss_now = _current_rss_mb()
    rss_max = _max_rss_mb()
    base: dict[str, Any] = {
        "service": "trader_koo-api",
        "now_utc": now.replace(microsecond=0).isoformat(),
        "db_exists": DB_PATH.exists(),
    }
    if EXPOSE_STATUS_INTERNAL:
        base["db_path"] = str(DB_PATH)
        base["process"] = {
            "pid": os.getpid(),
            "rss_mb": None if rss_now is None else round(rss_now, 2),
            "rss_max_mb": None if rss_max is None else round(rss_max, 2),
            "uptime_sec": int((now - PROCESS_START_UTC).total_seconds()),
        }
    if not DB_PATH.exists():
        return {**base, "ok": False, "error": "Database file not found"}

    conn = get_conn()
    try:
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM price_daily) AS price_rows,
                (SELECT COUNT(*) FROM finviz_fundamentals) AS fundamentals_rows,
                (SELECT COUNT(*) FROM options_iv) AS options_rows,
                (SELECT COUNT(DISTINCT ticker) FROM price_daily) AS tracked_tickers,
                (SELECT MAX(date) FROM price_daily) AS latest_price_date,
                (SELECT MAX(snapshot_ts) FROM finviz_fundamentals) AS latest_fund_snapshot,
                (SELECT MAX(snapshot_ts) FROM options_iv) AS latest_opt_snapshot
            """
        ).fetchone()

        run_row = None
        ticker_status_count = None
        if table_exists(conn, "ingest_runs"):
            run_row = conn.execute(
                """
                SELECT
                    run_id, started_ts, finished_ts, status, tickers_total, tickers_ok, tickers_failed, error_message
                FROM ingest_runs
                ORDER BY started_ts DESC
                LIMIT 1
                """
            ).fetchone()
            if run_row and table_exists(conn, "ingest_ticker_status"):
                ts_row = conn.execute(
                    "SELECT COUNT(*) AS c FROM ingest_ticker_status WHERE run_id = ?",
                    (run_row["run_id"],),
                ).fetchone()
                ticker_status_count = int(ts_row["c"]) if ts_row else 0

        latest_price_date = counts["latest_price_date"] if counts else None
        latest_fund_snapshot = counts["latest_fund_snapshot"] if counts else None
        latest_opt_snapshot = counts["latest_opt_snapshot"] if counts else None

        price_age_days = days_since(latest_price_date, now)
        fund_age_hours = hours_since(latest_fund_snapshot, now)
        opt_age_hours = hours_since(latest_opt_snapshot, now)

        warnings: list[str] = []
        if price_age_days is None or price_age_days > 3:
            warnings.append("price_daily stale")
        if fund_age_hours is None or fund_age_hours > 48:
            warnings.append("finviz_fundamentals stale")

        latest_run = dict(run_row) if run_row is not None else None
        if latest_run and latest_run.get("status") in {"failed"}:
            warnings.append("latest ingest run failed")
        if latest_run:
            if ticker_status_count is not None:
                latest_run["tickers_processed"] = ticker_status_count
            elif latest_run.get("status") in {"ok", "failed"}:
                completed = int(latest_run.get("tickers_ok") or 0) + int(latest_run.get("tickers_failed") or 0)
                latest_run["tickers_processed"] = completed or int(latest_run.get("tickers_total") or 0)

        pipeline_snap = _pipeline_status_snapshot(log_lines=60)
        resume_candidate = _post_ingest_resume_candidate(
            latest_run=latest_run,
            pipeline_active=bool(pipeline_snap.get("active")),
            now_utc=now,
        )
        pipeline_active = bool(pipeline_snap.get("active"))
        pipeline_stage = pipeline_snap.get("stage") or "unknown"
        pipeline_stage_line = pipeline_snap.get("stage_line")
        if latest_run and latest_run.get("status") == "running" and not pipeline_snap.get("running_stale"):
            pipeline_active = True
            if pipeline_stage in {"unknown", "idle"}:
                pipeline_stage = "ingest"
        if latest_run and latest_run.get("status") == "running" and pipeline_snap.get("running_stale"):
            warnings.append("latest ingest run appears stale-running")
        if resume_candidate:
            warnings.append("post-ingest yolo/report recovery recommended")

        activity = {
            "tracked_tickers": counts["tracked_tickers"] if counts else 0,
            "tickers_processed": int((latest_run or {}).get("tickers_processed") or 0),
            "tickers_total": int((latest_run or {}).get("tickers_total") or 0),
            "tickers_ok": int((latest_run or {}).get("tickers_ok") or 0),
            "tickers_failed": int((latest_run or {}).get("tickers_failed") or 0),
            "price_rows": counts["price_rows"] if counts else 0,
            "fundamentals_rows": counts["fundamentals_rows"] if counts else 0,
            "options_rows": counts["options_rows"] if counts else 0,
        }
        service_meta = {
            "service": "trader_koo-api",
            "contract": "control-center-v1",
            "contract_version": CONTROL_CENTER_CONTRACT_VERSION,
            "auth_header": "X-API-Key",
            "admin_auth_configured": bool(API_KEY),
        }
        if STATUS_BASE_URL:
            service_meta["base_url"] = STATUS_BASE_URL
        if STATUS_APP_URL:
            service_meta["app_url"] = STATUS_APP_URL
        if STATUS_REPO_URL:
            service_meta["repo_url"] = STATUS_REPO_URL

        payload = {
            **base,
            "ok": len(warnings) == 0,
            "warnings": warnings,
            "latest_run": latest_run,
            "pipeline_active": pipeline_active,
            "pipeline_stage": pipeline_stage,
            "service_meta": service_meta,
            "pipeline": {
                "active": pipeline_active,
                "stage": pipeline_stage,
                "stage_line": pipeline_stage_line,
                "stage_line_ts": pipeline_snap.get("stage_line_ts"),
                "stage_age_sec": pipeline_snap.get("stage_age_sec"),
                "stale_timeout_sec": pipeline_snap.get("stale_timeout_sec"),
                "stale_inference": pipeline_snap.get("stale_inference"),
                "running_age_sec": pipeline_snap.get("running_age_sec"),
                "running_stale_min": pipeline_snap.get("running_stale_min"),
                "running_stale": pipeline_snap.get("running_stale"),
                "last_completed_stage": pipeline_snap.get("last_completed_stage"),
                "last_completed_status": pipeline_snap.get("last_completed_status"),
                "last_completed_line": pipeline_snap.get("last_completed_line"),
                "last_completed_ts": pipeline_snap.get("last_completed_ts"),
                "post_ingest_resume": resume_candidate,
                "run_log_path": str(RUN_LOG_PATH),
            },
            "freshness": {
                "price_age_days": None if price_age_days is None else round(price_age_days, 2),
                "fund_age_hours": None if fund_age_hours is None else round(fund_age_hours, 2),
                "opt_age_hours": None if opt_age_hours is None else round(opt_age_hours, 2),
            },
            "counts": {
                "tracked_tickers": counts["tracked_tickers"] if counts else 0,
                "price_rows": counts["price_rows"] if counts else 0,
                "fundamentals_rows": counts["fundamentals_rows"] if counts else 0,
                "options_rows": counts["options_rows"] if counts else 0,
            },
            "activity": activity,
            "latest_data": {
                "price_date": latest_price_date,
                "fund_snapshot": latest_fund_snapshot,
                "options_snapshot": latest_opt_snapshot,
            },
        }
        if STATUS_CACHE_TTL_SEC > 0:
            with _STATUS_CACHE_LOCK:
                _STATUS_CACHE_AT = now
                _STATUS_CACHE_PAYLOAD = payload
        return payload
    finally:
        conn.close()


@app.get("/api/tickers")
def tickers(limit: int = Query(default=200, ge=1, le=2000)) -> dict[str, Any]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT ticker
            FROM price_daily
            ORDER BY ticker
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out = [r["ticker"] for r in rows]
        return {"count": len(out), "tickers": out}
    finally:
        conn.close()


@app.get("/api/dashboard/{ticker}")
def dashboard(ticker: str, months: int = Query(default=3, ge=0, le=240)) -> dict[str, Any]:
    conn = get_conn()
    try:
        return build_dashboard_payload(conn, ticker=ticker, months=months)
    finally:
        conn.close()


@app.get("/api/yolo/{ticker}")
def yolo_ticker(ticker: str) -> dict[str, Any]:
    """Return stored YOLO pattern detections for a ticker."""
    conn = get_conn()
    try:
        t = ticker.upper().strip()
        patterns = get_yolo_patterns(conn, t)
        return {"ticker": t, "count": len(patterns), "patterns": patterns}
    finally:
        conn.close()


@app.get("/api/opportunities")
def opportunities(
    limit: int = Query(default=500, ge=1, le=1000),
    min_discount: float = Query(default=10.0),
    max_peg: float = Query(default=2.0),
    view: str = Query(default="all", pattern="^(undervalued|overvalued|all)$"),
    overvalued_threshold: float = Query(default=-10.0),
) -> dict[str, Any]:
    conn = get_conn()
    try:
        snapshot_ts, universe_count = select_fund_snapshot(conn, min_complete_tickers=400)
        if snapshot_ts is None:
            return {"snapshot_ts": None, "count": 0, "rows": []}

        rows = conn.execute(
            """
            SELECT
                ticker,
                price,
                pe,
                peg,
                eps_ttm,
                eps_growth_5y,
                target_price,
                discount_pct,
                target_reason
            FROM finviz_fundamentals
            WHERE snapshot_ts = ?
            """,
            (snapshot_ts,),
        ).fetchall()

        all_rows = [dict(r) for r in rows]
        source_counts: dict[str, int] = {}
        enriched: list[dict[str, Any]] = []
        for r in all_rows:
            discount = r.get("discount_pct")
            peg = r.get("peg")
            reason = str(r.get("target_reason") or "")
            if reason.startswith("FINVIZ_"):
                target_source = "analyst_target"
            elif reason.startswith("MODEL_"):
                target_source = "model_eps_pe"
            else:
                target_source = "other"
            source_counts[target_source] = source_counts.get(target_source, 0) + 1

            valuation_label = "fair"
            if isinstance(discount, (int, float)):
                if discount >= 20:
                    valuation_label = "deep_undervalued"
                elif discount >= 10:
                    valuation_label = "undervalued"
                elif discount <= -20:
                    valuation_label = "deep_overvalued"
                elif discount <= -10:
                    valuation_label = "overvalued"

            if isinstance(peg, (int, float)) and peg > 3.0 and valuation_label in {"fair", "undervalued"}:
                valuation_label = "high_peg"

            r["target_source"] = target_source
            r["valuation_label"] = valuation_label
            enriched.append(r)

        def include_row(r: dict[str, Any]) -> bool:
            discount = r.get("discount_pct")
            peg = r.get("peg")
            if view == "all":
                return True
            if not isinstance(discount, (int, float)):
                return False
            if view == "undervalued":
                if not isinstance(peg, (int, float)) or peg <= 0 or peg > max_peg:
                    return False
                return discount >= min_discount
            if view == "overvalued":
                return discount <= overvalued_threshold
            return False

        filtered = [r for r in enriched if include_row(r)]
        if view == "overvalued":
            filtered.sort(key=lambda r: (r.get("discount_pct", 0.0), -(r.get("peg") or 0.0)))
        elif view == "all":
            filtered.sort(key=lambda r: str(r.get("ticker") or ""))
        else:
            filtered.sort(key=lambda r: (-(r.get("discount_pct") or 0.0), (r.get("peg") or 9999.0)))

        eligible_count = len(filtered)
        out_rows = filtered[:limit]

        return {
            "snapshot_ts": snapshot_ts,
            "count": len(out_rows),
            "eligible_count": eligible_count,
            "universe_count": universe_count,
            "rows": out_rows,
            "source_counts": source_counts,
            "filters": {
                "view": view,
                "min_discount": min_discount,
                "max_peg": max_peg,
                "overvalued_threshold": overvalued_threshold,
                "limit": limit,
            },
            "filter_help": {
                "view": "all: full universe, undervalued: upside candidates, overvalued: downside candidates",
                "min_discount": "Undervalued threshold: minimum upside discount to target (%)",
                "max_peg": "Valuation cap used in undervalued mode",
                "overvalued_threshold": "Overvalued threshold: maximum discount (negative means above target)",
                "limit": "Maximum rows returned",
            },
        }
    finally:
        conn.close()


@app.get("/api/market-summary")
def market_summary(days: int = Query(90, ge=7, le=365)) -> Any:
    """Public endpoint — SPY & QQQ price history for the portfolio chart. No API key required."""
    tickers = ["SPY", "QQQ"]
    conn = get_conn()
    try:
        result: dict[str, Any] = {"as_of": None, "tickers": {}}
        for ticker in tickers:
            rows = conn.execute(
                """
                SELECT date, CAST(close AS REAL) AS close
                FROM price_daily
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (ticker, days),
            ).fetchall()
            if not rows:
                result["tickers"][ticker] = None
                continue
            history = [{"date": r[0], "close": round(float(r[1]), 2)} for r in reversed(rows)]
            latest_close = history[-1]["close"]
            prev_close = history[-2]["close"] if len(history) >= 2 else latest_close
            first_close = history[0]["close"]
            change_pct_1d = round((latest_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0
            change_pct_period = round((latest_close - first_close) / first_close * 100, 2) if first_close else 0.0
            result["tickers"][ticker] = {
                "price": latest_close,
                "change_pct_1d": change_pct_1d,
                "change_pct_period": change_pct_period,
                "history": history,
            }
            if result["as_of"] is None:
                result["as_of"] = history[-1]["date"]
        return result
    finally:
        conn.close()

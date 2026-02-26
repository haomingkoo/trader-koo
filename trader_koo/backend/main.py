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
from fastapi.responses import FileResponse, JSONResponse

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trader_koo.data.schema import ensure_ohlcv_schema
from trader_koo.cv.compare import HybridCVCompareConfig, compare_hybrid_vs_cv
from trader_koo.cv.proxy_patterns import CVProxyConfig, detect_cv_proxy_patterns
from trader_koo.features.candle_patterns import CandlePatternConfig, detect_candlestick_patterns
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
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
PROCESS_START_UTC = dt.datetime.now(dt.timezone.utc)


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
_STATUS_CACHE_LOCK = threading.Lock()
_STATUS_CACHE_AT: dt.datetime | None = None
_STATUS_CACHE_PAYLOAD: dict[str, Any] | None = None
_MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")
try:
    MARKET_TZ = ZoneInfo(_MARKET_TZ_NAME)
except Exception:
    MARKET_TZ = dt.timezone.utc
MARKET_CLOSE_HOUR = min(23, max(0, int(os.getenv("TRADER_KOO_MARKET_CLOSE_HOUR", "16"))))
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


def _run_daily_update() -> None:
    script = SCRIPTS_DIR / "daily_update.sh"
    started = dt.datetime.now(dt.timezone.utc)
    rss_before = _current_rss_mb()
    _append_run_log("SCHED", f"daily_update invoked rss_before_mb={_fmt_mb(rss_before)}")
    LOG.info(
        "Scheduler: starting daily_update.sh (rss_before_mb=%s, run_log=%s)",
        _fmt_mb(rss_before),
        RUN_LOG_PATH,
    )
    result = subprocess.run(["bash", str(script)], capture_output=False)
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    rss_after = _current_rss_mb()
    delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    if result.returncode == 0:
        _append_run_log(
            "SCHED",
            f"daily_update completed rc={result.returncode} sec={elapsed:.1f} rss_after_mb={_fmt_mb(rss_after)} rss_delta_mb={_fmt_mb(delta)}",
        )
        LOG.info(
            "Scheduler: daily_update.sh completed OK (rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s)",
            result.returncode,
            elapsed,
            _fmt_mb(rss_after),
            _fmt_mb(delta),
        )
    else:
        _append_run_log(
            "SCHED",
            f"daily_update failed rc={result.returncode} sec={elapsed:.1f} rss_after_mb={_fmt_mb(rss_after)} rss_delta_mb={_fmt_mb(delta)}",
        )
        LOG.error(
            "Scheduler: daily_update.sh failed (rc=%d, sec=%.1f, rss_after_mb=%s, rss_delta_mb=%s, run_log=%s)",
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
    _scheduler.start()
    LOG.info("Scheduler started — daily_update: 22:00 UTC Mon–Fri | weekly_yolo: 00:30 UTC Sat")
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
    """Require X-API-Key on /api/admin/* routes only."""
    if API_KEY:
        path = request.url.path
        if path.startswith(ADMIN_API_PREFIX):
            provided = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(provided, API_KEY):
                xff = request.headers.get("x-forwarded-for", "")
                client_ip = (xff.split(",")[0].strip() if xff else "") or (request.client.host if request.client else "-")
                ua = request.headers.get("user-agent", "-")
                referer = request.headers.get("referer", "-")
                LOG.warning(
                    "Unauthorized request blocked method=%s path=%s client_ip=%s user_agent=%s referer=%s",
                    request.method,
                    path,
                    client_ip,
                    ua,
                    referer,
                )
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
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


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"DB not found at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


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


def get_yolo_patterns(conn: sqlite3.Connection, ticker: str) -> list[dict[str, Any]]:
    if not table_exists(conn, "yolo_patterns"):
        return []
    rows = conn.execute(
        """
        SELECT timeframe, pattern, confidence, x0_date, x1_date, y0, y1, lookback_days, as_of_date, detected_ts
        FROM yolo_patterns
        WHERE ticker = ?
        ORDER BY timeframe, confidence DESC
        """,
        (ticker,),
    ).fetchall()
    return [dict(r) for r in rows]


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
        # Fallback: infer active phase from recent markers even if the explicit
        # [START] line is outside the retained tail window.
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


def _pipeline_status_snapshot(log_lines: int = 160) -> dict[str, Any]:
    tail = _tail_text_file(RUN_LOG_PATH, lines=log_lines, max_bytes=256_000)
    inferred = _infer_pipeline_from_log_tail(tail)
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
    elif active and stage != "ingest" and stage_age_sec is not None and stage_age_sec > PIPELINE_STALE_SEC:
        # Prevent stale log tails from pinning pipeline_active=True forever.
        stale_inference = True
        active = False
        stage = "idle"
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
        "yolo_patterns": get_yolo_patterns(conn, ticker),
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
    return {"ok": db_exists, "db_path": str(DB_PATH), "db_exists": db_exists}


@app.post("/api/admin/trigger-update")
def trigger_update() -> dict[str, Any]:
    """Trigger daily_update.sh immediately (runs in background via scheduler)."""
    pipeline = _pipeline_status_snapshot(log_lines=120)
    if pipeline["active"]:
        stage = pipeline.get("stage", "unknown")
        return {
            "ok": False,
            "message": f"daily_update already running (stage={stage})",
            "stage": stage,
            "latest_run": pipeline.get("latest_run"),
            "run_log_path": pipeline.get("run_log_path"),
        }
    job = _scheduler.get_job("daily_update")
    if job is None:
        raise HTTPException(status_code=500, detail="Scheduler job not found")
    job.modify(next_run_time=dt.datetime.now(dt.timezone.utc))
    return {
        "ok": True,
        "message": "daily_update triggered — check /data/logs/cron_daily.log",
        "stage": "queued",
        "run_log_path": str(RUN_LOG_PATH),
    }


_yolo_seed_thread: threading.Thread | None = None


@app.post("/api/admin/run-yolo-seed")
def run_yolo_seed(timeframe: str = "both") -> dict[str, Any]:
    """Trigger full YOLO seed for all tickers in background (no --only-new)."""
    global _yolo_seed_thread
    if _yolo_seed_thread and _yolo_seed_thread.is_alive():
        return {"ok": False, "message": "Seed already running — wait for it to finish"}
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_yolo_patterns.py"
    cmd = [
        sys.executable, str(script),
        "--db-path", str(DB_PATH),
        "--timeframe", timeframe,
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
        "message": f"YOLO seed started (timeframe={timeframe}) — tail /data/logs/yolo_patterns.log or check Railway logs",
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


@app.get("/api/admin/pipeline-status")
def pipeline_status(log_lines: int = Query(default=120, ge=20, le=1000)) -> dict[str, Any]:
    """Return current pipeline phase inferred from run logs + latest ingest run."""
    snap = _pipeline_status_snapshot(log_lines=log_lines)
    return {
        "ok": True,
        **snap,
    }


@app.get("/api/admin/daily-report")
def daily_report(limit: int = Query(default=20, ge=1, le=200), include_markdown: bool = Query(default=False)) -> dict[str, Any]:
    """Return latest generated daily report and recent report files."""
    report_dir = REPORT_DIR
    latest_path, latest_payload = _latest_daily_report_json(report_dir)
    pipeline = _pipeline_status_snapshot(log_lines=120)
    detail: str | None = None
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
                    "Check /api/admin/logs?name=cron for [REPORT] errors."
                )
            elif generated_ts < (run_finished_ts - dt.timedelta(seconds=60)):
                detail = (
                    "Latest ingest run finished at "
                    f"{run_finished_ts.replace(microsecond=0).isoformat()}, "
                    "but report generated_ts is still "
                    f"{generated_ts.replace(microsecond=0).isoformat()}. "
                    "Report output is stale; check /api/admin/logs?name=cron for [REPORT] errors."
                )
    latest_md_path = report_dir / "daily_report_latest.md"
    md_text = ""
    if include_markdown and latest_md_path.exists():
        try:
            md_text = latest_md_path.read_text(encoding="utf-8")
        except Exception:
            md_text = ""
    return {
        "ok": latest_payload is not None,
        "report_dir": str(report_dir),
        "latest_file": str(latest_path) if latest_path else None,
        "latest": latest_payload or {},
        "history": _daily_report_history(report_dir, limit=limit),
        "detail": detail,
        "pipeline": {
            "active": pipeline.get("active"),
            "stage": pipeline.get("stage"),
            "latest_run": pipeline.get("latest_run"),
            "run_log_path": pipeline.get("run_log_path"),
        },
        "latest_markdown": md_text,
    }


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


@app.post("/api/admin/email-latest-report")
def email_latest_report(
    to: str | None = Query(default=None),
    include_markdown: bool = Query(default=True),
    attach_json: bool = Query(default=True),
) -> dict[str, Any]:
    """Send the latest daily report by email via SMTP."""
    smtp = _smtp_settings()
    recipient = (to or smtp["default_to"] or "").strip()

    missing: list[str] = []
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

    generated = (
        str(latest_payload.get("generated_at_utc") or latest_payload.get("snapshot_ts") or "").strip()
        or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    )

    status_block = latest_payload.get("status", {}) if isinstance(latest_payload, dict) else {}
    yolo_block = latest_payload.get("yolo", {}) if isinstance(latest_payload, dict) else {}
    counts_block = latest_payload.get("counts", {}) if isinstance(latest_payload, dict) else {}

    lines = [
        f"trader_koo daily report ({generated})",
        "",
        "Quick summary:",
        f"- tracked_tickers: {counts_block.get('tracked_tickers', 'n/a')}",
        f"- price_rows: {counts_block.get('price_rows', 'n/a')}",
        f"- fundamentals_rows: {counts_block.get('fundamentals_rows', 'n/a')}",
        f"- yolo_rows_total: {yolo_block.get('rows_total', 'n/a')}",
        f"- yolo_tickers_total: {yolo_block.get('tickers_total', 'n/a')}",
        f"- latest_ingest_status: {status_block.get('latest_run_status', 'n/a')}",
        "",
    ]
    if include_markdown and md_text:
        lines += ["Full markdown report:", "", md_text]
    else:
        lines += ["Use /api/admin/daily-report?include_markdown=true to fetch full markdown."]

    message = EmailMessage()
    message["Subject"] = f"[trader_koo] Daily report {generated}"
    message["From"] = smtp["from_email"]
    message["To"] = recipient
    message.set_content("\n".join(lines))

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
        _send_smtp_email(message, smtp)
    except Exception as exc:
        LOG.exception("Failed to send daily report email")
        raise HTTPException(status_code=500, detail=f"Email send failed: {exc}") from exc

    return {
        "ok": True,
        "to": recipient,
        "subject": message["Subject"],
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
    base = {
        "service": "trader_koo-api",
        "now_utc": now.replace(microsecond=0).isoformat(),
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
        "process": {
            "pid": os.getpid(),
            "rss_mb": None if rss_now is None else round(rss_now, 2),
            "rss_max_mb": None if rss_max is None else round(rss_max, 2),
            "uptime_sec": int((now - PROCESS_START_UTC).total_seconds()),
        },
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
        pipeline_active = bool(pipeline_snap.get("active"))
        pipeline_stage = pipeline_snap.get("stage") or "unknown"
        pipeline_stage_line = pipeline_snap.get("stage_line")
        if latest_run and latest_run.get("status") == "running" and not pipeline_snap.get("running_stale"):
            pipeline_active = True
            if pipeline_stage in {"unknown", "idle"}:
                pipeline_stage = "ingest"
        if latest_run and latest_run.get("status") == "running" and pipeline_snap.get("running_stale"):
            warnings.append("latest ingest run appears stale-running")

        payload = {
            **base,
            "ok": len(warnings) == 0,
            "warnings": warnings,
            "latest_run": latest_run,
            "pipeline_active": pipeline_active,
            "pipeline_stage": pipeline_stage,
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

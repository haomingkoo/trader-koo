from __future__ import annotations

import datetime as dt
import logging
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
LOG = logging.getLogger("trader_koo.api")
if not LOG.handlers:
    logging.basicConfig(
        level=os.getenv("TRADER_KOO_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
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


def _run_daily_update() -> None:
    script = SCRIPTS_DIR / "daily_update.sh"
    LOG.info("Scheduler: starting daily_update.sh")
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    if result.returncode == 0:
        LOG.info("Scheduler: daily_update.sh completed OK")
    else:
        LOG.error("Scheduler: daily_update.sh failed (rc=%d): %s", result.returncode, result.stderr[-500:])


_scheduler = BackgroundScheduler(timezone="UTC")
_scheduler.add_job(
    _run_daily_update,
    CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone="UTC"),
    id="daily_update",
    replace_existing=True,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _scheduler.start()
    LOG.info("Scheduler started — daily_update runs at 22:00 UTC Mon–Fri")
    yield
    _scheduler.shutdown(wait=False)


_ALLOWED_ORIGIN = os.getenv("TRADER_KOO_ALLOWED_ORIGIN", "*")

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
    """Require X-API-Key on all /api/* routes except /api/health."""
    if API_KEY:
        path = request.url.path
        if path.startswith("/api/") and path not in ("/api/health", "/api/config"):
            provided = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(provided, API_KEY):
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


def hours_since(ts: str | None, now: dt.datetime) -> float | None:
    parsed = parse_iso_utc(ts)
    if parsed is None:
        return None
    return (now - parsed).total_seconds() / 3600.0


def days_since(date_str: str | None, now: dt.datetime) -> float | None:
    if not date_str:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(date_str)).replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return (now - parsed).total_seconds() / 86400.0


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
    """Public client config — returns API key so the JS frontend can authenticate."""
    return {"api_key": API_KEY}


@app.get("/api/health")
def health() -> dict[str, Any]:
    db_exists = DB_PATH.exists()
    return {"ok": db_exists, "db_path": str(DB_PATH), "db_exists": db_exists}


@app.post("/api/admin/trigger-update")
def trigger_update() -> dict[str, Any]:
    """Trigger daily_update.sh immediately (runs in background via scheduler)."""
    job = _scheduler.get_job("daily_update")
    if job is None:
        raise HTTPException(status_code=500, detail="Scheduler job not found")
    job.modify(next_run_time=dt.datetime.now(dt.timezone.utc))
    return {"ok": True, "message": "daily_update triggered — check /data/logs/cron_daily.log"}


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


@app.get("/api/status")
def status() -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    base = {
        "service": "trader_koo-api",
        "now_utc": now.replace(microsecond=0).isoformat(),
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
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
            if run_row and run_row["status"] == "running" and table_exists(conn, "ingest_ticker_status"):
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
        if latest_run and ticker_status_count is not None:
            latest_run["tickers_processed"] = ticker_status_count

        return {
            **base,
            "ok": len(warnings) == 0,
            "warnings": warnings,
            "latest_run": latest_run,
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

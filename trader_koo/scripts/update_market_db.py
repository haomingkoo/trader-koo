from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from io import StringIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional

import finviz
import pandas as pd
import requests
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trader_koo.data.schema import ensure_ohlcv_schema

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = str((PROJECT_DIR / "data" / "trader_koo.db").resolve())
DEFAULT_LOG_PATH = str((PROJECT_DIR / "data" / "logs" / "update_market_db.log").resolve())


DEFAULT_TARGET_PE = 21.0
DEFAULT_MAX_SECS_PER_TICKER = float(os.getenv("TRADER_KOO_INGEST_MAX_SECS_PER_TICKER", "120"))
DEFAULT_PRICE_TIMEOUT_SEC = float(os.getenv("TRADER_KOO_PRICE_TIMEOUT_SEC", "25"))
LOG = logging.getLogger("trader_koo.ingest")


def setup_logging(level: str, log_file: str | None) -> None:
    LOG.handlers.clear()
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOG.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        LOG.addHandler(file_handler)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextmanager
def _ticker_timeout(seconds: float):
    """Raise TimeoutError if one ticker takes too long on Unix platforms."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"ticker exceeded timeout ({seconds:.1f}s)")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


def parse_iso_utc(value: str | None) -> Optional[dt.datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s in {"", "-", "nan", "None"}:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def to_percent(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace("%", "")
    base = to_float(s)
    return None if base is None else base / 100.0


def parse_tickers(raw: str) -> list[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return tickers


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS finviz_fundamentals (
            snapshot_ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            price REAL,
            pe REAL,
            peg REAL,
            eps_ttm REAL,
            eps_growth_5y REAL,
            target_price REAL,
            target_reason TEXT,
            discount_pct REAL,
            raw_json TEXT,
            PRIMARY KEY (snapshot_ts, ticker)
        );

        CREATE TABLE IF NOT EXISTS price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        );

        CREATE TABLE IF NOT EXISTS options_iv (
            snapshot_ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            last_price REAL,
            bid REAL,
            ask REAL,
            implied_vol REAL,
            open_interest REAL,
            volume REAL,
            moneyness REAL,
            PRIMARY KEY (snapshot_ts, ticker, expiration, option_type, strike)
        );

        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id TEXT PRIMARY KEY,
            started_ts TEXT NOT NULL,
            finished_ts TEXT,
            status TEXT NOT NULL,
            tickers_total INTEGER NOT NULL,
            tickers_ok INTEGER DEFAULT 0,
            tickers_failed INTEGER DEFAULT 0,
            args_json TEXT,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS ingest_ticker_status (
            run_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            started_ts TEXT NOT NULL,
            finished_ts TEXT,
            status TEXT NOT NULL,
            fundamentals_refreshed INTEGER DEFAULT 0,
            price_fetch_start TEXT,
            price_rows INTEGER DEFAULT 0,
            options_refreshed INTEGER DEFAULT 0,
            options_rows INTEGER DEFAULT 0,
            message TEXT,
            error_message TEXT,
            PRIMARY KEY (run_id, ticker),
            FOREIGN KEY (run_id) REFERENCES ingest_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_fund_ticker_snap ON finviz_fundamentals(ticker, snapshot_ts);
        CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_daily(ticker, date);
        CREATE INDEX IF NOT EXISTS idx_options_ticker_snap ON options_iv(ticker, snapshot_ts);
        CREATE INDEX IF NOT EXISTS idx_ingest_runs_started ON ingest_runs(started_ts);
        CREATE INDEX IF NOT EXISTS idx_ingest_ticker_status_run ON ingest_ticker_status(run_id, status);
        """
    )
    conn.commit()


def get_latest_snapshot_ts(conn: sqlite3.Connection, table: str, ticker: str) -> Optional[str]:
    allowed = {"finviz_fundamentals", "options_iv"}
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    row = conn.execute(
        f"SELECT MAX(snapshot_ts) AS snap FROM {table} WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if row is None:
        return None
    return row[0]


def get_latest_price_date(conn: sqlite3.Connection, ticker: str) -> Optional[str]:
    row = conn.execute(
        "SELECT MAX(date) AS max_date FROM price_daily WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if row is None:
        return None
    return row[0]


def should_refresh(last_ts: str | None, min_interval_hours: float, now: dt.datetime) -> bool:
    if min_interval_hours <= 0:
        return True
    prev = parse_iso_utc(last_ts)
    if prev is None:
        return True
    return (now - prev).total_seconds() >= min_interval_hours * 3600.0


def resolve_price_fetch_start(
    conn: sqlite3.Connection,
    ticker: str,
    default_start: str,
    lookback_days: int,
    full_refresh: bool,
) -> str:
    if full_refresh:
        return default_start
    max_date = get_latest_price_date(conn, ticker)
    if not max_date:
        return default_start
    try:
        latest = dt.date.fromisoformat(max_date)
        default_dt = dt.date.fromisoformat(default_start)
    except ValueError:
        return default_start
    start = latest - dt.timedelta(days=max(1, lookback_days))
    if start < default_dt:
        start = default_dt
    return start.isoformat()


def begin_run(conn: sqlite3.Connection, run_id: str, started_ts: str, tickers_total: int, args: argparse.Namespace) -> None:
    conn.execute(
        """
        INSERT INTO ingest_runs (
            run_id, started_ts, status, tickers_total, args_json
        ) VALUES (?, ?, 'running', ?, ?)
        """,
        (run_id, started_ts, tickers_total, json.dumps(vars(args), sort_keys=True)),
    )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    finished_ts: str,
    status: str,
    tickers_ok: int,
    tickers_failed: int,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE ingest_runs
        SET finished_ts = ?,
            status = ?,
            tickers_ok = ?,
            tickers_failed = ?,
            error_message = ?
        WHERE run_id = ?
        """,
        (finished_ts, status, tickers_ok, tickers_failed, error_message, run_id),
    )
    conn.commit()


def upsert_ticker_status(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    ticker: str,
    started_ts: str,
    finished_ts: str,
    status: str,
    fundamentals_refreshed: int,
    price_fetch_start: str | None,
    price_rows: int,
    options_refreshed: int,
    options_rows: int,
    message: str | None,
    error_message: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingest_ticker_status (
            run_id, ticker, started_ts, finished_ts, status,
            fundamentals_refreshed, price_fetch_start, price_rows,
            options_refreshed, options_rows, message, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, ticker)
        DO UPDATE SET
            finished_ts = excluded.finished_ts,
            status = excluded.status,
            fundamentals_refreshed = excluded.fundamentals_refreshed,
            price_fetch_start = excluded.price_fetch_start,
            price_rows = excluded.price_rows,
            options_refreshed = excluded.options_refreshed,
            options_rows = excluded.options_rows,
            message = excluded.message,
            error_message = excluded.error_message
        """,
        (
            run_id,
            ticker,
            started_ts,
            finished_ts,
            status,
            fundamentals_refreshed,
            price_fetch_start,
            price_rows,
            options_refreshed,
            options_rows,
            message,
            error_message,
        ),
    )


def fetch_finviz_row(ticker: str, retry_attempts: int = 3) -> dict:
    last_err: Optional[Exception] = None
    raw: dict = {}
    for attempt in range(1, retry_attempts + 1):
        try:
            raw = finviz.get_stock(ticker)
            if not isinstance(raw, dict) or not raw:
                raise RuntimeError("empty finviz response")
            break
        except Exception as exc:
            last_err = exc
            if attempt == retry_attempts:
                raise RuntimeError(f"Finviz fetch failed for {ticker}: {last_err}") from last_err
            time.sleep((1.8 ** (attempt - 1)) + random.uniform(0.0, 0.4))

    price = to_float(raw.get("Price"))
    pe = to_float(raw.get("P/E"))
    peg = to_float(raw.get("PEG"))
    eps_ttm = to_float(raw.get("EPS (ttm)") or raw.get("EPS"))
    eps_growth_5y = to_percent(raw.get("EPS next 5Y"))
    analyst_target = to_float(raw.get("Target Price"))

    model_target = eps_ttm * DEFAULT_TARGET_PE if (eps_ttm is not None and eps_ttm > 0) else None
    if analyst_target is not None:
        target_price = analyst_target
        target_reason = "FINVIZ_ANALYST_TARGET"
    elif model_target is not None:
        target_price = model_target
        target_reason = "MODEL_TARGET_EPSxPE"
    else:
        target_price = None
        target_reason = "NO_TARGET"

    discount_pct = None
    if price is not None and target_price is not None and target_price != 0:
        discount_pct = (target_price - price) / target_price * 100.0

    return {
        "ticker": ticker,
        "price": price,
        "pe": pe,
        "peg": peg,
        "eps_ttm": eps_ttm,
        "eps_growth_5y": eps_growth_5y,
        "target_price": target_price,
        "target_reason": target_reason,
        "discount_pct": discount_pct,
        "raw_json": json.dumps(raw),
    }


def write_fundamentals(conn: sqlite3.Connection, snapshot_ts: str, row: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO finviz_fundamentals (
            snapshot_ts, ticker, price, pe, peg, eps_ttm, eps_growth_5y,
            target_price, target_reason, discount_pct, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_ts,
            row["ticker"],
            row["price"],
            row["pe"],
            row["peg"],
            row["eps_ttm"],
            row["eps_growth_5y"],
            row["target_price"],
            row["target_reason"],
            row["discount_pct"],
            row["raw_json"],
        ),
    )


def fetch_price_daily(
    ticker: str,
    start: str,
    end: Optional[str],
    auto_adjust: bool = False,
    timeout_sec: float = DEFAULT_PRICE_TIMEOUT_SEC,
    retry_attempts: int = 3,
) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, retry_attempts) + 1):
        try:
            raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                actions=False,
                group_by="column",
                threads=False,
                timeout=timeout_sec,
            )
            break
        except Exception as exc:
            last_err = exc
            if attempt >= max(1, retry_attempts):
                raise RuntimeError(f"price fetch failed for {ticker}: {last_err}") from last_err
            sleep_s = (1.8 ** (attempt - 1)) + random.uniform(0.0, 0.4)
            LOG.warning(
                "price_fetch_retry ticker=%s attempt=%s/%s timeout=%.1fs err=%s sleep=%.2fs",
                ticker,
                attempt,
                max(1, retry_attempts),
                timeout_sec,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    raw_reset = raw.reset_index()
    try:
        df = ensure_ohlcv_schema(raw_reset)
    except Exception:
        LOG.exception(
            "schema_normalization_failed ticker=%s start=%s end=%s columns=%s",
            ticker,
            start,
            end,
            [str(c) for c in raw_reset.columns],
        )
        raise
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


def write_price_daily(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    rows = [
        (
            ticker,
            r.date,
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            float(r.volume) if pd.notna(r.volume) else None,
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO price_daily (
            ticker, date, open, high, low, close, volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def fetch_options_rows(
    ticker: str,
    max_expiries: int = 2,
    min_moneyness: float = 0.7,
    max_moneyness: float = 1.3,
) -> list[tuple]:
    out: list[tuple] = []
    tk = yf.Ticker(ticker)
    expiries = list(tk.options or [])[:max_expiries]
    if not expiries:
        return out

    spot_hist = tk.history(period="5d")
    if spot_hist.empty or "Close" not in spot_hist.columns:
        return out
    spot = float(spot_hist["Close"].iloc[-1])
    if spot <= 0:
        return out

    for exp in expiries:
        chain = tk.option_chain(exp)
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            work = df.copy()
            work["moneyness"] = work["strike"] / spot
            work = work[
                (work["moneyness"] >= min_moneyness)
                & (work["moneyness"] <= max_moneyness)
            ].copy()
            if work.empty:
                continue
            for r in work.itertuples(index=False):
                out.append(
                    (
                        ticker,
                        exp,
                        option_type,
                        float(r.strike),
                        float(r.lastPrice) if pd.notna(r.lastPrice) else None,
                        float(r.bid) if pd.notna(r.bid) else None,
                        float(r.ask) if pd.notna(r.ask) else None,
                        float(r.impliedVolatility) if pd.notna(r.impliedVolatility) else None,
                        float(r.openInterest) if pd.notna(r.openInterest) else None,
                        float(r.volume) if pd.notna(r.volume) else None,
                        float(r.moneyness) if pd.notna(r.moneyness) else None,
                    )
                )
    return out


def write_options_rows(conn: sqlite3.Connection, snapshot_ts: str, rows: Iterable[tuple]) -> int:
    data = list(rows)
    if not data:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO options_iv (
            snapshot_ts, ticker, expiration, option_type, strike,
            last_price, bid, ask, implied_vol, open_interest, volume, moneyness
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(snapshot_ts, *r) for r in data],
    )
    return len(data)


def run(args: argparse.Namespace) -> None:
    if not LOG.handlers:
        setup_logging(level="INFO", log_file=None)

    db_path = Path(args.db_path)
    conn = connect_db(db_path)
    ensure_schema(conn)

    if args.use_sp500:
        tickers = get_sp500_tickers()
    else:
        tickers = parse_tickers(args.tickers)
    if not tickers:
        raise ValueError("No tickers provided.")

    # Always include market context tickers regardless of mode
    # Trinity: SPY (S&P 500), QQQ (Nasdaq), ^DJI (Dow) + VIX + 10yr yield + inverse VIX
    ALWAYS_FETCH = ["^VIX", "^GSPC", "^DJI", "^TNX", "SPY", "QQQ", "SVIX"]
    for t in ALWAYS_FETCH:
        if t not in tickers:
            tickers.append(t)

    now = dt.datetime.now(dt.timezone.utc)
    snapshot_ts = utc_now_iso()
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    ok = 0
    fail = 0
    begin_run(conn, run_id=run_id, started_ts=snapshot_ts, tickers_total=len(tickers), args=args)
    LOG.info(
        "run_id=%s started db=%s tickers=%s snapshot=%s",
        run_id,
        db_path,
        len(tickers),
        snapshot_ts,
    )

    try:
        for i, tkr in enumerate(tickers, start=1):
            ticker_started_ts = utc_now_iso()
            price_rows = 0
            options_rows = 0
            fundamentals_refreshed = 0
            options_refreshed = 0
            message_parts: list[str] = []
            price_fetch_start: str | None = args.price_start

            try:
                LOG.info("run_id=%s [%s/%s] ticker=%s start", run_id, i, len(tickers), tkr)
                with _ticker_timeout(args.max_seconds_per_ticker):
                    # Skip Finviz fundamentals for index/non-stock tickers (^VIX etc.)
                    is_index = tkr.startswith("^")
                    last_fund = get_latest_snapshot_ts(conn, "finviz_fundamentals", tkr)
                    if not is_index and should_refresh(last_fund, args.fund_min_interval_hours, now=dt.datetime.now(dt.timezone.utc)):
                        row = fetch_finviz_row(tkr, retry_attempts=args.retry_attempts)
                        write_fundamentals(conn, snapshot_ts, row)
                        fundamentals_refreshed = 1
                        message_parts.append("fund:refreshed")
                    else:
                        message_parts.append("fund:skipped_index" if is_index else "fund:skipped_recent")

                    if args.skip_price:
                        price_fetch_start = None
                        price_rows = 0
                        message_parts.append("price:skipped")
                    else:
                        price_fetch_start = resolve_price_fetch_start(
                            conn=conn,
                            ticker=tkr,
                            default_start=args.price_start,
                            lookback_days=args.price_lookback_days,
                            full_refresh=args.full_price_refresh,
                        )
                        price_df = fetch_price_daily(
                            ticker=tkr,
                            start=price_fetch_start,
                            end=args.price_end,
                            auto_adjust=args.auto_adjust,
                            timeout_sec=args.price_timeout_sec,
                            retry_attempts=args.price_retry_attempts,
                        )
                        price_rows = len(price_df)
                        write_price_daily(conn, tkr, price_df)
                        message_parts.append(f"price:start={price_fetch_start}")

                    if args.include_options:
                        last_opt = get_latest_snapshot_ts(conn, "options_iv", tkr)
                        if should_refresh(last_opt, args.options_min_interval_hours, now=dt.datetime.now(dt.timezone.utc)):
                            option_rows = fetch_options_rows(
                                ticker=tkr,
                                max_expiries=args.max_expiries,
                                min_moneyness=args.min_moneyness,
                                max_moneyness=args.max_moneyness,
                            )
                            options_rows = write_options_rows(conn, snapshot_ts, option_rows)
                            options_refreshed = 1 if options_rows > 0 else 0
                            message_parts.append("opt:refreshed")
                        else:
                            message_parts.append("opt:skipped_recent")

                upsert_ticker_status(
                    conn,
                    run_id=run_id,
                    ticker=tkr,
                    started_ts=ticker_started_ts,
                    finished_ts=utc_now_iso(),
                    status="ok",
                    fundamentals_refreshed=fundamentals_refreshed,
                    price_fetch_start=price_fetch_start,
                    price_rows=price_rows,
                    options_refreshed=options_refreshed,
                    options_rows=options_rows,
                    message=" | ".join(message_parts),
                    error_message=None,
                )
                conn.commit()
                ok += 1
                LOG.info(
                    "run_id=%s [%s/%s] ticker=%s ok price_rows=%s options_rows=%s flags=%s",
                    run_id,
                    i,
                    len(tickers),
                    tkr,
                    price_rows,
                    options_rows,
                    ",".join(message_parts),
                )
            except Exception as exc:
                conn.rollback()
                fail += 1
                err = str(exc)
                try:
                    upsert_ticker_status(
                        conn,
                        run_id=run_id,
                        ticker=tkr,
                        started_ts=ticker_started_ts,
                        finished_ts=utc_now_iso(),
                        status="failed",
                        fundamentals_refreshed=fundamentals_refreshed,
                        price_fetch_start=price_fetch_start,
                        price_rows=price_rows,
                        options_refreshed=options_refreshed,
                        options_rows=options_rows,
                        message=" | ".join(message_parts) if message_parts else None,
                        error_message=err,
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                LOG.exception("run_id=%s [%s/%s] ticker=%s failed: %s", run_id, i, len(tickers), tkr, err)
            finally:
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))

        final_status = "ok" if fail == 0 else ("failed" if ok == 0 else "partial_failed")
        finish_run(
            conn,
            run_id=run_id,
            finished_ts=utc_now_iso(),
            status=final_status,
            tickers_ok=ok,
            tickers_failed=fail,
        )
        LOG.info("run_id=%s finished status=%s ok=%s failed=%s", run_id, final_status, ok, fail)
    except Exception as exc:
        conn.rollback()
        finish_run(
            conn,
            run_id=run_id,
            finished_ts=utc_now_iso(),
            status="failed",
            tickers_ok=ok,
            tickers_failed=max(fail, len(tickers) - ok),
            error_message=str(exc),
        )
        LOG.exception("run_id=%s fatal failure: %s", run_id, exc)
        raise
    finally:
        conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Update local market database (Finviz + prices + options).")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--use-sp500", action="store_true")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,DIA,NVDA,AAPL,MSFT,TSLA")
    p.add_argument("--include-options", action="store_true")
    p.add_argument("--max-expiries", type=int, default=2)
    p.add_argument("--min-moneyness", type=float, default=0.7)
    p.add_argument("--max-moneyness", type=float, default=1.3)
    p.add_argument("--price-start", default="2018-01-01")
    p.add_argument("--price-end", default=None)
    p.add_argument("--price-lookback-days", type=int, default=10)
    p.add_argument("--full-price-refresh", action="store_true")
    p.add_argument("--skip-price", action="store_true")
    p.add_argument("--auto-adjust", action="store_true")
    p.add_argument("--fund-min-interval-hours", type=float, default=12.0)
    p.add_argument("--options-min-interval-hours", type=float, default=4.0)
    p.add_argument("--retry-attempts", type=int, default=3)
    p.add_argument(
        "--max-seconds-per-ticker",
        type=float,
        default=DEFAULT_MAX_SECS_PER_TICKER,
        help="Fail-safe timeout per ticker; timeout marks ticker failed and continues (0 disables).",
    )
    p.add_argument(
        "--price-timeout-sec",
        type=float,
        default=DEFAULT_PRICE_TIMEOUT_SEC,
        help="HTTP timeout for yfinance download requests.",
    )
    p.add_argument(
        "--price-retry-attempts",
        type=int,
        default=3,
        help="Retry attempts for yfinance price fetch (per ticker).",
    )
    p.add_argument("--sleep-min", type=float, default=0.4)
    p.add_argument("--sleep-max", type=float, default=1.0)
    p.add_argument("--log-file", default=DEFAULT_LOG_PATH)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    run(args)

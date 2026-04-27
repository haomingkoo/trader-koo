"""Options positioning research from local option-chain snapshots.

This module intentionally does not model real-time option flow. The source
table is populated from Yahoo/yfinance option-chain snapshots, so the output is
best-effort context: implied-volatility rank, open-interest rank, skew, and
cheap/elevated volatility flags.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from trader_koo.config import (
    OPTIONS_CONTRACT_MULTIPLIER,
    OPTIONS_PERCENT_MULTIPLIER,
    OPTIONS_PREMIUM_BIAS_BALANCED,
    OPTIONS_PREMIUM_BIAS_CALL,
    OPTIONS_PREMIUM_BIAS_PUT,
    OPTIONS_PREMIUM_BIAS_UNKNOWN,
    OPTIONS_PREMIUM_BALANCED_SKEW_RATIO,
    OPTIONS_PREMIUM_DEFAULT_LIMIT,
    OPTIONS_PREMIUM_DEFAULT_SORT_BY,
    OPTIONS_PREMIUM_PROXY_NOTE,
    OPTIONS_SCORE_MAX,
    OPTIONS_SCORE_MIN,
    OPTIONS_SNAPSHOT_DEFAULT_MAX_EXPIRIES,
    OPTIONS_SNAPSHOT_DEFAULT_MAX_MONEYNESS,
    OPTIONS_SNAPSHOT_DEFAULT_MAX_TICKERS,
    OPTIONS_SNAPSHOT_DEFAULT_MIN_INTERVAL_HOURS,
    OPTIONS_SNAPSHOT_DEFAULT_MIN_MONEYNESS,
    OPTIONS_SNAPSHOT_DEFAULT_TICKERS,
    OPTIONS_SMART_HIGH_PUT_CALL_OI,
    OPTIONS_SMART_HIGH_SCORE,
    OPTIONS_SMART_HOT_IV_PCT,
    OPTIONS_SMART_LOW_HISTORY_SNAPSHOTS,
    OPTIONS_SMART_LOW_PUT_CALL_OI,
    OPTIONS_SMART_OI_MISMATCH_SCORE,
    OPTIONS_SMART_SCORE_WEIGHTS,
    OPTIONS_SMART_SIGNAL_BEARISH_OR_HEDGE,
    OPTIONS_SMART_SIGNAL_BULLISH,
    OPTIONS_SMART_SIGNAL_MOMENTUM_CHASE,
    OPTIONS_SMART_SIGNAL_RELATIVE_VALUE,
    OPTIONS_SMART_SIGNAL_WATCH,
    OPTIONS_SMART_SINGLE_NAME_SCORE,
    OPTIONS_SMART_STRONG_SCORE,
    OPTIONS_SMART_STRONG_SKEW_PCT,
    OPTIONS_SMART_TAG_CALL_OI_LEAD,
    OPTIONS_SMART_TAG_HOT_IV,
    OPTIONS_SMART_TAG_LIMITED_HISTORY,
    OPTIONS_SMART_TAG_LIQUID,
    OPTIONS_SMART_TAG_OI_CONFIRMED,
    OPTIONS_SMART_TAG_PUT_OI_HEAVY,
    OPTIONS_SMART_TAG_RELATIVE_VALUE_IV,
    OPTIONS_SMART_TAG_STRONG_FLOW,
    OPTIONS_SMART_TAG_TOP_SCORE,
    OPTIONS_SOURCE,
    OPTIONS_SOURCE_NOTE,
    normalize_options_limit,
    normalize_options_sort,
)

SOURCE = OPTIONS_SOURCE
SOURCE_NOTE = OPTIONS_SOURCE_NOTE
PREMIUM_PROXY_NOTE = OPTIONS_PREMIUM_PROXY_NOTE
DEFAULT_OPTIONS_SNAPSHOT_TICKERS = OPTIONS_SNAPSHOT_DEFAULT_TICKERS

OptionRow = tuple[
    str,
    str,
    str,
    float,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        is not None
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str | None) -> dt.datetime | None:
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


def normalize_options_tickers(raw: str | Iterable[str] | None) -> list[str]:
    """Return de-duplicated Yahoo-compatible tickers for option snapshots."""
    if raw is None:
        parts: list[str] = []
    elif isinstance(raw, str):
        parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    else:
        parts = [str(p).strip() for p in raw]

    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        symbol = part.upper().replace(".", "-")
        if not symbol or symbol in seen:
            continue
        # Yahoo index symbols often do not expose chains through yfinance's
        # equity option-chain path; keep ETFs/stocks in this bounded crawl.
        if symbol.startswith("^"):
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def ensure_options_iv_schema(conn: sqlite3.Connection) -> None:
    """Ensure the canonical local option-chain snapshot table exists."""
    conn.execute(
        """
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
        )
        """
    )
    existing = _columns(conn, "options_iv")
    for name, ddl in (
        ("expiration", "ALTER TABLE options_iv ADD COLUMN expiration TEXT"),
        ("strike", "ALTER TABLE options_iv ADD COLUMN strike REAL"),
        ("last_price", "ALTER TABLE options_iv ADD COLUMN last_price REAL"),
        ("bid", "ALTER TABLE options_iv ADD COLUMN bid REAL"),
        ("ask", "ALTER TABLE options_iv ADD COLUMN ask REAL"),
        ("volume", "ALTER TABLE options_iv ADD COLUMN volume REAL"),
        ("moneyness", "ALTER TABLE options_iv ADD COLUMN moneyness REAL"),
    ):
        if name in existing:
            continue
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_options_ticker_snap ON options_iv(ticker, snapshot_ts)")


def _percent_rank(values: list[float], current: float | None) -> float | None:
    if current is None or len(values) < 2:
        return None
    clean = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if len(clean) < 2:
        return None
    lower_or_equal = sum(1 for v in clean if v <= current)
    return round((lower_or_equal / len(clean)) * OPTIONS_PERCENT_MULTIPLIER, 1)


def _realized_vol_pct(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    lookback: int,
) -> float | None:
    if not _table_exists(conn, "price_daily"):
        return None
    rows = conn.execute(
        """
        SELECT close
        FROM price_daily
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker.upper(), lookback + 1),
    ).fetchall()
    closes = [float(r[0]) for r in reversed(rows) if _safe_float(r[0]) and float(r[0]) > 0]
    if len(closes) < max(6, min(lookback, 30)):
        return None
    returns: list[float] = []
    for prev, cur in zip(closes, closes[1:]):
        if prev > 0 and cur > 0:
            returns.append(math.log(cur / prev))
    if len(returns) < 5:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
    return round(math.sqrt(variance) * math.sqrt(252) * OPTIONS_PERCENT_MULTIPLIER, 1)


def _latest_close_from_history(history: Any) -> float | None:
    if history is None or getattr(history, "empty", False):
        return None
    close_col = None
    for candidate in ("Close", "close"):
        if candidate in getattr(history, "columns", []):
            close_col = candidate
            break
    if close_col is None:
        return None
    try:
        series = history[close_col].dropna()
        if series.empty:
            return None
        close = _safe_float(series.iloc[-1])
        return close if close and close > 0 else None
    except Exception:
        return None


def _series_get(row: Any, *names: str) -> float | None:
    for name in names:
        try:
            if name in row:
                value = _safe_float(row.get(name))
                if value is not None:
                    return value
        except Exception:
            continue
    return None


def fetch_yfinance_options_rows(
    ticker: str,
    *,
    max_expiries: int = OPTIONS_SNAPSHOT_DEFAULT_MAX_EXPIRIES,
    min_moneyness: float = OPTIONS_SNAPSHOT_DEFAULT_MIN_MONEYNESS,
    max_moneyness: float = OPTIONS_SNAPSHOT_DEFAULT_MAX_MONEYNESS,
    ticker_factory: Callable[[str], Any] | None = None,
) -> list[OptionRow]:
    """Fetch a bounded Yahoo/yfinance option-chain snapshot for one ticker."""
    symbol = normalize_options_tickers([ticker])
    if not symbol:
        return []
    symbol = symbol[0]
    factory = ticker_factory
    if factory is None:
        import yfinance as yf

        factory = yf.Ticker

    tk = factory(symbol)
    expiries = list(getattr(tk, "options", None) or [])[: max(0, int(max_expiries))]
    if not expiries:
        return []

    spot = _latest_close_from_history(tk.history(period="5d"))
    if spot is None or spot <= 0:
        return []

    rows: list[OptionRow] = []
    lo = float(min(min_moneyness, max_moneyness))
    hi = float(max(min_moneyness, max_moneyness))
    for exp in expiries:
        chain = tk.option_chain(exp)
        for option_type, frame in (("call", getattr(chain, "calls", None)), ("put", getattr(chain, "puts", None))):
            if frame is None or getattr(frame, "empty", False):
                continue
            for _, row in frame.iterrows():
                strike = _series_get(row, "strike")
                if strike is None or strike <= 0:
                    continue
                moneyness = strike / spot
                if moneyness < lo or moneyness > hi:
                    continue
                rows.append(
                    (
                        symbol,
                        str(exp),
                        option_type,
                        float(strike),
                        _series_get(row, "lastPrice", "last_price"),
                        _series_get(row, "bid"),
                        _series_get(row, "ask"),
                        _series_get(row, "impliedVolatility", "implied_volatility", "implied_vol"),
                        _series_get(row, "openInterest", "open_interest"),
                        _series_get(row, "volume"),
                        float(moneyness),
                    )
                )
    return rows


def write_options_rows(conn: sqlite3.Connection, snapshot_ts: str, rows: Iterable[OptionRow]) -> int:
    """Write canonical option-chain rows to ``options_iv``."""
    data = list(rows)
    if not data:
        return 0
    ensure_options_iv_schema(conn)
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


def _latest_options_snapshot_ts(conn: sqlite3.Connection, ticker: str) -> str | None:
    if not _table_exists(conn, "options_iv"):
        return None
    try:
        row = conn.execute(
            "SELECT MAX(snapshot_ts) FROM options_iv WHERE ticker = ?",
            (ticker.upper(),),
        ).fetchone()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def _should_refresh_snapshot(
    last_snapshot_ts: str | None,
    *,
    min_interval_hours: float,
    now: dt.datetime | None = None,
) -> bool:
    parsed = _parse_iso_utc(last_snapshot_ts)
    if parsed is None:
        return True
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.timezone.utc)
    age_hours = (current.astimezone(dt.timezone.utc) - parsed).total_seconds() / 3600.0
    return age_hours >= max(0.0, float(min_interval_hours))


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    seen = set(target)
    for value in normalize_options_tickers(values):
        if value not in seen:
            target.append(value)
            seen.add(value)


def load_options_snapshot_tickers(
    conn: sqlite3.Connection | None = None,
    *,
    explicit_tickers: str | Iterable[str] | None = None,
    latest_report_path: str | Path | None = None,
    max_tickers: int = OPTIONS_SNAPSHOT_DEFAULT_MAX_TICKERS,
) -> list[str]:
    """Choose a bounded ticker list for the nightly option-chain snapshot."""
    limit = max(1, int(max_tickers))
    tickers: list[str] = []
    explicit = normalize_options_tickers(explicit_tickers)
    if explicit:
        return explicit[:limit]

    report_path = Path(latest_report_path) if latest_report_path else None
    if report_path and report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            signals = payload.get("signals") if isinstance(payload, dict) else {}
            suggestions = (signals or {}).get("suggestions") or {}
            _append_unique(
                tickers,
                [
                    item.get("ticker")
                    for item in suggestions.get("items") or []
                    if isinstance(item, dict)
                ],
            )
            _append_unique(
                tickers,
                [
                    row.get("ticker")
                    for row in (signals or {}).get("setup_quality_top") or []
                    if isinstance(row, dict)
                ],
            )
        except Exception:
            pass

    if conn is not None and len(tickers) < limit and _table_exists(conn, "setup_call_evaluations"):
        try:
            rows = conn.execute(
                """
                SELECT ticker
                FROM setup_call_evaluations
                WHERE asof_date = (SELECT MAX(asof_date) FROM setup_call_evaluations)
                ORDER BY COALESCE(score, 0) DESC, ticker
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            _append_unique(tickers, [row[0] for row in rows])
        except Exception:
            pass

    if len(tickers) < limit:
        _append_unique(tickers, DEFAULT_OPTIONS_SNAPSHOT_TICKERS)
    return tickers[:limit]


def snapshot_options_iv(
    conn: sqlite3.Connection,
    tickers: Iterable[str],
    *,
    snapshot_ts: str | None = None,
    max_expiries: int = OPTIONS_SNAPSHOT_DEFAULT_MAX_EXPIRIES,
    min_moneyness: float = OPTIONS_SNAPSHOT_DEFAULT_MIN_MONEYNESS,
    max_moneyness: float = OPTIONS_SNAPSHOT_DEFAULT_MAX_MONEYNESS,
    min_interval_hours: float = OPTIONS_SNAPSHOT_DEFAULT_MIN_INTERVAL_HOURS,
    force: bool = False,
    sleep_sec: float = 0.0,
    ticker_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Fetch and persist yfinance option-chain snapshots for a bounded list."""
    ensure_options_iv_schema(conn)
    snap = snapshot_ts or _utc_now_iso()
    selected = normalize_options_tickers(tickers)
    summary: dict[str, Any] = {
        "source": SOURCE,
        "source_note": SOURCE_NOTE,
        "snapshot_ts": snap,
        "tickers_total": len(selected),
        "tickers_refreshed": 0,
        "tickers_skipped_recent": 0,
        "tickers_empty": 0,
        "tickers_failed": 0,
        "rows_inserted": 0,
        "errors": {},
    }
    now = _parse_iso_utc(snap) or dt.datetime.now(dt.timezone.utc)
    for ticker in selected:
        try:
            last_ts = _latest_options_snapshot_ts(conn, ticker)
            if not force and not _should_refresh_snapshot(
                last_ts,
                min_interval_hours=min_interval_hours,
                now=now,
            ):
                summary["tickers_skipped_recent"] += 1
                continue
            rows = fetch_yfinance_options_rows(
                ticker,
                max_expiries=max_expiries,
                min_moneyness=min_moneyness,
                max_moneyness=max_moneyness,
                ticker_factory=ticker_factory,
            )
            written = write_options_rows(conn, snap, rows)
            conn.commit()
            summary["rows_inserted"] += written
            if written > 0:
                summary["tickers_refreshed"] += 1
            else:
                summary["tickers_empty"] += 1
        except Exception as exc:
            conn.rollback()
            summary["tickers_failed"] += 1
            summary["errors"][ticker] = str(exc)
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    return summary


def _snapshot_aggregates(conn: sqlite3.Connection, ticker: str) -> list[dict[str, Any]]:
    cols = _columns(conn, "options_iv")
    if not {"snapshot_ts", "ticker", "option_type"}.issubset(cols):
        return []

    has_moneyness = "moneyness" in cols
    has_iv = "implied_vol" in cols
    has_oi = "open_interest" in cols
    has_volume = "volume" in cols

    iv_expr = "CAST(implied_vol AS REAL)" if has_iv else "NULL"
    oi_expr = "CAST(open_interest AS REAL)" if has_oi else "NULL"
    vol_expr = "CAST(volume AS REAL)" if has_volume else "NULL"
    atm_iv_expr = (
        "AVG(CASE WHEN moneyness BETWEEN 0.95 AND 1.05 THEN CAST(implied_vol AS REAL) END)"
        if has_moneyness and has_iv
        else f"AVG({iv_expr})"
    )

    rows = conn.execute(
        f"""
        SELECT
            snapshot_ts,
            COUNT(*) AS contracts,
            AVG({iv_expr}) AS avg_iv,
            {atm_iv_expr} AS atm_iv,
            SUM(COALESCE({oi_expr}, 0)) AS total_oi,
            SUM(CASE WHEN LOWER(option_type)='call' THEN COALESCE({oi_expr}, 0) ELSE 0 END) AS call_oi,
            SUM(CASE WHEN LOWER(option_type)='put' THEN COALESCE({oi_expr}, 0) ELSE 0 END) AS put_oi,
            SUM(COALESCE({vol_expr}, 0)) AS total_volume
        FROM options_iv
        WHERE ticker = ?
        GROUP BY snapshot_ts
        ORDER BY snapshot_ts ASC
        """,
        (ticker.upper(),),
    ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row) if isinstance(row, sqlite3.Row) else {
            "snapshot_ts": row[0],
            "contracts": row[1],
            "avg_iv": row[2],
            "atm_iv": row[3],
            "total_oi": row[4],
            "call_oi": row[5],
            "put_oi": row[6],
            "total_volume": row[7],
        }
        out.append(d)
    return out


def _round_money(value: float | None) -> float | None:
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), 2)


def _premium_direction(net_value: float | None, gross_value: float | None) -> str:
    if net_value is None or gross_value is None or gross_value <= 0:
        return OPTIONS_PREMIUM_BIAS_UNKNOWN
    threshold = gross_value * OPTIONS_PREMIUM_BALANCED_SKEW_RATIO
    if abs(net_value) < threshold:
        return OPTIONS_PREMIUM_BIAS_BALANCED
    return OPTIONS_PREMIUM_BIAS_CALL if net_value > 0 else OPTIONS_PREMIUM_BIAS_PUT


def _rank_score(value: float | None, values: Iterable[float | None], *, higher_is_better: bool = True) -> float | None:
    """Return a 0-100 cross-sectional score for a metric."""
    number = _safe_float(value)
    nums = sorted(v for item in values if (v := _safe_float(item)) is not None)
    if number is None or not nums:
        return None
    if len(nums) == 1:
        return OPTIONS_SMART_SINGLE_NAME_SCORE
    lower = sum(1 for item in nums if item < number)
    equal = sum(1 for item in nums if item == number)
    rank = lower + ((equal - 1) / 2)
    score = (rank / (len(nums) - 1)) * OPTIONS_SCORE_MAX
    if not higher_is_better:
        score = OPTIONS_SCORE_MAX - score
    return round(max(OPTIONS_SCORE_MIN, min(OPTIONS_SCORE_MAX, score)), 1)


def _avg_scores(*values: float | None) -> float | None:
    nums = [v for item in values if (v := _safe_float(item)) is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 1)


def _ratio_pct(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * OPTIONS_PERCENT_MULTIPLIER, 1)


def _smart_options_signal(row: dict[str, Any]) -> tuple[str, list[str]]:
    tags: list[str] = []
    bias = str(row.get("premium_bias") or "")
    score = _safe_float(row.get("smart_score")) or OPTIONS_SCORE_MIN
    volume_skew = _safe_float(row.get("volume_skew_pct"))
    oi_skew = _safe_float(row.get("oi_skew_pct"))
    iv_value_score = _safe_float(row.get("iv_value_score")) or OPTIONS_SCORE_MIN
    liquidity_score = _safe_float(row.get("liquidity_score")) or OPTIONS_SCORE_MIN
    avg_iv_pct = _safe_float(row.get("avg_iv_pct"))
    put_call = _safe_float(row.get("put_call_oi_ratio"))

    if score >= OPTIONS_SMART_HIGH_SCORE:
        tags.append(OPTIONS_SMART_TAG_TOP_SCORE)
    if abs(volume_skew or OPTIONS_SCORE_MIN) >= OPTIONS_SMART_STRONG_SKEW_PCT:
        tags.append(OPTIONS_SMART_TAG_STRONG_FLOW)
    if liquidity_score >= OPTIONS_SMART_STRONG_SCORE:
        tags.append(OPTIONS_SMART_TAG_LIQUID)
    if iv_value_score >= OPTIONS_SMART_STRONG_SCORE:
        tags.append(OPTIONS_SMART_TAG_RELATIVE_VALUE_IV)
    if avg_iv_pct is not None and avg_iv_pct >= OPTIONS_SMART_HOT_IV_PCT:
        tags.append(OPTIONS_SMART_TAG_HOT_IV)
    if int(row.get("historical_snapshots") or 0) < OPTIONS_SMART_LOW_HISTORY_SNAPSHOTS:
        tags.append(OPTIONS_SMART_TAG_LIMITED_HISTORY)
    if put_call is not None and put_call <= OPTIONS_SMART_LOW_PUT_CALL_OI:
        tags.append(OPTIONS_SMART_TAG_CALL_OI_LEAD)
    if put_call is not None and put_call >= OPTIONS_SMART_HIGH_PUT_CALL_OI:
        tags.append(OPTIONS_SMART_TAG_PUT_OI_HEAVY)

    if bias == OPTIONS_PREMIUM_BIAS_CALL and (oi_skew or OPTIONS_SCORE_MIN) > OPTIONS_SCORE_MIN:
        tags.append(OPTIONS_SMART_TAG_OI_CONFIRMED)
    elif bias == OPTIONS_PREMIUM_BIAS_PUT and (oi_skew or OPTIONS_SCORE_MIN) < OPTIONS_SCORE_MIN:
        tags.append(OPTIONS_SMART_TAG_OI_CONFIRMED)

    if bias == OPTIONS_PREMIUM_BIAS_CALL and score >= OPTIONS_SMART_STRONG_SCORE:
        signal = OPTIONS_SMART_SIGNAL_BULLISH
    elif bias == OPTIONS_PREMIUM_BIAS_PUT and score >= OPTIONS_SMART_STRONG_SCORE:
        signal = OPTIONS_SMART_SIGNAL_BEARISH_OR_HEDGE
    elif OPTIONS_SMART_TAG_RELATIVE_VALUE_IV in tags and bias in {
        OPTIONS_PREMIUM_BIAS_CALL,
        OPTIONS_PREMIUM_BIAS_PUT,
    }:
        signal = OPTIONS_SMART_SIGNAL_RELATIVE_VALUE
    elif OPTIONS_SMART_TAG_HOT_IV in tags and OPTIONS_SMART_TAG_STRONG_FLOW in tags:
        signal = OPTIONS_SMART_SIGNAL_MOMENTUM_CHASE
    else:
        signal = OPTIONS_SMART_SIGNAL_WATCH

    return signal, tags


def build_options_premium_proxy(
    conn: sqlite3.Connection,
    *,
    limit: int = OPTIONS_PREMIUM_DEFAULT_LIMIT,
    sort_by: str = OPTIONS_PREMIUM_DEFAULT_SORT_BY,
) -> dict[str, Any]:
    """Return a dashboard-ready options premium proxy from local snapshots.

    This intentionally does not claim true net premium. The source table is
    yfinance option-chain snapshots, so the best available approximation is
    call premium minus put premium from displayed volume/open interest.
    """
    resolved_limit = normalize_options_limit(limit)
    resolved_sort_by = normalize_options_sort(sort_by)

    if not _table_exists(conn, "options_iv"):
        return {
            "available": False,
            "source": SOURCE,
            "source_note": SOURCE_NOTE,
            "premium_proxy_note": PREMIUM_PROXY_NOTE,
            "count": 0,
            "rows": [],
            "totals": {},
            "detail": "options_iv table missing",
        }

    cols = _columns(conn, "options_iv")
    if not {"snapshot_ts", "ticker", "option_type"}.issubset(cols):
        return {
            "available": False,
            "source": SOURCE,
            "source_note": SOURCE_NOTE,
            "premium_proxy_note": PREMIUM_PROXY_NOTE,
            "count": 0,
            "rows": [],
            "totals": {},
            "detail": "options_iv table is missing required columns",
        }

    has_bid_ask = {"bid", "ask"}.issubset(cols)
    has_last = "last_price" in cols
    has_iv = "implied_vol" in cols
    has_oi = "open_interest" in cols
    has_volume = "volume" in cols
    has_expiration = "expiration" in cols

    if has_bid_ask and has_last:
        price_expr = (
            "CASE "
            "WHEN CAST(bid AS REAL) > 0 AND CAST(ask AS REAL) > 0 "
            "THEN (CAST(bid AS REAL) + CAST(ask AS REAL)) / 2.0 "
            "WHEN CAST(last_price AS REAL) > 0 THEN CAST(last_price AS REAL) "
            "ELSE NULL END"
        )
    elif has_bid_ask:
        price_expr = (
            "CASE "
            "WHEN CAST(bid AS REAL) > 0 AND CAST(ask AS REAL) > 0 "
            "THEN (CAST(bid AS REAL) + CAST(ask AS REAL)) / 2.0 "
            "ELSE NULL END"
        )
    elif has_last:
        price_expr = "CASE WHEN CAST(last_price AS REAL) > 0 THEN CAST(last_price AS REAL) ELSE NULL END"
    else:
        price_expr = "NULL"

    iv_expr = "CAST(implied_vol AS REAL)" if has_iv else "NULL"
    oi_expr = "CAST(open_interest AS REAL)" if has_oi else "NULL"
    volume_expr = "CAST(volume AS REAL)" if has_volume else "NULL"
    expiration_select = "o.expiration AS expiration," if has_expiration else ""
    expiration_expr = "COUNT(DISTINCT b.expiration)" if has_expiration else "NULL"

    rows = conn.execute(
        f"""
        WITH latest AS (
            SELECT ticker, MAX(snapshot_ts) AS snapshot_ts
            FROM options_iv
            GROUP BY ticker
        ),
        history AS (
            SELECT ticker, COUNT(DISTINCT snapshot_ts) AS historical_snapshots
            FROM options_iv
            GROUP BY ticker
        ),
        base AS (
            SELECT
                o.ticker,
                o.snapshot_ts,
                LOWER(o.option_type) AS option_type,
                {expiration_select}
                {price_expr} AS proxy_price,
                {iv_expr} AS implied_vol,
                {oi_expr} AS open_interest,
                {volume_expr} AS volume
            FROM options_iv o
            JOIN latest l
              ON o.ticker = l.ticker
             AND o.snapshot_ts = l.snapshot_ts
        )
        SELECT
            b.ticker,
            b.snapshot_ts,
            h.historical_snapshots,
            COUNT(*) AS contracts,
            {expiration_expr} AS expirations,
            AVG(b.implied_vol) AS avg_iv,
            SUM(CASE WHEN b.option_type = 'call' THEN COALESCE(b.open_interest, 0) ELSE 0 END) AS call_open_interest,
            SUM(CASE WHEN b.option_type = 'put' THEN COALESCE(b.open_interest, 0) ELSE 0 END) AS put_open_interest,
            SUM(CASE WHEN b.option_type = 'call' THEN COALESCE(b.volume, 0) ELSE 0 END) AS call_volume,
            SUM(CASE WHEN b.option_type = 'put' THEN COALESCE(b.volume, 0) ELSE 0 END) AS put_volume,
            SUM(CASE WHEN b.option_type = 'call' THEN COALESCE(b.proxy_price, 0) * COALESCE(b.volume, 0) * {OPTIONS_CONTRACT_MULTIPLIER} ELSE 0 END) AS call_volume_premium,
            SUM(CASE WHEN b.option_type = 'put' THEN COALESCE(b.proxy_price, 0) * COALESCE(b.volume, 0) * {OPTIONS_CONTRACT_MULTIPLIER} ELSE 0 END) AS put_volume_premium,
            SUM(CASE WHEN b.option_type = 'call' THEN COALESCE(b.proxy_price, 0) * COALESCE(b.open_interest, 0) * {OPTIONS_CONTRACT_MULTIPLIER} ELSE 0 END) AS call_oi_premium,
            SUM(CASE WHEN b.option_type = 'put' THEN COALESCE(b.proxy_price, 0) * COALESCE(b.open_interest, 0) * {OPTIONS_CONTRACT_MULTIPLIER} ELSE 0 END) AS put_oi_premium
        FROM base b
        LEFT JOIN history h ON b.ticker = h.ticker
        GROUP BY b.ticker, b.snapshot_ts, h.historical_snapshots
        """,
    ).fetchall()

    output_rows: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw) if isinstance(raw, sqlite3.Row) else dict(raw)
        call_volume_premium = _safe_float(row.get("call_volume_premium")) or 0.0
        put_volume_premium = _safe_float(row.get("put_volume_premium")) or 0.0
        call_oi_premium = _safe_float(row.get("call_oi_premium")) or 0.0
        put_oi_premium = _safe_float(row.get("put_oi_premium")) or 0.0
        net_volume_premium = call_volume_premium - put_volume_premium
        gross_volume_premium = call_volume_premium + put_volume_premium
        net_oi_premium = call_oi_premium - put_oi_premium
        gross_oi_premium = call_oi_premium + put_oi_premium
        call_oi = _safe_float(row.get("call_open_interest")) or 0.0
        put_oi = _safe_float(row.get("put_open_interest")) or 0.0
        call_volume = _safe_float(row.get("call_volume")) or 0.0
        put_volume = _safe_float(row.get("put_volume")) or 0.0
        put_call_ratio = (put_oi / call_oi) if call_oi > 0 else None
        volume_direction = _premium_direction(net_volume_premium, gross_volume_premium)
        oi_direction = _premium_direction(net_oi_premium, gross_oi_premium)
        if volume_direction != OPTIONS_PREMIUM_BIAS_UNKNOWN and gross_volume_premium > 0:
            premium_bias = volume_direction
            primary_premium_source = "volume"
        else:
            premium_bias = oi_direction
            primary_premium_source = "open_interest"

        output_rows.append(
            {
                "ticker": str(row.get("ticker") or "").upper(),
                "snapshot_ts": row.get("snapshot_ts"),
                "historical_snapshots": int(row.get("historical_snapshots") or 0),
                "contracts": int(row.get("contracts") or 0),
                "expirations": int(row.get("expirations") or 0) if row.get("expirations") is not None else None,
                "avg_iv_pct": (
                    round(float(row["avg_iv"]) * OPTIONS_PERCENT_MULTIPLIER, 1)
                    if row.get("avg_iv") is not None
                    else None
                ),
                "call_open_interest": round(call_oi, 1),
                "put_open_interest": round(put_oi, 1),
                "put_call_oi_ratio": round(put_call_ratio, 2) if put_call_ratio is not None else None,
                "call_volume": round(call_volume, 1),
                "put_volume": round(put_volume, 1),
                "total_volume": round(call_volume + put_volume, 1),
                "call_volume_premium": _round_money(call_volume_premium),
                "put_volume_premium": _round_money(put_volume_premium),
                "net_volume_premium": _round_money(net_volume_premium),
                "gross_volume_premium": _round_money(gross_volume_premium),
                "volume_skew_pct": _ratio_pct(net_volume_premium, gross_volume_premium),
                "volume_premium_direction": volume_direction,
                "call_oi_premium": _round_money(call_oi_premium),
                "put_oi_premium": _round_money(put_oi_premium),
                "net_oi_premium": _round_money(net_oi_premium),
                "gross_oi_premium": _round_money(gross_oi_premium),
                "oi_skew_pct": _ratio_pct(net_oi_premium, gross_oi_premium),
                "oi_premium_direction": oi_direction,
                "premium_bias": premium_bias,
                "primary_premium_source": primary_premium_source,
            }
        )

    abs_net_volume_premiums = [abs(float(row.get("net_volume_premium") or 0.0)) for row in output_rows]
    abs_net_oi_premiums = [abs(float(row.get("net_oi_premium") or 0.0)) for row in output_rows]
    gross_volume_premiums = [float(row.get("gross_volume_premium") or 0.0) for row in output_rows]
    total_volumes = [float(row.get("total_volume") or 0.0) for row in output_rows]
    contracts = [float(row.get("contracts") or 0.0) for row in output_rows]
    avg_ivs = [row.get("avg_iv_pct") for row in output_rows]

    for row in output_rows:
        volume_rank = _rank_score(
            abs(float(row.get("net_volume_premium") or 0.0)),
            abs_net_volume_premiums,
        )
        oi_rank = _rank_score(
            abs(float(row.get("net_oi_premium") or 0.0)),
            abs_net_oi_premiums,
        )
        liquidity_score = _avg_scores(
            _rank_score(row.get("gross_volume_premium"), gross_volume_premiums),
            _rank_score(row.get("total_volume"), total_volumes),
            _rank_score(row.get("contracts"), contracts),
        )
        iv_value_score = _rank_score(row.get("avg_iv_pct"), avg_ivs, higher_is_better=False)
        oi_confirms = row.get("oi_premium_direction") == row.get("volume_premium_direction")
        flow_quality_score = _avg_scores(
            volume_rank,
            abs(_safe_float(row.get("volume_skew_pct")) or OPTIONS_SCORE_MIN),
            oi_rank if oi_confirms else OPTIONS_SMART_OI_MISMATCH_SCORE,
        )
        smart_score = (
            ((volume_rank or OPTIONS_SCORE_MIN) * OPTIONS_SMART_SCORE_WEIGHTS["volume_rank"])
            + ((liquidity_score or OPTIONS_SCORE_MIN) * OPTIONS_SMART_SCORE_WEIGHTS["liquidity"])
            + ((iv_value_score or OPTIONS_SCORE_MIN) * OPTIONS_SMART_SCORE_WEIGHTS["iv_value"])
            + ((flow_quality_score or OPTIONS_SCORE_MIN) * OPTIONS_SMART_SCORE_WEIGHTS["flow_quality"])
        )
        row["volume_rank_pct"] = volume_rank
        row["oi_rank_pct"] = oi_rank
        row["liquidity_score"] = liquidity_score
        row["iv_value_score"] = iv_value_score
        row["flow_quality_score"] = flow_quality_score
        row["smart_score"] = round(max(OPTIONS_SCORE_MIN, min(OPTIONS_SCORE_MAX, smart_score)), 1)
        signal, tags = _smart_options_signal(row)
        row["smart_signal"] = signal
        row["smart_tags"] = tags

    if resolved_sort_by == "ticker":
        output_rows.sort(key=lambda item: str(item.get("ticker") or ""))
    elif resolved_sort_by == "oi_premium":
        output_rows.sort(key=lambda item: abs(float(item.get("net_oi_premium") or 0.0)), reverse=True)
    else:
        output_rows.sort(
            key=lambda item: (
                abs(float(item.get("net_volume_premium") or 0.0)),
                abs(float(item.get("net_oi_premium") or 0.0)),
            ),
            reverse=True,
        )

    capped = output_rows[:resolved_limit]
    totals = {
        "call_volume_premium": _round_money(sum(float(row.get("call_volume_premium") or 0.0) for row in capped)),
        "put_volume_premium": _round_money(sum(float(row.get("put_volume_premium") or 0.0) for row in capped)),
        "net_volume_premium": _round_money(sum(float(row.get("net_volume_premium") or 0.0) for row in capped)),
        "call_oi_premium": _round_money(sum(float(row.get("call_oi_premium") or 0.0) for row in capped)),
        "put_oi_premium": _round_money(sum(float(row.get("put_oi_premium") or 0.0) for row in capped)),
        "net_oi_premium": _round_money(sum(float(row.get("net_oi_premium") or 0.0) for row in capped)),
    }

    latest_snapshot_ts = max(
        (str(row.get("snapshot_ts") or "") for row in capped if row.get("snapshot_ts")),
        default=None,
    )
    return {
        "available": bool(capped),
        "source": SOURCE,
        "source_note": SOURCE_NOTE,
        "premium_proxy_note": PREMIUM_PROXY_NOTE,
        "latest_snapshot_ts": latest_snapshot_ts,
        "count": len(capped),
        "eligible_count": len(output_rows),
        "rows": capped,
        "totals": totals,
        "filters": {"limit": resolved_limit, "sort_by": resolved_sort_by},
    }


def build_options_positioning_context(
    conn: sqlite3.Connection,
    ticker: str,
) -> dict[str, Any]:
    """Return MarketChameleon-style IV/OI context for one ticker.

    The rank math uses the local history in ``options_iv``. A rank of 100 means
    the latest value is higher than all observed local snapshots; ``None`` means
    there is not enough snapshot history yet.
    """
    symbol = str(ticker or "").upper().strip()
    if not symbol:
        return {"available": False, "source": SOURCE, "note": "No ticker supplied"}
    if not _table_exists(conn, "options_iv"):
        return {"available": False, "source": SOURCE, "note": "options_iv table missing"}

    snapshots = _snapshot_aggregates(conn, symbol)
    if not snapshots:
        return {"available": False, "source": SOURCE, "note": f"No option snapshots for {symbol}"}

    latest = snapshots[-1]
    latest_iv = _safe_float(latest.get("atm_iv")) or _safe_float(latest.get("avg_iv"))
    avg_iv = _safe_float(latest.get("avg_iv"))
    total_oi = _safe_float(latest.get("total_oi")) or 0.0
    call_oi = _safe_float(latest.get("call_oi")) or 0.0
    put_oi = _safe_float(latest.get("put_oi")) or 0.0
    total_volume = _safe_float(latest.get("total_volume")) or 0.0

    iv_history = [
        value
        for snap in snapshots
        if (value := (_safe_float(snap.get("atm_iv")) or _safe_float(snap.get("avg_iv")))) is not None
    ]
    oi_history = [
        value
        for snap in snapshots
        if (value := _safe_float(snap.get("total_oi"))) is not None
    ]

    iv_rank = _percent_rank(iv_history, latest_iv)
    oi_rank = _percent_rank(oi_history, total_oi)
    put_call = (put_oi / call_oi) if call_oi > 0 else None
    rv20 = _realized_vol_pct(conn, symbol, lookback=20)
    rv1y = _realized_vol_pct(conn, symbol, lookback=252)
    iv_pct = round(latest_iv * 100.0, 1) if latest_iv is not None else None
    avg_iv_pct = round(avg_iv * 100.0, 1) if avg_iv is not None else None
    iv_minus_rv20 = round(iv_pct - rv20, 1) if iv_pct is not None and rv20 is not None else None

    underpriced_score = 50.0
    if iv_rank is not None:
        underpriced_score += (50.0 - iv_rank) * 0.45
    if oi_rank is not None:
        underpriced_score += (oi_rank - 50.0) * 0.25
    if iv_minus_rv20 is not None:
        if iv_minus_rv20 <= -10.0:
            underpriced_score += 10.0
        elif iv_minus_rv20 >= 20.0:
            underpriced_score -= 12.0
    underpriced_score = round(max(0.0, min(100.0, underpriced_score)), 1)

    if put_call is None:
        skew = "unknown"
    elif put_call <= 0.7:
        skew = "call_oi_skew"
    elif put_call >= 1.3:
        skew = "put_oi_skew"
    else:
        skew = "balanced"

    if iv_rank is not None and oi_rank is not None and iv_rank <= 30 and oi_rank >= 60:
        signal = "underpriced_positioning"
        interpretation = "Open interest is elevated while implied volatility is subdued versus local history."
    elif iv_rank is not None and iv_rank <= 30:
        signal = "subdued_iv"
        interpretation = "Implied volatility is low versus local history; options may be cheaper than usual."
    elif iv_rank is not None and iv_rank >= 70:
        signal = "elevated_iv_event_risk"
        interpretation = "Implied volatility is elevated versus local history; options may already price a large move."
    elif oi_rank is not None and oi_rank >= 80:
        signal = "crowded_open_interest"
        interpretation = "Open interest is high versus local history; watch pinning or crowded positioning risk."
    else:
        signal = "neutral"
        interpretation = "Options positioning is not extreme on the available local history."

    return {
        "available": True,
        "source": SOURCE,
        "source_note": SOURCE_NOTE,
        "ticker": symbol,
        "latest_snapshot_ts": latest.get("snapshot_ts"),
        "contracts": int(latest.get("contracts") or 0),
        "historical_snapshots": len(snapshots),
        "atm_iv_pct": iv_pct,
        "avg_iv_pct": avg_iv_pct,
        "iv_rank_pct": iv_rank,
        "oi_rank_pct": oi_rank,
        "total_open_interest": round(total_oi, 1),
        "call_open_interest": round(call_oi, 1),
        "put_open_interest": round(put_oi, 1),
        "put_call_oi_ratio": round(put_call, 2) if put_call is not None else None,
        "option_volume": round(total_volume, 1),
        "realized_vol_20d_pct": rv20,
        "realized_vol_1y_pct": rv1y,
        "iv_minus_rv_20d_pct": iv_minus_rv20,
        "positioning_skew": skew,
        "underpriced_score": underpriced_score,
        "signal": signal,
        "interpretation": interpretation,
    }

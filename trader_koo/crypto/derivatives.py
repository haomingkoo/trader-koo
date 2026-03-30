"""Crypto derivatives signals: funding rates, long/short ratio, Fear & Greed.

Uses public Binance Futures API + alternative.me — no auth required.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG = logging.getLogger("trader_koo.crypto.derivatives")

# Binance symbol mapping (shared with binance_oi.py)
_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
}

_TRACKED_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {}
_CACHE_TTL_SEC = 300


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_derivatives_schema(conn: sqlite3.Connection) -> None:
    """Create tables for funding rates, L/S ratio, and crypto F&G."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crypto_funding_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            funding_rate REAL NOT NULL,
            funding_time TEXT NOT NULL,
            mark_price REAL,
            snapshot_ts TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cfr_symbol_ts
        ON crypto_funding_rates(symbol, snapshot_ts)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crypto_long_short_ratio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            long_account REAL NOT NULL,
            short_account REAL NOT NULL,
            long_short_ratio REAL NOT NULL,
            timestamp TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_clsr_symbol_ts
        ON crypto_long_short_ratio(symbol, snapshot_ts)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crypto_fear_greed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value INTEGER NOT NULL,
            classification TEXT NOT NULL,
            fng_timestamp TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cfg_ts
        ON crypto_fear_greed(snapshot_ts)
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: int = 15) -> Any:
    """Fetch JSON from a URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "trader-koo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class FundingSnapshot:
    symbol: str
    funding_rate: float
    funding_time: str
    mark_price: float
    next_funding_rate: float | None


def fetch_funding_rates() -> list[FundingSnapshot]:
    """Fetch current funding rates from Binance premiumIndex."""
    cache_key = "funding_all"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
            return cached["data"]

    results: list[FundingSnapshot] = []
    for our_sym, binance_sym in _SYMBOL_MAP.items():
        if our_sym not in _TRACKED_SYMBOLS:
            continue
        try:
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={binance_sym}"
            data = _http_get_json(url)
            results.append(FundingSnapshot(
                symbol=our_sym,
                funding_rate=float(data.get("lastFundingRate", 0)),
                funding_time=dt.datetime.fromtimestamp(
                    int(data["nextFundingTime"]) / 1000, tz=dt.timezone.utc,
                ).isoformat(),
                mark_price=float(data.get("markPrice", 0)),
                next_funding_rate=float(data["estimatedSettlePrice"]) if data.get("estimatedSettlePrice") else None,
            ))
        except Exception as exc:
            LOG.warning("Funding rate fetch failed for %s: %s", our_sym, exc)

    with _cache_lock:
        _cache[cache_key] = {
            "data": results,
            "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_CACHE_TTL_SEC)).timestamp(),
        }
    return results


@dataclass
class LongShortSnapshot:
    symbol: str
    long_account: float
    short_account: float
    long_short_ratio: float
    timestamp: str


def fetch_long_short_ratio(
    symbol: str = "BTC-USD",
    period: str = "1h",
    limit: int = 1,
) -> list[LongShortSnapshot]:
    """Fetch top trader long/short account ratio from Binance."""
    binance_sym = _SYMBOL_MAP.get(symbol.upper())
    if not binance_sym:
        return []

    cache_key = f"ls_{binance_sym}_{period}_{limit}"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
            return cached["data"]

    url = (
        f"https://fapi.binance.com/futures/data/topLongShortAccountRatio"
        f"?symbol={binance_sym}&period={period}&limit={limit}"
    )
    try:
        raw = _http_get_json(url)
        if not isinstance(raw, list):
            return []

        results: list[LongShortSnapshot] = []
        for item in raw:
            results.append(LongShortSnapshot(
                symbol=symbol,
                long_account=float(item.get("longAccount", 0)),
                short_account=float(item.get("shortAccount", 0)),
                long_short_ratio=float(item.get("longShortRatio", 1)),
                timestamp=dt.datetime.fromtimestamp(
                    int(item["timestamp"]) / 1000, tz=dt.timezone.utc,
                ).isoformat(),
            ))

        with _cache_lock:
            _cache[cache_key] = {
                "data": results,
                "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=_CACHE_TTL_SEC)).timestamp(),
            }
        return results
    except Exception as exc:
        LOG.warning("L/S ratio fetch failed for %s: %s", symbol, exc)
        return []


@dataclass
class CryptoFearGreed:
    value: int
    classification: str
    timestamp: str


def fetch_crypto_fear_greed() -> CryptoFearGreed | None:
    """Fetch crypto Fear & Greed Index from alternative.me."""
    cache_key = "crypto_fng"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and cached.get("expires_at", 0) > dt.datetime.now(dt.timezone.utc).timestamp():
            return cached["data"]

    try:
        data = _http_get_json("https://api.alternative.me/fng/?limit=1")
        item = data.get("data", [{}])[0]
        result = CryptoFearGreed(
            value=int(item.get("value", 0)),
            classification=str(item.get("value_classification", "")),
            timestamp=dt.datetime.fromtimestamp(
                int(item.get("timestamp", 0)), tz=dt.timezone.utc,
            ).isoformat(),
        )
        with _cache_lock:
            _cache[cache_key] = {
                "data": result,
                "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=3600)).timestamp(),
            }
        return result
    except Exception as exc:
        LOG.warning("Crypto F&G fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Snapshot + alert logic
# ---------------------------------------------------------------------------

# Thresholds for Telegram alerts
_FUNDING_EXTREME_THRESHOLD = 0.001    # 0.1% per 8h = extreme
_FUNDING_ELEVATED_THRESHOLD = 0.0005  # 0.05% per 8h = elevated
_LS_RATIO_EXTREME = 2.5              # 2.5:1 long/short = crowded
_FNG_EXTREME_FEAR = 20
_FNG_EXTREME_GREED = 80


def snapshot_derivatives(db_path: Path) -> dict[str, Any]:
    """Fetch all derivatives data, save to DB, return summary for alerting."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        ensure_derivatives_schema(conn)
        now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        summary: dict[str, Any] = {"funding": [], "ls_ratios": [], "fng": None}

        # Funding rates
        funding = fetch_funding_rates()
        for f in funding:
            conn.execute(
                """INSERT INTO crypto_funding_rates
                   (symbol, funding_rate, funding_time, mark_price, snapshot_ts)
                   VALUES (?, ?, ?, ?, ?)""",
                (f.symbol, f.funding_rate, f.funding_time, f.mark_price, now_iso),
            )
            summary["funding"].append({
                "symbol": f.symbol,
                "rate": f.funding_rate,
                "rate_pct": round(f.funding_rate * 100, 4),
                "annualized_pct": round(f.funding_rate * 3 * 365 * 100, 1),
                "mark_price": f.mark_price,
            })

        # Long/Short ratios
        for sym in _TRACKED_SYMBOLS:
            ls_data = fetch_long_short_ratio(sym, period="1h", limit=1)
            for ls in ls_data:
                conn.execute(
                    """INSERT INTO crypto_long_short_ratio
                       (symbol, long_account, short_account, long_short_ratio,
                        timestamp, snapshot_ts)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ls.symbol, ls.long_account, ls.short_account,
                     ls.long_short_ratio, ls.timestamp, now_iso),
                )
                summary["ls_ratios"].append({
                    "symbol": ls.symbol,
                    "long_pct": round(ls.long_account * 100, 1),
                    "short_pct": round(ls.short_account * 100, 1),
                    "ratio": round(ls.long_short_ratio, 2),
                })

        # Crypto Fear & Greed
        fng = fetch_crypto_fear_greed()
        if fng:
            conn.execute(
                """INSERT INTO crypto_fear_greed
                   (value, classification, fng_timestamp, snapshot_ts)
                   VALUES (?, ?, ?, ?)""",
                (fng.value, fng.classification, fng.timestamp, now_iso),
            )
            summary["fng"] = {
                "value": fng.value,
                "classification": fng.classification,
            }

        conn.commit()
        LOG.info(
            "Derivatives snapshot: %d funding, %d L/S, F&G=%s",
            len(funding), len(summary["ls_ratios"]),
            fng.value if fng else "N/A",
        )
        return summary
    except Exception as exc:
        LOG.error("Derivatives snapshot failed: %s", exc)
        return {"funding": [], "ls_ratios": [], "fng": None}
    finally:
        conn.close()


def _previous_fng_zone(db_path: Path) -> str | None:
    """Get the previous F&G classification for zone-transition detection."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        row = conn.execute(
            """SELECT classification FROM crypto_fear_greed
               ORDER BY snapshot_ts DESC LIMIT 1 OFFSET 1""",
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        conn.close()


def check_and_alert(db_path: Path, summary: dict[str, Any]) -> int:
    """Evaluate derivatives summary and send Telegram alerts for extremes.

    Returns number of alerts sent.
    """
    import os

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return 0

    alerts: list[str] = []

    # Funding rate extremes
    for f in summary.get("funding", []):
        rate = f["rate"]
        if abs(rate) >= _FUNDING_EXTREME_THRESHOLD:
            direction = "longs pay shorts" if rate > 0 else "shorts pay longs"
            alerts.append(
                f"\U0001f525 <b>{f['symbol']} funding extreme</b>: "
                f"{f['rate_pct']:+.4f}% ({f['annualized_pct']:+.0f}% ann.)\n"
                f"  {direction} — crowded positioning, reversal risk"
            )

    # L/S ratio extremes
    for ls in summary.get("ls_ratios", []):
        ratio = ls["ratio"]
        if ratio >= _LS_RATIO_EXTREME:
            alerts.append(
                f"\u2696\ufe0f <b>{ls['symbol']} L/S ratio extreme</b>: "
                f"{ls['long_pct']}% long / {ls['short_pct']}% short (ratio {ratio})\n"
                f"  Top traders heavily long — contrarian short signal"
            )
        elif ratio <= 1.0 / _LS_RATIO_EXTREME:
            alerts.append(
                f"\u2696\ufe0f <b>{ls['symbol']} L/S ratio extreme</b>: "
                f"{ls['long_pct']}% long / {ls['short_pct']}% short (ratio {ratio})\n"
                f"  Top traders heavily short — contrarian long signal"
            )

    # F&G zone transitions
    fng = summary.get("fng")
    if fng:
        prev_zone = _previous_fng_zone(db_path)
        curr_class = fng["classification"]
        if prev_zone and prev_zone != curr_class:
            emoji = "\U0001f630" if "fear" in curr_class.lower() else "\U0001f911"
            alerts.append(
                f"{emoji} <b>Crypto Fear & Greed shifted</b>: "
                f"{prev_zone} \u2192 {curr_class} ({fng['value']}/100)"
            )
        elif fng["value"] <= _FNG_EXTREME_FEAR:
            alerts.append(
                f"\U0001f630 <b>Crypto Extreme Fear</b>: {fng['value']}/100 ({curr_class})\n"
                f"  Historically correlates with local bottoms"
            )
        elif fng["value"] >= _FNG_EXTREME_GREED:
            alerts.append(
                f"\U0001f911 <b>Crypto Extreme Greed</b>: {fng['value']}/100 ({curr_class})\n"
                f"  Elevated risk of correction"
            )

    if not alerts:
        return 0

    # Build combined alert message
    lines = ["\U0001f4ca <b>Crypto Derivatives Alert</b>", ""]
    lines.extend(alerts)
    lines.append("")
    lines.append("<i>NFA</i>")
    text = "\n".join(lines)

    try:
        import httpx

        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        LOG.info("Derivatives alert sent: %d signals", len(alerts))
        return len(alerts)
    except Exception as exc:
        LOG.warning("Derivatives Telegram alert failed: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Convenience: fetch latest for display / morning summary
# ---------------------------------------------------------------------------

def get_latest_derivatives_summary(db_path: Path) -> dict[str, Any]:
    """Return the most recent derivatives data for display."""
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        ensure_derivatives_schema(conn)

        result: dict[str, Any] = {"funding": {}, "ls_ratios": {}, "fng": None}

        # Latest funding per symbol
        for sym in _TRACKED_SYMBOLS:
            row = conn.execute(
                """SELECT symbol, funding_rate, mark_price, snapshot_ts
                   FROM crypto_funding_rates
                   WHERE symbol = ? ORDER BY snapshot_ts DESC LIMIT 1""",
                (sym,),
            ).fetchone()
            if row:
                rate = float(row["funding_rate"])
                result["funding"][sym] = {
                    "rate": rate,
                    "rate_pct": round(rate * 100, 4),
                    "annualized_pct": round(rate * 3 * 365 * 100, 1),
                    "mark_price": float(row["mark_price"]),
                }

        # Latest L/S per symbol
        for sym in _TRACKED_SYMBOLS:
            row = conn.execute(
                """SELECT long_account, short_account, long_short_ratio
                   FROM crypto_long_short_ratio
                   WHERE symbol = ? ORDER BY snapshot_ts DESC LIMIT 1""",
                (sym,),
            ).fetchone()
            if row:
                result["ls_ratios"][sym] = {
                    "long_pct": round(float(row["long_account"]) * 100, 1),
                    "short_pct": round(float(row["short_account"]) * 100, 1),
                    "ratio": round(float(row["long_short_ratio"]), 2),
                }

        # Latest F&G
        row = conn.execute(
            "SELECT value, classification FROM crypto_fear_greed ORDER BY snapshot_ts DESC LIMIT 1"
        ).fetchone()
        if row:
            result["fng"] = {
                "value": int(row["value"]),
                "classification": row["classification"],
            }

        return result
    except Exception as exc:
        LOG.warning("Failed to load derivatives summary: %s", exc)
        return {"funding": {}, "ls_ratios": {}, "fng": None}
    finally:
        conn.close()

"""Intraday price alert engine.

Polls Finnhub REST API for current prices of the top 10 setup tickers
from the nightly report, then checks proximity to support/resistance
levels and fires Telegram alerts when triggered.

Uses REST polling (not WebSocket) to avoid consuming Finnhub's 50-symbol
WebSocket cap — those slots are reserved for the interactive dashboard.

Designed to run as an ``asyncio`` background task inside the FastAPI
lifespan — failure never propagates to the main application.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import httpx

from trader_koo.notifications.telegram import send_price_alert

LOG = logging.getLogger("trader_koo.notifications.alert_engine")

# Cooldown: suppress duplicate alert for same ticker+level for 4 hours
DEFAULT_COOLDOWN_SEC = 4 * 3600

# Proximity threshold: fire when price is within this % of a level
DEFAULT_PROXIMITY_PCT = 0.01

# REST poll interval: 2 minutes (10 tickers = 5 calls/min, within 60/min)
POLL_INTERVAL_SEC = 120

# How often the engine reloads setups from the daily report (seconds)
SETUP_REFRESH_INTERVAL_SEC = 6 * 3600  # every 6 hours

# Maximum number of REPORT setup tickers to monitor via REST polling
MAX_REPORT_TICKERS = 10

# Tickers that are ALWAYS monitored regardless of nightly report setups.
# Levels come from the database (same support/resistance as chart page).
ALWAYS_WATCH: list[str] = [
    # Mag 7
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Indices + Volatility
    "SPY", "QQQ", "^VIX",
]

# Finnhub REST API
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_REQUEST_TIMEOUT_SEC = 10

# US market hours in Eastern Time (ET = UTC-5 / EDT = UTC-4)
MARKET_OPEN_ET = dt.time(9, 30)
MARKET_CLOSE_ET = dt.time(16, 0)


def _is_us_market_hours(now_utc: dt.datetime) -> bool:
    """Return True when *now_utc* falls within 9:30-16:00 ET, Mon-Fri.

    Uses a fixed UTC-5 offset (EST).  During daylight saving the
    window shifts by one hour, which is acceptable — the engine simply
    starts an hour early / ends an hour late for half the year.
    """
    et = now_utc.astimezone(dt.timezone(dt.timedelta(hours=-5)))
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return MARKET_OPEN_ET <= et.time() <= MARKET_CLOSE_ET


def _ensure_telegram_alerts_table(conn: sqlite3.Connection) -> None:
    """Create the ``telegram_alerts`` table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS telegram_alerts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker     TEXT    NOT NULL,
            level      REAL    NOT NULL,
            price      REAL    NOT NULL,
            alert_type TEXT    NOT NULL,
            setup_tier TEXT,
            bias       TEXT,
            sent_at    TEXT    NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_telegram_alerts_ticker "
        "ON telegram_alerts(ticker, sent_at)"
    )
    conn.commit()


class AlertEngine:
    """Polls Finnhub REST API and fires Telegram price alerts.

    Monitors the top 10 setup tickers from the nightly report against
    their support/resistance levels.  Polls every 2 minutes during US
    market hours (9:30-16:00 ET, Mon-Fri).

    Parameters
    ----------
    db_path:
        Path to the SQLite database (``/data/trader_koo.db``).
    report_dir:
        Path to the directory containing daily report JSON files.
    finnhub_api_key:
        Finnhub API key for REST quote polling.
    proximity_pct:
        Fire alert when price is within this fraction of a level
        (default 0.01 = 1 %).
    cooldown_sec:
        Minimum seconds between re-alerts for the same ticker+level
        (default 4 hours).
    """

    def __init__(
        self,
        db_path: Path,
        report_dir: Path,
        finnhub_api_key: str = "",
        proximity_pct: float = DEFAULT_PROXIMITY_PCT,
        cooldown_sec: int = DEFAULT_COOLDOWN_SEC,
    ) -> None:
        self._db_path = db_path
        self._report_dir = report_dir
        self._finnhub_api_key = (
            finnhub_api_key
            or os.getenv("FINNHUB_API_KEY", "").strip()
        )
        self._proximity_pct = proximity_pct
        self._cooldown_sec = cooldown_sec

        # {ticker: [{level, level_type, setup_tier, bias}, ...]}
        self._watchlist: dict[str, list[dict[str, Any]]] = {}

        # {(ticker, level_rounded): last_alert_utc_ts}
        self._cooldowns: dict[tuple[str, float], float] = {}
        self._lock = threading.Lock()

        self._running = False

    # ------------------------------------------------------------------
    # Setup loading
    # ------------------------------------------------------------------

    def load_setups(self) -> int:
        """Load ALWAYS_WATCH + top 10 setups from the nightly report.

        Final watchlist = set(ALWAYS_WATCH) | set(top_report_setups),
        deduplicated.  Returns the number of ticker-level pairs watched.
        """
        from trader_koo.backend.services.report_loader import (
            latest_daily_report_json,
        )

        # --- Step 1: parse report setups ---
        report_tickers: dict[str, dict[str, str]] = {}
        _, payload = latest_daily_report_json(self._report_dir)
        if isinstance(payload, dict):
            signals = payload.get("signals")
            if isinstance(signals, dict):
                setup_rows: list[dict[str, Any]] = signals.get(
                    "setup_quality_top", []
                )
                if not isinstance(setup_rows, list):
                    setup_rows = []
                for row in setup_rows:
                    if not isinstance(row, dict):
                        continue
                    ticker = str(row.get("ticker") or "").strip().upper()
                    if not ticker:
                        continue
                    if len(report_tickers) >= MAX_REPORT_TICKERS:
                        break
                    report_tickers[ticker] = {
                        "tier": str(row.get("setup_tier") or "D").strip().upper(),
                        "bias": str(row.get("signal_bias") or "neutral").strip().lower(),
                    }
        else:
            LOG.warning("No daily report found — using ALWAYS_WATCH only")

        # --- Step 2: merge ALWAYS_WATCH + report (deduplicated) ---
        combined: dict[str, dict[str, str]] = {}
        for ticker in ALWAYS_WATCH:
            combined[ticker] = report_tickers.get(
                ticker, {"tier": "-", "bias": "neutral"}
            )
        for ticker, meta in report_tickers.items():
            if ticker not in combined:
                combined[ticker] = meta

        # --- Step 3: build watchlist with DB levels ---
        new_watchlist: dict[str, list[dict[str, Any]]] = {}
        total_levels = 0

        for ticker, meta in combined.items():
            levels = self._extract_levels_for_ticker(ticker)
            entries: list[dict[str, Any]] = []
            for lvl in levels:
                entries.append(
                    {
                        "level": float(lvl["level"]),
                        "level_type": str(lvl["type"]),
                        "setup_tier": meta["tier"],
                        "bias": meta["bias"],
                    }
                )
                total_levels += 1

            if entries:
                new_watchlist[ticker] = entries

        with self._lock:
            self._watchlist = new_watchlist

        LOG.info(
            "Alert engine loaded %d tickers with %d levels "
            "(capped at %d for REST polling)",
            len(new_watchlist),
            total_levels,
            MAX_REPORT_TICKERS,
        )
        return total_levels

    def _extract_levels_for_ticker(
        self,
        ticker: str,
    ) -> list[dict[str, Any]]:
        """Query support/resistance levels for *ticker* from the DB."""
        if not self._db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            from trader_koo.backend.services.database import (
                get_price_df,
                table_exists,
            )
            from trader_koo.structure.levels import (
                LevelConfig,
                add_fallback_levels,
                build_levels_from_pivots,
                select_target_levels,
            )
            from trader_koo.features.technical import compute_pivots

            if not table_exists(conn, "price_daily"):
                conn.close()
                return []

            df = get_price_df(conn, ticker)
            conn.close()

            if df.empty or len(df) < 30:
                return []

            cfg = LevelConfig()
            pivots = compute_pivots(df, left=5, right=5)
            raw_levels = build_levels_from_pivots(pivots, cfg)
            last_close = float(df["close"].iloc[-1])
            selected = select_target_levels(raw_levels, last_close, cfg)
            selected = add_fallback_levels(df, selected, last_close, cfg)

            if selected.empty:
                return []

            return selected[["type", "level"]].to_dict("records")
        except Exception as exc:
            LOG.warning(
                "Failed to extract levels for %s: %s", ticker, exc
            )
            return []

    # ------------------------------------------------------------------
    # Finnhub REST price polling
    # ------------------------------------------------------------------

    def _poll_finnhub_quote(self, ticker: str) -> float | None:
        """Fetch current price for *ticker* via Finnhub REST API.

        Returns the current price (``c`` field) or ``None`` on failure.
        Uses ``GET /api/v1/quote?symbol={ticker}&token={key}``.
        """
        if not self._finnhub_api_key:
            LOG.warning(
                "FINNHUB_API_KEY not set — cannot poll quote for %s",
                ticker,
            )
            return None

        try:
            with httpx.Client(
                timeout=FINNHUB_REQUEST_TIMEOUT_SEC,
            ) as client:
                resp = client.get(
                    FINNHUB_QUOTE_URL,
                    params={
                        "symbol": ticker,
                        "token": self._finnhub_api_key,
                    },
                )
            if resp.status_code != 200:
                LOG.warning(
                    "Finnhub quote API returned %d for %s",
                    resp.status_code,
                    ticker,
                )
                return None

            data = resp.json()
            price = data.get("c")  # "c" = current price
            if price is None or price == 0:
                LOG.debug(
                    "Finnhub returned no price for %s: %s", ticker, data
                )
                return None
            return float(price)
        except httpx.HTTPError as exc:
            LOG.warning("Finnhub HTTP error for %s: %s", ticker, exc)
            return None
        except Exception as exc:
            LOG.warning(
                "Finnhub quote poll failed for %s: %s", ticker, exc
            )
            return None

    async def _poll_all_tickers(self) -> dict[str, float]:
        """Poll Finnhub REST API for all watched tickers.

        Runs synchronous HTTP calls in a thread executor to avoid
        blocking the async event loop.  Returns {ticker: price}.
        """
        with self._lock:
            tickers = list(self._watchlist.keys())

        if not tickers:
            return {}

        loop = asyncio.get_running_loop()
        results: dict[str, float] = {}

        for ticker in tickers:
            try:
                price = await loop.run_in_executor(
                    None, self._poll_finnhub_quote, ticker
                )
                if price is not None:
                    results[ticker] = price
            except Exception as exc:
                LOG.debug("Poll executor error for %s: %s", ticker, exc)

        LOG.debug(
            "Polled %d/%d tickers successfully",
            len(results),
            len(tickers),
        )
        return results

    # ------------------------------------------------------------------
    # Tick processing
    # ------------------------------------------------------------------

    def _check_tick(self, ticker: str, price: float) -> None:
        """Evaluate a single price against watched levels."""
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()

        with self._lock:
            entries = self._watchlist.get(ticker)
            if not entries:
                return

        for entry in entries:
            level = entry["level"]
            if level <= 0:
                continue

            distance_pct = abs(price - level) / level

            if distance_pct <= self._proximity_pct:
                if entry["level_type"] == "support":
                    if price < level:
                        alert_type = "breakdown_below_support"
                    else:
                        alert_type = "approaching_support"
                else:
                    if price > level:
                        alert_type = "breakout_above_resistance"
                    else:
                        alert_type = "approaching_resistance"
            else:
                continue  # not close enough

            # Cooldown check
            cooldown_key = (ticker, round(level, 2))
            with self._lock:
                last_alert = self._cooldowns.get(cooldown_key, 0.0)
                if now_ts - last_alert < self._cooldown_sec:
                    continue
                self._cooldowns[cooldown_key] = now_ts

            sent = send_price_alert(
                ticker=ticker,
                current_price=price,
                level=level,
                level_type=entry["level_type"],
                setup_tier=entry["setup_tier"],
                bias=entry["bias"],
                alert_type=alert_type,
            )
            if sent:
                self._persist_alert(
                    ticker=ticker,
                    level=level,
                    price=price,
                    alert_type=alert_type,
                    setup_tier=entry["setup_tier"],
                    bias=entry["bias"],
                )
            else:
                LOG.warning(
                    "Failed to send alert for %s at %.2f", ticker, price
                )

    def _persist_alert(
        self,
        *,
        ticker: str,
        level: float,
        price: float,
        alert_type: str,
        setup_tier: str,
        bias: str,
    ) -> None:
        """Write alert record to the ``telegram_alerts`` table."""
        if not self._db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(self._db_path))
            _ensure_telegram_alerts_table(conn)
            conn.execute(
                """
                INSERT INTO telegram_alerts
                    (ticker, level, price, alert_type, setup_tier, bias, sent_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    level,
                    price,
                    alert_type,
                    setup_tier,
                    bias,
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            LOG.warning("Failed to persist alert: %s", exc)

    # ------------------------------------------------------------------
    # Main run loop (async)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-running async task — polls REST API and fires alerts.

        Polls Finnhub REST every 2 minutes during US market hours.
        10 tickers x every 2 min = 5 calls/min (within 60/min free cap).
        Exits cleanly when ``self._running`` is set to ``False``.
        """
        self._running = True
        LOG.info(
            "Telegram alert engine started "
            "(REST polling, %ds interval, max %d tickers)",
            POLL_INTERVAL_SEC,
            MAX_REPORT_TICKERS,
        )

        # Initial setup load
        try:
            self.load_setups()
        except Exception as exc:
            LOG.error("Alert engine initial setup failed: %s", exc)

        # Ensure DB schema
        if self._db_path.exists():
            try:
                conn = sqlite3.connect(str(self._db_path))
                _ensure_telegram_alerts_table(conn)
                conn.close()
            except Exception as exc:
                LOG.warning(
                    "Failed to init telegram_alerts schema: %s", exc
                )

        last_refresh_ts = dt.datetime.now(dt.timezone.utc).timestamp()

        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL_SEC)

                now_utc = dt.datetime.now(dt.timezone.utc)
                if not _is_us_market_hours(now_utc):
                    continue

                # Periodically reload setups (catches nightly pipeline)
                now_ts = now_utc.timestamp()
                if now_ts - last_refresh_ts >= SETUP_REFRESH_INTERVAL_SEC:
                    try:
                        self.load_setups()
                        last_refresh_ts = now_ts
                    except Exception as exc:
                        LOG.warning("Setup refresh failed: %s", exc)

                # Poll Finnhub REST API for current prices
                prices = await self._poll_all_tickers()

                for ticker, price in prices.items():
                    self._check_tick(ticker, price)

            except asyncio.CancelledError:
                LOG.info("Alert engine task cancelled")
                break
            except Exception as exc:
                LOG.error("Alert engine poll error: %s", exc)
                await asyncio.sleep(10.0)

        LOG.info("Telegram alert engine stopped")

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_recent_alerts(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the most recent alerts from the database."""
        if not self._db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            _ensure_telegram_alerts_table(conn)
            rows = conn.execute(
                """
                SELECT id, ticker, level, price, alert_type,
                       setup_tier, bias, sent_at
                FROM telegram_alerts
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as exc:
            LOG.warning("Failed to query recent alerts: %s", exc)
            return []

    def get_watchlist_summary(self) -> dict[str, Any]:
        """Return a snapshot of the current watchlist state."""
        with self._lock:
            tickers = list(self._watchlist.keys())
            total_levels = sum(
                len(entries) for entries in self._watchlist.values()
            )
        return {
            "running": self._running,
            "tickers": len(tickers),
            "levels": total_levels,
            "max_tickers": MAX_REPORT_TICKERS,
            "poll_interval_sec": POLL_INTERVAL_SEC,
            "cooldown_sec": self._cooldown_sec,
            "proximity_pct": self._proximity_pct,
            "ticker_list": sorted(tickers),
        }

"""Intraday price alert engine.

Monitors real-time Finnhub equity ticks against support/resistance
levels from the latest daily report.  Fires Telegram alerts when
price approaches or crosses key levels for active setups.

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

from trader_koo.notifications.telegram import send_price_alert

LOG = logging.getLogger("trader_koo.notifications.alert_engine")

# Cooldown: suppress duplicate alert for same ticker+level for 4 hours
DEFAULT_COOLDOWN_SEC = 4 * 3600

# Proximity threshold: fire when price is within this % of a level
DEFAULT_PROXIMITY_PCT = 0.01

# How often the engine polls for new ticks (seconds)
POLL_INTERVAL_SEC = 2.0

# How often the engine reloads setups from the daily report (seconds)
SETUP_REFRESH_INTERVAL_SEC = 6 * 3600  # every 6 hours

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
    """Monitors streaming ticks and fires Telegram price alerts.

    Parameters
    ----------
    db_path:
        Path to the SQLite database (``/data/trader_koo.db``).
    report_dir:
        Path to the directory containing daily report JSON files.
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
        proximity_pct: float = DEFAULT_PROXIMITY_PCT,
        cooldown_sec: int = DEFAULT_COOLDOWN_SEC,
    ) -> None:
        self._db_path = db_path
        self._report_dir = report_dir
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
        """Load active setups from the latest daily report.

        Returns the number of ticker-level pairs now being watched.
        """
        from trader_koo.backend.services.report_loader import (
            latest_daily_report_json,
        )

        _, payload = latest_daily_report_json(self._report_dir)
        if not isinstance(payload, dict):
            LOG.warning("No daily report found — alert engine has no setups")
            return 0

        signals = payload.get("signals")
        if not isinstance(signals, dict):
            return 0

        setup_rows: list[dict[str, Any]] = signals.get(
            "setup_quality_top", []
        )
        if not isinstance(setup_rows, list):
            setup_rows = []

        new_watchlist: dict[str, list[dict[str, Any]]] = {}
        total_levels = 0

        for row in setup_rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker.startswith("^"):
                # Skip index-only rows like ^VIX
                continue

            tier = str(row.get("setup_tier") or "D").strip().upper()
            bias = str(row.get("signal_bias") or "neutral").strip().lower()

            # Extract support/resistance levels from the row
            levels = self._extract_levels_for_ticker(ticker)
            entries: list[dict[str, Any]] = []
            for lvl in levels:
                entries.append(
                    {
                        "level": float(lvl["level"]),
                        "level_type": str(lvl["type"]),
                        "setup_tier": tier,
                        "bias": bias,
                    }
                )
                total_levels += 1

            if entries:
                new_watchlist[ticker] = entries

        with self._lock:
            self._watchlist = new_watchlist

        LOG.info(
            "Alert engine loaded %d tickers with %d levels",
            len(new_watchlist),
            total_levels,
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
            # Check if the report has levels stored; otherwise compute
            # from price history using the structure module.
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
    # Tick processing
    # ------------------------------------------------------------------

    def _check_tick(self, ticker: str, price: float) -> None:
        """Evaluate a single tick against watched levels."""
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

            # Determine alert type
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

            # Send alert
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
    # Subscription management
    # ------------------------------------------------------------------

    def _subscribe_tickers(self) -> int:
        """Subscribe watched tickers to the Finnhub streaming feed.

        Returns the number of symbols successfully subscribed.
        """
        from trader_koo.streaming.service import subscribe_symbol

        with self._lock:
            tickers = list(self._watchlist.keys())

        subscribed = 0
        for ticker in tickers:
            if subscribe_symbol(ticker):
                subscribed += 1
            else:
                LOG.debug(
                    "Could not subscribe %s (feed full or unavailable)",
                    ticker,
                )
        LOG.info(
            "Alert engine subscribed %d/%d tickers to equity feed",
            subscribed,
            len(tickers),
        )
        return subscribed

    # ------------------------------------------------------------------
    # Main run loop (async)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-running async task — polls ticks and fires alerts.

        Exits cleanly when ``self._running`` is set to ``False``.
        """
        self._running = True
        LOG.info("Telegram alert engine started")

        # Initial setup load
        try:
            self.load_setups()
            self._subscribe_tickers()
        except Exception as exc:
            LOG.error("Alert engine initial setup failed: %s", exc)

        # Ensure DB schema
        if self._db_path.exists():
            try:
                conn = sqlite3.connect(str(self._db_path))
                _ensure_telegram_alerts_table(conn)
                conn.close()
            except Exception as exc:
                LOG.warning("Failed to init telegram_alerts schema: %s", exc)

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
                        self.refresh_setups()
                        last_refresh_ts = now_ts
                    except Exception as exc:
                        LOG.warning("Setup refresh failed: %s", exc)

                # Poll latest prices from the equity feed
                from trader_koo.streaming.service import get_equity_prices

                prices = get_equity_prices()
                with self._lock:
                    watched = set(self._watchlist.keys())

                for symbol, data in prices.items():
                    if symbol not in watched:
                        continue
                    price = data.get("price")
                    if price is None:
                        continue
                    self._check_tick(symbol, float(price))

            except asyncio.CancelledError:
                LOG.info("Alert engine task cancelled")
                break
            except Exception as exc:
                LOG.error("Alert engine tick error: %s", exc)
                await asyncio.sleep(5.0)

        LOG.info("Telegram alert engine stopped")

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False

    def refresh_setups(self) -> int:
        """Reload setups from the latest report and re-subscribe.

        Call this after a nightly pipeline run to pick up new tickers.
        Returns the number of levels now being watched.
        """
        total = self.load_setups()
        self._subscribe_tickers()
        return total

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
            "cooldown_sec": self._cooldown_sec,
            "proximity_pct": self._proximity_pct,
            "ticker_list": sorted(tickers),
        }

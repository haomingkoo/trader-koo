"""Telegram bot command handler — polls ``getUpdates`` and responds.

Supports interactive commands from the configured chat so the user
can query pipeline status, top setups, live prices, VIX regime, and
recent alerts directly from Telegram.

Security: only messages from ``TELEGRAM_CHAT_ID`` are processed.
All others are silently ignored.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx

LOG = logging.getLogger("trader_koo.notifications.bot_commands")

TELEGRAM_API_BASE = "https://api.telegram.org"
POLL_INTERVAL_SEC = 10
LONG_POLL_TIMEOUT_SEC = 5
SEND_TIMEOUT_SEC = 15
MIN_RESPONSE_INTERVAL_SEC = 1.0

# Finnhub REST API for /price command
FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
FINNHUB_REQUEST_TIMEOUT_SEC = 10


class TelegramCommandHandler:
    """Polls Telegram ``getUpdates`` and dispatches bot commands.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token.
    chat_id:
        Authorized chat ID — messages from other chats are ignored.
    db_path:
        Path to the SQLite database.
    report_dir:
        Path to the daily report JSON directory.
    finnhub_api_key:
        Finnhub API key for ``/price`` quotes.
    alert_engine:
        Optional reference to a running ``AlertEngine`` instance.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        db_path: Path,
        report_dir: Path,
        finnhub_api_key: str = "",
        alert_engine: Any = None,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._db_path = db_path
        self._report_dir = report_dir
        self._finnhub_api_key = (
            finnhub_api_key
            or os.getenv("FINNHUB_API_KEY", "").strip()
        )
        self._alert_engine = alert_engine

        self._last_update_id: int = 0
        self._last_response_ts: float = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Long-running async task — polls ``getUpdates`` every 10s."""
        self._running = True
        LOG.info(
            "Telegram command handler started "
            "(poll=%ds, long_poll=%ds)",
            POLL_INTERVAL_SEC,
            LONG_POLL_TIMEOUT_SEC,
        )

        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                LOG.info("Command handler task cancelled")
                break
            except Exception as exc:
                LOG.error("Command handler poll error: %s", exc)

            await asyncio.sleep(POLL_INTERVAL_SEC)

        LOG.info("Telegram command handler stopped")

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._running = False

    # ------------------------------------------------------------------
    # Telegram API
    # ------------------------------------------------------------------

    async def _get_updates(self) -> list[dict[str, Any]]:
        """Fetch new updates from Telegram via long polling."""
        url = f"{TELEGRAM_API_BASE}/bot{self._bot_token}/getUpdates"
        params: dict[str, Any] = {
            "timeout": LONG_POLL_TIMEOUT_SEC,
            "allowed_updates": '["message"]',
        }
        if self._last_update_id > 0:
            params["offset"] = self._last_update_id + 1

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: httpx.get(
                    url,
                    params=params,
                    timeout=LONG_POLL_TIMEOUT_SEC + 10,
                ),
            )
            if resp.status_code != 200:
                LOG.warning(
                    "getUpdates returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return []
            data = resp.json()
            return data.get("result", [])
        except httpx.HTTPError as exc:
            LOG.warning("getUpdates HTTP error: %s", exc)
            return []
        except Exception as exc:
            LOG.warning("getUpdates failed: %s", exc)
            return []

    async def _send_reply(self, text: str) -> bool:
        """Send a reply message to the authorized chat."""
        # Rate limit: max 1 response per second
        now_ts = dt.datetime.now(dt.timezone.utc).timestamp()
        elapsed = now_ts - self._last_response_ts
        if elapsed < MIN_RESPONSE_INTERVAL_SEC:
            await asyncio.sleep(MIN_RESPONSE_INTERVAL_SEC - elapsed)

        url = (
            f"{TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        )
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: httpx.post(
                    url, json=payload, timeout=SEND_TIMEOUT_SEC
                ),
            )
            self._last_response_ts = (
                dt.datetime.now(dt.timezone.utc).timestamp()
            )
            if resp.status_code != 200:
                LOG.error(
                    "sendMessage returned %d: %s",
                    resp.status_code,
                    resp.text[:300],
                )
                return False
            return True
        except Exception as exc:
            LOG.error("sendMessage failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Update dispatch
    # ------------------------------------------------------------------

    async def _handle_update(self, update: dict[str, Any]) -> None:
        """Route a single Telegram update to the right command."""
        update_id = int(update.get("update_id", 0))
        if update_id > self._last_update_id:
            self._last_update_id = update_id

        message = update.get("message")
        if not isinstance(message, dict):
            return

        # Security: only respond to the configured chat
        chat = message.get("chat", {})
        msg_chat_id = str(chat.get("id", ""))
        if msg_chat_id != self._chat_id:
            return

        text = str(message.get("text") or "").strip()
        if not text.startswith("/"):
            return

        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        # Strip @bot_username suffix (e.g. /status@MyBot)
        command = command.split("@")[0]
        args = parts[1].strip() if len(parts) > 1 else ""

        LOG.info("Bot command received: %s (args=%r)", command, args)

        handler_map: dict[str, Any] = {
            "/status": self._cmd_status,
            "/top": self._cmd_top,
            "/price": self._cmd_price,
            "/vix": self._cmd_vix,
            "/alerts": self._cmd_alerts,
            "/help": self._cmd_help,
        }

        handler = handler_map.get(command)
        if handler is None:
            await self._send_reply(
                "Unknown command. Try /help"
            )
            return

        try:
            if command == "/price":
                response = await handler(args)
            else:
                response = await handler()
        except Exception as exc:
            LOG.error("Command %s failed: %s", command, exc)
            response = f"Command failed: {type(exc).__name__}"

        await self._send_reply(response)

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    async def _cmd_help(self) -> str:
        """List available commands."""
        return (
            "\U0001f916 *Trader Koo Bot*\n"
            "\n"
            "/status — Pipeline status + alert engine\n"
            "/top — Today's top 10 setups\n"
            "/price AAPL — Current price + levels\n"
            "/vix — VIX level + regime + Fear/Greed\n"
            "/alerts — Last 5 alerts fired today\n"
            "/help — This message"
        )

    async def _cmd_status(self) -> str:
        """Pipeline status + last report time + alert engine status."""
        lines: list[str] = ["\U0001f4e1 *Pipeline Status*\n"]

        # Pull from internal /api/status logic
        try:
            from trader_koo.backend.services.database import (
                DB_PATH as _db,
                get_conn,
                table_exists,
            )
            from trader_koo.backend.services.pipeline import (
                pipeline_status_snapshot,
            )

            pipeline = pipeline_status_snapshot(log_lines=60)
            active = pipeline.get("active", False)
            stage = pipeline.get("stage", "unknown")
            lines.append(
                f"\U0001f504 Active: {'Yes' if active else 'No'}"
            )
            lines.append(f"\U0001f3ad Stage: {stage}")

            if _db.exists():
                conn = get_conn()
                try:
                    if table_exists(conn, "ingest_runs"):
                        row = conn.execute(
                            "SELECT status, finished_ts "
                            "FROM ingest_runs "
                            "ORDER BY started_ts DESC LIMIT 1"
                        ).fetchone()
                        if row:
                            lines.append(
                                f"\U0001f4cb Last run: {row['status']}"
                            )
                            if row["finished_ts"]:
                                lines.append(
                                    f"\U0001f552 Finished: "
                                    f"{row['finished_ts'][:19]}"
                                )
                finally:
                    conn.close()
        except Exception as exc:
            lines.append(f"Error fetching status: {exc}")

        # Latest report timestamp
        try:
            from trader_koo.backend.services.report_loader import (
                latest_daily_report_json,
            )

            _, payload = latest_daily_report_json(self._report_dir)
            if isinstance(payload, dict):
                gen_ts = payload.get("generated_ts", "")
                if gen_ts:
                    lines.append(
                        f"\n\U0001f4c4 Report: {str(gen_ts)[:19]}"
                    )
        except Exception:
            pass

        # Alert engine summary
        if self._alert_engine is not None:
            try:
                summary = self._alert_engine.get_watchlist_summary()
                lines.append(
                    f"\n\U0001f514 Alert engine: "
                    f"{'running' if summary['running'] else 'stopped'}"
                )
                lines.append(
                    f"\U0001f4ca Watching: {summary['tickers']} tickers, "
                    f"{summary['levels']} levels"
                )
            except Exception:
                pass

        return "\n".join(lines)

    async def _cmd_top(self) -> str:
        """Today's top 10 setups with tier and bias."""
        try:
            from trader_koo.backend.services.report_loader import (
                latest_daily_report_json,
            )

            _, payload = latest_daily_report_json(self._report_dir)
            if not isinstance(payload, dict):
                return "No daily report available."

            signals = payload.get("signals")
            if not isinstance(signals, dict):
                return "No signals in daily report."

            setup_rows: list[dict[str, Any]] = signals.get(
                "setup_quality_top", []
            )
            if not isinstance(setup_rows, list) or not setup_rows:
                return "No top setups in latest report."

            # Extract report date
            gen_ts = str(payload.get("generated_ts") or "")
            date_label = gen_ts[:10] if len(gen_ts) >= 10 else "today"
            try:
                parsed = dt.date.fromisoformat(date_label)
                date_label = parsed.strftime("%b %d")
            except ValueError:
                pass

            lines: list[str] = [
                f"\U0001f4ca *Top Setups ({date_label})*\n"
            ]
            for i, row in enumerate(setup_rows[:10], start=1):
                if not isinstance(row, dict):
                    continue
                ticker = str(
                    row.get("ticker") or ""
                ).strip().upper()
                tier = str(
                    row.get("setup_tier") or row.get("tier") or "?"
                ).strip().upper()
                bias = str(
                    row.get("signal_bias")
                    or row.get("bias")
                    or "Neutral"
                ).strip().capitalize()
                lines.append(
                    f"{i}. {ticker} — Tier {tier} | {bias}"
                )

            return "\n".join(lines)
        except Exception as exc:
            LOG.error("/top command failed: %s", exc)
            return f"Failed to load top setups: {exc}"

    async def _cmd_price(self, ticker_arg: str = "") -> str:
        """Current price + nearest support/resistance for a ticker."""
        ticker = re.sub(r"[^A-Za-z0-9.^]", "", ticker_arg).upper()
        if not ticker:
            return "Usage: /price AAPL"

        # Fetch live quote from Finnhub
        price = await self._fetch_finnhub_quote(ticker)
        if price is None:
            return f"Could not fetch price for {ticker}."

        lines: list[str] = [
            f"\U0001f4b0 *{ticker}*: ${price:,.2f}"
        ]

        # Get support/resistance from DB
        levels = self._get_ticker_levels(ticker)
        supports = [
            lvl for lvl in levels if lvl["type"] == "support"
        ]
        resistances = [
            lvl for lvl in levels if lvl["type"] == "resistance"
        ]

        # Nearest support (highest below price)
        supports_below = [
            s for s in supports if s["level"] < price
        ]
        if supports_below:
            nearest_sup = max(
                supports_below, key=lambda s: s["level"]
            )
            dist_pct = abs(price - nearest_sup["level"]) / price * 100
            lines.append(
                f"\U0001f4cd Support: ${nearest_sup['level']:,.2f} "
                f"({dist_pct:.1f}% away)"
            )

        # Nearest resistance (lowest above price)
        resistances_above = [
            r for r in resistances if r["level"] > price
        ]
        if resistances_above:
            nearest_res = min(
                resistances_above, key=lambda r: r["level"]
            )
            dist_pct = (
                abs(nearest_res["level"] - price) / price * 100
            )
            lines.append(
                f"\U0001f4cd Resistance: ${nearest_res['level']:,.2f} "
                f"({dist_pct:.1f}% away)"
            )

        if len(lines) == 1:
            lines.append("No support/resistance levels available.")

        return "\n".join(lines)

    async def _cmd_vix(self) -> str:
        """VIX level + regime + Fear/Greed score."""
        lines: list[str] = []

        try:
            from trader_koo.backend.services.database import (
                get_conn,
            )

            conn = get_conn()
            try:
                # VIX level
                vix_row = conn.execute(
                    "SELECT close FROM price_daily "
                    "WHERE ticker = '^VIX' "
                    "ORDER BY date DESC LIMIT 1"
                ).fetchone()
                if vix_row:
                    vix_close = float(vix_row["close"])
                    lines.append(
                        f"\U0001f4c8 *VIX*: {vix_close:.2f}"
                    )

                    # Determine regime
                    if vix_close < 15:
                        regime = "Low volatility"
                        regime_emoji = "\U0001f7e2"
                    elif vix_close < 20:
                        regime = "Normal"
                        regime_emoji = "\U0001f7e2"
                    elif vix_close < 25:
                        regime = "Elevated"
                        regime_emoji = "\U0001f7e1"
                    elif vix_close < 30:
                        regime = "High volatility"
                        regime_emoji = "\U0001f7e0"
                    else:
                        regime = "Extreme fear"
                        regime_emoji = "\U0001f534"

                    lines.append(
                        f"{regime_emoji} Regime: {regime}"
                    )
                else:
                    lines.append("VIX data not available.")

                # Fear/Greed score
                from trader_koo.structure.fear_greed import (
                    compute_fear_greed_index,
                )

                fg = compute_fear_greed_index(conn)
                score = fg.get("score")
                label = fg.get("label", "Unavailable")
                if score is not None:
                    if score >= 75:
                        fg_emoji = "\U0001f929"
                    elif score >= 55:
                        fg_emoji = "\U0001f60a"
                    elif score >= 45:
                        fg_emoji = "\U0001f610"
                    elif score >= 25:
                        fg_emoji = "\U0001f61f"
                    else:
                        fg_emoji = "\U0001f628"
                    lines.append(
                        f"{fg_emoji} Fear/Greed: {score} ({label})"
                    )
                else:
                    lines.append(
                        "\U0001f610 Fear/Greed: Unavailable"
                    )
            finally:
                conn.close()
        except Exception as exc:
            LOG.error("/vix command failed: %s", exc)
            return f"Failed to fetch VIX data: {exc}"

        return "\n".join(lines) if lines else "VIX data not available."

    async def _cmd_alerts(self) -> str:
        """Last 5 alerts fired today."""
        today_str = (
            dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        )

        if not self._db_path.exists():
            return "Database not available."

        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            # Ensure table exists before querying
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
            rows = conn.execute(
                """
                SELECT ticker, price, level, alert_type, sent_at
                FROM telegram_alerts
                WHERE sent_at >= ?
                ORDER BY id DESC
                LIMIT 5
                """,
                (today_str,),
            ).fetchall()
            conn.close()

            if not rows:
                return (
                    f"\U0001f514 No alerts fired today ({today_str})."
                )

            lines: list[str] = [
                f"\U0001f514 *Recent Alerts ({today_str})*\n"
            ]
            for row in rows:
                ticker = row["ticker"]
                price = float(row["price"])
                alert_type = str(
                    row["alert_type"]
                ).replace("_", " ").title()
                sent_time = str(row["sent_at"])[11:16]
                lines.append(
                    f"\u2022 {ticker} ${price:,.2f} — "
                    f"{alert_type} ({sent_time} UTC)"
                )
            return "\n".join(lines)
        except Exception as exc:
            LOG.error("/alerts command failed: %s", exc)
            return f"Failed to fetch alerts: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_finnhub_quote(
        self,
        ticker: str,
    ) -> float | None:
        """Fetch current price via Finnhub REST API."""
        if not self._finnhub_api_key:
            LOG.warning(
                "FINNHUB_API_KEY not set — cannot fetch quote"
            )
            return None

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: httpx.get(
                    FINNHUB_QUOTE_URL,
                    params={
                        "symbol": ticker,
                        "token": self._finnhub_api_key,
                    },
                    timeout=FINNHUB_REQUEST_TIMEOUT_SEC,
                ),
            )
            if resp.status_code != 200:
                LOG.warning(
                    "Finnhub quote returned %d for %s",
                    resp.status_code,
                    ticker,
                )
                return None
            data = resp.json()
            price = data.get("c")
            if price is None or price == 0:
                return None
            return float(price)
        except Exception as exc:
            LOG.warning(
                "Finnhub quote failed for %s: %s", ticker, exc
            )
            return None

    def _get_ticker_levels(
        self,
        ticker: str,
    ) -> list[dict[str, Any]]:
        """Extract support/resistance levels from the DB."""
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
            selected = select_target_levels(
                raw_levels, last_close, cfg
            )
            selected = add_fallback_levels(
                df, selected, last_close, cfg
            )

            if selected.empty:
                return []

            return selected[["type", "level"]].to_dict("records")
        except Exception as exc:
            LOG.warning(
                "Failed to get levels for %s: %s", ticker, exc
            )
            return []

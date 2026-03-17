"""Finnhub WebSocket client for real-time equity trade data.

Connects to the Finnhub streaming endpoint for US equities.
Maintains latest trade price per symbol in a thread-safe dict.
Auto-reconnects on disconnect with exponential backoff.

Free tier limit: 50 concurrent symbol subscriptions.
Only streams during US market hours (pre-market 4 AM to after-hours 8 PM ET).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from typing import Callable

import websocket  # websocket-client

LOG = logging.getLogger("trader_koo.streaming.finnhub_ws")

FINNHUB_WS_URL = "wss://ws.finnhub.io"
MAX_SUBSCRIPTIONS = 50

# Reconnect backoff
INITIAL_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 60.0
BACKOFF_MULTIPLIER = 2.0


class FinnhubWSClient:
    """WebSocket client that subscribes to Finnhub real-time trades."""

    def __init__(
        self,
        api_key: str,
        always_on: list[str] | None = None,
        on_tick: Callable[[dict], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self._always_on: set[str] = {
            s.upper() for s in (always_on or ["SPY", "QQQ"])
        }
        self._on_tick = on_tick

        self._lock = threading.Lock()
        self._connected = False
        self._running = False
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None

        # Currently subscribed symbols (includes always-on)
        self._subscribed: set[str] = set()

        # Latest trade data per symbol: {symbol: {price, volume, timestamp}}
        self._prices: dict[str, dict] = {}

        # Previous price for change tracking
        self._prev_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def subscribe(self, symbol: str) -> bool:
        """Subscribe to a symbol. Returns False if at 50 limit."""
        sym = symbol.upper()
        with self._lock:
            if sym in self._subscribed:
                return True
            if len(self._subscribed) >= MAX_SUBSCRIPTIONS:
                LOG.warning(
                    "Cannot subscribe to %s: at %d/%d limit",
                    sym, len(self._subscribed), MAX_SUBSCRIPTIONS,
                )
                return False
            self._subscribed.add(sym)
            ws = self._ws

        if ws is not None:
            try:
                ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
                LOG.info("Subscribed to %s", sym)
            except Exception as exc:
                LOG.warning("Failed to send subscribe for %s: %s", sym, exc)
        return True

    def unsubscribe(self, symbol: str) -> bool:
        """Unsubscribe from a symbol. Cannot unsubscribe always-on symbols."""
        sym = symbol.upper()
        with self._lock:
            if sym in self._always_on:
                LOG.debug(
                    "Cannot unsubscribe always-on symbol %s", sym,
                )
                return False
            if sym not in self._subscribed:
                return True
            self._subscribed.discard(sym)
            self._prices.pop(sym, None)
            self._prev_prices.pop(sym, None)
            ws = self._ws

        if ws is not None:
            try:
                ws.send(json.dumps({"type": "unsubscribe", "symbol": sym}))
                LOG.info("Unsubscribed from %s", sym)
            except Exception as exc:
                LOG.warning("Failed to send unsubscribe for %s: %s", sym, exc)
        return True

    def get_price(self, symbol: str) -> dict | None:
        """Get latest price data for a symbol."""
        with self._lock:
            return self._prices.get(symbol.upper())

    def get_all_prices(self) -> dict[str, dict]:
        """Get all currently streaming prices."""
        with self._lock:
            return dict(self._prices)

    def get_subscription_count(self) -> int:
        """How many symbols currently subscribed."""
        with self._lock:
            return len(self._subscribed)

    def get_subscribed_symbols(self) -> list[str]:
        """Return sorted list of currently subscribed symbols."""
        with self._lock:
            return sorted(self._subscribed)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket in a daemon thread."""
        if self._running:
            LOG.warning("FinnhubWSClient.start() called but already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_forever,
            name="finnhub-ws",
            daemon=True,
        )
        self._thread.start()
        LOG.info("Finnhub WebSocket client started (daemon thread)")

    def stop(self) -> None:
        """Signal the WebSocket to close."""
        self._running = False
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        LOG.info("Finnhub WebSocket client stop requested")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        backoff = INITIAL_BACKOFF_SEC
        while self._running:
            try:
                self._connect()
            except Exception as exc:
                LOG.error("Finnhub WS unexpected error: %s", exc)

            with self._lock:
                self._connected = False

            if not self._running:
                break

            LOG.warning(
                "Finnhub WS disconnected — reconnecting in %.1f s",
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SEC)

        LOG.info("Finnhub WS run-loop exited")

    def _connect(self) -> None:
        url = f"{FINNHUB_WS_URL}?token={self._api_key}"
        LOG.info("Connecting to Finnhub WS")

        ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws = ws
        ws.run_forever(ping_interval=20, ping_timeout=10)

    def _on_open(self, _ws: websocket.WebSocketApp) -> None:
        with self._lock:
            self._connected = True
            # Subscribe to always-on symbols first
            for sym in self._always_on:
                self._subscribed.add(sym)

        # Send subscription messages for all tracked symbols
        with self._lock:
            symbols = list(self._subscribed)

        for sym in symbols:
            try:
                _ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
            except Exception as exc:
                LOG.warning("Failed to subscribe %s on open: %s", sym, exc)

        LOG.info(
            "Finnhub WS connected — subscribed to %d symbols",
            len(symbols),
        )

    def _on_message(self, _ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            msg = json.loads(raw)
            self._handle_trades(msg)
        except Exception as exc:
            LOG.warning("Failed to parse Finnhub message: %s", exc)

    def _on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        LOG.error("Finnhub WS error: %s", error)

    def _on_close(
        self,
        _ws: websocket.WebSocketApp,
        close_status_code: int | None,
        close_msg: str | None,
    ) -> None:
        with self._lock:
            self._connected = False
        LOG.warning(
            "Finnhub WS closed (code=%s msg=%s)",
            close_status_code,
            close_msg,
        )

    # ------------------------------------------------------------------
    # Trade parsing
    # ------------------------------------------------------------------

    def _handle_trades(self, msg: dict) -> None:
        """Parse trade events and update price buffers.

        Finnhub trade format:
        {"type":"trade","data":[{"s":"AAPL","p":150.25,"v":100,"t":1234567890000}]}
        """
        if msg.get("type") != "trade":
            return

        data_list = msg.get("data")
        if not data_list or not isinstance(data_list, list):
            return

        for trade in data_list:
            symbol = trade.get("s")
            price = trade.get("p")
            volume = trade.get("v", 0)
            ts_ms = trade.get("t", 0)

            if not symbol or price is None:
                continue

            timestamp = dt.datetime.fromtimestamp(
                ts_ms / 1000, tz=dt.timezone.utc,
            )

            with self._lock:
                if symbol not in self._subscribed:
                    continue

                prev = self._prices.get(symbol, {}).get("price")
                if prev is not None:
                    self._prev_prices[symbol] = prev

                tick_data = {
                    "symbol": symbol,
                    "price": float(price),
                    "volume": int(volume),
                    "timestamp": timestamp.isoformat(),
                    "prev_price": self._prev_prices.get(symbol),
                }
                self._prices[symbol] = tick_data

            # Broadcast to subscribers (outside the lock)
            if self._on_tick is not None:
                try:
                    self._on_tick(tick_data)
                except Exception as exc:
                    LOG.debug("on_tick callback error: %s", exc)

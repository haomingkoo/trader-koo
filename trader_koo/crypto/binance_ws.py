"""Binance WebSocket client for real-time BTC and ETH kline data.

Connects to the public Binance stream endpoint — no API key required.
Maintains an in-memory buffer of 1-minute bars (last 24 h) per symbol
and the latest tick price. Thread-safe via ``threading.Lock``.

Auto-reconnects on disconnect with exponential backoff.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from collections import deque
from typing import Callable, Deque

import websocket  # websocket-client

from trader_koo.crypto.models import CryptoBar, CryptoTick

LOG = logging.getLogger("trader_koo.crypto.binance_ws")

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# Map Binance symbol → display symbol
SYMBOL_MAP: dict[str, str] = {
    "btcusdt": "BTC-USD",
    "ethusdt": "ETH-USD",
}

# Streams to subscribe to
KLINE_STREAMS: list[str] = ["btcusdt@kline_1m", "ethusdt@kline_1m"]

# Buffer size: 1440 bars = 24 hours of 1-minute candles
MAX_BARS = 1440

# Reconnect backoff
INITIAL_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 60.0
BACKOFF_MULTIPLIER = 2.0


class BinanceWSClient:
    """WebSocket client that subscribes to Binance kline streams."""

    def __init__(
        self,
        on_tick: Callable[[CryptoTick], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._connected = False
        self._running = False
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._on_tick = on_tick

        # Latest tick per symbol
        self._ticks: dict[str, CryptoTick] = {}

        # Bars buffer per symbol: deque of CryptoBar
        self._bars: dict[str, Deque[CryptoBar]] = {
            "BTC-USD": deque(maxlen=MAX_BARS),
            "ETH-USD": deque(maxlen=MAX_BARS),
        }

        # 24h tracking for change_pct calculation
        self._open_24h: dict[str, float | None] = {
            "BTC-USD": None,
            "ETH-USD": None,
        }
        self._volume_24h: dict[str, float] = {
            "BTC-USD": 0.0,
            "ETH-USD": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    def get_latest_ticks(self) -> dict[str, CryptoTick]:
        with self._lock:
            return dict(self._ticks)

    def get_bars(self, symbol: str, limit: int = 100) -> list[CryptoBar]:
        with self._lock:
            buf = self._bars.get(symbol, deque())
            bars = list(buf)
        # Return the last ``limit`` bars
        return bars[-limit:]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket in a daemon thread."""
        if self._running:
            LOG.warning("BinanceWSClient.start() called but already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_forever,
            name="binance-ws",
            daemon=True,
        )
        self._thread.start()
        LOG.info("Binance WebSocket client started (daemon thread)")

    def stop(self) -> None:
        """Signal the WebSocket to close."""
        self._running = False
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        LOG.info("Binance WebSocket client stop requested")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        backoff = INITIAL_BACKOFF_SEC
        while self._running:
            try:
                self._connect()
            except Exception as exc:
                LOG.error("Binance WS unexpected error: %s", exc)

            with self._lock:
                self._connected = False

            if not self._running:
                break

            LOG.warning(
                "Binance WS disconnected — reconnecting in %.1f s",
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_SEC)

        LOG.info("Binance WS run-loop exited")

    def _connect(self) -> None:
        url = f"{BINANCE_WS_URL}/{'/'.join(KLINE_STREAMS)}"
        LOG.info("Connecting to Binance WS: %s", url)

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
        LOG.info("Binance WS connected — subscribed to %s", KLINE_STREAMS)

    def _on_message(self, _ws: websocket.WebSocketApp, raw: str) -> None:
        try:
            msg = json.loads(raw)
            self._handle_kline(msg)
        except Exception as exc:
            LOG.warning("Failed to parse Binance message: %s", exc)

    def _on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        LOG.error("Binance WS error: %s", error)

    def _on_close(
        self,
        _ws: websocket.WebSocketApp,
        close_status_code: int | None,
        close_msg: str | None,
    ) -> None:
        with self._lock:
            self._connected = False
        LOG.warning(
            "Binance WS closed (code=%s msg=%s)",
            close_status_code,
            close_msg,
        )

    # ------------------------------------------------------------------
    # Kline parsing
    # ------------------------------------------------------------------

    def _handle_kline(self, msg: dict) -> None:
        """Parse a kline/candlestick event and update buffers."""
        # Combined stream format: {"stream": "btcusdt@kline_1m", "data": {...}}
        # or direct format with "e": "kline"
        if "data" in msg:
            data = msg["data"]
        elif msg.get("e") == "kline":
            data = msg
        else:
            return

        kline = data.get("k")
        if kline is None:
            return

        binance_symbol = str(kline.get("s", "")).lower()
        display_symbol = SYMBOL_MAP.get(binance_symbol)
        if display_symbol is None:
            return

        close_price = float(kline["c"])
        open_price = float(kline["o"])
        high_price = float(kline["h"])
        low_price = float(kline["l"])
        volume = float(kline["v"])
        event_time_ms = int(data.get("E", kline.get("t", 0)))
        timestamp = dt.datetime.fromtimestamp(
            event_time_ms / 1000, tz=dt.timezone.utc,
        )
        is_closed = kline.get("x", False)

        with self._lock:
            # Update 24h open reference (first bar we see)
            if self._open_24h[display_symbol] is None:
                self._open_24h[display_symbol] = open_price

            # Approximate 24h volume from buffer
            self._volume_24h[display_symbol] = float(kline.get("q", volume))

            # Compute 24h change %
            open_24h = self._open_24h[display_symbol]
            if open_24h and open_24h > 0:
                change_pct = ((close_price - open_24h) / open_24h) * 100
            else:
                change_pct = 0.0

            # Update latest tick
            tick = CryptoTick(
                symbol=display_symbol,
                price=close_price,
                volume_24h=self._volume_24h[display_symbol],
                change_pct_24h=round(change_pct, 4),
                timestamp=timestamp,
            )
            self._ticks[display_symbol] = tick

        # Broadcast to browser subscribers (outside the lock)
        if self._on_tick is not None:
            try:
                self._on_tick(tick)
            except Exception as exc:
                LOG.debug("on_tick callback error: %s", exc)

        with self._lock:
            # If this kline bar is closed, append to bar buffer
            if is_closed:
                bar_ts = dt.datetime.fromtimestamp(
                    int(kline["t"]) / 1000, tz=dt.timezone.utc,
                )
                bar = CryptoBar(
                    symbol=display_symbol,
                    timestamp=bar_ts,
                    interval="1m",
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                )
                self._bars[display_symbol].append(bar)

    # ------------------------------------------------------------------
    # 24h reference reset
    # ------------------------------------------------------------------

    def _prune_old_bars(self) -> None:
        """Remove bars older than 24 h and reset 24h open reference."""
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
        with self._lock:
            for symbol in list(self._bars):
                buf = self._bars[symbol]
                while buf and buf[0].timestamp < cutoff:
                    buf.popleft()
                # Reset 24h open to earliest remaining bar
                if buf:
                    self._open_24h[symbol] = buf[0].open
                else:
                    self._open_24h[symbol] = None

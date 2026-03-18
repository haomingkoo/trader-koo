import { useCallback, useEffect, useRef, useState } from "react";
import type { CryptoBar } from "../api/types";

export interface FormingCandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  progress_pct: number;
}

interface WsMessage {
  type?: string;
  symbol?: string;
  interval?: string;
  timestamp?: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  progress_pct?: number;
  bar?: CryptoBar;
}

export function useCryptoSubscription(
  symbol: string,
  interval: string,
): {
  formingCandle: FormingCandleData | null;
  closedBar: CryptoBar | null;
  wsConnected: boolean;
} {
  const [formingCandle, setFormingCandle] = useState<FormingCandleData | null>(null);
  const [closedBar, setClosedBar] = useState<CryptoBar | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);
  const disposedRef = useRef(false);
  const currentSub = useRef({ symbol, interval });

  currentSub.current = { symbol, interval };

  const sendSubscribe = useCallback((ws: WebSocket, sym: string, iv: string) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "subscribe", symbol: sym, interval: iv }));
    }
  }, []);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/crypto`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setWsConnected(true);
      backoff.current = 1000;
      sendSubscribe(ws, currentSub.current.symbol, currentSub.current.interval);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage;
        if (
          msg.type === "forming" &&
          msg.symbol === currentSub.current.symbol &&
          msg.interval === currentSub.current.interval
        ) {
          setFormingCandle({
            timestamp: msg.timestamp ?? "",
            open: msg.open ?? 0,
            high: msg.high ?? 0,
            low: msg.low ?? 0,
            close: msg.close ?? 0,
            volume: msg.volume ?? 0,
            progress_pct: msg.progress_pct ?? 0,
          });
          return;
        }

        if (
          msg.type === "candle_close" &&
          msg.symbol === currentSub.current.symbol &&
          msg.interval === currentSub.current.interval &&
          msg.bar
        ) {
          setClosedBar(msg.bar);
          setFormingCandle(null);
        }
      } catch {
        // Ignore malformed messages and keep the socket alive.
      }
    };

    ws.onclose = () => {
      setWsConnected(false);
      wsRef.current = null;
      if (disposedRef.current) {
        return;
      }
      reconnectTimer.current = setTimeout(() => {
        if (disposedRef.current) {
          return;
        }
        backoff.current = Math.min(backoff.current * 2, 30000);
        connect();
      }, backoff.current);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, [sendSubscribe]);

  useEffect(() => {
    disposedRef.current = false;
    connect();
    return () => {
      disposedRef.current = true;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  useEffect(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      sendSubscribe(ws, symbol, interval);
      setFormingCandle(null);
      setClosedBar(null);
    }
  }, [symbol, interval, sendSubscribe]);

  return { formingCandle, closedBar, wsConnected };
}

export function mergeClosedBarIntoHistory(
  bars: CryptoBar[],
  closedBar: CryptoBar | null,
): CryptoBar[] {
  if (!closedBar) {
    return bars;
  }
  if (bars.length === 0) {
    return [closedBar];
  }

  const nextBars = [...bars];
  const existingIndex = nextBars.findIndex((bar) => bar.timestamp === closedBar.timestamp);
  if (existingIndex >= 0) {
    nextBars[existingIndex] = closedBar;
    return nextBars;
  }

  nextBars.push(closedBar);
  nextBars.sort((left, right) => left.timestamp.localeCompare(right.timestamp));
  return nextBars;
}

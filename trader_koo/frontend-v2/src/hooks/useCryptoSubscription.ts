import { useCallback, useEffect, useRef, useState } from "react";
import type { CryptoBar } from "../api/types";
import { useCryptoWs } from "./useCryptoWs";

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
  const { connected, send, addListener } = useCryptoWs();
  const [formingCandle, setFormingCandle] = useState<FormingCandleData | null>(null);
  const [closedBar, setClosedBar] = useState<CryptoBar | null>(null);
  const currentSub = useRef({ symbol, interval });

  currentSub.current = { symbol, interval };

  // Subscribe when connected or when symbol/interval changes
  useEffect(() => {
    if (connected) {
      send({ action: "subscribe", symbol, interval });
      setFormingCandle(null);
      setClosedBar(null);
    }
  }, [connected, symbol, interval, send]);

  // Listen for candle messages on the shared socket
  const handleMessage = useCallback((data: unknown) => {
    const msg = data as WsMessage;
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
  }, []);

  useEffect(() => {
    return addListener(handleMessage);
  }, [addListener, handleMessage]);

  return { formingCandle, closedBar, wsConnected: connected };
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

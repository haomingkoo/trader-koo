import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { EquityTick } from "../api/types";
import { EquityPriceStoreContext, EquityWsContext } from "./equityWsContext";

export function EquityWsProvider({ children }: { children: ReactNode }) {
  const [prices, setPrices] = useState<Record<string, EquityTick>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);
  const reconnectRef = useRef<() => void>(() => {});
  const latestRef = useRef<Record<string, EquityTick>>({});
  const storeListenersRef = useRef<Set<() => void>>(new Set());

  const store = useMemo(() => ({
    getPrice: (symbol: string) => latestRef.current[symbol],
    subscribe: (onChange: () => void) => {
      storeListenersRef.current.add(onChange);
      return () => {
        storeListenersRef.current.delete(onChange);
      };
    },
    connected,
  }), [connected]);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/equities`);

    ws.onopen = () => {
      setConnected(true);
      backoff.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const tick = JSON.parse(event.data) as EquityTick;
        latestRef.current = { ...latestRef.current, [tick.symbol]: tick };
        setPrices(latestRef.current);
        for (const fn of storeListenersRef.current) fn();
      } catch {
        // Ignore malformed messages.
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      reconnectTimer.current = setTimeout(() => {
        backoff.current = Math.min(backoff.current * 2, 30_000);
        reconnectRef.current();
      }, backoff.current);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    reconnectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const value = useMemo(() => ({ prices, connected }), [prices, connected]);

  return (
    <EquityPriceStoreContext.Provider value={store}>
      <EquityWsContext.Provider value={value}>
        {children}
      </EquityWsContext.Provider>
    </EquityPriceStoreContext.Provider>
  );
}

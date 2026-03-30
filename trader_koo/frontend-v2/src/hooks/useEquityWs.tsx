import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { EquityTick } from "../api/types";

interface EquityWsContextValue {
  /** All broadcast equity ticks keyed by symbol. */
  prices: Record<string, EquityTick>;
  connected: boolean;
}

const EquityWsContext = createContext<EquityWsContextValue | null>(null);

export function EquityWsProvider({ children }: { children: ReactNode }) {
  const [prices, setPrices] = useState<Record<string, EquityTick>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);

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
        setPrices((prev) => ({ ...prev, [tick.symbol]: tick }));
      } catch {
        // Ignore malformed messages.
      }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      reconnectTimer.current = setTimeout(() => {
        backoff.current = Math.min(backoff.current * 2, 30_000);
        connect();
      }, backoff.current);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return (
    <EquityWsContext.Provider value={{ prices, connected }}>
      {children}
    </EquityWsContext.Provider>
  );
}

export function useEquityWs(): EquityWsContextValue {
  const ctx = useContext(EquityWsContext);
  if (!ctx) throw new Error("useEquityWs must be used within EquityWsProvider");
  return ctx;
}

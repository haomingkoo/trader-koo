import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { CryptoPrice } from "../api/types";
import { CryptoWsContext } from "./cryptoWsContext";
import type { CryptoWsListener } from "./cryptoWsContext";

export function CryptoWsProvider({ children }: { children: ReactNode }) {
  const [prices, setPrices] = useState<Record<string, CryptoPrice>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);
  const listenersRef = useRef<Set<CryptoWsListener>>(new Set());
  const reconnectRef = useRef<() => void>(() => {});

  const send = useCallback((msg: unknown) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }, []);

  const addListener = useCallback((fn: CryptoWsListener) => {
    listenersRef.current.add(fn);
    return () => {
      listenersRef.current.delete(fn);
    };
  }, []);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/crypto`);

    ws.onopen = () => {
      setConnected(true);
      backoff.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.symbol && typeof data.price === "number" && !data.type) {
          setPrices((prev) => ({ ...prev, [data.symbol]: data as CryptoPrice }));
        }
        for (const fn of listenersRef.current) {
          fn(data);
        }
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

  return (
    <CryptoWsContext.Provider value={{ prices, connected, send, addListener }}>
      {children}
    </CryptoWsContext.Provider>
  );
}

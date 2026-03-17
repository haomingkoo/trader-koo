import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../api/client";
import type { EquityTick } from "../api/types";

interface LivePriceResponse {
  ok: boolean;
  price?: number | null;
  volume?: number | null;
  timestamp?: string | null;
  prev_price?: number | null;
}

export function useLiveEquityPrice(ticker: string) {
  const [livePrice, setLivePrice] = useState<EquityTick | null>(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const [subscriptionReady, setSubscriptionReady] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);
  const tickerRef = useRef("");
  const shouldReconnect = useRef(true);

  useEffect(() => {
    tickerRef.current = ticker.trim().toUpperCase();
  }, [ticker]);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/equities`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setSocketConnected(true);
      backoff.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const tick = JSON.parse(event.data) as EquityTick;
        if (tick.symbol === tickerRef.current) {
          setLivePrice(tick);
        }
      } catch {
        // Ignore malformed websocket messages.
      }
    };

    ws.onclose = () => {
      setSocketConnected(false);
      wsRef.current = null;
      if (!shouldReconnect.current) return;
      reconnectTimer.current = setTimeout(() => {
        backoff.current = Math.min(backoff.current * 2, 30000);
        connect();
      }, backoff.current);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    shouldReconnect.current = true;
    connect();
    return () => {
      shouldReconnect.current = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  useEffect(() => {
    const symbol = ticker.trim().toUpperCase();
    if (!symbol) {
      setLivePrice(null);
      setSubscriptionReady(false);
      return;
    }

    let active = true;
    setLivePrice((current) => (current?.symbol === symbol ? current : null));
    setSubscriptionReady(false);

    apiFetch<{ ok: boolean }>(`/api/streaming/subscribe/${symbol}`, {
      method: "POST",
    })
      .then((response) => {
        if (!active) return;
        setSubscriptionReady(Boolean(response.ok));
        return apiFetch<LivePriceResponse>(`/api/streaming/price/${symbol}`);
      })
      .then((response) => {
        if (!active || !response || response.price == null) return;
        setLivePrice({
          symbol,
          price: response.price,
          volume: response.volume ?? 0,
          timestamp: response.timestamp ?? "",
          prev_price: response.prev_price ?? null,
        });
      })
      .catch(() => {
        if (active) setSubscriptionReady(false);
      });

    return () => {
      active = false;
      apiFetch(`/api/streaming/unsubscribe/${symbol}`, {
        method: "POST",
      }).catch(() => {});
    };
  }, [ticker]);

  return {
    livePrice,
    streamingActive: socketConnected && subscriptionReady,
    socketConnected,
  };
}

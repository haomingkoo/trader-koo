import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../api/client";
import type { EquityTick } from "../api/types";
import { useEquityWs } from "./useEquityWs";

interface LivePriceResponse {
  ok: boolean;
  price?: number | null;
  volume?: number | null;
  timestamp?: string | null;
  prev_price?: number | null;
}

export function useLiveEquityPrice(ticker: string) {
  const { prices, connected: socketConnected } = useEquityWs();
  const [livePrice, setLivePrice] = useState<EquityTick | null>(null);
  const [subscriptionReady, setSubscriptionReady] = useState(false);
  const tickerRef = useRef("");

  useEffect(() => {
    tickerRef.current = ticker.trim().toUpperCase();
  }, [ticker]);

  // Pick up ticks from the shared WS for the active ticker
  useEffect(() => {
    const symbol = ticker.trim().toUpperCase();
    if (!symbol) return;
    const tick = prices[symbol];
    if (tick) {
      setLivePrice(tick);
    }
  }, [prices, ticker]);

  // Subscribe via HTTP API so the backend streams this ticker
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

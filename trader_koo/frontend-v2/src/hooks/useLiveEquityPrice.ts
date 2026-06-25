import { useEffect, useMemo, useState } from "react";
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
  const symbol = useMemo(() => ticker.trim().toUpperCase(), [ticker]);
  const [apiPrice, setApiPrice] = useState<EquityTick | null>(null);
  const [subscriptionReady, setSubscriptionReady] = useState(false);
  const livePrice = prices[symbol] ?? apiPrice;

  // Subscribe via HTTP API so the backend streams this ticker
  useEffect(() => {
    let active = true;

    if (!symbol) {
      const resetTimer = window.setTimeout(() => {
        if (!active) return;
        setApiPrice(null);
        setSubscriptionReady(false);
      }, 0);
      return () => {
        active = false;
        window.clearTimeout(resetTimer);
      };
    }

    const resetTimer = window.setTimeout(() => {
      if (!active) return;
      setApiPrice((current) => (current?.symbol === symbol ? current : null));
      setSubscriptionReady(false);
    }, 0);

    apiFetch<{ ok: boolean }>(`/api/streaming/subscribe/${symbol}`, {
      method: "POST",
    })
      .then((response) => {
        if (!active) return undefined;
        setSubscriptionReady(Boolean(response.ok));
        return apiFetch<LivePriceResponse>(`/api/streaming/price/${symbol}`);
      })
      .then((response) => {
        if (!active || !response || response.price == null) return;
        setApiPrice({
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
      window.clearTimeout(resetTimer);
      apiFetch(`/api/streaming/unsubscribe/${symbol}`, {
        method: "POST",
      }).catch(() => {});
    };
  }, [symbol]);

  return {
    livePrice,
    streamingActive: socketConnected && subscriptionReady,
    socketConnected,
  };
}

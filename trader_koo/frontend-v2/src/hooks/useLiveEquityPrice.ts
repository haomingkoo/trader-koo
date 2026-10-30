import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "../api/client";
import type { EquityTick } from "../api/types";
import { useEquityPrice, useEquityWsConnected } from "./useEquityWs";

interface LivePriceResponse {
  ok: boolean;
  subscribed?: boolean;
  price?: number | null;
  volume?: number | null;
  timestamp?: string | null;
  prev_price?: number | null;
}

export function useLiveEquityPrice(ticker: string) {
  const symbol = useMemo(() => ticker.trim().toUpperCase(), [ticker]);
  const wsPrice = useEquityPrice(symbol);
  const socketConnected = useEquityWsConnected();
  const [apiPrice, setApiPrice] = useState<EquityTick | null>(null);
  const [subscriptionReady, setSubscriptionReady] = useState(false);
  const livePrice = wsPrice ?? apiPrice;

  // The price route subscribes server-side; the feed evicts stale symbols itself
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

    apiFetch<LivePriceResponse>(`/api/streaming/price/${symbol}`)
      .then((response) => {
        if (!active) return;
        setSubscriptionReady(Boolean(response.subscribed));
        if (response.price == null) return;
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
    };
  }, [symbol]);

  return {
    livePrice,
    streamingActive: socketConnected && subscriptionReady,
    socketConnected,
  };
}

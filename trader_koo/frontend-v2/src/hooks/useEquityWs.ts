import { useCallback, useContext, useSyncExternalStore } from "react";
import type { EquityTick } from "../api/types";
import { EquityPriceStoreContext, EquityWsContext } from "./equityWsContext";
import type { EquityWsContextValue } from "./equityWsContext";

export function useEquityWs(): EquityWsContextValue {
  const ctx = useContext(EquityWsContext);
  if (!ctx) throw new Error("useEquityWs must be used within EquityWsProvider");
  return ctx;
}

/** Latest tick for one symbol; re-renders only when that symbol ticks. */
export function useEquityPrice(symbol: string): EquityTick | undefined {
  const store = useContext(EquityPriceStoreContext);
  if (!store) throw new Error("useEquityPrice must be used within EquityWsProvider");
  const getSnapshot = useCallback(() => store.getPrice(symbol), [store, symbol]);
  return useSyncExternalStore(store.subscribe, getSnapshot);
}

/** Socket status without subscribing to price ticks. */
export function useEquityWsConnected(): boolean {
  const store = useContext(EquityPriceStoreContext);
  if (!store) throw new Error("useEquityWsConnected must be used within EquityWsProvider");
  return store.connected;
}

import { createContext } from "react";
import type { EquityTick } from "../api/types";

export interface EquityWsContextValue {
  prices: Record<string, EquityTick>;
  connected: boolean;
}

export const EquityWsContext = createContext<EquityWsContextValue | null>(null);

export interface EquityPriceStore {
  getPrice: (symbol: string) => EquityTick | undefined;
  subscribe: (onChange: () => void) => () => void;
  connected: boolean;
}

// Stable store so a single-symbol reader skips re-renders for other symbols' ticks.
export const EquityPriceStoreContext = createContext<EquityPriceStore | null>(null);

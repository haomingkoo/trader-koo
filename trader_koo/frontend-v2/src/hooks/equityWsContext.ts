import { createContext } from "react";
import type { EquityTick } from "../api/types";

export interface EquityWsContextValue {
  prices: Record<string, EquityTick>;
  connected: boolean;
}

export const EquityWsContext = createContext<EquityWsContextValue | null>(null);

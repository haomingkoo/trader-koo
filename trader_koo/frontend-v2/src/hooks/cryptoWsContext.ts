import { createContext } from "react";
import type { CryptoPrice } from "../api/types";

export type CryptoWsListener = (data: unknown) => void;

export interface CryptoWsContextValue {
  prices: Record<string, CryptoPrice>;
  connected: boolean;
  send: (msg: unknown) => void;
  addListener: (fn: CryptoWsListener) => () => void;
}

export const CryptoWsContext = createContext<CryptoWsContextValue | null>(null);

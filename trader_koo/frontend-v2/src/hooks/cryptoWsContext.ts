import { createContext } from "react";
import type { CryptoPrice } from "../api/types";

export type CryptoWsListener = (data: unknown) => void;

export interface CryptoWsApiValue {
  connected: boolean;
  send: (msg: unknown) => void;
  addListener: (fn: CryptoWsListener) => () => void;
}

export interface CryptoWsContextValue extends CryptoWsApiValue {
  prices: Record<string, CryptoPrice>;
}

// Prices live in their own context so a tick only re-renders components that read prices.
export const CryptoWsApiContext = createContext<CryptoWsApiValue | null>(null);
export const CryptoPricesContext = createContext<Record<string, CryptoPrice> | null>(null);

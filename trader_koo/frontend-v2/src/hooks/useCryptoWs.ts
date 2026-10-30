import { useContext } from "react";
import type { CryptoPrice } from "../api/types";
import { CryptoPricesContext, CryptoWsApiContext } from "./cryptoWsContext";
import type { CryptoWsApiValue, CryptoWsContextValue } from "./cryptoWsContext";

/** Socket controls only; does not re-render on price ticks. */
export function useCryptoWsApi(): CryptoWsApiValue {
  const ctx = useContext(CryptoWsApiContext);
  if (!ctx) throw new Error("useCryptoWsApi must be used within CryptoWsProvider");
  return ctx;
}

export function useCryptoPrices(): Record<string, CryptoPrice> {
  const prices = useContext(CryptoPricesContext);
  if (!prices) throw new Error("useCryptoPrices must be used within CryptoWsProvider");
  return prices;
}

export function useCryptoWs(): CryptoWsContextValue {
  return { ...useCryptoWsApi(), prices: useCryptoPrices() };
}

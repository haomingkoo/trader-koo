import { useContext } from "react";
import { CryptoWsContext } from "./cryptoWsContext";
import type { CryptoWsContextValue } from "./cryptoWsContext";

export function useCryptoWs(): CryptoWsContextValue {
  const ctx = useContext(CryptoWsContext);
  if (!ctx) throw new Error("useCryptoWs must be used within CryptoWsProvider");
  return ctx;
}

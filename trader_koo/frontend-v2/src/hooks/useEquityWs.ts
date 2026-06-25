import { useContext } from "react";
import { EquityWsContext } from "./equityWsContext";
import type { EquityWsContextValue } from "./equityWsContext";

export function useEquityWs(): EquityWsContextValue {
  const ctx = useContext(EquityWsContext);
  if (!ctx) throw new Error("useEquityWs must be used within EquityWsProvider");
  return ctx;
}

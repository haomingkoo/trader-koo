import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ChartState {
  ticker: string;
  timeframe: "daily" | "weekly" | "monthly";
  setTicker: (ticker: string) => void;
  setTimeframe: (tf: "daily" | "weekly" | "monthly") => void;
}

export const useChartStore = create<ChartState>()(
  persist(
    (set) => ({
      ticker: "SPY",
      timeframe: "daily",
      setTicker: (ticker: string) =>
        set({ ticker: ticker.trim().toUpperCase() }),
      setTimeframe: (tf: "daily" | "weekly" | "monthly") =>
        set({ timeframe: tf }),
    }),
    { name: "trader-koo-chart" },
  ),
);

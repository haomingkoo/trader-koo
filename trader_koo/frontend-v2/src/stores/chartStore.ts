import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ChartState {
  ticker: string;
  timeframe: "daily" | "weekly";
  setTicker: (ticker: string) => void;
  setTimeframe: (tf: "daily" | "weekly") => void;
}

export const useChartStore = create<ChartState>()(
  persist(
    (set) => ({
      ticker: "SPY",
      timeframe: "daily",
      setTicker: (ticker: string) =>
        set({ ticker: ticker.trim().toUpperCase() }),
      setTimeframe: (tf: "daily" | "weekly") => set({ timeframe: tf }),
    }),
    { name: "trader-koo-chart" },
  ),
);

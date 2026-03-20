import { useMemo } from "react";
import PlotlyWrapper from "../PlotlyWrapper";
import Spinner from "../ui/Spinner";

function useIsMobile(): boolean {
  return useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.innerWidth < 640;
  }, []);
}

interface CryptoChartPanelProps {
  historyLoading: boolean;
  hasChart: boolean;
  connected: boolean;
  chartData: Record<string, unknown>[] | null;
  chartLayout: Record<string, unknown> | null;
  onRelayout: (eventData: Record<string, unknown>) => void;
}

export default function CryptoChartPanel({
  historyLoading,
  hasChart,
  connected,
  chartData,
  chartLayout,
  onRelayout,
}: CryptoChartPanelProps) {
  const isMobile = useIsMobile();

  if (historyLoading) {
    return <Spinner className="mt-8" />;
  }

  if (hasChart && chartData && chartLayout) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-1 sm:p-2">
        <PlotlyWrapper
          data={chartData}
          layout={chartLayout}
          config={{
            responsive: true,
            displayModeBar: !isMobile,
            scrollZoom: !isMobile,
          }}
          onRelayout={onRelayout}
          style={{ width: "100%", height: isMobile ? 350 : 500 }}
        />
      </div>
    );
  }

  if (connected) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--muted)]">
        No chart data available yet. The app will backfill Binance history for this symbol and timeframe on demand.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--red)]">
      Crypto feed disconnected — no chart data available.
    </div>
  );
}

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
  onRetry?: () => void;
}

export default function CryptoChartPanel({
  historyLoading,
  hasChart,
  connected,
  chartData,
  chartLayout,
  onRelayout,
  onRetry,
}: CryptoChartPanelProps) {
  const isMobile = useIsMobile();

  if (historyLoading) {
    return <Spinner className="mt-8" />;
  }

  if (hasChart && chartData && chartLayout) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-1 sm:p-2 mb-4 min-h-[500px] sm:min-h-[700px] overflow-hidden">
        <PlotlyWrapper
          data={chartData}
          layout={chartLayout}
          config={{
            responsive: true,
            displayModeBar: !isMobile,
            scrollZoom: !isMobile,
          }}
          onRelayout={onRelayout}
          style={{ width: "100%", height: isMobile ? 600 : 700 }}
        />
      </div>
    );
  }

  if (connected) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center">
        <p className="text-sm text-[var(--muted)] mb-3">
          Chart data is loading. The server is backfilling history from
          Binance — this typically takes 30–60 seconds after a deploy.
        </p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="rounded-md border border-[var(--line)] bg-[var(--bg)] px-4 py-1.5 text-xs text-[var(--text)] hover:bg-[var(--panel)] transition-colors"
          >
            Retry now
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center">
      <p className="text-sm text-[var(--red)] mb-2">
        Crypto feed disconnected — no chart data available.
      </p>
      <p className="text-xs text-[var(--muted)]">
        The server may be restarting. Data will appear automatically when
        the connection is restored.
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 rounded-md border border-[var(--line)] bg-[var(--bg)] px-4 py-1.5 text-xs text-[var(--text)] hover:bg-[var(--panel)] transition-colors"
        >
          Retry now
        </button>
      )}
    </div>
  );
}

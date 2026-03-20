import { useMemo } from "react";
import PlotlyWrapper from "../PlotlyWrapper";

interface ChartPlotPanelProps {
  chartData: Record<string, unknown>[] | null;
  chartLayout: Record<string, unknown> | null;
}

export default function ChartPlotPanel({
  chartData,
  chartLayout,
}: ChartPlotPanelProps) {
  const isMobile = useMemo(
    () => typeof window !== "undefined" && window.innerWidth < 768,
    [],
  );

  if (chartData && chartLayout) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
        <PlotlyWrapper
          data={chartData}
          layout={chartLayout}
          config={{
            responsive: true,
            displayModeBar: isMobile ? false : "hover",
            scrollZoom: true,
          }}
          style={{ width: "100%", height: (chartLayout.height as number) ?? 580 }}
        />
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--muted)]">
      No chart data available.
    </div>
  );
}

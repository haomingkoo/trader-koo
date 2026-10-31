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
  // Stable identities keep the memoized Plotly wrapper from redrawing on every parent render.
  const plotConfig = useMemo(
    () => ({ responsive: true, displayModeBar: isMobile ? false : "hover", scrollZoom: true }),
    [isMobile],
  );
  const plotHeight = (chartLayout?.height as number | undefined) ?? 580;
  const plotStyle = useMemo(() => ({ width: "100%", height: plotHeight }), [plotHeight]);

  if (chartData && chartLayout) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
        <PlotlyWrapper
          data={chartData}
          layout={chartLayout}
          config={plotConfig}
          style={plotStyle}
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

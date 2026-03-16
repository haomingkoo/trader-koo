import { useState, useCallback, lazy, Suspense } from "react";
import { useChart } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import type { Fundamentals, OptionsSummary } from "../api/types";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";

const Plot = lazy(() => import("react-plotly.js"));

export default function ChartPage() {
  const { ticker, timeframe, setTicker, setTimeframe } = useChartStore();
  const [inputValue, setInputValue] = useState(ticker);
  const { data, isLoading, error, refetch } = useChart(ticker);

  const handleLoad = useCallback(() => {
    const clean = inputValue.trim().toUpperCase();
    if (clean) {
      setTicker(clean);
    }
  }, [inputValue, setTicker]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") handleLoad();
    },
    [handleLoad],
  );

  const fundamentals: Fundamentals = data?.fundamentals ?? { price: null, pe: null, peg: null, target_price: null, discount_pct: null };
  const options: OptionsSummary = data?.options_summary ?? { put_call_oi_ratio: null };
  const commentary = data?.chart_commentary;

  const chart = data?.chart ?? [];
  const chartX = chart.map((r) => r.date);
  const chartOpen = chart.map((r) => r.open);
  const chartHigh = chart.map((r) => r.high);
  const chartLow = chart.map((r) => r.low);
  const chartClose = chart.map((r) => r.close);
  const chartVol = chart.map((r) => r.volume);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-xl font-bold tracking-tight">Chart</h2>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value.toUpperCase())}
          onKeyDown={handleKeyDown}
          placeholder="Ticker (e.g. SPY)"
          className="w-28 rounded-lg border border-[var(--line)] bg-[var(--bg)] px-3 py-1.5 text-sm text-[var(--text)] placeholder-[var(--muted)]"
        />
        <button
          onClick={handleLoad}
          disabled={isLoading}
          className="rounded-lg bg-[var(--accent)] px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-[var(--blue)] disabled:opacity-50"
        >
          Load
        </button>
        <div className="flex gap-1">
          {(["daily", "weekly"] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                timeframe === tf
                  ? "bg-[var(--blue)] text-white"
                  : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {tf.charAt(0).toUpperCase() + tf.slice(1)}
            </button>
          ))}
        </div>
        <button
          onClick={() => { refetch(); }}
          disabled={isLoading}
          className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-xs text-[var(--muted)] transition-colors hover:text-[var(--text)] disabled:opacity-50"
        >
          Refresh
        </button>
      </div>

      {isLoading && <Spinner className="mt-12" />}
      {error && (
        <div className="mt-4 text-center text-sm text-[var(--red)]">
          Failed to load chart: {(error as Error).message}
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Fundamental cards */}
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <Card label="Price" value={fundamentals.price ?? "\u2014"} />
            <Card label="P/E" value={fundamentals.pe ?? "\u2014"} />
            <Card label="PEG" value={fundamentals.peg ?? "\u2014"} />
            <Card
              label="Target"
              value={fundamentals.target_price ?? "\u2014"}
            />
            <Card
              label="Discount %"
              value={fundamentals.discount_pct ?? "\u2014"}
            />
            <Card
              label="Put/Call OI"
              value={
                options.put_call_oi_ratio != null
                  ? options.put_call_oi_ratio.toFixed(3)
                  : "\u2014"
              }
            />
          </div>

          {/* Commentary */}
          {commentary && (
            <Card label="Chart Commentary">
              <div className="mt-2 space-y-2 text-xs text-[var(--muted)]">
                {commentary.observation && (
                  <p>{commentary.observation}</p>
                )}
                {commentary.action && (
                  <p>
                    <strong className="text-[var(--text)]">Action:</strong>{" "}
                    {commentary.action}
                  </p>
                )}
                {commentary.risk_note && (
                  <p>
                    <strong className="text-[var(--text)]">Risk:</strong>{" "}
                    {commentary.risk_note}
                  </p>
                )}
                {commentary.technical_read && (
                  <p>
                    <strong className="text-[var(--text)]">
                      Technical:
                    </strong>{" "}
                    {commentary.technical_read}
                  </p>
                )}
              </div>
            </Card>
          )}

          {/* Plotly chart */}
          {chart.length > 0 && (
            <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
              <Suspense fallback={<Spinner className="py-24" />}>
                <Plot
                  data={[
                    {
                      type: "candlestick",
                      x: chartX,
                      open: chartOpen,
                      high: chartHigh,
                      low: chartLow,
                      close: chartClose,
                      name: data.ticker,
                      increasing: {
                        line: { color: "#38d39f" },
                        fillcolor: "rgba(56,211,159,0.85)",
                      },
                      decreasing: {
                        line: { color: "#ff6b6b" },
                        fillcolor: "rgba(255,107,107,0.85)",
                      },
                      xaxis: "x",
                      yaxis: "y",
                    },
                    {
                      type: "bar",
                      x: chartX,
                      y: chartVol,
                      name: "Volume",
                      marker: {
                        color: chart.map((r) =>
                          r.close >= r.open
                            ? "rgba(56,211,159,0.5)"
                            : "rgba(255,107,107,0.5)",
                        ),
                      },
                      xaxis: "x",
                      yaxis: "y2",
                    },
                  ]}
                  layout={{
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    font: { color: "#8ea0bd", size: 11 },
                    margin: { t: 20, r: 40, b: 40, l: 60 },
                    xaxis: {
                      gridcolor: "rgba(255,255,255,0.04)",
                      rangeslider: { visible: false },
                    },
                    yaxis: {
                      gridcolor: "rgba(255,255,255,0.06)",
                      domain: [0.22, 1],
                    },
                    yaxis2: {
                      gridcolor: "rgba(255,255,255,0.04)",
                      domain: [0, 0.18],
                    },
                    showlegend: false,
                    height: 520,
                  }}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: "100%", height: 520 }}
                />
              </Suspense>
            </div>
          )}

          {/* Levels, gaps, trendlines tables placeholder */}
          <div className="grid gap-4 lg:grid-cols-2">
            <Card label={`Support/Resistance Levels (${(data.levels ?? []).length})`}>
              <p className="mt-1 text-xs text-[var(--muted)]">
                Detailed levels table coming soon.
              </p>
            </Card>
            <Card label={`Gaps (${(data.gaps ?? []).length})`}>
              <p className="mt-1 text-xs text-[var(--muted)]">
                Gaps table coming soon.
              </p>
            </Card>
          </div>

          <div className="text-xs text-[var(--muted)]">
            As of {data.asof ?? "\u2014"} &middot; {data.ticker}
          </div>
        </>
      )}
    </div>
  );
}

import { useState, lazy, Suspense } from "react";
import { usePaperTradeSummary, usePaperTrades } from "../api/hooks";
import type { PaperTradeSummaryOverall } from "../api/types";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";

const Plot = lazy(() => import("react-plotly.js"));

const tradeColumns = [
  { key: "ticker" as const, label: "Ticker" },
  { key: "direction" as const, label: "Dir" },
  { key: "entry_price" as const, label: "Entry" },
  { key: "current_price" as const, label: "Current" },
  { key: "stop_loss" as const, label: "Stop" },
  { key: "target_price" as const, label: "Target" },
  { key: "status" as const, label: "Status" },
  { key: "setup_family" as const, label: "Setup" },
  { key: "setup_tier" as const, label: "Tier" },
  { key: "entry_date" as const, label: "Entry Date" },
];

export default function PaperTradePage() {
  const [statusFilter, setStatusFilter] = useState("all");
  const [dirFilter, setDirFilter] = useState("all");

  const { data: summary, isLoading: summaryLoading } = usePaperTradeSummary();
  const {
    data: tradesData,
    isLoading: tradesLoading,
    error,
  } = usePaperTrades(statusFilter, dirFilter);

  const isLoading = summaryLoading || tradesLoading;
  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load paper trades: {(error as Error).message}
      </div>
    );
  }

  const overall: PaperTradeSummaryOverall = summary?.overall ?? { total_trades: null, win_rate_pct: null, avg_pnl_pct: null, total_pnl_pct: null, avg_r_multiple: null };
  const trades = tradesData?.trades ?? [];
  const equityCurve = summary?.equity_curve ?? [];

  const fmtPct = (v: number | null | undefined, suffix: string = "%"): string =>
    v != null ? `${v.toFixed(2)}${suffix}` : "\u2014";

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">Paper Trades</h2>

      {/* Summary cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Card label="Total Trades" value={overall.total_trades ?? "\u2014"} />
        <Card label="Win Rate" value={fmtPct(overall.win_rate_pct)} />
        <Card label="Avg P&L" value={fmtPct(overall.avg_pnl_pct)} />
        <Card label="Total P&L" value={fmtPct(overall.total_pnl_pct)} />
        <Card
          label="Avg R"
          value={
            overall.avg_r_multiple != null
              ? `${overall.avg_r_multiple.toFixed(2)}R`
              : "\u2014"
          }
        />
      </div>

      {/* Equity curve */}
      {equityCurve.length > 1 && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
          <Suspense fallback={<Spinner className="py-12" />}>
            <Plot
              data={[
                {
                  x: equityCurve.map((p) => p.date),
                  y: equityCurve.map((p) => p.equity_index),
                  type: "scatter" as const,
                  mode: "lines" as const,
                  fill: "tozeroy",
                  line: { color: "#22c55e", width: 2 },
                  fillcolor: "rgba(34,197,94,0.08)",
                  name: "Equity Index",
                },
                {
                  x: [
                    equityCurve[0].date,
                    equityCurve[equityCurve.length - 1].date,
                  ],
                  y: [100, 100],
                  type: "scatter" as const,
                  mode: "lines" as const,
                  line: { color: "#6b7280", width: 1, dash: "dot" },
                  name: "Baseline (100)",
                },
              ]}
              layout={{
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                margin: { t: 20, r: 16, b: 40, l: 50 },
                font: { color: "#9ca3af", size: 11 },
                xaxis: { gridcolor: "rgba(255,255,255,0.04)" },
                yaxis: {
                  gridcolor: "rgba(255,255,255,0.06)",
                  title: { text: "Equity", font: { size: 11 } },
                },
                legend: {
                  orientation: "h",
                  y: 1.12,
                  x: 0.5,
                  xanchor: "center",
                  font: { size: 10 },
                },
                showlegend: true,
                height: 280,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%", height: 280 }}
            />
          </Suspense>
        </div>
      )}

      {/* Direction / exit breakdown */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card label="By Direction">
          {summary?.by_direction && Object.keys(summary.by_direction).length > 0 ? (
            <div className="mt-1 space-y-1 text-xs text-[var(--muted)]">
              {Object.entries(summary.by_direction).map(([dir, stats]) => (
                <div key={dir}>
                  <strong className="capitalize text-[var(--text)]">
                    {dir}
                  </strong>
                  : {stats.total} trades, {stats.win_rate_pct}% win, avg{" "}
                  {stats.avg_pnl_pct > 0 ? "+" : ""}
                  {stats.avg_pnl_pct}%
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-1 text-xs text-[var(--muted)]">
              No closed trades yet
            </p>
          )}
        </Card>
        <Card label="By Exit Reason">
          {summary?.by_exit_reason &&
          Object.keys(summary.by_exit_reason).length > 0 ? (
            <div className="mt-1 space-y-1 text-xs text-[var(--muted)]">
              {Object.entries(summary.by_exit_reason).map(([reason, count]) => (
                <div key={reason}>
                  <strong className="capitalize text-[var(--text)]">
                    {reason.replace(/_/g, " ")}
                  </strong>
                  : {count}
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-1 text-xs text-[var(--muted)]">
              No closed trades yet
            </p>
          )}
        </Card>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 text-xs text-[var(--muted)]">
        <label>
          Status:
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="ml-1 rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-[var(--text)]"
          >
            <option value="all">All</option>
            <option value="open">Open</option>
            <option value="closed">Closed</option>
          </select>
        </label>
        <label>
          Direction:
          <select
            value={dirFilter}
            onChange={(e) => setDirFilter(e.target.value)}
            className="ml-1 rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-[var(--text)]"
          >
            <option value="all">All</option>
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
        </label>
      </div>

      {/* Trade log */}
      <Table
        columns={tradeColumns}
        data={trades as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}

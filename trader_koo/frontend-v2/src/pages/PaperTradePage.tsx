import { useState, lazy, Suspense } from "react";
import { Link } from "react-router-dom";
import { usePaperTradeSummary, usePaperTrades } from "../api/hooks";
import type { PaperTrade, PaperTradeSummaryOverall } from "../api/types";
import Card from "../components/ui/Card";
import Badge, { tierVariant } from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";

const Plot = lazy(() => import("react-plotly.js"));

const fmtPct = (v: number | null | undefined, suffix: string = "%", sign: boolean = false): string =>
  typeof v === "number" ? `${sign && v > 0 ? "+" : ""}${v.toFixed(2)}${suffix}` : "\u2014";

const fmtPrice = (v: number | null | undefined): string =>
  typeof v === "number" ? `$${v.toFixed(2)}` : "\u2014";

const pnlColor = (v: number | null | undefined): string => {
  if (v == null) return "";
  if (v > 0) return "text-[var(--green)]";
  if (v < 0) return "text-[var(--red)]";
  return "";
};

const tradeColumns = [
  {
    key: "ticker" as const,
    label: "Ticker",
    render: (v: unknown) => {
      const ticker = String(v ?? "");
      if (!ticker) return "\u2014";
      return (
        <Link
          to={`/chart?t=${ticker}`}
          className="font-mono font-bold text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          {ticker}
        </Link>
      );
    },
  },
  {
    key: "direction" as const,
    label: "Dir",
    render: (v: unknown) => {
      const dir = String(v ?? "").toLowerCase();
      const variant =
        dir === "long" ? "green" : dir === "short" ? "red" : "muted";
      return (
        <Badge variant={variant}>
          {String(v ?? "\u2014").toUpperCase()}
        </Badge>
      );
    },
  },
  {
    key: "entry_price" as const,
    label: "Entry",
    render: (v: unknown) => fmtPrice(v as number | null),
  },
  {
    key: "current_price" as const,
    label: "Current",
    render: (v: unknown) => fmtPrice(v as number | null),
  },
  {
    key: "stop_loss" as const,
    label: "Stop",
    render: (v: unknown) => fmtPrice(v as number | null),
  },
  {
    key: "target_price" as const,
    label: "Target",
    render: (v: unknown) => fmtPrice(v as number | null),
  },
  {
    key: "pnl_pct" as const,
    label: "P&L %",
    render: (_v: unknown, row: unknown) => {
      const trade = row as PaperTrade;
      const raw =
        trade.status === "open" ? trade.unrealized_pnl_pct : trade.pnl_pct;
      const pnl = typeof raw === "number" ? raw : null;
      if (pnl == null) return "\u2014";
      return (
        <span className={`font-medium ${pnlColor(pnl)}`}>
          {pnl > 0 ? "+" : ""}
          {pnl.toFixed(2)}%
        </span>
      );
    },
  },
  {
    key: "r_multiple" as const,
    label: "R",
    render: (v: unknown) => {
      const n = typeof v === "number" ? v : null;
      if (n == null) return "\u2014";
      return (
        <span className={pnlColor(n)}>
          {n > 0 ? "+" : ""}
          {n.toFixed(2)}R
        </span>
      );
    },
  },
  {
    key: "status" as const,
    label: "Status",
    render: (v: unknown) => {
      const s = String(v ?? "").toLowerCase();
      const variant =
        s === "open" ? "blue" : s === "closed" ? "muted" : "default";
      return (
        <Badge variant={variant}>
          {String(v ?? "\u2014").replace(/_/g, " ").toUpperCase()}
        </Badge>
      );
    },
  },
  {
    key: "exit_reason" as const,
    label: "Exit",
    render: (v: unknown) => {
      const val = typeof v === "string" ? v : null;
      return val ? val.replace(/_/g, " ") : "\u2014";
    },
  },
  {
    key: "setup_family" as const,
    label: "Setup",
    render: (v: unknown) => String(v ?? "\u2014"),
  },
  {
    key: "setup_tier" as const,
    label: "Tier",
    render: (v: unknown) => {
      const tier = typeof v === "string" ? v : null;
      return tier ? (
        <Badge variant={tierVariant(tier)}>{tier}</Badge>
      ) : (
        "\u2014"
      );
    },
  },
  {
    key: "entry_date" as const,
    label: "Entry Date",
    render: (v: unknown) => String(v ?? "\u2014"),
  },
];

export default function PaperTradePage() {
  const [statusFilter, setStatusFilter] = useState("all");
  const [dirFilter, setDirFilter] = useState("all");

  const { data: summary, isLoading: summaryLoading } =
    usePaperTradeSummary();
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
        Failed to load paper trades: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  const overall: PaperTradeSummaryOverall = summary?.overall ?? {
    total_trades: 0,
    open_count: 0,
    win_rate_pct: null,
    avg_pnl_pct: null,
    total_pnl_pct: null,
    avg_r_multiple: null,
  };
  const trades = tradesData?.trades ?? [];
  const equityCurve = summary?.equity_curve ?? [];
  const latestEquity =
    equityCurve.length > 0
      ? equityCurve[equityCurve.length - 1].equity_index
      : null;
  const maxDrawdown = (() => {
    if (equityCurve.length < 2) return null;
    let peak = equityCurve[0].equity_index;
    let maxDd = 0;
    for (const point of equityCurve) {
      if (point.equity_index > peak) peak = point.equity_index;
      const dd = ((peak - point.equity_index) / peak) * 100;
      if (dd > maxDd) maxDd = dd;
    }
    return maxDd;
  })();

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">Paper Trades</h2>

      {/* Summary KPI cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
        <Card
          glass
          label="Total Trades"
          value={overall.total_trades}
        />
        <Card
          glass
          label="Win Rate"
          value={fmtPct(overall.win_rate_pct)}
        />
        <Card
          glass
          label="Avg P&L"
          value={fmtPct(overall.avg_pnl_pct, "%", true)}
        />
        <Card
          glass
          label="Total P&L"
          value={fmtPct(overall.total_pnl_pct, "%", true)}
        />
        <Card
          glass
          label="Avg R-Multiple"
          value={
            overall.avg_r_multiple != null
              ? `${overall.avg_r_multiple.toFixed(2)}R`
              : "\u2014"
          }
        />
        <Card
          glass
          label="Max Drawdown"
          value={
            maxDrawdown != null ? `${maxDrawdown.toFixed(2)}%` : "\u2014"
          }
        />
        <Card
          glass
          label="Equity Index"
          value={latestEquity != null ? latestEquity.toFixed(2) : "\u2014"}
        />
      </div>

      {/* Equity curve chart */}
      {equityCurve.length > 1 ? (
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
      ) : (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
          No equity curve data available. Equity curve requires 2+ data
          points.
        </div>
      )}

      {/* By-direction and By-exit-reason panels */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card label="By Direction">
          {summary?.by_direction &&
          Object.keys(summary.by_direction).length > 0 ? (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                    <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">
                      Direction
                    </th>
                    <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">
                      Trades
                    </th>
                    <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">
                      Wins
                    </th>
                    <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">
                      Win Rate
                    </th>
                    <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">
                      Avg P&L
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.by_direction).map(([dir, stats]) => (
                    <tr
                      key={dir}
                      className="border-b border-[var(--line)] last:border-b-0"
                    >
                      <td className="px-2 py-1.5">
                        <Badge
                          variant={
                            dir === "long"
                              ? "green"
                              : dir === "short"
                                ? "red"
                                : "muted"
                          }
                        >
                          {dir.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="px-2 py-1.5 text-[var(--text)]">
                        {String(stats.total ?? "\u2014")}
                      </td>
                      <td className="px-2 py-1.5 text-[var(--text)]">
                        {String(stats.wins ?? "\u2014")}
                      </td>
                      <td className="px-2 py-1.5 text-[var(--text)]">
                        {typeof stats.win_rate_pct === "number" ? `${stats.win_rate_pct.toFixed(1)}%` : "\u2014"}
                      </td>
                      <td
                        className={`px-2 py-1.5 ${pnlColor(stats.avg_pnl_pct)}`}
                      >
                        {typeof stats.avg_pnl_pct === "number" ? `${stats.avg_pnl_pct > 0 ? "+" : ""}${stats.avg_pnl_pct.toFixed(2)}%` : "\u2014"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-1 text-xs text-[var(--muted)]">
              No closed trades yet.
            </p>
          )}
        </Card>

        <Card label="By Exit Reason">
          {summary?.by_exit_reason &&
          Object.keys(summary.by_exit_reason).length > 0 ? (
            <div className="mt-2 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                    <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">
                      Reason
                    </th>
                    <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">
                      Count
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.by_exit_reason).map(
                    ([reason, count]) => (
                      <tr
                        key={reason}
                        className="border-b border-[var(--line)] last:border-b-0"
                      >
                        <td className="px-2 py-1.5 capitalize text-[var(--text)]">
                          {reason.replace(/_/g, " ")}
                        </td>
                        <td className="px-2 py-1.5 text-[var(--text)]">
                          {String(count)}
                        </td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-1 text-xs text-[var(--muted)]">
              No closed trades yet.
            </p>
          )}
        </Card>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 text-xs text-[var(--muted)]">
        <label className="flex items-center gap-1.5">
          Status:
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-[var(--text)]"
          >
            <option value="all">All</option>
            <option value="open">Open</option>
            <option value="closed">Closed</option>
            <option value="stopped_out">Stopped Out</option>
            <option value="target_hit">Target Hit</option>
            <option value="expired">Expired</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5">
          Direction:
          <select
            value={dirFilter}
            onChange={(e) => setDirFilter(e.target.value)}
            className="rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-[var(--text)]"
          >
            <option value="all">All</option>
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
        </label>
        <span className="text-[var(--muted)]">
          Showing {trades.length} trade(s)
        </span>
      </div>

      {/* Trade log table */}
      <Table
        columns={tradeColumns}
        data={trades}
        sortable
      />
    </div>
  );
}

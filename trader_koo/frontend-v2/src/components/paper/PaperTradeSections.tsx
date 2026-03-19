import { Link } from "react-router-dom";
import { useMemo } from "react";
import type {
  PaperTrade,
  PaperTradeDirectionStats,
  PaperTradeSummary,
  PaperTradeSummaryOverall,
} from "../../api/types";
import { getPlotlyColors } from "../../lib/plotlyTheme";
import PlotlyWrapper from "../PlotlyWrapper";
import Badge, { tierVariant } from "../ui/Badge";
import Table from "../ui/Table";

const fmtPct = (
  value: number | null | undefined,
  suffix: string = "%",
  sign: boolean = false,
): string =>
  typeof value === "number"
    ? `${sign && value > 0 ? "+" : ""}${value.toFixed(2)}${suffix}`
    : "\u2014";

const fmtPrice = (value: number | null | undefined): string =>
  typeof value === "number" ? `$${value.toFixed(2)}` : "\u2014";

const fmtDollars = (value: number | null | undefined, compact = false): string => {
  if (typeof value !== "number") return "\u2014";
  if (compact) {
    const abs = Math.abs(value);
    const formatted =
      abs >= 1_000_000
        ? `$${(abs / 1_000_000).toFixed(2)}M`
        : abs >= 1_000
          ? `$${(abs / 1_000).toFixed(1)}K`
          : `$${abs.toFixed(0)}`;
    return value < 0 ? `-${formatted}` : formatted;
  }
  return `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const pnlColor = (value: number | null | undefined): string => {
  if (value == null) return "";
  if (value > 0) return "text-[var(--green)]";
  if (value < 0) return "text-[var(--red)]";
  return "";
};

/* ── Portfolio Hero ── */
export function PaperTradePortfolioHero({
  overall,
}: {
  overall: PaperTradeSummaryOverall;
}) {
  const portfolioValue = overall.portfolio_value ?? 1_000_000;
  const totalReturn = overall.total_return_pct ?? 0;
  const realizedPnl = overall.realized_pnl ?? 0;
  const unrealizedPnl = overall.unrealized_pnl ?? 0;

  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-5">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="label-sm tracking-widest">
            Portfolio Value
          </div>
          <div className="mt-1 text-3xl font-bold tracking-tight text-[var(--text)]">
            {fmtDollars(portfolioValue)}
          </div>
          <div className={`mt-1 text-sm font-medium ${pnlColor(totalReturn)}`}>
            {totalReturn > 0 ? "+" : ""}
            {totalReturn.toFixed(2)}% total return
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-5">
          <Stat label="Realized P&L" value={fmtDollars(realizedPnl, true)} tone={pnlColor(realizedPnl)} />
          <Stat label="Unrealized P&L" value={fmtDollars(unrealizedPnl, true)} tone={pnlColor(unrealizedPnl)} />
          <Stat label="Open" value={String(overall.open_count ?? 0)} />
          <Stat label="Win Rate" value={fmtPct(overall.win_rate_pct)} />
          <Stat
            label="Profit Factor"
            value={overall.profit_factor != null ? overall.profit_factor.toFixed(2) : "\u2014"}
          />
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tone = "text-[var(--text)]",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div>
      <div className="label-xs tracking-widest">
        {label}
      </div>
      <div className={`mt-0.5 text-lg font-semibold ${tone}`}>{value}</div>
    </div>
  );
}

/* ── Open Positions Table ── */
export function PaperTradeOpenPositions({ trades }: { trades: PaperTrade[] }) {
  const openTrades = trades.filter((t) => t.status === "open");

  if (openTrades.length === 0) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 text-center text-sm text-[var(--muted)]">
        No open positions
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-[var(--text)]">
          Open Positions
        </div>
        <Badge variant="blue">{openTrades.length} open</Badge>
      </div>
      <div className="overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--panel)]">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-[var(--line)] text-[var(--muted)]">
              <th className="px-3 py-2 font-semibold">Ticker</th>
              <th className="px-3 py-2 font-semibold">Dir</th>
              <th className="px-3 py-2 font-semibold">Entry</th>
              <th className="px-3 py-2 font-semibold">Current</th>
              <th className="px-3 py-2 font-semibold">Stop</th>
              <th className="px-3 py-2 font-semibold">Target</th>
              <th className="px-3 py-2 font-semibold">P&L %</th>
              <th className="px-3 py-2 font-semibold">Plan R</th>
              <th className="px-3 py-2 font-semibold">Tier</th>
              <th className="px-3 py-2 font-semibold">Entry Date</th>
            </tr>
          </thead>
          <tbody>
            {openTrades.map((t) => {
              const pnl = t.unrealized_pnl_pct;
              return (
                <tr key={t.id} className="border-b border-[var(--line)]/60 last:border-b-0">
                  <td className="px-3 py-2">
                    <Link
                      to={`/chart?t=${t.ticker}`}
                      className="font-mono font-bold text-[var(--accent)] hover:text-[var(--blue)]"
                    >
                      {t.ticker}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <Badge variant={t.direction === "long" ? "green" : "red"}>
                      {t.direction.toUpperCase()}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 text-[var(--text)]">{fmtPrice(t.entry_price)}</td>
                  <td className="px-3 py-2 text-[var(--text)]">{fmtPrice(t.current_price)}</td>
                  <td className="px-3 py-2 text-[var(--muted)]">{fmtPrice(t.stop_loss)}</td>
                  <td className="px-3 py-2 text-[var(--muted)]">{fmtPrice(t.target_price)}</td>
                  <td className={`px-3 py-2 font-medium ${pnlColor(pnl)}`}>
                    {typeof pnl === "number" ? `${pnl > 0 ? "+" : ""}${pnl.toFixed(2)}%` : "\u2014"}
                  </td>
                  <td className="px-3 py-2 text-[var(--muted)]">
                    {typeof t.expected_r_multiple === "number"
                      ? `${t.expected_r_multiple.toFixed(1)}R`
                      : "\u2014"}
                  </td>
                  <td className="px-3 py-2">
                    {t.setup_tier ? (
                      <Badge variant={tierVariant(t.setup_tier)}>{t.setup_tier}</Badge>
                    ) : (
                      "\u2014"
                    )}
                  </td>
                  <td className="px-3 py-2 text-[var(--muted)]">{t.entry_date ?? "\u2014"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Equity Curve ── */
export function PaperTradeEquityCurve({
  equityCurve,
}: {
  equityCurve: PaperTradeSummary["equity_curve"];
}) {
  if (equityCurve.length <= 1) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
        Equity curve requires 2+ data points.
      </div>
    );
  }

  const theme = getPlotlyColors();

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
      <PlotlyWrapper
        data={[
          {
            x: equityCurve.map((point) => point.date),
            y: equityCurve.map((point) => point.equity_index),
            type: "scatter" as const,
            mode: "lines" as const,
            fill: "tozeroy",
            line: { color: "#22c55e", width: 2 },
            fillcolor: "rgba(34,197,94,0.08)",
            name: "Equity Index",
          },
          {
            x: [equityCurve[0].date, equityCurve[equityCurve.length - 1].date],
            y: [100, 100],
            type: "scatter" as const,
            mode: "lines" as const,
            line: { color: "#6b7280", width: 1, dash: "dot" },
            name: "Baseline (100)",
          },
        ]}
        layout={{
          paper_bgcolor: theme.bg,
          plot_bgcolor: theme.bg,
          margin: { t: 20, r: 16, b: 40, l: 50 },
          font: { color: theme.font, size: 11 },
          xaxis: { gridcolor: theme.grid },
          yaxis: {
            gridcolor: theme.grid,
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
    </div>
  );
}

/* ── Performance Attribution ── */
function AttributionTable({
  title,
  data,
  showAvgR = false,
}: {
  title: string;
  data: Record<string, PaperTradeDirectionStats>;
  showAvgR?: boolean;
}) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return (
      <div className="text-xs text-[var(--muted)]">No {title.toLowerCase()} data yet.</div>
    );
  }

  return (
    <div>
      <div className="mb-2 text-xs font-semibold text-[var(--text)]">{title}</div>
      <div className="overflow-x-auto rounded-lg border border-[var(--line)]">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-[var(--line)] text-[var(--muted)]">
              <th className="px-3 py-1.5 font-semibold">Category</th>
              <th className="px-3 py-1.5 font-semibold text-right">Trades</th>
              <th className="px-3 py-1.5 font-semibold text-right">Win Rate</th>
              <th className="px-3 py-1.5 font-semibold text-right">Avg P&L</th>
              {showAvgR && (
                <th className="px-3 py-1.5 font-semibold text-right">Avg R</th>
              )}
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, stats]) => (
              <tr key={key} className="border-b border-[var(--line)]/60 last:border-b-0">
                <td className="px-3 py-1.5 font-medium text-[var(--text)] capitalize">
                  {key.replace(/_/g, " ")}
                </td>
                <td className="px-3 py-1.5 text-right text-[var(--text)]">{stats.total}</td>
                <td className="px-3 py-1.5 text-right text-[var(--text)]">
                  {fmtPct(stats.win_rate_pct)}
                </td>
                <td className={`px-3 py-1.5 text-right font-medium ${pnlColor(stats.avg_pnl_pct)}`}>
                  {fmtPct(stats.avg_pnl_pct, "%", true)}
                </td>
                {showAvgR && (
                  <td className={`px-3 py-1.5 text-right font-medium ${pnlColor((stats as unknown as Record<string, unknown>).avg_r_multiple as number | null)}`}>
                    {typeof (stats as unknown as Record<string, unknown>).avg_r_multiple === "number"
                      ? `${((stats as unknown as Record<string, unknown>).avg_r_multiple as number).toFixed(2)}R`
                      : "\u2014"}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ExitReasonTable({
  data,
}: {
  data: Record<string, number>;
}) {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return (
      <div className="text-xs text-[var(--muted)]">No exit reason data yet.</div>
    );
  }
  const total = entries.reduce((sum, [, count]) => sum + count, 0);

  return (
    <div>
      <div className="mb-2 text-xs font-semibold text-[var(--text)]">By Exit Reason</div>
      <div className="overflow-x-auto rounded-lg border border-[var(--line)]">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-[var(--line)] text-[var(--muted)]">
              <th className="px-3 py-1.5 font-semibold">Reason</th>
              <th className="px-3 py-1.5 font-semibold text-right">Count</th>
              <th className="px-3 py-1.5 font-semibold text-right">% of Total</th>
            </tr>
          </thead>
          <tbody>
            {entries
              .sort(([, a], [, b]) => b - a)
              .map(([reason, count]) => (
                <tr key={reason} className="border-b border-[var(--line)]/60 last:border-b-0">
                  <td className="px-3 py-1.5 font-medium text-[var(--text)] capitalize">
                    {reason.replace(/_/g, " ")}
                  </td>
                  <td className="px-3 py-1.5 text-right text-[var(--text)]">{count}</td>
                  <td className="px-3 py-1.5 text-right text-[var(--muted)]">
                    {total > 0 ? `${((count / total) * 100).toFixed(1)}%` : "\u2014"}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function PaperTradePerformanceAttribution({
  summary,
}: {
  summary: PaperTradeSummary;
}) {
  const hasData =
    Object.keys(summary.by_direction).length > 0 ||
    Object.keys(summary.by_family).length > 0 ||
    Object.keys(summary.by_exit_reason).length > 0;

  if (!hasData) return null;

  return (
    <details className="group rounded-xl border border-[var(--line)] bg-[var(--panel)]">
      <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-[var(--text)] select-none">
        Performance Attribution
        <span className="ml-2 text-xs text-[var(--muted)] group-open:hidden">
          (click to expand)
        </span>
      </summary>
      <div className="grid gap-6 border-t border-[var(--line)] px-4 py-4 sm:grid-cols-2 lg:grid-cols-3">
        <AttributionTable
          title="By Direction"
          data={summary.by_direction}
          showAvgR
        />
        <AttributionTable
          title="By Setup Family"
          data={summary.by_family}
        />
        <ExitReasonTable data={summary.by_exit_reason} />
      </div>
    </details>
  );
}

/* ── ML Calibration ── */
interface CalibrationBucket {
  label: string;
  min: number;
  max: number;
  total: number;
  wins: number;
  actualWinRate: number | null;
}

const ML_BUCKETS: Array<{ label: string; min: number; max: number }> = [
  { label: "0-40%", min: 0, max: 0.4 },
  { label: "40-50%", min: 0.4, max: 0.5 },
  { label: "50-60%", min: 0.5, max: 0.6 },
  { label: "60-70%", min: 0.6, max: 0.7 },
  { label: "70%+", min: 0.7, max: 1.01 },
];

export function PaperTradeMLCalibration({
  trades,
}: {
  trades: PaperTrade[];
}) {
  const buckets = useMemo<CalibrationBucket[]>(() => {
    const closedWithML = trades.filter(
      (t) =>
        t.status !== "open" &&
        typeof t.ml_predicted_win_prob === "number" &&
        t.ml_predicted_win_prob != null,
    );

    return ML_BUCKETS.map(({ label, min, max }) => {
      const inBucket = closedWithML.filter(
        (t) => t.ml_predicted_win_prob! >= min && t.ml_predicted_win_prob! < max,
      );
      const wins = inBucket.filter((t) => {
        const pnl = t.pnl_pct ?? 0;
        return pnl > 0;
      }).length;
      const total = inBucket.length;
      return {
        label,
        min,
        max,
        total,
        wins,
        actualWinRate: total > 0 ? (wins / total) * 100 : null,
      };
    });
  }, [trades]);

  const totalScored = buckets.reduce((sum, b) => sum + b.total, 0);

  if (totalScored === 0) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
        <div className="text-sm font-semibold text-[var(--text)]">ML Calibration</div>
        <div className="mt-2 text-xs text-[var(--muted)]">
          No closed trades with ML predictions yet. Calibration data will appear once trades with
          ml_predicted_win_prob are resolved.
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-[var(--text)]">ML Calibration</div>
        <span className="text-xs text-[var(--muted)]">{totalScored} scored trades</span>
      </div>
      <p className="mt-1 text-[10px] text-[var(--muted)]">
        Compares ML predicted win probability buckets to actual outcomes. A well-calibrated model
        shows actual win rates rising with predicted probability.
      </p>
      <div className="mt-3 overflow-x-auto rounded-lg border border-[var(--line)]">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-[var(--line)] text-[var(--muted)]">
              <th className="px-3 py-1.5 font-semibold">Predicted Prob</th>
              <th className="px-3 py-1.5 font-semibold text-right">Trades</th>
              <th className="px-3 py-1.5 font-semibold text-right">Wins</th>
              <th className="px-3 py-1.5 font-semibold text-right">Actual Win Rate</th>
              <th className="px-3 py-1.5 font-semibold text-right">Calibration</th>
            </tr>
          </thead>
          <tbody>
            {buckets.map((bucket) => {
              const midpoint = ((bucket.min + Math.min(bucket.max, 1)) / 2) * 100;
              const diff =
                bucket.actualWinRate != null ? bucket.actualWinRate - midpoint : null;
              const calibrationColor =
                diff == null
                  ? ""
                  : Math.abs(diff) <= 10
                    ? "text-[var(--green)]"
                    : Math.abs(diff) <= 20
                      ? "text-[var(--amber)]"
                      : "text-[var(--red)]";

              return (
                <tr
                  key={bucket.label}
                  className="border-b border-[var(--line)]/60 last:border-b-0"
                >
                  <td className="px-3 py-1.5 font-medium text-[var(--text)]">
                    {bucket.label}
                  </td>
                  <td className="px-3 py-1.5 text-right text-[var(--text)]">
                    {bucket.total}
                  </td>
                  <td className="px-3 py-1.5 text-right text-[var(--text)]">
                    {bucket.wins}
                  </td>
                  <td className="px-3 py-1.5 text-right text-[var(--text)]">
                    {bucket.actualWinRate != null
                      ? `${bucket.actualWinRate.toFixed(1)}%`
                      : "\u2014"}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-medium ${calibrationColor}`}>
                    {diff != null
                      ? `${diff > 0 ? "+" : ""}${diff.toFixed(1)}pp`
                      : "\u2014"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── CSV Export ── */
const CSV_COLUMNS = [
  "ticker",
  "direction",
  "entry_price",
  "exit_price",
  "entry_date",
  "exit_date",
  "pnl_pct",
  "r_multiple",
  "status",
  "exit_reason",
  "setup_family",
  "setup_tier",
  "ml_predicted_win_prob",
] as const;

function escapeCsvField(value: unknown): string {
  if (value == null) return "";
  const str = String(value);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function exportTradesToCsv(trades: PaperTrade[]): void {
  const header = CSV_COLUMNS.join(",");
  const rows = trades.map((trade) =>
    CSV_COLUMNS.map((col) =>
      escapeCsvField((trade as unknown as Record<string, unknown>)[col]),
    ).join(","),
  );
  const csvContent = [header, ...rows].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `paper_trades_${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

/* ── Filters ── */
export function PaperTradeFilters({
  statusFilter,
  directionFilter,
  tradeCount,
  trades,
  onStatusChange,
  onDirectionChange,
}: {
  statusFilter: string;
  directionFilter: string;
  tradeCount: number;
  trades: PaperTrade[];
  onStatusChange: (value: string) => void;
  onDirectionChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-[var(--muted)]">
      <label className="flex items-center gap-1.5">
        Status:
        <select
          value={statusFilter}
          onChange={(event) => onStatusChange(event.target.value)}
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
          value={directionFilter}
          onChange={(event) => onDirectionChange(event.target.value)}
          className="rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-[var(--text)]"
        >
          <option value="all">All</option>
          <option value="long">Long</option>
          <option value="short">Short</option>
        </select>
      </label>
      <span className="text-[var(--muted)]">Showing {tradeCount} trade(s)</span>
      <button
        type="button"
        onClick={() => exportTradesToCsv(trades)}
        disabled={trades.length === 0}
        className="rounded border border-[var(--line)] bg-[var(--bg)] px-3 py-1 text-[var(--text)] transition-colors hover:bg-[var(--panel)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        Export CSV
      </button>
    </div>
  );
}

/* ── Closed Trades Table ── */
export const tradeColumns = [
  {
    key: "ticker" as const,
    label: "Ticker",
    render: (value: unknown) => {
      const ticker = String(value ?? "");
      if (!ticker) return "\u2014";
      return (
        <Link
          to={`/chart?t=${ticker}`}
          className="font-mono font-bold text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
        >
          {ticker}
        </Link>
      );
    },
  },
  {
    key: "direction" as const,
    label: "Dir",
    render: (value: unknown) => {
      const direction = String(value ?? "").toLowerCase();
      const variant =
        direction === "long" ? "green" : direction === "short" ? "red" : "muted";
      return <Badge variant={variant}>{String(value ?? "\u2014").toUpperCase()}</Badge>;
    },
  },
  {
    key: "entry_price" as const,
    label: "Entry",
    render: (value: unknown) => fmtPrice(value as number | null),
  },
  {
    key: "current_price" as const,
    label: "Current",
    render: (value: unknown) => fmtPrice(value as number | null),
  },
  {
    key: "pnl_pct" as const,
    label: "P&L %",
    render: (_value: unknown, row: unknown) => {
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
    render: (value: unknown) => {
      const multiple = typeof value === "number" ? value : null;
      if (multiple == null) return "\u2014";
      return (
        <span className={pnlColor(multiple)}>
          {multiple > 0 ? "+" : ""}
          {multiple.toFixed(2)}R
        </span>
      );
    },
  },
  {
    key: "status" as const,
    label: "Status",
    render: (value: unknown) => {
      const status = String(value ?? "").toLowerCase();
      const variant =
        status === "open" ? "blue" : status === "closed" ? "muted" : "default";
      return (
        <Badge variant={variant}>
          {String(value ?? "\u2014").replace(/_/g, " ").toUpperCase()}
        </Badge>
      );
    },
  },
  {
    key: "exit_reason" as const,
    label: "Exit",
    render: (value: unknown) => {
      const reason = typeof value === "string" ? value : null;
      return reason ? reason.replace(/_/g, " ") : "\u2014";
    },
  },
  {
    key: "setup_tier" as const,
    label: "Tier",
    render: (value: unknown) => {
      const tier = typeof value === "string" ? value : null;
      return tier ? <Badge variant={tierVariant(tier)}>{tier}</Badge> : "\u2014";
    },
  },
  {
    key: "entry_date" as const,
    label: "Entry Date",
    render: (value: unknown) => String(value ?? "\u2014"),
  },
];

export function PaperTradeLogTable({ trades }: { trades: PaperTrade[] }) {
  return <Table columns={tradeColumns} data={trades} sortable />;
}

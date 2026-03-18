import { Link } from "react-router-dom";
import type {
  PaperTrade,
  PaperTradeDirectionStats,
  PaperTradeSummary,
  PaperTradeSummaryOverall,
} from "../../api/types";
import PlotlyWrapper from "../PlotlyWrapper";
import Badge, { tierVariant } from "../ui/Badge";
import Card from "../ui/Card";
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

const pnlColor = (value: number | null | undefined): string => {
  if (value == null) return "";
  if (value > 0) return "text-[var(--green)]";
  if (value < 0) return "text-[var(--red)]";
  return "";
};

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
    key: "stop_loss" as const,
    label: "Stop",
    render: (value: unknown) => fmtPrice(value as number | null),
  },
  {
    key: "target_price" as const,
    label: "Target",
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
    key: "decision_state" as const,
    label: "Decision",
    render: (_value: unknown, row: unknown) => {
      const trade = row as PaperTrade;
      const state = String(
        trade.decision_state ?? trade.portfolio_decision ?? "",
      ).toLowerCase();
      const variant =
        state === "approved"
          ? "green"
          : state === "approved_with_flags"
            ? "amber"
            : state === "rejected"
              ? "red"
              : "muted";
      const label = state ? state.replace(/_/g, " ").toUpperCase() : "\u2014";
      return (
        <div className="space-y-1">
          <Badge variant={variant}>{label}</Badge>
          {trade.decision_summary ? (
            <div className="max-w-[16rem] text-[11px] leading-snug text-[var(--muted)]">
              {trade.decision_summary}
            </div>
          ) : null}
          {trade.sizing_summary ? (
            <div className="max-w-[16rem] text-[11px] leading-snug text-[var(--muted)]">
              {trade.sizing_summary}
            </div>
          ) : null}
          {typeof trade.expected_r_multiple === "number" ? (
            <div className="text-[11px] leading-snug text-[var(--muted)]">
              Plan {trade.expected_r_multiple.toFixed(2)}R
            </div>
          ) : null}
          {trade.status !== "open" && trade.review_summary ? (
            <div className="max-w-[16rem] text-[11px] leading-snug text-[var(--muted)]">
              {trade.review_summary}
            </div>
          ) : null}
        </div>
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
    key: "setup_family" as const,
    label: "Setup",
    render: (value: unknown) => String(value ?? "\u2014"),
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

export function PaperTradeSummaryGrid({
  overall,
  maxDrawdown,
  latestEquity,
}: {
  overall: PaperTradeSummaryOverall;
  maxDrawdown: number | null;
  latestEquity: number | null;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-9">
      <Card glass label="Total Trades" value={overall.total_trades} />
      <Card glass label="Win Rate" value={fmtPct(overall.win_rate_pct)} />
      <Card glass label="Avg P&L" value={fmtPct(overall.avg_pnl_pct, "%", true)} />
      <Card glass label="Expectancy" value={fmtPct(overall.expectancy_pct, "%", true)} />
      <Card glass label="Total P&L" value={fmtPct(overall.total_pnl_pct, "%", true)} />
      <Card
        glass
        label="Avg R-Multiple"
        value={overall.avg_r_multiple != null ? `${overall.avg_r_multiple.toFixed(2)}R` : "\u2014"}
      />
      <Card
        glass
        label="Profit Factor"
        value={overall.profit_factor != null ? overall.profit_factor.toFixed(2) : "\u2014"}
      />
      <Card
        glass
        label="Max Drawdown"
        value={maxDrawdown != null ? `${maxDrawdown.toFixed(2)}%` : "\u2014"}
      />
      <Card
        glass
        label="Equity Index"
        value={latestEquity != null ? latestEquity.toFixed(2) : "\u2014"}
      />
    </div>
  );
}

export function PaperTradeEquityCurve({
  equityCurve,
}: {
  equityCurve: PaperTradeSummary["equity_curve"];
}) {
  if (equityCurve.length <= 1) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
        No equity curve data available. Equity curve requires 2+ data
        points.
      </div>
    );
  }

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
    </div>
  );
}

function DirectionStatsTable({ rows }: { rows: Array<[string, PaperTradeDirectionStats]> }) {
  return (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--line)] text-[var(--muted)]">
            <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Direction</th>
            <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Trades</th>
            <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Wins</th>
            <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Win Rate</th>
            <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Avg P&L</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([direction, stats]) => (
            <tr key={direction} className="border-b border-[var(--line)] last:border-b-0">
              <td className="px-2 py-1.5">
                <Badge
                  variant={
                    direction === "long"
                      ? "green"
                      : direction === "short"
                        ? "red"
                        : "muted"
                  }
                >
                  {direction.toUpperCase()}
                </Badge>
              </td>
              <td className="px-2 py-1.5 text-[var(--text)]">{String(stats.total ?? "\u2014")}</td>
              <td className="px-2 py-1.5 text-[var(--text)]">{String(stats.wins ?? "\u2014")}</td>
              <td className="px-2 py-1.5 text-[var(--text)]">
                {typeof stats.win_rate_pct === "number" ? `${stats.win_rate_pct.toFixed(1)}%` : "\u2014"}
              </td>
              <td className={`px-2 py-1.5 ${pnlColor(stats.avg_pnl_pct)}`}>
                {typeof stats.avg_pnl_pct === "number"
                  ? `${stats.avg_pnl_pct > 0 ? "+" : ""}${stats.avg_pnl_pct.toFixed(2)}%`
                  : "\u2014"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExitReasonTable({ rows }: { rows: Array<[string, number]> }) {
  return (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-[var(--line)] text-[var(--muted)]">
            <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Reason</th>
            <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Count</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([reason, count]) => (
            <tr key={reason} className="border-b border-[var(--line)] last:border-b-0">
              <td className="px-2 py-1.5 capitalize text-[var(--text)]">
                {reason.replace(/_/g, " ")}
              </td>
              <td className="px-2 py-1.5 text-[var(--text)]">{String(count)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PaperTradeBreakdownPanels({
  summary,
}: {
  summary: PaperTradeSummary | undefined;
}) {
  const byDirection = summary?.by_direction ? Object.entries(summary.by_direction) : [];
  const byExitReason = summary?.by_exit_reason ? Object.entries(summary.by_exit_reason) : [];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card label="By Direction">
        {byDirection.length > 0 ? (
          <DirectionStatsTable rows={byDirection} />
        ) : (
          <p className="mt-1 text-xs text-[var(--muted)]">No closed trades yet.</p>
        )}
      </Card>

      <Card label="By Exit Reason">
        {byExitReason.length > 0 ? (
          <ExitReasonTable rows={byExitReason} />
        ) : (
          <p className="mt-1 text-xs text-[var(--muted)]">No closed trades yet.</p>
        )}
      </Card>
    </div>
  );
}

export function PaperTradeFilters({
  statusFilter,
  directionFilter,
  tradeCount,
  onStatusChange,
  onDirectionChange,
}: {
  statusFilter: string;
  directionFilter: string;
  tradeCount: number;
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
    </div>
  );
}

export function PaperTradeLogTable({ trades }: { trades: PaperTrade[] }) {
  return <Table columns={tradeColumns} data={trades} sortable />;
}

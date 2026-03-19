import { Link } from "react-router-dom";
import type {
  PaperTrade,
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

const fmtDollars = (value: number | null | undefined): string => {
  if (typeof value !== "number") return "\u2014";
  const abs = Math.abs(value);
  const formatted =
    abs >= 1_000_000
      ? `$${(abs / 1_000_000).toFixed(2)}M`
      : abs >= 1_000
        ? `$${(abs / 1_000).toFixed(1)}K`
        : `$${abs.toFixed(0)}`;
  return value < 0 ? `-${formatted}` : formatted;
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
          <Stat label="Realized P&L" value={fmtDollars(realizedPnl)} tone={pnlColor(realizedPnl)} />
          <Stat label="Unrealized P&L" value={fmtDollars(unrealizedPnl)} tone={pnlColor(unrealizedPnl)} />
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

/* ── Filters ── */
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

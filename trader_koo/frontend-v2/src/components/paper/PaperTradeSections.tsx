import { Link } from "react-router-dom";
import type {
  PaperTrade,
  PaperTradeFamilyEdgeRow,
  PaperTradeFeedbackItem,
  PaperTradePolicy,
  PaperTradeDirectionStats,
  PaperTradeRegimeEdgeRow,
  PaperTradeSummary,
  PaperTradeSummaryOverall,
  PaperTradeVixBucketEdgeRow,
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

const severityVariant = (value: string | null | undefined) => {
  const severity = String(value ?? "").toLowerCase();
  if (severity === "high" || severity === "red" || severity === "risk") return "red";
  if (severity === "medium" || severity === "amber") return "amber";
  if (severity === "green" || severity === "positive") return "green";
  return "muted";
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

function HeroMetric({
  label,
  value,
  tone = "text-[var(--text)]",
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] px-4 py-3 backdrop-blur-sm">
      <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-[var(--muted)]">
        {label}
      </div>
      <div className={`mt-2 text-2xl font-semibold tracking-tight ${tone}`}>{value}</div>
    </div>
  );
}

export function PaperTradeHero({
  overall,
  latestEquity,
  maxDrawdown,
  policy,
}: {
  overall: PaperTradeSummaryOverall;
  latestEquity: number | null;
  maxDrawdown: number | null;
  policy: PaperTradePolicy | null | undefined;
}) {
  return (
    <div className="relative overflow-hidden rounded-[28px] border border-[var(--line)] bg-[linear-gradient(135deg,rgba(38,64,122,0.28),rgba(12,18,32,0.96)_42%,rgba(20,64,56,0.32))] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.28)] sm:p-6">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-16 right-[-4rem] h-52 w-52 rounded-full bg-[rgba(96,165,250,0.12)] blur-3xl" />
        <div className="absolute bottom-[-3rem] left-[-2rem] h-40 w-40 rounded-full bg-[rgba(34,197,94,0.10)] blur-3xl" />
      </div>
      <div className="relative grid gap-5 xl:grid-cols-[1.4fr_1fr]">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="blue">Auto Paper Trader</Badge>
            {policy?.bot_version ? <Badge variant="muted">{policy.bot_version}</Badge> : null}
            {policy?.decision_version ? <Badge variant="muted">{policy.decision_version}</Badge> : null}
            {typeof overall.open_count === "number" ? (
              <Badge variant={overall.open_count > 0 ? "green" : "muted"}>
                {overall.open_count} open
              </Badge>
            ) : null}
          </div>
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-[var(--text)] sm:text-3xl">
              Versioned swing-trade paper bot
            </h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)] sm:text-[15px]">
              Nightly setups are filtered, sized, planned, simulated, and reviewed as a governed
              paper-trading system. This is the proving ground before any live execution.
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <HeroMetric label="Win Rate" value={fmtPct(overall.win_rate_pct)} tone="text-[var(--green)]" />
            <HeroMetric
              label="Expectancy"
              value={fmtPct(overall.expectancy_pct, "%", true)}
              tone={pnlColor(overall.expectancy_pct)}
            />
            <HeroMetric
              label="Profit Factor"
              value={overall.profit_factor != null ? overall.profit_factor.toFixed(2) : "\u2014"}
            />
            <HeroMetric
              label="Open Risk"
              value={typeof overall.open_count === "number" ? String(overall.open_count) : "\u2014"}
            />
          </div>
        </div>
        <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
          <div className="rounded-[24px] border border-[var(--line)] bg-[var(--panel)] p-4 backdrop-blur-sm">
            <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-[var(--muted)]">
              Capital Discipline
            </div>
            <div className="mt-3 space-y-2 text-sm text-[var(--muted)]">
              <div className="flex items-center justify-between gap-4">
                <span>Equity Index</span>
                <span className="font-semibold text-[var(--text)]">
                  {latestEquity != null ? latestEquity.toFixed(2) : "\u2014"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Max Drawdown</span>
                <span className="font-semibold text-[var(--text)]">
                  {maxDrawdown != null ? `${maxDrawdown.toFixed(2)}%` : "\u2014"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-4">
                <span>Total Closed</span>
                <span className="font-semibold text-[var(--text)]">{overall.total_trades}</span>
              </div>
            </div>
          </div>
          <div className="rounded-[24px] border border-[var(--line)] bg-[var(--panel)] p-4 backdrop-blur-sm sm:col-span-2 xl:col-span-1">
            <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-[var(--muted)]">
              Policy Guardrails
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {policy ? (
                <>
                  <Badge variant="green">Min Tier {policy.min_tier}</Badge>
                  <Badge variant="amber">Min Score {policy.min_score.toFixed(0)}</Badge>
                  <Badge variant="muted">Max Open {policy.max_open}</Badge>
                  <Badge variant="muted">{policy.expiry_days}d expiry</Badge>
                  <Badge variant="muted">Min {policy.min_reward_r_multiple.toFixed(1)}R</Badge>
                </>
              ) : (
                <span className="text-sm text-[var(--muted)]">Policy metadata unavailable.</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

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

export function PaperTradeBotOverview({
  overall,
  policy,
}: {
  overall: PaperTradeSummaryOverall;
  policy: PaperTradePolicy | null | undefined;
}) {
  if (!policy) {
    return (
      <Card label="Auto Paper Trader">
        <p className="text-sm text-[var(--muted)]">
          Bot policy metadata is not available yet.
        </p>
      </Card>
    );
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
      <Card label="Auto Paper Trader">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge variant="blue">{policy.decision_version}</Badge>
            <Badge variant="muted">{policy.bot_version}</Badge>
            <Badge variant="green">Min Tier {policy.min_tier}</Badge>
            <Badge variant="amber">Min Score {policy.min_score.toFixed(0)}</Badge>
            <Badge variant="muted">Max Open {policy.max_open}</Badge>
            <Badge variant="muted">{policy.expiry_days}d expiry</Badge>
            <Badge variant="muted">Min {policy.min_reward_r_multiple.toFixed(1)}R</Badge>
          </div>
          <p className="text-sm text-[var(--text)]">
            The bot paper-trades nightly report setups that clear tier, score,
            actionability, debate, and risk gates. Each approved trade gets a
            size, invalidation, target, and review loop.
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-[var(--line)] bg-[var(--bg)] p-3">
                <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
                  Sizing Rules
                </div>
                <div className="mt-2 space-y-1 text-xs text-[var(--muted)]">
                <div>A: {policy.position_size_pct["A"]?.toFixed(1)}%</div>
                <div>B: {policy.position_size_pct["B"]?.toFixed(1)}%</div>
                <div>C: {policy.position_size_pct["C"]?.toFixed(1)}%</div>
                <div>Caution scale: {(policy.caution_position_scale * 100).toFixed(0)}%</div>
                <div>High-vol scale: {(policy.high_vol_position_scale * 100).toFixed(0)}%</div>
              </div>
            </div>
            <div className="rounded-xl border border-[var(--line)] bg-[var(--bg)] p-3">
              <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
                Current Edge
              </div>
              <div className="mt-2 space-y-1 text-xs text-[var(--muted)]">
                <div>Closed trades: {overall.total_trades}</div>
                <div>Open trades: {overall.open_count}</div>
                <div>Win rate: {fmtPct(overall.win_rate_pct)}</div>
                <div>Expectancy: {fmtPct(overall.expectancy_pct, "%", true)}</div>
                <div>Avg R: {overall.avg_r_multiple != null ? `${overall.avg_r_multiple.toFixed(2)}R` : "\u2014"}</div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      <Card label="Promotion Gate">
        <div className="space-y-2 text-sm text-[var(--muted)]">
          <p>
            Treat this as a versioned paper-trading bot, not a self-editing live
            system. Promote changes only after a larger sample shows stable
            expectancy, controlled drawdown, and repeatable family-level edge.
          </p>
          <p>
            Current version:{" "}
            <strong className="text-[var(--text)]">{policy.bot_version}</strong>{" "}
            <span className="text-[var(--muted)]">({policy.decision_version})</span>
          </p>
        </div>
      </Card>
    </div>
  );
}

type EdgeTableRow = {
  trade_count: number;
  win_rate_pct: number;
  avg_pnl_pct: number;
  avg_r_multiple: number | null;
};

function EdgeTable<T extends EdgeTableRow>({
  title,
  rows,
  labelKey,
  labelHeader,
}: {
  title: string;
  rows: T[];
  labelKey: keyof T;
  labelHeader: string;
}) {
  if (rows.length === 0) {
    return (
      <Card label={title}>
        <p className="text-sm text-[var(--muted)]">
          Not enough closed-trade history yet for this slice.
        </p>
      </Card>
    );
  }

  return (
    <Card label={title}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-[var(--line)] text-[var(--muted)]">
              <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">{labelHeader}</th>
              <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Trades</th>
              <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Win Rate</th>
              <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Avg P&L</th>
              <th className="px-2 py-1.5 font-semibold uppercase tracking-wider">Avg R</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => {
              const label = String(row[labelKey] ?? "\u2014");
              const tradeCount = Number(row.trade_count ?? 0);
              const winRate = Number(row.win_rate_pct ?? 0);
              const avgPnl = typeof row.avg_pnl_pct === "number" ? row.avg_pnl_pct : Number(row.avg_pnl_pct ?? 0);
              const avgR = row.avg_r_multiple;
              return (
                <tr key={`${label}-${idx}`} className="border-b border-[var(--line)]/60">
                  <td className="px-2 py-2 text-[var(--text)]">{label}</td>
                  <td className="px-2 py-2 text-[var(--muted)]">{tradeCount}</td>
                  <td className="px-2 py-2 text-[var(--muted)]">{winRate.toFixed(1)}%</td>
                  <td className={`px-2 py-2 ${pnlColor(avgPnl)}`}>{fmtPct(avgPnl, "%", true)}</td>
                  <td className="px-2 py-2 text-[var(--muted)]">
                    {typeof avgR === "number" ? `${avgR.toFixed(2)}R` : "\u2014"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export function PaperTradeEdgePanels({
  familyEdges,
  regimeEdges,
  vixBucketEdges,
}: {
  familyEdges: PaperTradeFamilyEdgeRow[];
  regimeEdges: PaperTradeRegimeEdgeRow[];
  vixBucketEdges: PaperTradeVixBucketEdgeRow[];
}) {
  return (
    <div className="space-y-3">
      <div className="text-sm font-semibold text-[var(--text)]">Rolling Edge Monitor</div>
      <div className="grid gap-3 xl:grid-cols-3">
        <EdgeTable
          title="Family Edge"
          rows={familyEdges.slice(0, 8).map((row) => ({
            ...row,
            label: `${row.setup_family} (${row.direction})`,
          }))}
          labelKey={"label"}
          labelHeader="Family"
        />
        <EdgeTable
          title="Regime Edge"
          rows={regimeEdges}
          labelKey={"regime"}
          labelHeader="Regime"
        />
        <EdgeTable
          title="VIX Bucket Edge"
          rows={vixBucketEdges}
          labelKey={"vix_bucket"}
          labelHeader="VIX Bucket"
        />
      </div>
    </div>
  );
}

export function PaperTradeOpenPlans({
  trades,
}: {
  trades: PaperTrade[];
}) {
  const openTrades = trades.filter((trade) => trade.status === "open").slice(0, 4);
  if (openTrades.length === 0) {
    return (
      <Card label="Open Trade Plans">
        <p className="text-sm text-[var(--muted)]">
          No open paper trades right now.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[var(--text)]">Execution Board</div>
          <div className="text-xs text-[var(--muted)]">
            Live paper positions with size, invalidation, and planned exits.
          </div>
        </div>
        <Badge variant="blue">{openTrades.length} live plans</Badge>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {openTrades.map((trade) => (
          <Card key={trade.id} label={`${trade.ticker} ${trade.direction.toUpperCase()}`}>
            <div className="space-y-3 text-sm">
              <div className="flex flex-wrap gap-2">
                {trade.setup_tier ? <Badge variant={tierVariant(trade.setup_tier)}>{trade.setup_tier}</Badge> : null}
                {trade.decision_state ? (
                  <Badge variant={severityVariant(trade.decision_state)}>
                    {trade.decision_state.replace(/_/g, " ").toUpperCase()}
                  </Badge>
                ) : null}
                {typeof trade.expected_r_multiple === "number" ? (
                  <Badge variant="muted">Plan {trade.expected_r_multiple.toFixed(2)}R</Badge>
                ) : null}
              </div>
              <div className="grid gap-2 sm:grid-cols-3 text-xs text-[var(--muted)]">
                <div>
                  <div className="uppercase tracking-wider">Entry</div>
                  <div className="mt-1 text-[var(--text)]">{fmtPrice(trade.entry_price)}</div>
                </div>
                <div>
                  <div className="uppercase tracking-wider">Stop</div>
                  <div className="mt-1 text-[var(--text)]">{fmtPrice(trade.stop_loss)}</div>
                </div>
                <div>
                  <div className="uppercase tracking-wider">Target</div>
                  <div className="mt-1 text-[var(--text)]">{fmtPrice(trade.target_price)}</div>
                </div>
              </div>
              <div className="grid gap-2 sm:grid-cols-3 text-xs text-[var(--muted)]">
                <div>
                  <div className="uppercase tracking-wider">Size</div>
                  <div className="mt-1 text-[var(--text)]">{fmtPct(trade.position_size_pct)}</div>
                </div>
                <div>
                  <div className="uppercase tracking-wider">Risk Budget</div>
                  <div className="mt-1 text-[var(--text)]">{fmtPct(trade.risk_budget_pct)}</div>
                </div>
                <div>
                  <div className="uppercase tracking-wider">Current P&L</div>
                  <div className={`mt-1 ${pnlColor(trade.unrealized_pnl_pct)}`}>
                    {fmtPct(trade.unrealized_pnl_pct, "%", true)}
                  </div>
                </div>
              </div>
              {trade.entry_plan ? (
                <p className="text-xs text-[var(--text)]">
                  <strong>Entry:</strong> {trade.entry_plan}
                </p>
              ) : null}
              {trade.exit_plan ? (
                <p className="text-xs text-[var(--muted)]">
                  <strong className="text-[var(--text)]">Exit:</strong> {trade.exit_plan}
                </p>
              ) : null}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

export function PaperTradeFeedbackPanel({
  feedback,
}: {
  feedback: PaperTradeFeedbackItem[];
}) {
  if (feedback.length === 0) {
    return (
      <Card label="Model Feedback">
        <p className="text-sm text-[var(--muted)]">
          Not enough closed-trade evidence yet to generate tuning notes.
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[var(--text)]">Post-Trade Feedback Loop</div>
          <div className="text-xs text-[var(--muted)]">
            Evidence-based tuning notes from closed paper trades and rolling edge data.
          </div>
        </div>
        <Badge variant="muted">{feedback.length} notes</Badge>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {feedback.map((item, index) => (
          <Card key={`${item.kind}-${index}`} label={item.title}>
            <div className="space-y-2 text-sm">
              <Badge variant={severityVariant(item.severity)}>
                {item.severity.toUpperCase()}
              </Badge>
              <p className="text-[var(--muted)]">{item.detail}</p>
              <p className="text-[var(--text)]">
                <strong>Tune:</strong> {item.action}
              </p>
            </div>
          </Card>
        ))}
      </div>
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

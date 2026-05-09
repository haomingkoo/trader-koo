import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../../api/client";
import PlotlyWrapper from "../PlotlyWrapper";
import Spinner from "../ui/Spinner";
import { getPlotlyColors } from "../../lib/plotlyTheme";

/* ── Live Signal Panel ── */
interface LivePosition {
  coin: string;
  side: string;
  notional_usd: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  leverage: string;
  liquidation_price: number | null;
  mark_price_source: string;
  notional_source: string;
  data_warnings: string[];
}

interface LiveData {
  ok: boolean;
  wallet: { account_value: number; margin_ratio: number };
  positions: LivePosition[];
  config: {
    min_counter_notional_usd: number;
  };
}

function LiveSignalPanel({ wallet }: { wallet: string }) {
  const { data } = useQuery<LiveData>({
    queryKey: ["hl-live", wallet],
    queryFn: () => apiFetch(`/api/hyperliquid/live/${wallet}`),
    refetchInterval: 60_000,
  });

  if (!data?.ok || !data.positions?.length) return null;

  const threshold = data.config.min_counter_notional_usd;
  const totalNotional = data.positions.reduce((s, p) => s + p.notional_usd, 0);
  const acctLeverage = data.wallet.account_value > 0 ? totalNotional / data.wallet.account_value : 0;
  const counterPositions = data.positions.filter((p) => p.notional_usd >= threshold);
  const hasSignal = counterPositions.length > 0;

  return (
    <div className={`rounded-xl border-2 p-4 sm:p-6 ${hasSignal ? "border-[var(--green)] bg-[var(--green)]/5" : "border-[var(--line)] bg-[var(--panel)]"}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`inline-block h-3 w-3 rounded-full ${hasSignal ? "bg-[var(--green)] animate-pulse" : "bg-[var(--muted)]"}`} />
          <h4 className="text-sm font-bold text-[var(--text)]">
            {hasSignal ? "Counter-Trade Signal Active" : "Monitoring"}
          </h4>
        </div>
        <span className="text-[10px] text-[var(--muted)]">Updates every 60s</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4 text-xs">
        <div>
          <div className="text-[var(--muted)]">Account</div>
          <div className="font-semibold text-[var(--text)]">${data.wallet.account_value.toLocaleString()}</div>
        </div>
        <div>
          <div className="text-[var(--muted)]">Total Exposure</div>
          <div className="font-semibold text-[var(--text)]">{formatUsd(totalNotional)}</div>
        </div>
        <div>
          <div className="text-[var(--muted)]">Account Leverage</div>
          <div className={`font-semibold ${acctLeverage > 10 ? "text-[var(--red)]" : "text-[var(--text)]"}`}>{acctLeverage.toFixed(0)}x</div>
        </div>
        <div>
          <div className="text-[var(--muted)]">Research Threshold</div>
          <div className="font-semibold text-[var(--text)]">{formatUsd(threshold)}</div>
        </div>
      </div>

      {data.positions.map((p) => {
        const above = p.notional_usd >= threshold;
        const counterSide = p.side === "long" ? "SHORT" : "LONG";
        const liqDist = p.liquidation_price && p.mark_price > 0
          ? Math.abs(p.mark_price - p.liquidation_price) / p.mark_price * 100
          : null;
        return (
          <div key={p.coin} className={`rounded-lg border px-3 py-2 mb-2 ${above ? "border-[var(--green)]/50 bg-[var(--green)]/5" : "border-[var(--line)] bg-[var(--bg)]"}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {above && <span className="rounded bg-[var(--green)] px-1.5 py-0.5 text-[9px] font-bold text-black">{counterSide} {p.coin}</span>}
                <span className="text-xs font-medium text-[var(--text)]">{p.coin} {p.side.toUpperCase()}</span>
                <span className="text-[10px] text-[var(--muted)]">{p.leverage}</span>
              </div>
              <span className="text-xs font-medium text-[var(--text)]">{formatUsd(p.notional_usd)}</span>
            </div>
            <div className="flex items-center justify-between mt-1 text-[10px] text-[var(--muted)]">
              <span>Entry: ${p.entry_price.toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
              <span className={p.unrealized_pnl >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}>
                uPnL: ${p.unrealized_pnl.toLocaleString()}
              </span>
              {liqDist !== null && liqDist < 10 && (
                <span className="text-[var(--red)]">Liq {liqDist.toFixed(1)}% away</span>
              )}
            </div>
            {above && (
              <div className="mt-1 text-[10px] text-[var(--green)]">
                {(p.notional_usd / threshold).toFixed(1)}x above research threshold - review study before acting
              </div>
            )}
          </div>
        );
      })}

      <div className="text-[9px] text-[var(--muted)] mt-2">
        Research signal only. Not financial advice. Past performance does not guarantee future results.
      </div>
    </div>
  );
}

interface BucketData {
  bucket: string;
  count: number;
  win_rate_pct: number;
  avg_pnl: number;
  total_pnl: number;
  counter_edge_total: number;
  action: string;
}

interface DurationData {
  duration: string;
  count: number;
  win_rate_pct: number;
  avg_pnl: number;
  total_pnl: number;
}

interface CoinData {
  coin: string;
  count: number;
  win_rate_pct: number;
  avg_pnl: number;
  total_pnl: number;
  liquidations: number;
}

interface MonthlyData {
  month: string;
  count: number;
  win_rate_pct: number;
  avg_pnl: number;
  total_pnl: number;
}

interface StrategyRule {
  condition: string;
  action: string;
  reason: string;
  counter_edge: number;
}

interface EquityPoint {
  date: string;
  equity: number;
  coin?: string;
  our_pnl?: number;
}

interface BacktestStrategy {
  trades: number;
  wins: number;
  win_rate_pct: number;
  total_pnl: number;
  return_pct: number;
  max_drawdown_pct: number;
  final_equity: number;
  description?: string;
  equity_curve: EquityPoint[];
}

interface StudyData {
  ok: boolean;
  wallet: string;
  overview: {
    total_cycles: number;
    total_pnl: number;
    win_rate_pct: number;
    avg_pnl: number;
    total_fees: number;
    liquidation_cycles: number;
    date_range: { start: string; end: string };
  };
  notional_analysis: BucketData[];
  duration_analysis: DurationData[];
  coin_analysis: CoinData[];
  monthly_analysis: MonthlyData[];
  tilt_analysis: {
    after_streak_3plus: { count: number; win_rate_pct: number; avg_pnl: number; total_pnl: number } | Record<string, never>;
    normal: { count: number; win_rate_pct: number; avg_pnl: number } | Record<string, never>;
  };
  backtest: Record<string, BacktestStrategy | EquityPoint[]>;
  strategy: {
    name: string;
    description: string;
    rules: StrategyRule[];
    key_findings: string[];
    disclaimer: string;
  };
}

const ACTION_COLORS: Record<string, string> = {
  COUNTER: "#38d39f",      // green = go, counter-trade
  LEAN_COUNTER: "#6366f1", // purple = lean counter
  SKIP: "#94a3b8",         // grey = no action
  COPY: "#f59e0b",         // amber = copy (follow him)
};

const ACTION_LABELS: Record<string, string> = {
  COUNTER: "Counter-Trade",
  LEAN_COUNTER: "Lean Counter",
  SKIP: "Skip",
  COPY: "Copy",
};

function formatUsd(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}

function isBacktestStrategy(value: unknown): value is BacktestStrategy {
  return Boolean(
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    "trades" in value &&
    "wins" in value &&
    "equity_curve" in value
  );
}

function strategyDisplayName(key: string, strategy?: BacktestStrategy): string {
  if (strategy?.description) return strategy.description;
  const labels: Record<string, string> = {
    counter_25m: "Counter >$25M",
    counter_25m_extended: "Counter >$25M (extended sample)",
    counter_5m: "Counter >$5M",
  };
  return labels[key] || key.replaceAll("_", " ");
}

function preferredBacktestStrategy(backtest: StudyData["backtest"]): { key: string; strategy: BacktestStrategy } | null {
  for (const key of ["counter_25m_extended", "counter_25m", "counter_5m"]) {
    const strategy = backtest[key];
    if (isBacktestStrategy(strategy)) return { key, strategy };
  }
  const firstKey = Object.keys(backtest).find((key) => isBacktestStrategy(backtest[key]));
  if (!firstKey) return null;
  return { key: firstKey, strategy: backtest[firstKey] as BacktestStrategy };
}

function formatDateRange(range: StudyData["overview"]["date_range"]): string {
  if (!range?.start || !range?.end) return "study window";
  return `${range.start} to ${range.end}`;
}

export default function CounterTradeStudy({ wallet }: { wallet: string }) {
  const { data, isLoading } = useQuery<StudyData>({
    queryKey: ["hl-study", wallet],
    queryFn: () => apiFetch(`/api/hyperliquid/study/${wallet}`),
    staleTime: 600_000,
  });

  if (isLoading) return <Spinner className="mt-8" />;
  if (!data?.ok) return null;

  const theme = getPlotlyColors();
  const { overview, notional_analysis, duration_analysis, coin_analysis, monthly_analysis, tilt_analysis, backtest, strategy } = data;

  return (
    <div className="space-y-6">
      {/* Live Signal Panel */}
      <LiveSignalPanel wallet={wallet} />

      {/* NFA Banner */}
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-3 text-xs text-[var(--amber)]">
        <strong>Research Study</strong> - {strategy.disclaimer}
      </div>

      {/* Title + Overview */}
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
        <h3 className="text-lg font-bold text-[var(--text)] mb-2">
          Counter-Trade Study: {wallet}
        </h3>
        <p className="text-sm text-[var(--muted)] mb-4">{strategy.description}</p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
            <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Cycles Analyzed</div>
            <div className="text-xl font-bold text-[var(--text)]">{overview.total_cycles}</div>
          </div>
          <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
            <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Trader Win Rate</div>
            <div className="text-xl font-bold text-[var(--text)]">{overview.win_rate_pct}%</div>
          </div>
          <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
            <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Trader Total PnL</div>
            <div className={`text-xl font-bold ${overview.total_pnl >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
              {formatUsd(overview.total_pnl)}
            </div>
          </div>
          <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
            <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Liquidations</div>
            <div className="text-xl font-bold text-[var(--red)]">{overview.liquidation_cycles}</div>
          </div>
        </div>

        <div className="text-[10px] text-[var(--muted)]">
          Data: {overview.date_range.start} to {overview.date_range.end}
        </div>
      </div>

      {/* Strategy Rules */}
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
        <h4 className="text-sm font-bold text-[var(--text)] mb-3">Strategy Rules</h4>
        <div className="space-y-2">
          {strategy.rules.map((rule) => (
            <div
              key={rule.condition}
              className="flex items-center justify-between rounded-lg border border-[var(--line)] bg-[var(--bg)] px-4 py-2"
            >
              <div className="flex items-center gap-3">
                <span
                  className="rounded-full px-2 py-0.5 text-[10px] font-bold text-white"
                  style={{ backgroundColor: ACTION_COLORS[rule.action] || "#94a3b8" }}
                >
                  {ACTION_LABELS[rule.action] || rule.action}
                </span>
                <span className="text-sm text-[var(--text)]">{rule.condition}</span>
              </div>
              <span className="text-xs text-[var(--muted)]">{rule.reason}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Notional Bucket Chart */}
      {notional_analysis.length > 0 && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
          <h4 className="text-sm font-bold text-[var(--text)] mb-3">
            Win Rate by Position Size
          </h4>
          <p className="text-xs text-[var(--muted)] mb-3">
            Win rate by notional bucket. Below 50% = potential counter-trade opportunity.
          </p>
          <div className="h-[300px]">
            <PlotlyWrapper
              data={[
                {
                  type: "bar" as const,
                  x: notional_analysis.map((b) => b.bucket),
                  y: notional_analysis.map((b) => b.win_rate_pct),
                  marker: {
                    color: notional_analysis.map((b) =>
                      b.win_rate_pct < 50 ? "#ff6b6b" : "#38d39f"
                    ),
                  },
                  text: notional_analysis.map((b) => `${b.win_rate_pct}%`),
                  textposition: "outside" as const,
                  textfont: { color: theme.font, size: 11 },
                  hovertemplate:
                    "%{x}<br>Win Rate: %{y:.1f}%<br>Trades: %{customdata[0]}<br>PnL: %{customdata[1]}<extra></extra>",
                  customdata: notional_analysis.map((b) => [
                    b.count,
                    formatUsd(b.total_pnl),
                  ]),
                },
              ]}
              layout={{
                paper_bgcolor: theme.bg,
                plot_bgcolor: theme.bg,
                font: { color: theme.font, size: 11 },
                margin: { t: 20, r: 20, b: 40, l: 50 },
                xaxis: { gridcolor: theme.grid, title: { text: "Position Notional", font: { size: 10 } } },
                yaxis: {
                  gridcolor: theme.grid,
                  title: { text: "Trader Win Rate %", font: { size: 10 } },
                  range: [0, 100],
                },
                shapes: [
                  {
                    type: "line" as const,
                    x0: -0.5,
                    x1: notional_analysis.length - 0.5,
                    y0: 50,
                    y1: 50,
                    line: { color: "#f59e0b", width: 1.5, dash: "dash" as const },
                  },
                ],
                annotations: [
                  {
                    x: notional_analysis.length - 0.7,
                    y: 52,
                    text: "50% (breakeven)",
                    showarrow: false,
                    font: { color: "#f59e0b", size: 9 },
                  },
                ],
                autosize: true,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%", height: 300 }}
            />
          </div>
        </div>
      )}

      {/* Counter Edge Chart */}
      {notional_analysis.length > 0 && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
          <h4 className="text-sm font-bold text-[var(--text)] mb-3">
            Counter-Trade Edge by Bucket
          </h4>
          <p className="text-xs text-[var(--muted)] mb-3">
            Potential edge from taking the opposite side of each bucket (before costs).
          </p>
          <div className="h-[300px]">
            <PlotlyWrapper
              data={[
                {
                  type: "bar" as const,
                  x: notional_analysis.map((b) => b.bucket),
                  y: notional_analysis.map((b) => b.counter_edge_total),
                  marker: {
                    color: notional_analysis.map((b) =>
                      b.counter_edge_total > 0 ? "#38d39f" : "#ff6b6b"
                    ),
                  },
                  text: notional_analysis.map((b) => formatUsd(b.counter_edge_total)),
                  textposition: "outside" as const,
                  textfont: { color: theme.font, size: 11 },
                  hovertemplate: "%{x}<br>Counter Edge: %{text}<extra></extra>",
                },
              ]}
              layout={{
                paper_bgcolor: theme.bg,
                plot_bgcolor: theme.bg,
                font: { color: theme.font, size: 11 },
                margin: { t: 20, r: 20, b: 40, l: 60 },
                xaxis: { gridcolor: theme.grid },
                yaxis: {
                  gridcolor: theme.grid,
                  title: { text: "Counter-Trade Edge ($)", font: { size: 10 } },
                },
                autosize: true,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%", height: 300 }}
            />
          </div>
        </div>
      )}

      {/* Strategy Comparison Table */}
      {backtest ? (() => {
        const stratKeys = Object.keys(backtest).filter((key) => isBacktestStrategy(backtest[key]));
        if (!stratKeys.length) return null;
        const strats = stratKeys.map((key) => ({ key, strategy: backtest[key] as BacktestStrategy }));
        const dateRange = formatDateRange(overview.date_range);
        return (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
            <h4 className="text-sm font-bold text-[var(--text)] mb-3">Strategy Backtest Comparison</h4>
            <p className="text-xs text-[var(--muted)] mb-4">
              All strategies: 1x leverage, 5% position sizing, max $50K per trade, realistic costs.
              $100K starting capital, {dateRange}.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                    <th className="py-2 text-left">Strategy</th>
                    <th className="py-2 text-right">Trades</th>
                    <th className="py-2 text-right">Win Rate</th>
                    <th className="py-2 text-right">Return</th>
                    <th className="py-2 text-right">Max DD</th>
                  </tr>
                </thead>
                <tbody>
                  {strats.map(({ key, strategy }) => (
                    <tr key={key} className="border-b border-[var(--line)]/50">
                      <td className="py-2 text-[var(--text)]">{strategyDisplayName(key, strategy)}</td>
                      <td className="py-2 text-right text-[var(--muted)]">{strategy.trades}</td>
                      <td className={`py-2 text-right font-medium ${strategy.win_rate_pct >= 75 ? "text-[var(--green)]" : strategy.win_rate_pct >= 60 ? "text-[var(--text)]" : "text-[var(--amber)]"}`}>{strategy.win_rate_pct}%</td>
                      <td className={`py-2 text-right font-medium ${strategy.return_pct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
                        {strategy.return_pct >= 0 ? "+" : ""}{strategy.return_pct}%
                      </td>
                      <td className="py-2 text-right text-[var(--muted)]">{strategy.max_drawdown_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })() : null}

      {/* Backtest Equity Curve */}
      {backtest ? (() => {
        const selected = preferredBacktestStrategy(backtest);
        if (!selected) return null;
        const { key, strategy: strat } = selected;
        const traderCurve = Array.isArray(backtest.trader_equity_curve)
          ? backtest.trader_equity_curve
          : undefined;
        if (!strat?.equity_curve?.length) return null;
        const label = strategyDisplayName(key, strat);
        const traderFinalEquity = Array.isArray(traderCurve) && traderCurve.length > 0
          ? traderCurve[traderCurve.length - 1]?.equity ?? null
          : null;
        const traderReturnPct = traderFinalEquity != null
          ? (traderFinalEquity - 100_000) / 1000
          : null;
        const spreadPct = traderReturnPct != null
          ? strat.return_pct - traderReturnPct
          : null;
        const smallSample = strat.trades < 30;
        return (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
            <h4 className="text-sm font-bold text-[var(--text)] mb-1">
              Counter strategy vs target trader
            </h4>
            <p className="text-xs text-[var(--muted)] mb-3">
              {label}: {strat.return_pct >= 0 ? "+" : ""}{strat.return_pct}% over {strat.trades} qualifying trades.
              {traderReturnPct != null && (
                <> Target trader: {traderReturnPct >= 0 ? "+" : ""}{traderReturnPct.toFixed(1)}% over the same window.</>
              )}
              {" "}Both curves normalized to $100K, 1× leverage, 5% sizing per trade.
            </p>

            {/* Stats row — only Spread is highlighted; Counter/Trader Return keep neutral
                chrome with colored value text to reduce visual noise */}
            <div className="grid grid-cols-2 gap-3 mb-4 sm:grid-cols-3 lg:grid-cols-6">
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase tracking-wider text-[var(--muted)]">Counter Return</div>
                <div className={`text-lg font-bold tabular-nums ${strat.return_pct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
                  {strat.return_pct >= 0 ? "+" : ""}{strat.return_pct}%
                </div>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase tracking-wider text-[var(--muted)]">Trader Return</div>
                <div className={`text-lg font-bold tabular-nums ${
                  traderReturnPct == null
                    ? "text-[var(--muted)]"
                    : traderReturnPct >= 0
                      ? "text-[var(--green)]"
                      : "text-[var(--red)]"
                }`}>
                  {traderReturnPct != null
                    ? `${traderReturnPct >= 0 ? "+" : ""}${traderReturnPct.toFixed(1)}%`
                    : "—"}
                </div>
              </div>
              {spreadPct != null && (
                <div className={`rounded-lg border-2 p-2 ${
                  spreadPct >= 0
                    ? "border-[var(--green)]/50 bg-[var(--green)]/10"
                    : "border-[var(--red)]/50 bg-[var(--red)]/10"
                }`}>
                  <div className={`text-[9px] uppercase tracking-wider ${spreadPct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
                    Spread
                  </div>
                  <div className={`text-lg font-bold tabular-nums ${spreadPct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
                    {spreadPct >= 0 ? "+" : ""}{spreadPct.toFixed(1)}pp
                  </div>
                </div>
              )}
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase tracking-wider text-[var(--muted)]">Win Rate</div>
                <div className="text-lg font-bold tabular-nums text-[var(--text)]">{strat.win_rate_pct}%</div>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase tracking-wider text-[var(--muted)]">
                  Trades{smallSample && <span className="text-[var(--amber)]"> · small n</span>}
                </div>
                <div className="text-lg font-bold tabular-nums text-[var(--text)]">{strat.trades}</div>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase tracking-wider text-[var(--muted)]">Max Drawdown</div>
                <div className="text-lg font-bold tabular-nums text-[var(--text)]">{strat.max_drawdown_pct}%</div>
              </div>
            </div>

            <div className="h-[350px]">
              <PlotlyWrapper
                data={[
                  ...(Array.isArray(traderCurve) && traderCurve.length > 0
                    ? [{
                        type: "scatter" as const,
                        mode: "lines" as const,
                        x: traderCurve.map((p) => p.date),
                        y: traderCurve.map((p) => p.equity),
                        name: "Trader (scaled)",
                        line: { color: "#ff6b6b", width: 1.5 },
                        fill: "tozeroy" as const,
                        fillcolor: "rgba(255,107,107,0.05)",
                      }]
                    : []),
                  {
                    type: "scatter" as const,
                    mode: "lines+markers" as const,
                    x: strat.equity_curve.map((p) => p.date),
                    y: strat.equity_curve.map((p) => p.equity),
                    name: label,
                    line: { color: "#38d39f", width: 2.5 },
                    marker: { size: 6 },
                  },
                ]}
                layout={{
                  paper_bgcolor: theme.bg,
                  plot_bgcolor: theme.bg,
                  font: { color: theme.font, size: 10 },
                  margin: { t: 10, r: 20, b: 40, l: 60 },
                  xaxis: { gridcolor: theme.grid },
                  yaxis: {
                    gridcolor: theme.grid,
                    title: { text: "Equity ($)", font: { size: 10 } },
                  },
                  showlegend: true,
                  legend: { x: 0.01, y: 0.99, bgcolor: "rgba(11,15,22,0.8)", font: { size: 10 } },
                  shapes: [
                    {
                      type: "line" as const,
                      x0: strat.equity_curve[0]?.date,
                      x1: strat.equity_curve[strat.equity_curve.length - 1]?.date,
                      y0: 100000,
                      y1: 100000,
                      line: { color: "#94a3b8", width: 1, dash: "dot" as const },
                    },
                  ],
                  autosize: true,
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: "100%", height: 350 }}
              />
            </div>
            <p className="text-[9px] text-[var(--muted)] mt-2">
              Backtest uses 5% of position notional (max $50K), with 10bps entry + 15bps exit slippage + 3.5bps taker fees.
              Past performance does not guarantee future results.
            </p>
          </div>
        );
      })() : null}

      {/* Duration Analysis */}
      {duration_analysis.length > 0 && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
          <h4 className="text-sm font-bold text-[var(--text)] mb-3">
            Performance by Trade Duration
          </h4>
          <div className="h-[280px]">
            <PlotlyWrapper
              data={[
                {
                  type: "bar" as const,
                  x: duration_analysis.map((d) => d.duration),
                  y: duration_analysis.map((d) => d.win_rate_pct),
                  name: "Win Rate %",
                  marker: { color: "#6366f1" },
                  yaxis: "y" as const,
                  text: duration_analysis.map((d) => `${d.win_rate_pct}%`),
                  textposition: "outside" as const,
                  textfont: { color: theme.font, size: 10 },
                },
                {
                  type: "scatter" as const,
                  mode: "lines+markers" as const,
                  x: duration_analysis.map((d) => d.duration),
                  y: duration_analysis.map((d) => d.avg_pnl),
                  name: "Avg PnL ($)",
                  yaxis: "y2" as const,
                  line: { color: "#f59e0b", width: 2 },
                  marker: { size: 6 },
                },
              ]}
              layout={{
                paper_bgcolor: theme.bg,
                plot_bgcolor: theme.bg,
                font: { color: theme.font, size: 10 },
                margin: { t: 20, r: 60, b: 60, l: 50 },
                xaxis: { gridcolor: theme.grid },
                yaxis: {
                  gridcolor: theme.grid,
                  title: { text: "Win Rate %", font: { size: 10 } },
                  range: [0, 100],
                },
                yaxis2: {
                  title: { text: "Avg PnL ($)", font: { size: 10 } },
                  overlaying: "y" as const,
                  side: "right" as const,
                  gridcolor: "transparent",
                },
                showlegend: true,
                legend: { x: 0.01, y: 0.99, bgcolor: "rgba(0,0,0,0.5)", font: { size: 9 } },
                autosize: true,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%", height: 280 }}
            />
          </div>
        </div>
      )}

      {/* Tilt Detection */}
      {tilt_analysis?.after_streak_3plus?.count > 0 && (() => {
        const tiltCounterPnl = -tilt_analysis.after_streak_3plus.total_pnl;
        return (
        <div className="rounded-xl border border-[var(--red)]/30 bg-[var(--red)]/5 p-4 sm:p-6">
          <h4 className="text-sm font-bold text-[var(--red)] mb-3">
            Tilt Detection Signal
          </h4>
          <p className="text-xs text-[var(--muted)] mb-4">
            After 3 or more consecutive losses, this sample stays negative in aggregate but is only
            {` ${tilt_analysis.after_streak_3plus.count} cycles`}. Treat it as research context, not a standalone trigger.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Normal WR</div>
              <div className="text-xl font-bold text-[var(--green)]">
                {tilt_analysis.normal?.win_rate_pct ?? "--"}%
              </div>
              <div className="text-[10px] text-[var(--muted)]">{tilt_analysis.normal?.count ?? 0} trades</div>
            </div>
            <div className="rounded-lg border border-[var(--red)]/40 bg-[var(--red)]/10 p-3">
              <div className="text-[10px] uppercase tracking-wider text-[var(--red)]">After 3+ Losses</div>
              <div className="text-xl font-bold text-[var(--red)]">
                {tilt_analysis.after_streak_3plus.win_rate_pct}%
              </div>
              <div className="text-[10px] text-[var(--muted)]">{tilt_analysis.after_streak_3plus.count} trades</div>
            </div>
            <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Counter PnL Impact</div>
              <div className={`text-xl font-bold ${tiltCounterPnl >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
                {formatUsd(tiltCounterPnl)}
              </div>
              <div className="text-[10px] text-[var(--muted)]">opposite side P&L</div>
            </div>
          </div>
        </div>
        );
      })()}

      {/* Coin Breakdown */}
      {coin_analysis.length > 0 && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
          <h4 className="text-sm font-bold text-[var(--text)] mb-3">Coin Breakdown</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                  <th className="py-2 text-left">Coin</th>
                  <th className="py-2 text-right">Cycles</th>
                  <th className="py-2 text-right">WR</th>
                  <th className="py-2 text-right">Trader Total PnL</th>
                  <th className="py-2 text-right">Liquidations</th>
                  <th className="py-2 text-right">Counter Edge</th>
                </tr>
              </thead>
              <tbody>
                {coin_analysis.map((c) => (
                  <tr key={c.coin} className="border-b border-[var(--line)]/50">
                    <td className="py-2 font-medium text-[var(--text)]">{c.coin}</td>
                    <td className="py-2 text-right text-[var(--muted)]">{c.count}</td>
                    <td className={`py-2 text-right ${c.win_rate_pct < 50 ? "text-[var(--red)]" : "text-[var(--green)]"}`}>
                      {c.win_rate_pct}%
                    </td>
                    <td className={`py-2 text-right ${c.total_pnl >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
                      {formatUsd(c.total_pnl)}
                    </td>
                    <td className="py-2 text-right text-[var(--muted)]">{c.liquidations}</td>
                    <td className={`py-2 text-right font-medium ${-c.total_pnl > 0 ? "text-[var(--green)]" : "text-[var(--red)]"}`}>
                      {formatUsd(-c.total_pnl)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Monthly Performance */}
      {monthly_analysis.length > 0 && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
          <h4 className="text-sm font-bold text-[var(--text)] mb-3">Monthly Breakdown</h4>
          <div className="h-[250px]">
            <PlotlyWrapper
              data={[
                {
                  type: "bar" as const,
                  x: monthly_analysis.map((m) => m.month),
                  y: monthly_analysis.map((m) => m.total_pnl),
                  marker: {
                    color: monthly_analysis.map((m) =>
                      m.total_pnl >= 0 ? "#38d39f" : "#ff6b6b"
                    ),
                  },
                  text: monthly_analysis.map((m) => `${formatUsd(m.total_pnl)} (${m.count} trades)`),
                  textposition: "outside" as const,
                  textfont: { color: theme.font, size: 10 },
                  hovertemplate: "%{x}<br>PnL: %{text}<extra></extra>",
                },
              ]}
              layout={{
                paper_bgcolor: theme.bg,
                plot_bgcolor: theme.bg,
                font: { color: theme.font, size: 11 },
                margin: { t: 20, r: 20, b: 40, l: 60 },
                xaxis: { gridcolor: theme.grid },
                yaxis: {
                  gridcolor: theme.grid,
                  title: { text: "Trader PnL ($)", font: { size: 10 } },
                },
                autosize: true,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%", height: 250 }}
            />
          </div>
        </div>
      )}

      {/* Statistical Review - Critic Panel */}
      {(() => {
        const selected = preferredBacktestStrategy(backtest);
        const recStats = selected?.strategy ?? null;
        const trades = recStats?.trades ?? 0;
        const wins = recStats?.wins ?? 0;
        const wr = recStats?.win_rate_pct ?? 0;
        const cycles = overview.total_cycles;
        const dateStart = overview.date_range?.start ?? "unknown";
        const dateEnd = overview.date_range?.end ?? "unknown";
        const selectedLabel = selected ? strategyDisplayName(selected.key, selected.strategy) : "No strategy";
        // Binomial test: p-value for WR > 50% with n trades and k wins
        // Using normal approximation: z = (k - n*0.5) / sqrt(n*0.25)
        const z = trades > 0 ? (wins - trades * 0.5) / Math.sqrt(trades * 0.25) : 0;
        const isSignificant = wr > 50 && z > 2.33; // p < 0.01
        const isBorderline = wr > 50 && z > 1.64 && !isSignificant; // p < 0.05
        const verdictLabel = isSignificant ? "SIGNIFICANT" : isBorderline ? "BORDERLINE" : trades < 30 ? "INCONCLUSIVE" : wr > 60 ? "PROMISING" : "INCONCLUSIVE";
        const verdictColor = isSignificant ? "var(--green)" : isBorderline ? "var(--amber)" : "var(--amber)";
        // Wilson CI
        const zCI = 1.96;
        const pHat = trades > 0 ? wins / trades : 0;
        const denom = trades > 0 ? 1 + zCI * zCI / trades : 1;
        const center = trades > 0 ? (pHat + zCI * zCI / (2 * trades)) / denom : 0;
        const margin = trades > 0 ? (zCI * Math.sqrt(pHat * (1 - pHat) / trades + zCI * zCI / (4 * trades * trades))) / denom : 0;
        const ciLow = Math.max(0, (center - margin) * 100).toFixed(1);
        const ciHigh = Math.min(100, (center + margin) * 100).toFixed(1);
        const ciIncludesFifty = trades === 0 || parseFloat(ciLow) <= 50;

        return (
          <div className="rounded-xl border border-[var(--red)]/20 bg-[var(--panel)] p-4 sm:p-6">
            <h4 className="text-sm font-bold text-[var(--text)] mb-1">Statistical Review</h4>
            <p className="text-[10px] text-[var(--muted)] mb-4">
              Independent critic panel assessment. We present limitations honestly.
            </p>

            <div className="space-y-3 text-xs">
              {/* Verdict */}
              <div className="flex items-center gap-2 rounded-lg border px-3 py-2" style={{ borderColor: `${verdictColor}30`, backgroundColor: `${verdictColor}0d` }}>
                <span className="rounded px-2 py-0.5 text-[10px] font-bold text-black" style={{ backgroundColor: verdictColor }}>{verdictLabel}</span>
                <span className="text-[var(--muted)]">
                  {isSignificant
                    ? `${selectedLabel}: z = ${z.toFixed(2)}, p < 0.01. Statistically significant edge with ${trades} trades over ${cycles} analyzed cycles.`
                    : isBorderline
                      ? `${selectedLabel}: z = ${z.toFixed(2)}. Near-significant at alpha = 0.05. ${trades} trades over ${cycles} analyzed cycles.`
                      : `${selectedLabel}: z = ${z.toFixed(2)}. ${wr}% WR across ${trades} trades over ${cycles} analyzed cycles (${dateStart} to ${dateEnd}).`
                  }
                </span>
              </div>

              {/* Stats table */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                      <th className="py-1.5 text-left">Test</th>
                      <th className="py-1.5 text-right">Result</th>
                      <th className="py-1.5 text-right">z-score</th>
                      <th className="py-1.5 text-right">Significant?</th>
                    </tr>
                  </thead>
                  <tbody className="text-[var(--muted)]">
                    <tr className="border-b border-[var(--line)]/50">
                      <td className="py-1.5">Binomial (WR &gt; 50%)</td>
                      <td className="py-1.5 text-right">{wins}/{trades} wins ({wr}%)</td>
                      <td className="py-1.5 text-right">{z.toFixed(2)}</td>
                      <td className={`py-1.5 text-right ${isSignificant ? "text-[var(--green)]" : isBorderline ? "text-[var(--amber)]" : "text-[var(--muted)]"}`}>
                        {isSignificant ? "Yes (p < 0.01)" : isBorderline ? "Borderline" : "No"}
                      </td>
                    </tr>
                    <tr className="border-b border-[var(--line)]/50">
                      <td className="py-1.5">Study sample</td>
                      <td className="py-1.5 text-right">{cycles} cycles</td>
                      <td className="py-1.5 text-right">-</td>
                      <td className="py-1.5 text-right text-[var(--green)]">Full dataset</td>
                    </tr>
                    <tr className="border-b border-[var(--line)]/50">
                      <td className="py-1.5">95% CI (Wilson)</td>
                      <td className="py-1.5 text-right">{trades > 0 ? `[${ciLow}%, ${ciHigh}%]` : "n/a"}</td>
                      <td className="py-1.5 text-right">-</td>
                      <td className={`py-1.5 text-right ${ciIncludesFifty ? "text-[var(--amber)]" : "text-[var(--green)]"}`}>
                        {trades > 0 ? (ciIncludesFifty ? "Includes 50%" : "Above 50%") : "No sample"}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-1.5">Sample adequacy</td>
                      <td className="py-1.5 text-right">Need ~90 for 80% power</td>
                      <td className="py-1.5 text-right">-</td>
                      <td className={`py-1.5 text-right ${trades >= 90 ? "text-[var(--green)]" : trades >= 60 ? "text-[var(--amber)]" : "text-[var(--muted)]"}`}>
                        Have {trades}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Risk factors */}
              <div>
                <div className="text-[10px] font-semibold text-[var(--muted)] uppercase tracking-wider mb-2">Known Risks</div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {[
                    { label: "Sample size", severity: trades >= 90 ? "medium" : "critical", detail: `${trades} qualifying trades over ${cycles} analyzed cycles. Need roughly 78-90 for 80% power.` },
                    { label: "Selection bias", severity: "critical", detail: "Target selected after observing PnL history. Forward performance unproven." },
                    { label: "Regime dependency", severity: "critical", detail: "One historical window only; not validated across multiple crypto regimes." },
                    { label: "Data coverage", severity: "high", detail: `Study sample covers ${dateStart} to ${dateEnd}; verify fill collection is complete before using live.` },
                    { label: "Execution latency", severity: "medium", detail: "Wallet entries and exits can scale across many fills. Entry timing can materially change results." },
                    { label: "Single point of failure", severity: "medium", detail: "Strategy dies if he stops trading or changes behavior." },
                  ].map((risk) => (
                    <div
                      key={risk.label}
                      className="flex items-start gap-2 rounded border border-[var(--line)] bg-[var(--bg)] px-3 py-2"
                    >
                      <span
                        className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase ${
                          risk.severity === "critical"
                            ? "bg-[var(--red)]/20 text-[var(--red)]"
                            : risk.severity === "high"
                              ? "bg-[var(--amber)]/20 text-[var(--amber)]"
                              : "bg-[var(--line)] text-[var(--muted)]"
                        }`}
                      >
                        {risk.severity}
                      </span>
                      <div>
                        <div className="font-medium text-[var(--text)]">{risk.label}</div>
                        <div className="text-[var(--muted)]">{risk.detail}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* What would change our mind */}
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] px-3 py-2">
                <div className="text-[10px] font-semibold text-[var(--muted)] uppercase tracking-wider mb-1">
                  What would make this conclusive
                </div>
                <ul className="space-y-0.5 text-[var(--muted)]">
                  {trades < 90 && <li>- {90 - trades}+ more qualifying signals for {selectedLabel} (currently {trades})</li>}
                  <li>- Positive results through a bull market regime change</li>
                  <li>- Out-of-sample validation on other whales (not pre-selected by PnL)</li>
                  <li>- Live paper trading for 6+ months before real capital</li>
                </ul>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Key Findings */}
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
        <h4 className="text-sm font-bold text-[var(--text)] mb-3">Key Findings</h4>
        <ul className="space-y-1.5">
          {strategy.key_findings.map((finding, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-[var(--muted)]">
              <span className="mt-0.5 text-[var(--accent)]">-</span>
              {finding}
            </li>
          ))}
        </ul>
      </div>

      {/* Agent Analysis */}
      {(strategy as Record<string, unknown>).agent_analysis ? (() => {
        const agents = (strategy as Record<string, unknown>).agent_analysis as Record<string, { title: string; findings: string[] }>;
        return (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
            <h4 className="text-sm font-bold text-[var(--text)] mb-4">Expert Panel Analysis</h4>
            <div className="grid gap-4 sm:grid-cols-2">
              {Object.values(agents).map((agent) => (
                <div key={agent.title} className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
                  <div className="text-xs font-semibold text-[var(--accent)] mb-2">{agent.title}</div>
                  <ul className="space-y-1">
                    {agent.findings.map((f: string, i: number) => (
                      <li key={i} className="text-[10px] text-[var(--muted)] flex items-start gap-1">
                        <span className="mt-0.5 shrink-0">-</span>
                        <span>{f}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        );
      })() : null}

      {/* Bottom NFA */}
      <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] px-4 py-3 text-[10px] text-[var(--muted)]">
        <strong>Disclaimer:</strong> {strategy.disclaimer}
      </div>
    </div>
  );
}

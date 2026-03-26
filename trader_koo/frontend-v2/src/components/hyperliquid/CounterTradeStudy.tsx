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
}

interface LiveData {
  ok: boolean;
  wallet: { account_value: number; margin_ratio: number };
  positions: LivePosition[];
}

const REF_NOTIONAL = 10_000_000;
const REF_ETH = 4400;

function LiveSignalPanel({ wallet }: { wallet: string }) {
  const { data } = useQuery<LiveData>({
    queryKey: ["hl-live", wallet],
    queryFn: () => apiFetch(`/api/hyperliquid/live/${wallet}`),
    refetchInterval: 60_000,
  });

  if (!data?.ok || !data.positions?.length) return null;

  const ethPrice = data.positions.find((p) => p.coin === "ETH")?.mark_price || 2100;
  const threshold = REF_NOTIONAL * (ethPrice / REF_ETH);
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
          <div className="text-[var(--muted)]">Threshold</div>
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
                {(p.notional_usd / threshold).toFixed(1)}x above threshold - counter-trade opportunity (backtest: 63.6% WR)
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
  backtest: Record<string, {
    trades: number;
    wins: number;
    win_rate_pct: number;
    total_pnl: number;
    return_pct: number;
    max_drawdown_pct: number;
    final_equity: number;
    equity_curve: { date: string; equity: number; coin?: string; our_pnl?: number }[];
  } | { date: string; equity: number }[]>;
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
        const stratKeys = Object.keys(backtest).filter(k => k !== "trader_equity_curve" && typeof backtest[k] === "object" && "trades" in (backtest[k] as Record<string, unknown>));
        if (!stratKeys.length) return null;
        type StratData = { trades: number; wins: number; win_rate_pct: number; total_pnl: number; return_pct: number; max_drawdown_pct: number; final_equity: number; description?: string; equity_curve: { date: string; equity: number }[] };
        const strats = stratKeys.map(k => ({ key: k, ...(backtest[k] as unknown as StratData) }));
        return (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
            <h4 className="text-sm font-bold text-[var(--text)] mb-3">Strategy Backtest Comparison</h4>
            <p className="text-xs text-[var(--muted)] mb-4">
              All strategies: 1x leverage, 5% position sizing, max $50K per trade, realistic costs.
              $100K starting capital, Jul 2025 - Nov 2025.
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
                  {strats.map((s) => (
                    <tr key={s.key} className="border-b border-[var(--line)]/50">
                      <td className="py-2 text-[var(--text)]">{(s as StratData & { key: string }).description || s.key}</td>
                      <td className="py-2 text-right text-[var(--muted)]">{s.trades}</td>
                      <td className={`py-2 text-right font-medium ${s.win_rate_pct >= 75 ? "text-[var(--green)]" : s.win_rate_pct >= 60 ? "text-[var(--text)]" : "text-[var(--amber)]"}`}>{s.win_rate_pct}%</td>
                      <td className="py-2 text-right font-medium text-[var(--green)]">+{s.return_pct}%</td>
                      <td className="py-2 text-right text-[var(--muted)]">{s.max_drawdown_pct}%</td>
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
        const strat = (backtest.counter_25m_held_7d || backtest.counter_25m) as { trades: number; wins: number; win_rate_pct: number; total_pnl: number; return_pct: number; max_drawdown_pct: number; final_equity: number; equity_curve: { date: string; equity: number }[] } | undefined;
        const traderCurve = backtest.trader_equity_curve as { date: string; equity: number }[] | undefined;
        if (!strat?.equity_curve?.length) return null;
        return (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 sm:p-6">
            <h4 className="text-sm font-bold text-[var(--text)] mb-1">
              Equity Curve: Best Strategy vs Trader
            </h4>
            <p className="text-xs text-[var(--muted)] mb-3">
              {"Counter >$25M held >7 days (88.9% WR). Both normalized to $100K, 1x leverage, 5% sizing."}
            </p>

            {/* Stats row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
              <div className="rounded-lg border border-[var(--green)]/30 bg-[var(--green)]/5 p-2">
                <div className="text-[9px] uppercase text-[var(--green)]">Counter Return</div>
                <div className="text-lg font-bold text-[var(--green)]">+{strat.return_pct}%</div>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase text-[var(--muted)]">Win Rate</div>
                <div className="text-lg font-bold text-[var(--text)]">{strat.win_rate_pct}%</div>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase text-[var(--muted)]">Trades</div>
                <div className="text-lg font-bold text-[var(--text)]">{strat.trades}</div>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-2">
                <div className="text-[9px] uppercase text-[var(--muted)]">Max Drawdown</div>
                <div className="text-lg font-bold text-[var(--text)]">{strat.max_drawdown_pct}%</div>
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
                    name: "Counter >$25M",
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
      {tilt_analysis?.after_streak_3plus?.count > 0 && (
        <div className="rounded-xl border border-[var(--red)]/30 bg-[var(--red)]/5 p-4 sm:p-6">
          <h4 className="text-sm font-bold text-[var(--red)] mb-3">
            Tilt Detection Signal
          </h4>
          <p className="text-xs text-[var(--muted)] mb-4">
            After 3 or more consecutive losses, his behavior changes dramatically. Classic martingale
            pattern: position sizing increases after consecutive losses, leading to worse outcomes. This is the highest-confidence
            counter-trade window.
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
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Tilt PnL Impact</div>
              <div className="text-xl font-bold text-[var(--red)]">
                {formatUsd(tilt_analysis.after_streak_3plus.total_pnl)}
              </div>
              <div className="text-[10px] text-[var(--muted)]">opposite side P&L</div>
            </div>
          </div>
        </div>
      )}

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
      <div className="rounded-xl border border-[var(--red)]/20 bg-[var(--panel)] p-4 sm:p-6">
        <h4 className="text-sm font-bold text-[var(--text)] mb-1">Statistical Review</h4>
        <p className="text-[10px] text-[var(--muted)] mb-4">
          Independent critic panel assessment. We present limitations honestly.
        </p>

        <div className="space-y-3 text-xs">
          {/* Verdict */}
          <div className="flex items-center gap-2 rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-3 py-2">
            <span className="rounded bg-[var(--amber)] px-2 py-0.5 text-[10px] font-bold text-black">INCONCLUSIVE</span>
            <span className="text-[var(--muted)]">
              p = 0.055 (barely misses significance at alpha = 0.05). Promising but not proven.
            </span>
          </div>

          {/* Stats table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                  <th className="py-1.5 text-left">Test</th>
                  <th className="py-1.5 text-right">Result</th>
                  <th className="py-1.5 text-right">p-value</th>
                  <th className="py-1.5 text-right">Significant?</th>
                </tr>
              </thead>
              <tbody className="text-[var(--muted)]">
                <tr className="border-b border-[var(--line)]/50">
                  <td className="py-1.5">Binomial (WR &gt; 50%)</td>
                  <td className="py-1.5 text-right">21/32 wins</td>
                  <td className="py-1.5 text-right">0.055</td>
                  <td className="py-1.5 text-right text-[var(--amber)]">Borderline</td>
                </tr>
                <tr className="border-b border-[var(--line)]/50">
                  <td className="py-1.5">Bootstrap (10K resamples)</td>
                  <td className="py-1.5 text-right">96.6% positive</td>
                  <td className="py-1.5 text-right">-</td>
                  <td className="py-1.5 text-right text-[var(--green)]">Encouraging</td>
                </tr>
                <tr className="border-b border-[var(--line)]/50">
                  <td className="py-1.5">95% CI (Wilson)</td>
                  <td className="py-1.5 text-right">[48.3%, 79.6%]</td>
                  <td className="py-1.5 text-right">-</td>
                  <td className="py-1.5 text-right text-[var(--amber)]">Includes 50%</td>
                </tr>
                <tr>
                  <td className="py-1.5">Power analysis</td>
                  <td className="py-1.5 text-right">Need ~90 trades</td>
                  <td className="py-1.5 text-right">-</td>
                  <td className="py-1.5 text-right text-[var(--muted)]">Have 32</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Risk factors */}
          <div>
            <div className="text-[10px] font-semibold text-[var(--muted)] uppercase tracking-wider mb-2">Known Risks</div>
            <div className="grid gap-2 sm:grid-cols-2">
              {[
                { label: "Sample size", severity: "critical", detail: "32 trades over 4 months. Need 78-90 for 80% power." },
                { label: "Selection bias", severity: "critical", detail: "Target selected after observing PnL history. Forward performance unproven." },
                { label: "Regime dependency", severity: "critical", detail: "Bear market only (BTC $95K->$70K). Untested in bull." },
                { label: "Data coverage", severity: "high", detail: "API only goes back to Dec 2025. Missing his $50M peak era." },
                { label: "Execution latency", severity: "medium", detail: "He scales in over 2K+ fills per cycle. Entry timing unclear." },
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
              <li>- 78+ more trade cycles at &gt;$1M notional (currently 32)</li>
              <li>- Positive results through a bull market regime change</li>
              <li>- Out-of-sample validation on other whales (not pre-selected by PnL)</li>
              <li>- Live paper trading for 6+ months before real capital</li>
            </ul>
          </div>
        </div>
      </div>

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

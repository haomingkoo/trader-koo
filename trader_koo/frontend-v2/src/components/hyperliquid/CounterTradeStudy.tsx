import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../../api/client";
import PlotlyWrapper from "../PlotlyWrapper";
import Spinner from "../ui/Spinner";
import { getPlotlyColors } from "../../lib/plotlyTheme";

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
  strategy: {
    name: string;
    description: string;
    rules: StrategyRule[];
    key_findings: string[];
    disclaimer: string;
  };
}

const ACTION_COLORS: Record<string, string> = {
  COUNTER: "#ff6b6b",
  LEAN_COUNTER: "#f59e0b",
  SKIP: "#94a3b8",
  COPY: "#38d39f",
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
  const { overview, notional_analysis, duration_analysis, coin_analysis, monthly_analysis, strategy } = data;

  return (
    <div className="space-y-6">
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
            <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">His Win Rate</div>
            <div className="text-xl font-bold text-[var(--text)]">{overview.win_rate_pct}%</div>
          </div>
          <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] p-3">
            <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">His Total PnL</div>
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
            His win rate by notional bucket. Below 50% = counter-trade edge.
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
                    "%{x}<br>Win Rate: %{y:.1f}%<br>Trades: %{customdata[0]}<br>His PnL: %{customdata[1]}<extra></extra>",
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
                  title: { text: "His Win Rate %", font: { size: 10 } },
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
            Potential profit from counter-trading each bucket (his loss = our gain, before costs).
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
                  <th className="py-2 text-right">His WR</th>
                  <th className="py-2 text-right">His Total PnL</th>
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
                  title: { text: "His PnL ($)", font: { size: 10 } },
                },
                autosize: true,
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: "100%", height: 250 }}
            />
          </div>
        </div>
      )}

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

      {/* Bottom NFA */}
      <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)] px-4 py-3 text-[10px] text-[var(--muted)]">
        <strong>Disclaimer:</strong> {strategy.disclaimer}
      </div>
    </div>
  );
}

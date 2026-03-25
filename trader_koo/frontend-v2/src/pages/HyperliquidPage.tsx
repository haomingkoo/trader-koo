import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import Spinner from "../components/ui/Spinner";
import Badge from "../components/ui/Badge";
import CounterTradeStudy from "../components/hyperliquid/CounterTradeStudy";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface Position {
  coin: string;
  side: string;
  size: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  leverage: string;
  notional_usd: number;
  liquidation_price: number | null;
}

interface CounterSignal {
  wallet: string;
  coin: string;
  counter_side: string;
  their_side: string;
  their_size: number;
  their_leverage: number;
  their_notional_usd: number;
  confidence: number;
  reasoning: string;
  timestamp: string;
}

interface WalletLive {
  ok: boolean;
  wallet: {
    label: string;
    address: string;
    account_value: number;
    total_margin_used: number;
    margin_ratio: number;
    timestamp: string;
  };
  positions: Position[];
  counter_signals: CounterSignal[];
}

interface WalletHistory {
  ok: boolean;
  fill_count: number;
  lookback_days: number;
  stats: {
    total_pnl: number;
    total_fees: number;
    net_pnl: number;
    wins: number;
    losses: number;
    win_rate_pct: number;
    liquidations: number;
  };
  by_coin: Record<string, { pnl: number; fills: number; win_rate_pct: number }>;
}

/* ------------------------------------------------------------------ */
/* Hooks                                                               */
/* ------------------------------------------------------------------ */

function useWalletLive(label: string) {
  return useQuery({
    queryKey: ["hl-live", label],
    queryFn: () => apiFetch<WalletLive>(`/api/hyperliquid/live/${label}`),
    refetchInterval: 60_000,
  });
}

function useWalletHistory(label: string, days: number) {
  return useQuery({
    queryKey: ["hl-history", label, days],
    queryFn: () => apiFetch<WalletHistory>(`/api/hyperliquid/history/${label}?days=${days}`),
    refetchInterval: 300_000,
  });
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function fmt(n: number, decimals = 0): string {
  return n.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function fmtUsd(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

function pnlColor(n: number): string {
  return n > 0 ? "text-[var(--green)]" : n < 0 ? "text-[var(--red)]" : "text-[var(--muted)]";
}

/* ------------------------------------------------------------------ */
/* Components                                                          */
/* ------------------------------------------------------------------ */

function AccountHero({ wallet }: { wallet: WalletLive["wallet"] }) {
  const marginPct = (wallet.margin_ratio * 100).toFixed(1);
  const isDanger = wallet.margin_ratio > 0.9;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-[var(--text)]">{wallet.label}</h2>
          <p className="mt-0.5 font-mono text-[10px] text-[var(--muted)]">
            {wallet.address.slice(0, 10)}...{wallet.address.slice(-6)}
          </p>
        </div>
        <Badge variant={isDanger ? "red" : "default"}>
          {marginPct}% margin used
        </Badge>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Account Value</p>
          <p className="text-xl font-bold tabular-nums text-[var(--text)]">{fmtUsd(wallet.account_value)}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Margin Used</p>
          <p className="text-xl font-bold tabular-nums text-[var(--text)]">{fmtUsd(wallet.total_margin_used)}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Last Updated</p>
          <p className="text-sm tabular-nums text-[var(--muted)]">
            {new Date(wallet.timestamp).toLocaleTimeString()}
          </p>
        </div>
      </div>
    </div>
  );
}

function PositionsTable({ positions }: { positions: Position[] }) {
  if (!positions.length) {
    return <p className="text-sm text-[var(--muted)]">No open positions</p>;
  }

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      <h3 className="mb-3 text-sm font-semibold text-[var(--text)]">Open Positions</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--line)] text-[var(--muted)]">
              <th className="pb-2 text-left font-medium">Coin</th>
              <th className="pb-2 text-left font-medium">Side</th>
              <th className="pb-2 text-right font-medium">Size</th>
              <th className="pb-2 text-right font-medium">Entry</th>
              <th className="pb-2 text-right font-medium">Mark</th>
              <th className="pb-2 text-right font-medium">uPnL</th>
              <th className="pb-2 text-right font-medium">Leverage</th>
              <th className="pb-2 text-right font-medium">Notional</th>
              <th className="pb-2 text-right font-medium">Liq Price</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => {
              const liqDist = p.liquidation_price && p.mark_price > 0
                ? Math.abs(p.mark_price - p.liquidation_price) / p.mark_price * 100
                : null;

              return (
                <tr key={`${p.coin}-${p.side}`} className="border-b border-[var(--line)]/30">
                  <td className="py-2 font-bold text-[var(--text)]">{p.coin}</td>
                  <td className="py-2">
                    <Badge variant={p.side === "long" ? "green" : "red"}>
                      {p.side.toUpperCase()}
                    </Badge>
                  </td>
                  <td className="py-2 text-right tabular-nums text-[var(--text)]">{fmt(p.size, 2)}</td>
                  <td className="py-2 text-right tabular-nums text-[var(--muted)]">${fmt(p.entry_price, 2)}</td>
                  <td className="py-2 text-right tabular-nums text-[var(--text)]">${fmt(p.mark_price, 2)}</td>
                  <td className={`py-2 text-right tabular-nums font-bold ${pnlColor(p.unrealized_pnl)}`}>
                    {p.unrealized_pnl >= 0 ? "+" : ""}{fmtUsd(p.unrealized_pnl)}
                  </td>
                  <td className="py-2 text-right tabular-nums text-[var(--muted)]">{p.leverage}</td>
                  <td className="py-2 text-right tabular-nums text-[var(--muted)]">{fmtUsd(p.notional_usd)}</td>
                  <td className="py-2 text-right tabular-nums">
                    {p.liquidation_price ? (
                      <span className={liqDist != null && liqDist < 10 ? "text-[var(--red)] font-bold" : "text-[var(--muted)]"}>
                        ${fmt(p.liquidation_price, 2)}
                        {liqDist != null && <span className="ml-1 text-[9px]">({liqDist.toFixed(1)}%)</span>}
                      </span>
                    ) : (
                      <span className="text-[var(--muted)]">-</span>
                    )}
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

function CounterSignals({ signals }: { signals: CounterSignal[] }) {
  if (!signals.length) return null;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      <h3 className="mb-3 text-sm font-semibold text-[var(--text)]">Counter-Trade Signals</h3>
      <div className="space-y-2">
        {signals.map((s, i) => (
          <div key={i} className="rounded-lg border border-[var(--line)]/30 bg-[var(--bg)]/50 p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Badge variant={s.counter_side === "long" ? "green" : "red"}>
                  {s.counter_side.toUpperCase()} {s.coin}
                </Badge>
                <span className="text-[10px] text-[var(--muted)]">
                  vs their {s.their_side} ({s.their_leverage}x)
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-[var(--muted)]">confidence</span>
                <span className={`text-sm font-bold tabular-nums ${
                  s.confidence >= 70 ? "text-[var(--green)]" : s.confidence >= 50 ? "text-[var(--amber)]" : "text-[var(--muted)]"
                }`}>
                  {s.confidence.toFixed(0)}
                </span>
              </div>
            </div>
            <p className="mt-1.5 text-[10px] text-[var(--muted)]">{s.reasoning}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function HistoryStats({ history }: { history: WalletHistory }) {
  const { stats, by_coin } = history;
  const coins = Object.entries(by_coin).sort((a, b) => a[1].pnl - b[1].pnl);

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      <h3 className="mb-3 text-sm font-semibold text-[var(--text)]">
        {history.lookback_days}-Day Performance ({history.fill_count} fills)
      </h3>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Net PnL</p>
          <p className={`text-lg font-bold tabular-nums ${pnlColor(stats.net_pnl)}`}>
            {stats.net_pnl >= 0 ? "+" : ""}{fmtUsd(stats.net_pnl)}
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Win Rate</p>
          <p className="text-lg font-bold tabular-nums text-[var(--text)]">{stats.win_rate_pct}%</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Fees Paid</p>
          <p className="text-lg font-bold tabular-nums text-[var(--red)]">-{fmtUsd(stats.total_fees)}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Liquidations</p>
          <p className={`text-lg font-bold tabular-nums ${stats.liquidations > 0 ? "text-[var(--red)]" : "text-[var(--muted)]"}`}>
            {stats.liquidations}
          </p>
        </div>
      </div>

      {coins.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-[10px] uppercase tracking-wider text-[var(--muted)]">By Coin</p>
          <div className="space-y-1">
            {coins.map(([coin, data]) => (
              <div key={coin} className="flex items-center justify-between rounded bg-[var(--bg)]/50 px-2.5 py-1.5">
                <span className="text-xs font-bold text-[var(--text)]">{coin}</span>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] text-[var(--muted)]">{data.fills} fills</span>
                  <span className="text-[10px] text-[var(--muted)]">{data.win_rate_pct}% win</span>
                  <span className={`text-xs font-bold tabular-nums ${pnlColor(data.pnl)}`}>
                    {data.pnl >= 0 ? "+" : ""}{fmtUsd(data.pnl)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function HyperliquidPage() {
  useEffect(() => {
    document.title = "Hyperliquid Tracker \u2014 Trader Koo";
  }, []);

  const [wallet] = useState("machibro");
  const [historyDays, setHistoryDays] = useState(7);

  const { data: live, isLoading: liveLoading } = useWalletLive(wallet);
  const { data: history, isLoading: historyLoading } = useWalletHistory(wallet, historyDays);

  if (liveLoading) return <Spinner />;

  if (!live?.ok) {
    return (
      <div className="p-6 text-center text-[var(--muted)]">
        Failed to load wallet data. Check if the API is running.
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-4 p-4 sm:p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-[var(--text)]">Hyperliquid Whale Tracker</h1>
        <p className="text-[10px] text-[var(--muted)]">
          Counter-trade signals based on tracked wallet positions
        </p>
      </div>

      <AccountHero wallet={live.wallet} />
      <PositionsTable positions={live.positions} />
      <CounterSignals signals={live.counter_signals} />

      {/* History section */}
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold text-[var(--text)]">Trade History</h3>
        <div className="flex gap-1">
          {[7, 14, 30].map((d) => (
            <button
              key={d}
              onClick={() => setHistoryDays(d)}
              className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
                historyDays === d
                  ? "bg-[var(--accent)] text-white"
                  : "bg-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>

      {historyLoading ? (
        <Spinner />
      ) : history?.ok && history.stats ? (
        <HistoryStats history={history} />
      ) : (
        <p className="text-sm text-[var(--muted)]">No history available</p>
      )}

      {/* Counter-Trade Research Study */}
      <div className="border-t border-[var(--line)] pt-6 mt-6">
        <h2 className="text-lg font-bold text-[var(--text)] mb-4">Counter-Trade Research</h2>
        <CounterTradeStudy wallet="machibro" />
      </div>

      <p className="text-[9px] text-[var(--muted)]">
        Data from Hyperliquid API. Positions refresh every 60s. Counter-trade signals are informational only. Not financial advice.
      </p>
    </div>
  );
}

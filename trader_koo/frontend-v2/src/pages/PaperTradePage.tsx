import { useState, useEffect } from "react";
import { usePaperTradeSummary, usePaperTrades } from "../api/hooks";
import type { PaperTradeSummaryOverall } from "../api/types";
import Spinner from "../components/ui/Spinner";
import Badge from "../components/ui/Badge";
import {
  PaperTradePortfolioHero,
  PaperTradeOpenPositions,
  PaperTradeEquityCurve,
  PaperTradePerformanceAttribution,
  PaperTradeMLCalibration,
  PaperTradeFilters,
  PaperTradeLogTable,
} from "../components/paper/PaperTradeSections";

export default function PaperTradePage() {
  useEffect(() => {
    document.title = "Paper Trades \u2014 Trader Koo";
  }, []);

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

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        Paper trades are simulated and do not represent real money. Simulated results may not reflect actual trading conditions.
      </div>

      <PaperTradePortfolioHero overall={overall} />

      {/* How it works — collapsible explainer */}
      <details className="group rounded-xl border border-[var(--line)] bg-[var(--panel)]">
        <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-[var(--text)] select-none">
          How Paper Trading Works
          <span className="ml-2 text-xs text-[var(--muted)] group-open:hidden">
            (click to expand)
          </span>
        </summary>
        <div className="space-y-3 border-t border-[var(--line)] px-4 py-4 text-xs text-[var(--muted)]">
          <div className="flex flex-wrap gap-2">
            <Badge variant="blue">Nightly Pipeline</Badge>
            <Badge variant="green">ML Scored</Badge>
            <Badge variant="amber">Critic Reviewed</Badge>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <div className="font-semibold text-[var(--text)]">1. Signal Generation</div>
              <p className="mt-1">Every market night, the pipeline scans 500+ S&P tickers for technical setups using YOLOv8 pattern detection, support/resistance levels, and multi-angle debate scoring.</p>
            </div>
            <div>
              <div className="font-semibold text-[var(--text)]">2. ML Filtering</div>
              <p className="mt-1">A LightGBM model (54 features, walk-forward trained) scores each setup. Trades with predicted win probability below 55% are rejected. The model uses momentum, volatility, macro, and sentiment features.</p>
            </div>
            <div>
              <div className="font-semibold text-[var(--text)]">3. Critic Review</div>
              <p className="mt-1">A 7-check critic bot plays devil's advocate: conviction grade, debate strength, risk/reward ratio, regime alignment, portfolio concentration, VIX environment, and caution flags. All checks must pass.</p>
            </div>
            <div>
              <div className="font-semibold text-[var(--text)]">4. Position Management</div>
              <p className="mt-1">Approved trades get ATR-based stops, volatility-scaled targets, and tier-adjusted sizing. Stops trail automatically: breakeven at +1R, trail at +1.5R. Max drawdown circuit breaker halts entries at -15%.</p>
            </div>
          </div>
          <p className="text-[10px] italic">
            Starting capital: $1M simulated. Each trade is sized 5-12% based on tier (A/B/C). This is a proving ground before any live execution.
          </p>
        </div>
      </details>

      <PaperTradeOpenPositions trades={trades} />

      <PaperTradeEquityCurve equityCurve={equityCurve} />

      {summary && (
        <PaperTradePerformanceAttribution summary={summary} />
      )}

      <PaperTradeMLCalibration trades={trades} />

      <PaperTradeFilters
        statusFilter={statusFilter}
        directionFilter={dirFilter}
        tradeCount={trades.length}
        trades={trades}
        onStatusChange={setStatusFilter}
        onDirectionChange={setDirFilter}
      />

      <PaperTradeLogTable trades={trades} />
    </div>
  );
}

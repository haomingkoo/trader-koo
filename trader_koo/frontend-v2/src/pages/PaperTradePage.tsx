import { useState } from "react";
import { usePaperTradeSummary, usePaperTrades } from "../api/hooks";
import type { PaperTradeSummaryOverall } from "../api/types";
import Spinner from "../components/ui/Spinner";
import {
  PaperTradePortfolioHero,
  PaperTradeOpenPositions,
  PaperTradeEquityCurve,
  PaperTradeFilters,
  PaperTradeLogTable,
} from "../components/paper/PaperTradeSections";

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

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        Paper trades are simulated and do not represent real money. Simulated results may not reflect actual trading conditions.
      </div>

      <PaperTradePortfolioHero overall={overall} />

      <PaperTradeOpenPositions trades={trades} />

      <PaperTradeEquityCurve equityCurve={equityCurve} />

      <PaperTradeFilters
        statusFilter={statusFilter}
        directionFilter={dirFilter}
        tradeCount={trades.length}
        onStatusChange={setStatusFilter}
        onDirectionChange={setDirFilter}
      />

      <PaperTradeLogTable trades={trades} />
    </div>
  );
}

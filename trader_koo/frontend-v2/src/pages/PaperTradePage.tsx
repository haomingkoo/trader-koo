import { useState } from "react";
import { usePaperTradeSummary, usePaperTrades } from "../api/hooks";
import type { PaperTradeSummaryOverall } from "../api/types";
import Spinner from "../components/ui/Spinner";
import {
  PaperTradeHero,
  PaperTradeBotOverview,
  PaperTradeBreakdownPanels,
  PaperTradeEdgePanels,
  PaperTradeEquityCurve,
  PaperTradeFeedbackPanel,
  PaperTradeFilters,
  PaperTradeLogTable,
  PaperTradeOpenPlans,
  PaperTradeSummaryGrid,
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
  const latestEquity =
    equityCurve.length > 0
      ? equityCurve[equityCurve.length - 1].equity_index
      : null;
  const maxDrawdown = (() => {
    if (equityCurve.length < 2) return null;
    let peak = equityCurve[0].equity_index;
    let maxDd = 0;
    for (const point of equityCurve) {
      if (point.equity_index > peak) peak = point.equity_index;
      const dd = ((peak - point.equity_index) / peak) * 100;
      if (dd > maxDd) maxDd = dd;
    }
    return maxDd;
  })();

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        Paper trades are simulated and do not represent real money. Simulated results may not reflect actual trading conditions.
      </div>

      <PaperTradeHero
        overall={overall}
        latestEquity={latestEquity}
        maxDrawdown={maxDrawdown}
        policy={summary?.policy}
      />

      <PaperTradeSummaryGrid
        overall={overall}
        maxDrawdown={maxDrawdown}
        latestEquity={latestEquity}
      />

      <PaperTradeBotOverview overall={overall} policy={summary?.policy} />

      <PaperTradeEquityCurve equityCurve={equityCurve} />

      <PaperTradeBreakdownPanels summary={summary} />

      <PaperTradeOpenPlans trades={trades} />

      <PaperTradeEdgePanels
        familyEdges={summary?.family_edges ?? []}
        regimeEdges={summary?.regime_edges ?? []}
        vixBucketEdges={summary?.vix_bucket_edges ?? []}
      />

      <PaperTradeFeedbackPanel feedback={summary?.feedback ?? []} />

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

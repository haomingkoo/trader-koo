import { useReport } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import type { YoloBlock } from "../api/types";
import Spinner from "../components/ui/Spinner";
import FearGreedGauge from "../components/FearGreedGauge";
import { PipelineStatusInline } from "../components/PipelineOpsPanel";
import {
  KeyChangesSection,
  RiskFiltersPanel,
  SummaryKpiRow,
  VixRegimeWidget,
  YoloDetectionCards,
} from "../components/report/ReportOverviewSections";
import SetupEvaluationPanel from "../components/report/SetupEvaluationPanel";
import SetupQualitySection from "../components/report/SetupQualitySection";

/* ── Main Page ── */

export default function ReportPage() {
  const { data, isLoading, error } = useReport();
  const activeTicker = useChartStore((s) => s.ticker);

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load report: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  const latest = data?.latest;
  if (!data?.ok || !latest || !Object.keys(latest).length) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--muted)]">
        No report data available yet.
      </div>
    );
  }

  const counts = latest.counts ?? { tracked_tickers: null, price_rows: null };
  const yoloBlock: YoloBlock = latest.yolo ?? {
    summary: { rows_total: null, tickers_with_patterns: null },
    timeframes: [],
  };
  const ingestRun = latest.latest_ingest_run ?? { status: null };
  const signals = latest.signals ?? {
    tonight_key_changes: [],
    regime_context: null,
    setup_quality_top: [],
    setup_evaluation: {},
  };
  const reportDetail = typeof data.detail === "string" && data.detail.trim().length > 0 ? data.detail.trim() : null;
  const risk = latest.risk_filters ?? {
    trade_mode: "normal",
    hard_blocks: 0,
    soft_flags: 0,
    conditions: [],
  };
  const setupRows = signals.setup_quality_top ?? [];

  // Build a debate map from setup rows (debate_v1 data is on chart_commentary per-ticker,
  // but for the report view we don't have individual chart data loaded. We pass an empty map
  // and the debate column will show "\u2014" for tickers without preloaded debate data.)
  const debateMap = new Map<
    string,
    NonNullable<import("../api/types").ChartCommentary["debate_v1"]>
  >();

  return (
    <div className="space-y-6">
      {/* NFA disclaimer banner */}
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        <strong>Research only. Not financial advice.</strong> This dashboard can be wrong, stale, or incomplete.
        Treat it as a decision-support tool, not an instruction to buy or sell. Past performance does not guarantee future results.
      </div>

      {/* Pipeline status inline */}
      <PipelineStatusInline />

      {reportDetail && (
        <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/8 px-4 py-3 text-xs text-[var(--amber)]">
          <strong>Report status:</strong> {reportDetail}
        </div>
      )}

      <h2 className="text-xl font-bold tracking-tight">Daily Report</h2>

      {/* 1. Summary KPI cards */}
      <SummaryKpiRow
        generatedTs={latest.generated_ts}
        trackedTickers={counts.tracked_tickers}
        priceRows={counts.price_rows}
        priceDate={latest.latest_data?.price_date ?? null}
        ingestStatus={ingestRun.status}
      />

      {/* 2. Risk Filters */}
      <RiskFiltersPanel
        tradeMode={String(risk.trade_mode ?? "normal")}
        hardBlocks={risk.hard_blocks ?? 0}
        softFlags={risk.soft_flags ?? 0}
        conditions={risk.conditions ?? []}
      />

      {/* 3. Key Changes */}
      <KeyChangesSection changes={signals.tonight_key_changes ?? []} />

      {/* 4. Macro context */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <FearGreedGauge />
        <VixRegimeWidget regime={signals.regime_context} />
      </div>

      {/* 5. YOLO Detection */}
      <YoloDetectionCards yolo={yoloBlock} />

      {/* 6 + 7. Setup Quality Table with Debate */}
      <SetupQualitySection rows={setupRows} debateMap={debateMap} activeTicker={activeTicker} />

      {/* 8. Setup Evaluation */}
      <SetupEvaluationPanel
        evaluation={signals.setup_evaluation ?? {}}
      />
    </div>
  );
}

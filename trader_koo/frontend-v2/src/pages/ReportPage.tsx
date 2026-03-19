import { useEffect } from "react";
import { useReport } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import Spinner from "../components/ui/Spinner";
import FearGreedGauge from "../components/FearGreedGauge";
import { PipelineStatusInline } from "../components/PipelineOpsPanel";
import {
  KeyChangesSection,
  RiskFiltersPanel,
  SummaryKpiRow,
  VixRegimeWidget,
} from "../components/report/ReportOverviewSections";
import SetupEvaluationPanel from "../components/report/SetupEvaluationPanel";
import SetupQualitySection from "../components/report/SetupQualitySection";

/* ── Main Page ── */

export default function ReportPage() {
  useEffect(() => {
    document.title = "Daily Report \u2014 Trader Koo";
  }, []);

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

  const signals = latest.signals ?? {
    tonight_key_changes: [],
    regime_context: null,
    setup_quality_top: [],
    setup_evaluation: {},
  };
  const reportDetail = typeof data.detail === "string" && data.detail.trim().length > 0 ? data.detail.trim() : null;
  const reportIncomplete = Boolean(reportDetail);
  const risk = latest.risk_filters ?? {
    trade_mode: "normal",
    hard_blocks: 0,
    soft_flags: 0,
    conditions: [],
  };
  const setupRows = signals.setup_quality_top ?? [];

  const debateMap = new Map<
    string,
    NonNullable<import("../api/types").ChartCommentary["debate_v1"]>
  >();

  return (
    <div className="space-y-6">
      {/* NFA disclaimer banner */}
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        <strong>For informational and educational purposes only.</strong> Nothing on this
        page constitutes investment advice, a recommendation, or a solicitation to buy or sell
        any security. All content may be inaccurate, incomplete, or outdated. Past performance
        is not indicative of future results. Consult a licensed financial advisor before making
        any investment decisions. Use entirely at your own risk.
      </div>

      {reportDetail && (
        <>
          <PipelineStatusInline />
          <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/8 px-4 py-3 text-sm text-[var(--amber)]">
            <strong>Report incomplete:</strong> {reportDetail}
            <div className="mt-2 text-xs text-[var(--muted)]">
              We are hiding the main report until the nightly output is fully populated.
            </div>
          </div>
        </>
      )}

      <h2 className="text-xl font-bold tracking-tight">Daily Report</h2>

      <SummaryKpiRow
        generatedTs={latest.generated_ts}
        priceDate={latest.latest_data?.price_date ?? null}
      />

      {!reportIncomplete && (
        <>
          <RiskFiltersPanel
            tradeMode={String(risk.trade_mode ?? "normal")}
            hardBlocks={risk.hard_blocks ?? 0}
            softFlags={risk.soft_flags ?? 0}
            conditions={risk.conditions ?? []}
          />

          <KeyChangesSection changes={signals.tonight_key_changes ?? []} />

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
            <FearGreedGauge />
            <VixRegimeWidget regime={signals.regime_context} />
          </div>

          <SetupQualitySection rows={setupRows} debateMap={debateMap} activeTicker={activeTicker} />

          <SetupEvaluationPanel
            evaluation={signals.setup_evaluation ?? {}}
          />
        </>
      )}
    </div>
  );
}

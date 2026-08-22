import { useEffect } from "react";
import { Link } from "react-router-dom";
import { usePaperTradeSummary, useReport } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import Spinner from "../components/ui/Spinner";
import FearGreedGauge from "../components/FearGreedGauge";
import { PipelineStatusInline } from "../components/PipelineOpsPanel";
import {
  EvidenceSourceStrip,
  KeyChangesSection,
  RiskFiltersPanel,
  SummaryKpiRow,
  VixRegimeWidget,
} from "../components/report/ReportOverviewSections";
import SectorHeatmap from "../components/report/SectorHeatmap";
import SetupEvaluationPanel from "../components/report/SetupEvaluationPanel";
import SetupQualitySection from "../components/report/SetupQualitySection";
import SuggestionSection from "../components/report/SuggestionSection";
import { StrategyEvidenceStatePanel } from "../components/paper/PaperTradeSections";

/* ── Main Page ── */

export default function ReportPage() {
  useEffect(() => {
    document.title = "Daily Report \u2014 Trader Koo";
  }, []);

  const { data, isLoading, error } = useReport();
  const { data: paperSummary } = usePaperTradeSummary();
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
  const reportDetailLevel =
    data.detail_level === "info" || data.detail_level === "warning" || data.detail_level === "error"
      ? data.detail_level
      : "warning";
  const reportBlocksMainReport = Boolean(data.detail_blocks_main_report);
  const risk = latest.risk_filters ?? {
    trade_mode: "normal",
    hard_blocks: 0,
    soft_flags: 0,
    conditions: [],
  };
  const setupRows = signals.setup_quality_top ?? [];
  const detailTone =
    reportDetailLevel === "error"
      ? "border-[var(--red)]/30 bg-[var(--red)]/8 text-[var(--red)]"
      : reportDetailLevel === "info"
        ? "border-[var(--accent)]/30 bg-[var(--accent)]/8 text-[var(--accent)]"
        : "border-[var(--amber)]/30 bg-[var(--amber)]/8 text-[var(--amber)]";

  const debateMap = new Map<
    string,
    NonNullable<import("../api/types").ChartCommentary["debate_v1"]>
  >();

  return (
    <div className="space-y-6">
      {/* NFA disclaimer banner */}
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        <strong>Research only. Not financial advice.</strong> Data may be stale, partial, or wrong.
        Past performance does not guarantee future results.
      </div>

      {reportDetail && (
        <>
          <PipelineStatusInline />
          <div className={`rounded-lg border px-4 py-3 text-sm ${detailTone}`}>
            <strong>{reportBlocksMainReport ? "Report unavailable:" : "Report update:"}</strong> {reportDetail}
            <div className="mt-2 text-xs text-[var(--muted)]">
              {reportBlocksMainReport
                ? "We are hiding the main report until the nightly output is fully populated."
                : "Showing the latest completed report snapshot below while the nightly refresh catches up."}
            </div>
          </div>
        </>
      )}

      <h2 className="text-xl font-bold tracking-tight">Daily Report</h2>

      <StrategyEvidenceStatePanel evidence={paperSummary?.strategy_evidence} />

      <SummaryKpiRow
        generatedTs={latest.generated_ts}
        priceDate={latest.latest_data?.price_date ?? null}
      />

      <EvidenceSourceStrip
        generatedTs={latest.generated_ts}
        latestData={latest.latest_data}
        freshness={latest.freshness}
        warnings={latest.warnings ?? []}
      />

      {!reportBlocksMainReport && (
        <>
          <RiskFiltersPanel
            tradeMode={String(risk.trade_mode ?? "normal")}
            hardBlocks={risk.hard_blocks ?? 0}
            softFlags={risk.soft_flags ?? 0}
            conditions={risk.conditions ?? []}
          />

          <KeyChangesSection changes={signals.tonight_key_changes ?? []} />

          {((signals.green_barrier_coverage?.stale_skipped_count ?? 0) > 0 ||
            (signals.green_barrier_coverage?.invalid_date_skipped_count ?? 0) > 0 ||
            (signals.green_barrier_coverage?.insufficient_history_skipped_count ?? 0) > 0) && (
            <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-3 text-xs text-[var(--amber)]">
              Green Barrier coverage is incomplete: {signals.green_barrier_coverage?.stale_skipped_count ?? 0} stale and{" "}
              {signals.green_barrier_coverage?.invalid_date_skipped_count ?? 0} invalid-date ticker(s), plus{" "}
              {signals.green_barrier_coverage?.insufficient_history_skipped_count ?? 0} ticker/timeframe pair(s) with insufficient history skipped.
            </div>
          )}

          {(signals.green_barrier_hits?.length ?? 0) > 0 && (
            <section className="rounded-xl border border-[var(--green)]/25 bg-[var(--green)]/5 p-4">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <h3 className="text-sm font-semibold text-[var(--green)]">
                    Green Barrier Current Conditions
                  </h3>
                  <p className="mt-1 text-xs text-[var(--muted)]">
                    Williams %R(14) at or below the configured threshold. Repeated daily while active.
                    Research context only—not a buy signal.
                  </p>
                </div>
                <span className="rounded-full border border-[var(--green)]/25 px-2.5 py-1 text-xs font-semibold text-[var(--green)]">
                  {signals.green_barrier_hits?.length} {signals.green_barrier_hits?.length === 1 ? "hit" : "hits"}
                </span>
              </div>
              <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {signals.green_barrier_hits?.slice(0, 12).map((hit) => (
                  <Link
                    key={`${hit.ticker}-${hit.timeframe}`}
                    to={`/chart?ticker=${encodeURIComponent(hit.ticker)}&timeframe=${hit.timeframe}&threshold=${encodeURIComponent(hit.threshold)}&asof=${encodeURIComponent(hit.asof)}&value=${encodeURIComponent(hit.value)}`}
                    className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-2 transition-colors hover:border-[var(--green)]/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--green)]"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-sm font-bold text-[var(--text)]">
                        {hit.ticker}
                      </span>
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--green)]">
                        {hit.timeframe}
                      </span>
                    </div>
                    <div className="mt-1 flex items-baseline justify-between gap-2 text-xs">
                      <span className="text-[var(--muted)]">Williams %R</span>
                      <strong className="tabular-nums text-[var(--green)]">
                        {hit.value.toFixed(1)}
                      </strong>
                    </div>
                    <div className="mt-1 text-[10px] text-[var(--muted)]">
                      As of {hit.asof} · trigger ≤ {hit.threshold.toFixed(1)} · close {hit.close.toLocaleString()}
                    </div>
                  </Link>
                ))}
              </div>
              {(signals.green_barrier_hits?.length ?? 0) > 12 && (
                <p className="mt-2 text-[10px] text-[var(--muted)]">
                  Showing the 12 readings closest to −100.
                </p>
              )}
            </section>
          )}

          <SuggestionSection suggestions={signals.suggestions} />

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
            <FearGreedGauge />
            <VixRegimeWidget regime={signals.regime_context} />
          </div>

          <SectorHeatmap rows={signals.sector_heatmap ?? []} />

          <SetupQualitySection rows={setupRows} debateMap={debateMap} activeTicker={activeTicker} />

          <SetupEvaluationPanel
            evaluation={signals.setup_evaluation ?? {}}
          />
        </>
      )}
    </div>
  );
}

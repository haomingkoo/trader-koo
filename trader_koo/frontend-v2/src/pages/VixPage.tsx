import { useReport, useVixMetrics } from "../api/hooks";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import {
  CommentaryCard,
  formatVixState,
  MAMatrixCard,
  MarketHealthCard,
  ParticipationBiasCard,
  RegimeSummaryCard,
  SpikeAlertBanner,
  VixMetricCardsGrid,
  VixPrimaryPanels,
} from "../components/vix/VixSections";

export default function VixPage() {
  const { data, isLoading, error } = useReport();
  const {
    data: metricsData,
    isLoading: metricsLoading,
    error: metricsError,
  } = useVixMetrics();

  if (isLoading || metricsLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load VIX data: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  const regime = data?.latest?.signals?.regime_context;
  if (!regime || !Object.keys(regime).length) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--muted)]">
        No regime context available yet. Data populates after the nightly
        pipeline run.
      </div>
    );
  }

  const vix = regime.vix;
  const health = regime.health ?? {
    state: "unknown",
    score: null,
    drivers: [],
    warnings: [],
  };
  const overall = regime.overall ?? { participation_bias: "unknown" };
  const maMatrix = regime.ma_matrix ?? [];
  const commentary = regime.llm_commentary;
  const metrics = metricsData?.ok ? metricsData : null;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">VIX / Regime Analysis</h2>

      {metrics && <SpikeAlertBanner metrics={metrics} />}

      <VixPrimaryPanels vix={vix} metrics={metrics} />

      <p className="text-[10px] text-[var(--muted)]">
        Position sizing recommendations are for educational purposes only and
        should not be construed as financial advice.
      </p>

      {metrics && <VixMetricCardsGrid metrics={metrics} />}
      {metricsError && (
        <div className="text-xs text-[var(--red)]">
          Failed to load VIX metrics: {String((metricsError as Error)?.message ?? "Unknown error")}
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card
          glass
          label="VIX Close"
          value={vix.close != null ? vix.close.toFixed(2) : "\u2014"}
        />
        <Card glass label="Risk State" value={formatVixState(vix.risk_state ?? "unknown")} />
        <Card
          glass
          label="Percentile 1Y"
          value={
            vix.percentile_1y != null
              ? `${Number(vix.percentile_1y).toFixed(1)}%`
              : "\u2014"
          }
        />
        <Card
          glass
          label="MA Cross State"
          value={formatVixState(vix.ma_cross_state ?? vix.ma_state ?? "unknown")}
        />
      </div>

      <MarketHealthCard health={health} />
      <ParticipationBiasCard participationBias={overall.participation_bias ?? "unknown"} />
      <MAMatrixCard rows={maMatrix} />
      <RegimeSummaryCard summary={typeof regime.summary === "string" ? regime.summary : ""} />
      <CommentaryCard commentary={commentary} />

      <div className="text-xs text-[var(--muted)]">
        As of {String(regime.asof_date ?? "\u2014")} &middot; Source:{" "}
        {String(regime.source ?? "unknown")}
      </div>
    </div>
  );
}

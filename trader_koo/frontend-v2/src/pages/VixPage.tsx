import { useEffect } from "react";
import { useReport, useVixMetrics } from "../api/hooks";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import {
  CommentaryCard,
  MAMatrixCard,
  MarketHealthCard,
  ParticipationBiasCard,
  RegimeSummaryCard,
  SpikeAlertBanner,
  VixMetricCardsGrid,
  VixPrimaryPanels,
} from "../components/vix/VixSections";
import { formatVixState } from "../components/vix/vixUtils";

export default function VixPage() {
  useEffect(() => {
    document.title = "VIX \u2014 Trader Koo";
  }, []);

  const { data, isLoading, error } = useReport();
  const {
    data: metricsData,
    isLoading: metricsLoading,
    error: metricsError,
  } = useVixMetrics();

  if (isLoading || metricsLoading) return <Spinner className="mt-12" />;
  if (error && !metricsData?.ok) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load VIX data: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  const regime = data?.latest?.signals?.regime_context;
  if (!regime || !Object.keys(regime).length) {
    if (metricsData?.ok) {
      return (
        <div className="space-y-6" data-testid="vix-live-metrics-only">
          <h2 className="text-xl font-bold tracking-tight">VIX / Regime Analysis</h2>
          <div className="rounded-lg border border-[var(--amber)]/40 bg-[rgba(248,194,78,0.06)] p-3 text-xs text-[var(--muted)]" role="status">
            The sealed daily regime report is unavailable. Live volatility metrics remain available below and are not a substitute for the missing report.
          </div>
          <SpikeAlertBanner metrics={metricsData} />
          <VixMetricCardsGrid metrics={metricsData} />
          <p className="text-[10px] text-[var(--muted)]">
            Live metrics are descriptive and research-only. No regime or sizing decision is inferred from missing report evidence.
          </p>
        </div>
      );
    }
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]" role="alert">
        VIX data is unavailable: neither a sealed regime report nor live volatility metrics could be loaded.
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
        Sizing guidance is research-only, not financial advice.
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

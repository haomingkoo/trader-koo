import { Link } from "react-router-dom";
import type {
  KeyChange,
  RegimeContext,
  RiskCondition,
  YoloBlock,
} from "../../api/types";
import Badge from "../ui/Badge";
import {
  formatReportNumber,
  formatReportTimestamp,
  GlassCard,
  severityVariant,
} from "./reportShared";

export function SummaryKpiRow({
  generatedTs,
  priceDate,
}: {
  generatedTs: string | null;
  priceDate: string | null;
}) {
  const ts = formatReportTimestamp(generatedTs);
  const reportFor = priceDate ?? ts.ny.split(",")[0] ?? "\u2014";
  return (
    <GlassCard label="Report For" value={reportFor}>
      <div className="mt-2 text-xs text-[var(--muted)]">
        Generated {ts.local}
      </div>
      <div className="mt-1 text-xs text-[var(--muted)]">NY: {ts.ny}</div>
    </GlassCard>
  );
}

export function RiskFiltersPanel({
  tradeMode,
  hardBlocks,
  softFlags,
  conditions,
}: {
  tradeMode: string;
  hardBlocks: number;
  softFlags: number;
  conditions: RiskCondition[];
}) {
  const modeUpper = tradeMode.toUpperCase();
  const modeVariant =
    modeUpper === "NORMAL"
      ? "green"
      : modeUpper === "CAUTION"
        ? "amber"
        : "red";

  return (
    <GlassCard>
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant={modeVariant} className="text-xs">
          {modeUpper} MODE
        </Badge>
        <span className="text-xs text-[var(--muted)]">
          Hard blocks:{" "}
          <strong className="text-[var(--text)]">{hardBlocks}</strong>
        </span>
        <span className="text-xs text-[var(--muted)]">
          Soft flags:{" "}
          <strong className="text-[var(--text)]">{softFlags}</strong>
        </span>
      </div>
      {conditions.length > 0 ? (
        <ul className="mt-3 space-y-1.5">
          {conditions.map((c, i) => (
            <li key={i} className="flex items-start gap-2 text-xs">
              <Badge variant={severityVariant(c.severity)} className="shrink-0">
                {c.severity.toUpperCase()}
              </Badge>
              <span className="text-[var(--muted)]">
                {c.code && (
                  <span className="font-mono text-[var(--text)]">
                    {String(c.code)}
                  </span>
                )}{" "}
                {String(c.reason)}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-xs text-[var(--muted)]">
          No active risk conditions.
        </p>
      )}
    </GlassCard>
  );
}

export function KeyChangesSection({ changes }: { changes: KeyChange[] }) {
  if (changes.length === 0) {
    return (
      <GlassCard label="Key Changes">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No key changes tonight.
        </p>
      </GlassCard>
    );
  }
  return (
    <GlassCard label="Key Changes">
      <ul className="mt-2 space-y-1.5">
        {changes.map((kc, i) => (
          <li key={i} className="text-xs">
            <strong className="text-[var(--text)]">{String(kc.title)}:</strong>{" "}
            <span className="text-[var(--muted)]">{String(kc.detail)}</span>
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}

export function VixRegimeWidget({ regime }: { regime: RegimeContext | null }) {
  if (!regime) {
    return (
      <GlassCard label="VIX Regime Context">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No VIX regime data available.
        </p>
      </GlassCard>
    );
  }
  const vix = regime.vix;
  const health = regime.health;
  const overall = regime.overall;
  const riskState = (vix.risk_state ?? "unknown").replace(/_/g, " ");
  const healthState = (health.state ?? "unknown").replace(/_/g, " ");
  const participationBias = (overall.participation_bias ?? "unknown").replace(
    /_/g,
    " ",
  );

  const riskVariant =
    riskState.toLowerCase().includes("low") ||
    riskState.toLowerCase().includes("normal")
      ? "green"
      : riskState.toLowerCase().includes("elevated") ||
          riskState.toLowerCase().includes("caution")
        ? "amber"
        : "red";

  return (
    <GlassCard>
      <div className="flex flex-wrap items-center gap-4">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          VIX Regime
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold tabular-nums text-[var(--text)]">
            {formatReportNumber(vix.close, 2)}
          </span>
          <Badge variant={riskVariant}>{riskState.toUpperCase()}</Badge>
        </div>
        <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
          <span>
            Health:{" "}
            <strong className="text-[var(--text)]">
              {healthState} ({formatReportNumber(health.score, 1)}/100)
            </strong>
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
          <span>
            Bias:{" "}
            <strong className="text-[var(--text)]">{participationBias}</strong>
          </span>
        </div>
        <Link
          to="/vix"
          className="ml-auto text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          Full VIX Analysis &rarr;
        </Link>
      </div>
    </GlassCard>
  );
}

export function YoloDetectionCards({ yolo }: { yolo: YoloBlock }) {
  const summary = yolo.summary;
  const timeframes = yolo.timeframes ?? [];
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        YOLO Pattern Detection
      </h3>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard label="Total Rows" value={summary.rows_total ?? "\u2014"} />
        <GlassCard
          label="Tickers w/ Patterns"
          value={summary.tickers_with_patterns ?? "\u2014"}
        />
        {timeframes.map((tf) => (
          <GlassCard
            key={tf.timeframe}
            label={`${tf.timeframe} Tickers`}
            value={tf.tickers_with_patterns ?? "\u2014"}
          />
        ))}
      </div>
    </div>
  );
}

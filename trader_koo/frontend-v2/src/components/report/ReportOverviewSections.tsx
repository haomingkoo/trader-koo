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

  const healthVariant =
    healthState.toLowerCase().includes("healthy") || healthState.toLowerCase().includes("strong")
      ? "green"
      : healthState.toLowerCase().includes("warning") || healthState.toLowerCase().includes("weak")
        ? "amber"
        : "muted";

  return (
    <GlassCard>
      <div className="space-y-3">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            VIX Regime
          </div>
          <Link
            to="/vix"
            className="text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
          >
            Full Analysis &rarr;
          </Link>
        </div>

        {/* VIX level + risk state */}
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold tabular-nums text-[var(--text)]">
            {formatReportNumber(vix.close, 2)}
          </span>
          <Badge variant={riskVariant}>{riskState.toUpperCase()}</Badge>
        </div>

        {/* Key metrics grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-2.5">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--muted)]">
              Market Health
            </div>
            <div className="mt-1 flex items-center gap-2">
              <Badge variant={healthVariant} className="text-[9px]">
                {healthState.toUpperCase()}
              </Badge>
              <span className="text-sm font-bold tabular-nums text-[var(--text)]">
                {formatReportNumber(health.score, 0)}/100
              </span>
            </div>
          </div>
          <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-2.5">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--muted)]">
              Participation Bias
            </div>
            <div className="mt-1 text-sm font-bold capitalize text-[var(--text)]">
              {participationBias}
            </div>
          </div>
          {vix.percentile_1y != null && (
            <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-2.5">
              <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                1Y Percentile
              </div>
              <div className="mt-1 text-sm font-bold tabular-nums text-[var(--text)]">
                {formatReportNumber(vix.percentile_1y, 0)}%
              </div>
            </div>
          )}
          {vix.ma20 != null && (
            <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-2.5">
              <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                VIX 20-Day MA
              </div>
              <div className="mt-1 text-sm font-bold tabular-nums text-[var(--text)]">
                {formatReportNumber(vix.ma20, 2)}
              </div>
            </div>
          )}
        </div>

        {/* Actionable takeaway */}
        <div className="text-xs text-[var(--muted)]">
          {vix.close != null && vix.close < 15 && "Low volatility environment. Wider stops may whipsaw; consider tighter position sizing."}
          {vix.close != null && vix.close >= 15 && vix.close < 20 && "Normal volatility. Standard trading conditions."}
          {vix.close != null && vix.close >= 20 && vix.close < 25 && "Elevated volatility. Consider reducing position sizes and widening stops."}
          {vix.close != null && vix.close >= 25 && vix.close < 30 && "High volatility. Defensive posture recommended. Reduce exposure."}
          {vix.close != null && vix.close >= 30 && "Extreme volatility. Risk-off. Consider sitting out or trading reduced size only."}
        </div>
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

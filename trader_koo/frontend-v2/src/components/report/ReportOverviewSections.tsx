import { Link } from "react-router-dom";
import type {
  KeyChange,
  RegimeContext,
  ReportLatest,
  RiskCondition,
  SectorHeatmapRow,
  YoloBlock,
} from "../../api/types";
import Badge from "../ui/Badge";
import GlassCard from "../ui/GlassCard";
import {
  DISCLOSURE_SUMMARY,
  formatReportNumber,
  formatReportTimestamp,
  severityVariant,
} from "./reportShared";
import { presentKeyChanges, type SetupHeadline } from "./reportText";

function compactAge(hours: number | null | undefined): string | null {
  if (typeof hours !== "number") return null;
  if (hours < 24) return `${hours.toFixed(0)}h old`;
  return `${(hours / 24).toFixed(1)}d old`;
}

/** A source timestamp in the viewer's local time, with New York time on hover. */
function SourceTime({ ts }: { ts: string | null | undefined }) {
  if (!ts) return <>Unavailable</>;
  // Date-only values (a price session) are already exchange dates.
  if (/^\d{4}-\d{2}-\d{2}$/.test(ts)) return <>{ts}</>;
  const { ny } = formatReportTimestamp(ts);
  const parsed = new Date(ts);
  const local = Number.isNaN(parsed.getTime())
    ? ts
    : parsed.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  return <span title={`New York: ${ny}`}>{local}</span>;
}

function EvidenceChip({
  label,
  ts,
  detail,
  stale = false,
}: {
  label: string;
  ts: string | null | undefined;
  detail?: string | null;
  stale?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border px-3 py-2 ${
        stale
          ? "border-[var(--amber)]/35 bg-[var(--amber)]/5"
          : "border-[var(--line)] bg-[var(--bg)]/40"
      }`}
    >
      <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
        {label}
        {stale && <span className="ml-1 text-[var(--amber)]">stale</span>}
      </div>
      <div className="mt-1 truncate text-xs font-semibold text-[var(--text)]">
        <SourceTime ts={ts} />
      </div>
      {detail && (
        <div className="mt-0.5 truncate text-[10px] text-[var(--muted)]">
          {detail}
        </div>
      )}
    </div>
  );
}

export function EvidenceSourceStrip({
  generatedTs,
  reportRun,
  latestData,
  freshness,
  warnings = [],
}: {
  generatedTs: string | null;
  reportRun: ReportLatest["report_run"];
  latestData: ReportLatest["latest_data"];
  freshness: ReportLatest["freshness"];
  warnings?: string[];
}) {
  const generated = formatReportTimestamp(generatedTs);
  const priceAge = compactAge(
    typeof freshness?.price_age_days === "number"
      ? freshness.price_age_days * 24
      : null,
  );
  const fundAge = compactAge(freshness?.fund_age_hours);
  const optionsAge = compactAge(freshness?.opt_age_hours);
  const yoloAge = compactAge(freshness?.yolo_age_hours);
  const reportFor = latestData?.price_date ?? generated.ny.split(",")[0] ?? "\u2014";

  return (
    <GlassCard>
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            Report for
          </div>
          <div className="mt-0.5 text-lg font-bold tabular-nums text-[var(--text)]">{reportFor}</div>
          <div className="mt-0.5 text-xs text-[var(--muted)]">
            Generated <span title={`New York: ${generated.ny}`}>{generated.local}</span>
          </div>
        </div>
        {warnings.length > 0 && (
          <Badge variant="amber">{warnings.length} warning{warnings.length === 1 ? "" : "s"}</Badge>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2 xl:grid-cols-4">
        <EvidenceChip
          label="Prices"
          ts={latestData?.price_date}
          detail={priceAge}
          stale={Boolean(freshness?.price_age_days != null && freshness.price_age_days > 1)}
        />
        <EvidenceChip
          label="Fundamentals"
          ts={latestData?.fund_snapshot}
          detail={fundAge}
          stale={Boolean(freshness?.fund_age_hours != null && freshness.fund_age_hours > 72)}
        />
        <EvidenceChip
          label="Options"
          ts={latestData?.options_snapshot}
          detail={optionsAge}
          stale={Boolean(freshness?.opt_age_hours != null && freshness.opt_age_hours > 72)}
        />
        <EvidenceChip
          label="YOLO"
          ts={latestData?.yolo_detected_ts}
          detail={yoloAge}
          stale={Boolean(freshness?.yolo_age_hours != null && freshness.yolo_age_hours > 72)}
        />
      </div>
      {reportRun?.lineage === "linked" ? (
        <details data-testid="report-run-provenance" className="mt-3 text-[10px] text-[var(--muted)]">
          <summary className={DISCLOSURE_SUMMARY}>
            Lineage: run {reportRun.run_id?.slice(0, 8)}, {reportRun.state}
          </summary>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 px-3 py-2">
            <span><strong className="text-[var(--text)]">Run</strong> {reportRun.run_id}</span>
            <span><strong className="text-[var(--text)]">State</strong> {reportRun.state}</span>
            <span><strong className="text-[var(--text)]">Generation</strong> {reportRun.generation_key || "Unavailable"}</span>
            <span><strong className="text-[var(--text)]">Canonical retry</strong> {reportRun.canonical_generation === false ? "No" : "Yes"}</span>
            <span><strong className="text-[var(--text)]">Started</strong> {reportRun.started_ts || "Unavailable"}</span>
            <span><strong className="text-[var(--text)]">Completed</strong> {reportRun.completed_ts || "Unavailable"}</span>
            <span><strong className="text-[var(--text)]">Published</strong> {reportRun.published_ts || "Unavailable"}</span>
            <span title={reportRun.content_hash || undefined}><strong className="text-[var(--text)]">Artifact SHA-256</strong> {reportRun.content_hash?.slice(0, 12) || "Unavailable"}</span>
            <span title={reportRun.config_hash || undefined}><strong className="text-[var(--text)]">Config</strong> {reportRun.config_hash?.slice(0, 12) || "Unavailable"}</span>
            <span title={reportRun.code_version || undefined}><strong className="text-[var(--text)]">Code</strong> {reportRun.code_version?.slice(0, 12) || "Unavailable"}</span>
          </div>
        </details>
      ) : (
        <div data-testid="report-run-provenance" className="mt-3 rounded-lg border border-[var(--amber)]/35 bg-[var(--amber)]/5 px-3 py-2 text-[10px] text-[var(--muted)]">
          <strong className="text-[var(--amber)]">Unlinked legacy</strong> — this report predates immutable publication lineage; no run identity was inferred.
        </div>
      )}
      {warnings.length > 0 && (
        <ul className="mt-3 space-y-1 border-t border-[var(--line)] pt-3">
          {warnings.slice(0, 3).map((warning, index) => (
            <li key={`${warning}-${index}`} className="text-[11px] text-[var(--muted)]">
              <span className="font-semibold text-[var(--amber)]">Warning:</span>{" "}
              {warning}
            </li>
          ))}
        </ul>
      )}
    </GlassCard>
  );
}

const DATA_CHECK_LABELS: Record<string, string> = {
  NORMAL: "Clear",
  CAUTION: "Caution",
  BLOCKED: "Blocked",
};

/**
 * The report's no-trade conditions (`risk_filters`) cover data freshness,
 * market sessions, choppy breadth and pattern churn. They do not score market
 * risk, so the panel says so rather than reading "normal" beside a risk-off
 * Market Health score.
 */
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
        <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          Data and session checks
        </span>
        <Badge variant={modeVariant} className="text-xs">
          {DATA_CHECK_LABELS[modeUpper] ?? modeUpper}
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
      <p className="mt-2 text-xs text-[var(--muted)]">
        Covers stale or missing data, failed ingest, holidays and early closes, choppy breadth and
        pattern churn. It does not measure market risk; Market Health under VIX Regime does.
      </p>
      {conditions.length > 0 && (
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
      )}
    </GlassCard>
  );
}

export function KeyChangesSection({
  changes,
  sectorRows,
}: {
  changes: KeyChange[];
  sectorRows: SectorHeatmapRow[];
}) {
  const views = presentKeyChanges(changes, sectorRows);
  if (views.length === 0) {
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
        {views.map((kc) => (
          <li key={kc.slug || kc.title} className="text-xs">
            <strong className="text-[var(--text)]">{kc.title}:</strong>{" "}
            {kc.setup ? (
              <TopSetupDetail setup={kc.setup} />
            ) : (
              <span className="text-[var(--muted)]">{kc.detail}</span>
            )}
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}

function TopSetupDetail({ setup }: { setup: SetupHeadline }) {
  return (
    <>
      <span className="text-[var(--muted)]">{setup.summary}</span>
      {setup.reasons.length > 0 && (
        <ul className="mt-1 list-disc space-y-0.5 pl-5 text-[var(--muted)]">
          {setup.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
      {setup.plan && (
        <p className="mt-1 text-[var(--muted)]">
          <span className="font-semibold text-[var(--text)]">Plan:</span> {setup.plan}
        </p>
      )}
      {setup.more.length > 0 && (
        <details className="mt-1">
          <summary className={DISCLOSURE_SUMMARY}>{setup.more.length} more</summary>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-[var(--muted)]">
            {setup.more.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </details>
      )}
    </>
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
            <div
              className="text-[9px] font-semibold uppercase tracking-wider text-[var(--muted)]"
              title="Scored from VIX level, trend and term structure plus SPY/QQQ volume participation"
            >
              Market Health (VIX + volume)
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

import { useState } from "react";
import type {
  SetupEvalAction,
  SetupEvalFamily,
  SetupEvaluation,
  SetupEvalValidity,
} from "../../api/types";
import Badge from "../ui/Badge";
import Table from "../ui/Table";
import GlassCard from "../ui/GlassCard";
import { formatReportNumber } from "./reportShared";

function ImprovementAction({ action }: { action: SetupEvalAction }) {
  const priorityVariant =
    action.priority.toLowerCase() === "high"
      ? "red"
      : action.priority.toLowerCase() === "medium"
        ? "amber"
        : "muted";
  return (
    <li className="flex items-start gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3 text-xs">
      <Badge variant={priorityVariant} className="shrink-0">
        {action.priority.toUpperCase()}
      </Badge>
      <div>
        <span className="font-medium text-[var(--text)]">
          {action.scope.replace(/_/g, " ")}:
        </span>{" "}
        <span className="text-[var(--muted)]">{String(action.reason)}</span>
        {action.recommendation && (
          <p className="mt-1 text-[var(--amber)]">
            Tune: {String(action.recommendation)}
          </p>
        )}
      </div>
    </li>
  );
}

function signedPct(value: number | null | undefined, decimals = 2): string {
  if (value == null || !Number.isFinite(value)) return "\u2014";
  return `${value > 0 ? "+" : ""}${value.toFixed(decimals)}%`;
}

function expectancyTone(value: number | null | undefined): string {
  if (value == null || value === 0) return "text-[var(--text)]";
  return value > 0 ? "text-[var(--green)]" : "text-[var(--red)]";
}

/**
 * The calibration headline: whether archived setup calls of this kind made
 * money afterwards. Shown near the top because it qualifies every setup below.
 */
export function CalibrationHeadline({
  evaluation,
}: {
  evaluation: SetupEvaluation | Record<string, never>;
}) {
  const eval_ = evaluation as SetupEvaluation;
  if (!eval_ || !eval_.enabled || eval_.error || !eval_.overall) return null;
  const overall = eval_.overall;
  const directions = (eval_.by_direction ?? []).filter((row) => (row.calls ?? 0) > 0);

  return (
    <GlassCard>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          Setup track record
        </div>
        <a
          href="#model-calibration"
          className="rounded text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
        >
          Full calibration
        </a>
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div>
          <dt className="text-[10px] text-[var(--muted)]">Expectancy per call</dt>
          <dd className={`text-lg font-bold tabular-nums ${expectancyTone(overall.expectancy_pct)}`}>
            {signedPct(overall.expectancy_pct)}
          </dd>
        </div>
        <div>
          <dt className="text-[10px] text-[var(--muted)]">
            Hit rate (return {"\u2265"} {eval_.hit_threshold_pct ?? "\u2014"}%)
          </dt>
          <dd className="text-lg font-bold tabular-nums text-[var(--text)]">
            {overall.hit_rate_pct != null ? `${overall.hit_rate_pct.toFixed(1)}%` : "\u2014"}
          </dd>
        </div>
        <div>
          <dt className="text-[10px] text-[var(--muted)]">Scored calls</dt>
          <dd className="text-lg font-bold tabular-nums text-[var(--text)]">
            {eval_.scored_calls?.toLocaleString() ?? "\u2014"}
          </dd>
        </div>
        <div>
          <dt className="text-[10px] text-[var(--muted)]">Window</dt>
          <dd className="text-sm font-semibold tabular-nums text-[var(--text)]">
            {eval_.window_days ?? "\u2014"}d
            {eval_.latest_scored_asof && (
              <span className="block text-[10px] font-normal text-[var(--muted)]">
                to {eval_.latest_scored_asof}
              </span>
            )}
          </dd>
        </div>
      </dl>
      {directions.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-1 border-t border-[var(--line)] pt-2 text-xs text-[var(--muted)]">
          {directions.map((row) => (
            <li key={row.direction}>
              <span className="font-semibold capitalize text-[var(--text)]">{row.direction}</span>{" "}
              <span className="tabular-nums">{row.calls} calls</span>, hit{" "}
              <span className="tabular-nums">{formatReportNumber(row.hit_rate_pct, 1)}%</span>, expectancy{" "}
              <span className={`font-semibold tabular-nums ${expectancyTone(row.expectancy_pct)}`}>
                {signedPct(row.expectancy_pct)}
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-[10px] text-[var(--muted)]">
        Archived report calls scored against later closes. History, not a forecast.
      </p>
    </GlassCard>
  );
}

export default function SetupEvaluationPanel({
  evaluation,
}: {
  evaluation: SetupEvaluation | Record<string, never>;
}) {
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const eval_ = evaluation as SetupEvaluation;

  if (!eval_ || !Object.keys(eval_).length) {
    return (
      <GlassCard label="Setup Evaluation">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No model calibration data available yet.
        </p>
      </GlassCard>
    );
  }

  if (!eval_.enabled || eval_.error) {
    return (
      <GlassCard label="Setup Evaluation">
        <p className="mt-1 text-xs text-[var(--muted)]">
          Model calibration unavailable ({eval_.error ?? eval_.reason ?? "disabled"}).
        </p>
      </GlassCard>
    );
  }

  const overall = eval_.overall ?? {};
  const families = eval_.by_family ?? [];
  const validities = eval_.by_validity_days ?? [];
  const actions = eval_.improvement_actions ?? [];

  const longFamilies = families.filter(
    (f) => f.call_direction.toLowerCase() === "long",
  );
  const shortFamilies = families.filter(
    (f) => f.call_direction.toLowerCase() === "short",
  );

  const bestValidity = validities
    .filter((v) => (v.calls ?? 0) > 0)
    .sort((a, b) => (b.expectancy_pct ?? -999) - (a.expectancy_pct ?? -999))[0];

  const familyColumns: Array<{
    key: keyof SetupEvalFamily & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "setup_family", label: "Setup Family" },
    { key: "calls", label: "Calls" },
    {
      key: "hit_rate_pct",
      label: "Hit Rate %",
      render: (v: unknown) => formatReportNumber(v as number | null, 1),
    },
    {
      key: "avg_signed_return_pct",
      label: "Avg Return %",
      render: (v: unknown) => formatReportNumber(v as number | null, 2),
    },
    {
      key: "expectancy_pct",
      label: "Expectancy %",
      render: (v: unknown) => formatReportNumber(v as number | null, 2),
    },
  ];

  const validityColumns: Array<{
    key: keyof SetupEvalValidity & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "validity_days", label: "Validity (days)" },
    { key: "calls", label: "Calls" },
    {
      key: "hit_rate_pct",
      label: "Hit Rate %",
      render: (v: unknown) => formatReportNumber(v as number | null, 1),
    },
    {
      key: "avg_signed_return_pct",
      label: "Avg Return %",
      render: (v: unknown) => formatReportNumber(v as number | null, 2),
    },
    {
      key: "expectancy_pct",
      label: "Expectancy %",
      render: (v: unknown) => formatReportNumber(v as number | null, 2),
    },
  ];

  return (
    <div id="model-calibration" className="scroll-mt-4 space-y-4">
      <h3 className="text-sm font-semibold text-[var(--muted)]">
        Model Calibration
      </h3>

      <p className="text-xs text-[var(--muted)]">
        Archived setup calls scored against the closes after their validity windows.
      </p>

      <p className="text-xs text-[var(--muted)]">
        Headline: hit rate {formatReportNumber(overall.hit_rate_pct, 1)}%, avg signed return{" "}
        {formatReportNumber(overall.avg_signed_return_pct, 2)}%, expectancy{" "}
        {formatReportNumber(overall.expectancy_pct, 2)}% over {eval_.scored_calls ?? "\u2014"} scored calls.
      </p>

      <p className="text-xs text-[var(--muted)]">
        Window: {eval_.window_days ?? "\u2014"}d | Min sample:{" "}
        {eval_.min_sample ?? "\u2014"} | Open calls: {eval_.open_calls ?? "\u2014"} |
        {" "}Hit threshold: {eval_.hit_threshold_pct ?? "\u2014"}%
        {eval_.latest_scored_asof && ` | Latest scored: ${eval_.latest_scored_asof}`}
        {bestValidity &&
          ` | Best horizon: ${bestValidity.validity_days}d (${formatReportNumber(bestValidity.expectancy_pct, 2)}%)`}
      </p>

      <div>
        <button
          type="button"
          onClick={() => setShowDiagnostics((prev) => !prev)}
          aria-expanded={showDiagnostics}
          className="text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
        >
          {showDiagnostics ? "Hide" : "Show"} model diagnostics
        </button>
      </div>

      {showDiagnostics && (
        <div className="space-y-4">
          {longFamilies.length > 0 && (
            <div>
              <h4 className="mb-1 text-xs font-semibold text-[var(--green)]">
                Long Families
              </h4>
              <Table columns={familyColumns} data={longFamilies} />
            </div>
          )}
          {shortFamilies.length > 0 && (
            <div>
              <h4 className="mb-1 text-xs font-semibold text-[var(--red)]">
                Short Families
              </h4>
              <Table columns={familyColumns} data={shortFamilies} />
            </div>
          )}

          {validities.length > 0 && (
            <div>
              <h4 className="mb-1 text-xs font-semibold text-[var(--muted)]">
                By Validity Window
              </h4>
              <Table columns={validityColumns} data={validities} />
            </div>
          )}

          <div>
            <h4 className="mb-1 text-xs font-semibold text-[var(--muted)]">
              Tuning Notes
            </h4>
            {actions.length > 0 ? (
              <ul className="space-y-2">
                {actions.map((action, i) => (
                  <ImprovementAction key={i} action={action} />
                ))}
              </ul>
            ) : (
              <p className="text-xs text-[var(--muted)]">
                No tuning notes yet. Collect more scored calls.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

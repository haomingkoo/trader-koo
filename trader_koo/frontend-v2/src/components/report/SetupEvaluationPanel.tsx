import type {
  SetupEvalAction,
  SetupEvalFamily,
  SetupEvaluation,
  SetupEvalValidity,
} from "../../api/types";
import Badge from "../ui/Badge";
import Table from "../ui/Table";
import { formatReportNumber, GlassCard } from "./reportShared";

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

export default function SetupEvaluationPanel({
  evaluation,
}: {
  evaluation: SetupEvaluation | Record<string, never>;
}) {
  const eval_ = evaluation as SetupEvaluation;

  if (!eval_ || !Object.keys(eval_).length) {
    return (
      <GlassCard label="Setup Evaluation">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No setup evaluation data available.
        </p>
      </GlassCard>
    );
  }

  if (!eval_.enabled || eval_.error) {
    return (
      <GlassCard label="Setup Evaluation">
        <p className="mt-1 text-xs text-[var(--muted)]">
          Setup evaluation unavailable ({eval_.error ?? eval_.reason ?? "disabled"}).
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
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-[var(--muted)]">
        Setup Evaluation
      </h3>

      <p className="text-xs text-[var(--muted)]">
        Historical calibration built from archived setup calls and the later price
        outcomes that followed their validity windows. These figures are computed
        from stored report snapshots and subsequent closes, not manually keyed in.
      </p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard
          label="Hit Rate"
          value={
            overall.hit_rate_pct != null
              ? `${overall.hit_rate_pct.toFixed(1)}%`
              : "\u2014"
          }
        />
        <GlassCard
          label="Avg Signed Return"
          value={
            overall.avg_signed_return_pct != null
              ? `${overall.avg_signed_return_pct.toFixed(2)}%`
              : "\u2014"
          }
        />
        <GlassCard
          label="Expectancy"
          value={
            overall.expectancy_pct != null
              ? `${overall.expectancy_pct.toFixed(2)}%`
              : "\u2014"
          }
        />
        <GlassCard label="Scored Calls" value={eval_.scored_calls ?? "\u2014"} />
      </div>

      <p className="text-xs text-[var(--muted)]">
        Window: {eval_.window_days ?? "\u2014"}d | Min sample:{" "}
        {eval_.min_sample ?? "\u2014"} | Open calls: {eval_.open_calls ?? "\u2014"} |
        {" "}Hit threshold: {eval_.hit_threshold_pct ?? "\u2014"}%
        {eval_.latest_scored_asof && ` | Latest scored: ${eval_.latest_scored_asof}`}
        {bestValidity &&
          ` | Best horizon: ${bestValidity.validity_days}d (${formatReportNumber(bestValidity.expectancy_pct, 2)}%)`}
      </p>

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
          Improvement Actions
        </h4>
        {actions.length > 0 ? (
          <ul className="space-y-2">
            {actions.map((action, i) => (
              <ImprovementAction key={i} action={action} />
            ))}
          </ul>
        ) : (
          <p className="text-xs text-[var(--muted)]">
            No tuning actions yet. Collect more scored calls.
          </p>
        )}
      </div>
    </div>
  );
}

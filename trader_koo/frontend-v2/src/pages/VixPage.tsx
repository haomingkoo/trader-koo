import { useReport } from "../api/hooks";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";

const riskReadBadgeVariant = (
  read: string,
): "green" | "red" | "amber" | "muted" => {
  const lower = read.toLowerCase().replace(/_/g, " ");
  if (lower.includes("risk off") || lower.includes("elevated") || lower.includes("above"))
    return "red";
  if (lower.includes("risk on") || lower.includes("relief") || lower.includes("below"))
    return "green";
  return "muted";
};

const formatState = (s: string): string =>
  s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

export default function VixPage() {
  const { data, isLoading, error } = useReport();

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load VIX data: {(error as Error).message}
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
  const commentarySource = (commentary.source ?? "rule").trim().toLowerCase();

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">
        VIX / Regime Analysis
      </h2>

      {/* VIX KPI cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card
          glass
          label="VIX Close"
          value={vix.close != null ? vix.close.toFixed(2) : "\u2014"}
        />
        <Card
          glass
          label="Risk State"
          value={formatState(vix.risk_state ?? "unknown")}
        />
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
          value={formatState(vix.ma_cross_state ?? vix.ma_state ?? "unknown")}
        />
      </div>

      {/* Health panel */}
      <Card label="Market Health">
        <div className="mt-2 space-y-3">
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span className="text-[var(--muted)]">State:</span>
            <Badge
              variant={
                health.state === "healthy"
                  ? "green"
                  : health.state === "caution"
                    ? "amber"
                    : health.state === "stress"
                      ? "red"
                      : "muted"
              }
            >
              {formatState(health.state)}
            </Badge>
            <span className="text-[var(--muted)]">Score:</span>
            <span className="font-semibold text-[var(--text)]">
              {health.score != null ? `${health.score} / 100` : "\u2014"}
            </span>
          </div>

          {health.drivers.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-medium text-[var(--muted)]">
                Drivers
              </div>
              <ul className="space-y-0.5 text-xs text-[var(--text)]">
                {health.drivers.map((d: string, i: number) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="mt-0.5 text-[var(--green)]" aria-hidden="true">&#8226;</span>
                    {d}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {health.warnings.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-medium text-[var(--muted)]">
                Warnings
              </div>
              <ul className="space-y-0.5 text-xs text-[var(--red)]">
                {health.warnings.map((w: string, i: number) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="mt-0.5" aria-hidden="true">&#9888;</span>
                    <span>Warning: {w}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {health.drivers.length === 0 && health.warnings.length === 0 && (
            <p className="text-xs text-[var(--muted)]">
              No health drivers or warnings for this snapshot.
            </p>
          )}
        </div>
      </Card>

      {/* Participation Bias */}
      <Card label="Participation Bias">
        <div className="mt-1 flex items-center gap-2">
          <Badge
            variant={
              overall.participation_bias?.toLowerCase().includes("bull")
                ? "green"
                : overall.participation_bias?.toLowerCase().includes("bear")
                  ? "red"
                  : "muted"
            }
          >
            {formatState(overall.participation_bias ?? "unknown")}
          </Badge>
        </div>
      </Card>

      {/* MA Matrix table */}
      <Card label="MA Matrix">
        {maMatrix.length > 0 ? (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    Value %
                  </th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    State
                  </th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    Risk Read
                  </th>
                </tr>
              </thead>
              <tbody>
                {maMatrix.map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-[var(--line)] last:border-b-0"
                  >
                    <td className="px-3 py-2 font-medium text-[var(--text)]">
                      {row.metric}
                    </td>
                    <td className="px-3 py-2 tabular-nums text-[var(--text)]">
                      {row.value_pct != null
                        ? `${row.value_pct > 0 ? "+" : ""}${row.value_pct.toFixed(2)}%`
                        : "\u2014"}
                    </td>
                    <td className="px-3 py-2">
                      <Badge
                        variant={
                          row.state.toLowerCase().includes("above")
                            ? "red"
                            : row.state.toLowerCase().includes("below")
                              ? "green"
                              : "muted"
                        }
                      >
                        {formatState(row.state)}
                      </Badge>
                    </td>
                    <td className="px-3 py-2">
                      <Badge variant={riskReadBadgeVariant(row.risk_read)}>
                        {row.risk_read.replace(/_/g, " ")}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-2 text-xs text-[var(--muted)]">
            No MA matrix data available for this snapshot.
          </p>
        )}
      </Card>

      {/* Regime Summary */}
      <Card label="Regime Summary">
        <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
          {regime.summary ?? "No summary available."}
        </p>
      </Card>

      {/* LLM Commentary */}
      <Card label="Commentary">
        <div className="mt-2 space-y-2">
          <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--muted)]">
            Source:{" "}
            {commentarySource === "llm"
              ? "LLM (generated from current regime snapshot)"
              : "Rule (rule-based fallback)"}
          </div>
          {commentary.observation ? (
            <div className="space-y-2 text-xs text-[var(--muted)]">
              <p>{commentary.observation}</p>
              {commentary.action && (
                <p>
                  <strong className="text-[var(--text)]">Action:</strong>{" "}
                  {commentary.action}
                </p>
              )}
              {commentary.risk_note && (
                <p>
                  <strong className="text-[var(--text)]">Risk:</strong>{" "}
                  {commentary.risk_note}
                </p>
              )}
            </div>
          ) : (
            <p className="text-xs text-[var(--muted)]">
              No LLM commentary available for this snapshot.
            </p>
          )}
        </div>
      </Card>

      {/* Footer */}
      <div className="text-xs text-[var(--muted)]">
        As of {regime.asof_date ?? "\u2014"} &middot; Source:{" "}
        {regime.source ?? "unknown"}
      </div>
    </div>
  );
}

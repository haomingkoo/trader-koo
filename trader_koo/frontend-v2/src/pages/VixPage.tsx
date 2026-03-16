import { useReport } from "../api/hooks";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";

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

  const vix = regime.vix ?? {};
  const health = regime.health ?? { state: "unknown", score: null, drivers: [], warnings: [] };
  const overall = regime.overall ?? { participation_bias: "unknown" };

  const formatState = (s: string): string =>
    s
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">VIX / Regime Analysis</h2>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card
          label="VIX Close"
          value={vix.close != null ? vix.close.toFixed(2) : "\u2014"}
        />
        <Card
          label="Risk State"
          value={formatState(vix.risk_state ?? "unknown")}
        />
        <Card
          label="Health"
          value={`${formatState(health.state)} (${health.score != null ? `${health.score}/100` : "\u2014"})`}
        />
        <Card
          label="Participation Bias"
          value={formatState(overall.participation_bias ?? "unknown")}
        />
      </div>

      <Card label="Summary">
        <p className="mt-1 text-xs text-[var(--muted)]">
          {regime.summary ?? "No summary available."}
        </p>
      </Card>

      {health.drivers.length > 0 && (
        <Card label="Health Drivers">
          <ul className="mt-1 space-y-0.5 text-xs text-[var(--muted)]">
            {health.drivers.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ul>
        </Card>
      )}

      {health.warnings.length > 0 && (
        <Card label="Warnings">
          <ul className="mt-1 space-y-0.5 text-xs text-[var(--red)]">
            {health.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Card>
      )}

      {/* MA Matrix */}
      {(regime.ma_matrix ?? []).length > 0 && (
        <Card label="MA Matrix">
          <div className="mt-2 overflow-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                  <th className="px-2 py-1">Metric</th>
                  <th className="px-2 py-1">Value %</th>
                  <th className="px-2 py-1">State</th>
                  <th className="px-2 py-1">Risk Read</th>
                </tr>
              </thead>
              <tbody>
                {regime.ma_matrix.map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-[var(--line)] last:border-b-0"
                  >
                    <td className="px-2 py-1 text-[var(--text)]">
                      {row.metric}
                    </td>
                    <td className="px-2 py-1 tabular-nums">
                      {row.value_pct != null
                        ? `${row.value_pct.toFixed(2)}%`
                        : "\u2014"}
                    </td>
                    <td className="px-2 py-1">{row.state}</td>
                    <td className="px-2 py-1">
                      {row.risk_read.replace(/_/g, " ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card label="As of">
        <p className="mt-1 text-xs text-[var(--muted)]">
          {regime.asof_date ?? "\u2014"} &middot; Source:{" "}
          {regime.source ?? "unknown"}
        </p>
      </Card>
    </div>
  );
}

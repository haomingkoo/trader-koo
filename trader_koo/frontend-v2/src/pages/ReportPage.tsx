import { useReport } from "../api/hooks";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import Badge, { tierVariant } from "../components/ui/Badge";
import Table from "../components/ui/Table";

const setupColumns = [
  { key: "ticker" as const, label: "Ticker" },
  {
    key: "setup_score" as const,
    label: "Score",
    render: (v: unknown) => String(v ?? "\u2014"),
  },
  {
    key: "setup_tier" as const,
    label: "Tier",
    render: (v: unknown) => {
      const tier = v as string | null;
      return tier ? <Badge variant={tierVariant(tier)}>{tier}</Badge> : "\u2014";
    },
  },
  { key: "setup_label" as const, label: "Setup" },
  { key: "bias_label" as const, label: "Bias" },
  { key: "yolo_context" as const, label: "YOLO Context" },
  { key: "level_event" as const, label: "Level Event" },
  { key: "observation_short" as const, label: "What It Is" },
  { key: "next_step_short" as const, label: "Next Step" },
];

export default function ReportPage() {
  const { data, isLoading, error } = useReport();

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load report: {(error as Error).message}
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

  const counts = latest.counts ?? {};
  const yoloBlock = latest.yolo ?? { summary: {}, timeframes: [] };
  const yoloSummary = yoloBlock.summary ?? {};
  const ingestRun = latest.latest_ingest_run ?? {};
  const signals = latest.signals ?? {};
  const risk = latest.risk_filters ?? {};
  const setupRows = (signals.setup_quality_top ?? []).slice(0, 40);

  const formatTs = (ts: string | null): string => {
    if (!ts) return "\u2014";
    try {
      return new Date(ts).toLocaleString();
    } catch {
      return ts;
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">Daily Report</h2>

      {/* Summary cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <Card label="Generated" value={formatTs(latest.generated_ts)} />
        <Card label="Tracked Tickers" value={counts.tracked_tickers ?? "\u2014"} />
        <Card
          label="Price Rows"
          value={
            counts.price_rows != null
              ? counts.price_rows.toLocaleString()
              : "\u2014"
          }
        />
        <Card
          label="Latest Price Date"
          value={latest.latest_data?.price_date ?? "\u2014"}
        />
        <Card label="Last Ingest" value={ingestRun.status ?? "\u2014"} />
        <Card
          label="Risk Mode"
          value={String(risk.trade_mode ?? "normal").toUpperCase()}
        />
      </div>

      {/* YOLO cards */}
      <div>
        <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
          YOLO Pattern Detection
        </h3>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Card label="Total Rows" value={yoloSummary.rows_total ?? "\u2014"} />
          <Card
            label="Tickers w/ Patterns"
            value={yoloSummary.tickers_with_patterns ?? "\u2014"}
          />
          {(yoloBlock.timeframes ?? []).map((tf) => (
            <Card
              key={tf.timeframe}
              label={`${tf.timeframe} Tickers`}
              value={tf.tickers_with_patterns ?? "\u2014"}
            />
          ))}
        </div>
      </div>

      {/* Risk summary */}
      <Card label="Risk Filters">
        <div className="mt-1 flex gap-3 text-xs text-[var(--muted)]">
          <span>
            Mode:{" "}
            <strong className="text-[var(--text)]">
              {String(risk.trade_mode ?? "normal").toUpperCase()}
            </strong>
          </span>
          <span>
            Hard blocks:{" "}
            <strong className="text-[var(--text)]">
              {risk.hard_blocks ?? 0}
            </strong>
          </span>
          <span>
            Soft flags:{" "}
            <strong className="text-[var(--text)]">
              {risk.soft_flags ?? 0}
            </strong>
          </span>
        </div>
      </Card>

      {/* Key changes */}
      {(signals.tonight_key_changes ?? []).length > 0 && (
        <Card label="Key Changes">
          <ul className="mt-1 space-y-1 text-xs text-[var(--muted)]">
            {signals.tonight_key_changes.slice(0, 5).map((kc, i) => (
              <li key={i}>
                <strong className="text-[var(--text)]">
                  {kc.title ?? "Change"}:
                </strong>{" "}
                {kc.detail ?? "-"}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Setup quality table */}
      <div>
        <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
          Setup Quality ({setupRows.length} setups)
        </h3>
        <Table
          columns={setupColumns}
          data={setupRows as unknown as Record<string, unknown>[]}
          sortable
        />
      </div>
    </div>
  );
}

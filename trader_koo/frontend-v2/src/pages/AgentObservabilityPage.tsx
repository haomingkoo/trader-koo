import { useEffect, useState } from "react";
import { KeyRound, RefreshCw, ShieldCheck } from "lucide-react";
import { apiFetch } from "../api/client";
import Badge from "../components/ui/Badge";

const ADMIN_KEY_STORAGE = "trader_koo_admin_key";

type Trace = {
  trace_id: string;
  run_id: string;
  role: string;
  stage: string;
  source: string;
  provider: string;
  model: string | null;
  deployment: string | null;
  prompt_template_version: string;
  evaluator_version: string | null;
  evaluation_result: {
    passed: boolean;
    semantic_outcome: string;
    errors: string[];
    prose_quality_scored: false;
  } | null;
  cache_identity_sha256: string | null;
  ticker: string | null;
  terminal_status: string;
  validator_result: string;
  fallback_reason: string | null;
  latency_ms: number;
  total_tokens: number;
  estimated_cost_usd: number | null;
  decision_scope: string;
  decision_changed: number;
  started_ts: string;
};

type ObservabilityPayload = {
  schema_version: string;
  retention: {
    prompt_storage: string;
    trace_retention_days: number;
    credentials_stored: false;
  };
  aggregate: {
    traces: number;
    success_rate_pct: number | null;
    cache_hit_rate_pct: number | null;
    fallback_rate_pct: number | null;
    error_rate_pct: number | null;
    unresolved_traces: number;
    p50_latency_ms: number | null;
    p95_latency_ms: number | null;
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    estimated_cost_usd: number | null;
    validator_failures: number;
    decision_change_rate_pct: number | null;
    run_graphs: number;
    disagreements: number;
  };
  traces: Trace[];
  legacy_health_counters: { label: "legacy"; counts?: { total?: number } };
};

function getStoredKey() {
  try { return localStorage.getItem(ADMIN_KEY_STORAGE) ?? ""; } catch { return ""; }
}

function format(value: number | null, suffix = "") {
  return value === null ? "N/A" : `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`;
}

function statusVariant(status: string) {
  if (status === "success" || status === "cache_hit") return "green" as const;
  if (status === "fallback") return "amber" as const;
  if (status === "error") return "red" as const;
  return "muted" as const;
}

export default function AgentObservabilityPage() {
  useEffect(() => { document.title = "Agent Traces | Trader Koo"; }, []);
  const [apiKey, setApiKey] = useState(getStoredKey);
  const [data, setData] = useState<ObservabilityPayload | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setPending(true);
    setError(null);
    try {
      const payload = await apiFetch<ObservabilityPayload>(
        "/api/admin/agent-observability?limit=100",
        { headers: { "X-API-Key": apiKey } },
      );
      localStorage.setItem(ADMIN_KEY_STORAGE, apiKey);
      setData(payload);
    } catch (caught) {
      setData(null);
      setError(String((caught as Error)?.message ?? "Trace access failed"));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="space-y-7" data-testid="agent-observability-page">
      <header className="grid gap-4 border-b border-[var(--line)] pb-5 lg:grid-cols-[1fr_auto] lg:items-end">
        <div>
          <p className="label-xs mb-2">Authenticated operations</p>
          <h1 className="text-2xl font-semibold tracking-tight text-[var(--text)]">Agent Observability</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">
            Real model calls, bounded contributions, and fallbacks. Deterministic rules do not appear as agents.
          </p>
        </div>
        <form
          className="flex gap-2"
          onSubmit={(event) => { event.preventDefault(); void load(); }}
        >
          <label className="sr-only" htmlFor="agent-admin-key">Admin API key</label>
          <div className="relative">
            <KeyRound className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-[var(--muted)]" aria-hidden="true" />
            <input
              id="agent-admin-key"
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              autoComplete="off"
              className="w-56 rounded-md border border-[var(--line)] bg-[var(--panel)] py-2 pl-8 pr-3 text-xs text-[var(--text)]"
              placeholder="Admin API key"
            />
          </div>
          <button
            type="submit"
            disabled={pending}
            className="inline-flex items-center gap-2 rounded-md border border-[var(--line)] bg-[var(--panel-hover)] px-3 py-2 text-xs font-semibold text-[var(--text)] disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${pending ? "animate-spin" : ""}`} aria-hidden="true" />
            Load
          </button>
        </form>
      </header>

      {error && (
        <div className="rounded-md border border-[var(--red)]/30 bg-[var(--red)]/5 p-3 text-sm text-[var(--red)]" role="alert">
          {error}
        </div>
      )}

      {!data && !error && (
        <div className="flex min-h-64 items-center justify-center border-y border-dashed border-[var(--line)] bg-[var(--surface-subtle)] px-6 text-center">
          <div className="max-w-md">
            <ShieldCheck className="mx-auto mb-3 h-5 w-5 text-[var(--muted)]" aria-hidden="true" />
            <p className="text-sm font-semibold text-[var(--text)]">Trace data stays behind admin authentication</p>
            <p className="mt-1 text-xs leading-5 text-[var(--muted)]">Enter the API key stored locally in this browser. Prompts and credentials are never returned.</p>
          </div>
        </div>
      )}

      {data && (
        <>
          <section className="grid border-y border-[var(--line)] sm:grid-cols-2 lg:grid-cols-4" aria-label="LLM aggregate health">
            {[
              ["Recorded traces", format(data.aggregate.traces)],
              ["Success", format(data.aggregate.success_rate_pct, "%")],
              ["Fallback", format(data.aggregate.fallback_rate_pct, "%")],
              ["Errors", format(data.aggregate.error_rate_pct, "%")],
              ["p50 latency", format(data.aggregate.p50_latency_ms, " ms")],
              ["p95 latency", format(data.aggregate.p95_latency_ms, " ms")],
              ["Total tokens", format(data.aggregate.total_tokens)],
              ["Decision change", format(data.aggregate.decision_change_rate_pct, "%")],
            ].map(([label, value], index) => (
              <div key={label} className={`px-3 py-3 ${index % 4 !== 0 ? "lg:border-l lg:border-[var(--line)]" : ""}`}>
                <div className="text-[10px] font-medium uppercase text-[var(--muted)]">{label}</div>
                <div className="mt-1 text-sm font-semibold text-[var(--text)]">{value}</div>
              </div>
            ))}
          </section>

          <section className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-start">
            <div>
              <h2 className="text-base font-semibold text-[var(--text)]">Retention and truth labels</h2>
              <p className="mt-1 text-xs text-[var(--muted)]">Trace records are append-only. Outcome links are observational, not causal.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="muted">prompts: {data.retention.prompt_storage}</Badge>
              <Badge variant="green">credentials stored: no</Badge>
              <Badge variant="muted">legacy counters: {data.legacy_health_counters.counts?.total ?? 0}</Badge>
            </div>
          </section>

          <section aria-labelledby="trace-table-heading">
            <div className="mb-3 flex items-center justify-between gap-4">
              <h2 id="trace-table-heading" className="text-base font-semibold text-[var(--text)]">Recent model spans</h2>
              <span className="text-xs text-[var(--muted)]">{data.aggregate.unresolved_traces} unresolved</span>
            </div>
            {data.traces.length === 0 ? (
              <div className="border-y border-dashed border-[var(--line)] py-12 text-center text-xs text-[var(--muted)]">
                No real LLM calls have been recorded. Rule fallbacks are not agent spans.
              </div>
            ) : (
              <div className="overflow-x-auto border-y border-[var(--line)]">
                <table className="w-full min-w-[70rem] text-left text-xs">
                  <thead className="bg-[var(--surface-subtle)] text-[var(--muted)]">
                    <tr>
                      {[
                        "Time", "Role / stage", "Provider / model", "Context",
                        "Validation", "Semantic check", "Fallback", "Latency", "Tokens", "Contribution",
                      ].map((heading) => <th key={heading} className="px-3 py-2 font-medium">{heading}</th>)}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--line)]">
                    {data.traces.map((trace) => (
                      <tr key={trace.trace_id}>
                        <td className="whitespace-nowrap px-3 py-3 text-[var(--muted)]">{new Date(trace.started_ts).toLocaleString()}</td>
                        <td className="px-3 py-3"><div className="font-medium text-[var(--text)]">{trace.role}</div><div className="text-[var(--muted)]">{trace.stage}</div></td>
                        <td className="px-3 py-3"><div className="text-[var(--text)]">{trace.provider}</div><div className="text-[var(--muted)]">{trace.model ?? trace.deployment ?? "N/A"}</div></td>
                        <td className="px-3 py-3 text-[var(--muted)]">{trace.ticker ?? trace.source}</td>
                        <td className="px-3 py-3"><Badge variant={statusVariant(trace.terminal_status)}>{trace.validator_result}</Badge></td>
                        <td className="px-3 py-3"><div className="text-[var(--text)]">{trace.evaluation_result?.semantic_outcome ?? "N/A"}</div><div className="text-[var(--muted)]">{trace.evaluator_version ?? "not evaluated"}</div></td>
                        <td className="px-3 py-3 text-[var(--muted)]">{trace.fallback_reason?.replaceAll("_", " ") ?? "None"}</td>
                        <td className="px-3 py-3 text-[var(--muted)]">{format(trace.latency_ms, " ms")}</td>
                        <td className="px-3 py-3 text-[var(--muted)]">{trace.total_tokens}</td>
                        <td className="px-3 py-3"><Badge variant={trace.decision_changed ? "blue" : "muted"}>{trace.decision_scope.replaceAll("_", " ")}: {trace.decision_changed ? "changed" : "unchanged"}</Badge></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { AlertTriangle, Check, Download, FileWarning, LockKeyhole, Send } from "lucide-react";
import { useExperimentResults } from "../api/hooks";
import type { ExperimentResult } from "../api/types";
import { apiFetch } from "../api/client";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";

const METRIC_LABELS: Record<string, string> = {
  net_total_return_pct: "Net total return",
  cagr_pct: "CAGR",
  volatility_pct: "Volatility",
  sharpe: "Sharpe",
  sortino: "Sortino",
  max_drawdown_pct: "Max drawdown",
  calmar: "Calmar",
  profit_factor: "Profit factor",
  win_rate_pct: "Win rate",
  average_r: "Average R",
  exposure_pct: "Exposure",
  turnover_pct: "Turnover",
  capacity: "Capacity",
  trade_count: "Trades",
};

function words(value: string | null | undefined) {
  return value ? value.replaceAll("_", " ") : "N/A";
}

function shortHash(value: string | null | undefined) {
  return value ? `${value.slice(0, 12)}…${value.slice(-6)}` : "N/A";
}

function metricValue(name: string, value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  if (name.endsWith("_pct")) return `${value.toFixed(2)}%`;
  if (name === "capacity") return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function EvidenceBadge({ experiment }: { experiment: ExperimentResult }) {
  const variant = experiment.evidence_label === "promotion eligible"
    ? "green"
    : experiment.evidence_label === "invalid" ? "red" : "blue";
  return <Badge variant={variant}>{experiment.evidence_label}</Badge>;
}

function ManifestRows({ experiment }: { experiment: ExperimentResult }) {
  const rows = [
    ["Strategy version", experiment.manifest.strategy_version ?? "N/A"],
    ["Universe", words(experiment.manifest.universe_basis)],
    ["Return basis", words(experiment.manifest.return_basis)],
    ["Benchmark", experiment.manifest.benchmark ?? "N/A"],
    ["Code SHA", shortHash(experiment.manifest.code_sha)],
    ["Implementation", shortHash(experiment.manifest.implementation_sha256)],
    ["Data snapshot", shortHash(experiment.manifest.data_snapshot_hash)],
    ["Config", shortHash(experiment.manifest.config_hash)],
    ["Artifact", shortHash(experiment.manifest.artifact_hash)],
    ["Execution ledger", shortHash(experiment.manifest.ledger_hash)],
  ];
  return (
    <dl className="divide-y divide-[var(--line)] border-y border-[var(--line)]">
      {rows.map(([label, value]) => (
        <div key={label} className="grid gap-1 py-2.5 sm:grid-cols-[10rem_1fr] sm:gap-4">
          <dt className="text-xs text-[var(--muted)]">{label}</dt>
          <dd className="break-all font-mono text-xs text-[var(--text)]">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function CurveState({ experiment }: { experiment: ExperimentResult }) {
  const hasComparableCurves = experiment.curves.strategy.length > 1
    && experiment.curves.spy_total_return.length > 1
    && experiment.curves.cash.length > 1;
  if (!hasComparableCurves) {
    return (
      <div className="flex min-h-52 items-center justify-center border-y border-dashed border-[var(--line)] bg-[var(--surface-subtle)] px-6 text-center">
        <div className="max-w-md">
          <FileWarning className="mx-auto mb-3 h-5 w-5 text-[var(--amber)]" aria-hidden="true" />
          <p className="text-sm font-semibold text-[var(--text)]">Comparable curves are unavailable</p>
          <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
            Strategy, SPY total return, and cash must share exact dates and starting capital. This run failed before that contract was satisfied.
          </p>
        </div>
      </div>
    );
  }
  return (
    <div className="min-h-52 border-y border-[var(--line)] bg-[var(--surface-subtle)] p-4 text-xs text-[var(--muted)]">
      Verified comparison curves are present in the downloadable execution ledger.
    </div>
  );
}

function ChallengerTable({ experiment }: { experiment: ExperimentResult }) {
  if (!experiment.challengers) return null;
  return (
    <section aria-labelledby="challenger-heading">
      <div className="mb-3 flex items-end justify-between gap-4">
        <div>
          <h2 id="challenger-heading" className="text-base font-semibold text-[var(--text)]">Challenger record</h2>
          <p className="mt-1 text-xs text-[var(--muted)]">Every preregistered candidate remains visible, including failures.</p>
        </div>
        <Badge variant={experiment.heldout?.accessed ? "amber" : "muted"}>
          {experiment.heldout?.accessed ? "held-out accessed" : "held-out untouched"}
        </Badge>
      </div>
      <div className="overflow-x-auto border-y border-[var(--line)]">
        <table className="w-full min-w-[42rem] text-left text-xs">
          <thead className="bg-[var(--surface-subtle)] text-[var(--muted)]">
            <tr>
              <th className="px-3 py-2 font-medium">Candidate</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium">Failure evidence</th>
              <th className="px-3 py-2 font-medium">Config hash</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--line)]">
            {Object.entries(experiment.challengers).map(([name, result]) => (
              <tr key={name}>
                <td className="px-3 py-3 font-semibold text-[var(--text)]">{name}</td>
                <td className="px-3 py-3"><Badge variant="red">{words(result.status)}</Badge></td>
                <td className="px-3 py-3 text-[var(--muted)]">{result.reasons.map(words).join(", ")}</td>
                <td className="px-3 py-3 font-mono text-[var(--muted)]">{shortHash(result.config_sha256)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ArtifactAnalyst({ experiment }: { experiment: ExperimentResult }) {
  const [question, setQuestion] = useState("Why is this experiment invalid?");
  const [answer, setAnswer] = useState<string | null>(null);
  const [citations, setCitations] = useState<string[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setPending(true);
    setError(null);
    try {
      const response = await apiFetch<{
        analysis: { answer: string; citations: string[] };
      }>(`/api/research/experiments/${experiment.experiment_id}/analysis`, {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      setAnswer(response.analysis.answer);
      setCitations(response.analysis.citations);
    } catch (caught) {
      setError(String((caught as Error)?.message ?? "Analysis unavailable"));
    } finally {
      setPending(false);
    }
  }

  return (
    <section aria-labelledby="analyst-heading" className="border-y border-[var(--line)] py-5">
      <div className="grid gap-5 lg:grid-cols-[16rem_1fr]">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <h2 id="analyst-heading" className="text-base font-semibold text-[var(--text)]">Artifact analyst</h2>
            <Badge variant="muted">rules, not an agent</Badge>
          </div>
          <p className="text-xs leading-5 text-[var(--muted)]">
            Ask about the sealed manifest. Questions are not retained, no external model runs, and no decision can change.
          </p>
        </div>
        <div>
          <form onSubmit={submit} className="flex gap-2">
            <label className="sr-only" htmlFor="artifact-question">Question about this experiment</label>
            <input
              id="artifact-question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              minLength={3}
              maxLength={500}
              className="min-w-0 flex-1 rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--muted)]"
              placeholder="Why did this fail?"
            />
            <button
              type="submit"
              disabled={pending || question.trim().length < 3}
              className="inline-flex items-center gap-2 rounded-md bg-[var(--accent)] px-3 py-2 text-xs font-semibold text-[var(--bg)] disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Send className="h-4 w-4" aria-hidden="true" />
              {pending ? "Checking" : "Ask"}
            </button>
          </form>
          {error && <p className="mt-3 text-xs text-[var(--red)]" role="alert">{error}</p>}
          {answer && (
            <div className="mt-3 rounded-md bg-[var(--surface-subtle)] p-3" aria-live="polite">
              <p className="text-sm leading-6 text-[var(--text)]">{answer}</p>
              <p className="mt-2 font-mono text-[10px] text-[var(--muted)]">
                Evidence: {citations.join(", ")}
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export default function ExperimentResultsPage() {
  useEffect(() => { document.title = "Experiment Results | Trader Koo"; }, []);
  const { data, isLoading, error } = useExperimentResults();
  const [selectedId, setSelectedId] = useState<string>("");
  const experiments = data?.experiments ?? [];
  const selected = experiments.find((item) => item.experiment_id === selectedId)
    ?? experiments.find((item) => item.available)
    ?? experiments.find((item) => Object.keys(item.challengers ?? {}).length > 0)
    ?? experiments[0];

  if (isLoading) return <Spinner className="mt-12" />;
  if (error || !selected) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]" role="alert">
        Experiment results are unavailable: {String((error as Error | undefined)?.message ?? "no verified artifacts")}
      </div>
    );
  }

  const metrics = selected.metrics ?? {};
  return (
    <div className="space-y-8" data-testid="experiment-results-page">
      <header className="grid gap-4 border-b border-[var(--line)] pb-5 lg:grid-cols-[1fr_auto] lg:items-end">
        <div>
          <p className="label-xs mb-2">Research evidence</p>
          <h1 className="text-2xl font-semibold tracking-tight text-[var(--text)]">Experiment Results</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--muted)]">
            Reproducible manifests, comparable evidence, and failures that stay on the record.
          </p>
        </div>
        <nav className="flex flex-wrap gap-2" aria-label="Experiment selector">
          {experiments.map((experiment) => (
            <button
              key={experiment.experiment_id}
              type="button"
              onClick={() => setSelectedId(experiment.experiment_id)}
              aria-pressed={selected.experiment_id === experiment.experiment_id}
              className={`rounded-md border px-3 py-2 text-xs font-medium transition-colors duration-200 ${
                selected.experiment_id === experiment.experiment_id
                  ? "border-[var(--accent)] bg-[var(--panel-hover)] text-[var(--text)]"
                  : "border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {experiment.title}
            </button>
          ))}
        </nav>
      </header>

      <section className="grid gap-5 lg:grid-cols-[minmax(0,1.4fr)_minmax(18rem,.6fr)]">
        <div>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <EvidenceBadge experiment={selected} />
            <Badge variant="muted">{words(selected.status)}</Badge>
            {!selected.automatic_promotion && <Badge variant="muted">no automatic activation</Badge>}
          </div>
          <h2 className="text-xl font-semibold text-[var(--text)]">{selected.title}</h2>
          <div className="mt-4 rounded-md border border-[color-mix(in_srgb,var(--red)_32%,var(--line))] bg-[color-mix(in_srgb,var(--red)_7%,var(--panel))] p-4" role="status">
            <div className="flex gap-3">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-[var(--red)]" aria-hidden="true" />
              <div>
                <p className="text-sm font-semibold text-[var(--text)]">This result cannot support a strategy decision</p>
                <ul className="mt-2 space-y-1 text-xs leading-5 text-[var(--muted)]">
                  {selected.warnings.map((warning) => <li key={warning}>{words(warning)}</li>)}
                </ul>
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-md border border-[var(--line)] bg-[var(--panel)] p-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-[var(--text)]">
            <LockKeyhole className="h-4 w-4 text-[var(--muted)]" aria-hidden="true" />
            Evidence controls
          </div>
          <ul className="mt-3 space-y-2 text-xs text-[var(--muted)]">
            <li className="flex gap-2"><Check className="h-4 w-4 text-[var(--green)]" />Failed runs remain listed</li>
            <li className="flex gap-2"><Check className="h-4 w-4 text-[var(--green)]" />Missing metrics render as N/A</li>
            <li className="flex gap-2"><Check className="h-4 w-4 text-[var(--green)]" />Activation is unavailable here</li>
          </ul>
        </div>
      </section>

      <section aria-labelledby="metrics-heading">
        <h2 id="metrics-heading" className="mb-3 text-base font-semibold text-[var(--text)]">Comparable metrics</h2>
        <div className="grid border-y border-[var(--line)] sm:grid-cols-2 lg:grid-cols-4">
          {Object.entries(METRIC_LABELS).map(([name, label], index) => (
            <div key={name} className={`px-3 py-3 ${index % 4 !== 0 ? "lg:border-l lg:border-[var(--line)]" : ""}`}>
              <div className="text-[10px] font-medium uppercase text-[var(--muted)]">{label}</div>
              <div className="mt-1 text-sm font-semibold text-[var(--text)]">{metricValue(name, metrics[name])}</div>
            </div>
          ))}
        </div>
      </section>

      <section aria-labelledby="curves-heading">
        <div className="mb-3 flex items-center justify-between gap-4">
          <div>
            <h2 id="curves-heading" className="text-base font-semibold text-[var(--text)]">Equity and drawdown comparison</h2>
            <p className="mt-1 text-xs text-[var(--muted)]">Identical dates and starting capital are required.</p>
          </div>
        </div>
        <CurveState experiment={selected} />
      </section>

      <ChallengerTable experiment={selected} />

      <ArtifactAnalyst key={selected.experiment_id} experiment={selected} />

      <section className="grid gap-6 lg:grid-cols-[1fr_18rem]" aria-labelledby="manifest-heading">
        <div>
          <h2 id="manifest-heading" className="mb-3 text-base font-semibold text-[var(--text)]">Reproduction manifest</h2>
          <ManifestRows experiment={selected} />
        </div>
        <div>
          <h2 className="mb-3 text-base font-semibold text-[var(--text)]">Downloads</h2>
          <div className="space-y-2">
            <a href={selected.downloads.manifest} download className="flex items-center justify-between rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-xs text-[var(--text)] hover:bg-[var(--panel-hover)]">
              Manifest <Download className="h-4 w-4" aria-hidden="true" />
            </a>
            {selected.downloads.ledger ? (
              <a href={selected.downloads.ledger} download className="flex items-center justify-between rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-xs text-[var(--text)] hover:bg-[var(--panel-hover)]">
                Complete ledger <Download className="h-4 w-4" aria-hidden="true" />
              </a>
            ) : (
              <div className="rounded-md border border-dashed border-[var(--line)] px-3 py-2 text-xs text-[var(--muted)]">
                Complete ledger unavailable
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

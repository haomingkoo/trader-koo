import { useState, useCallback } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { usePipelineStatus, useTriggerUpdate } from "../api/hooks";
import { apiFetch } from "../api/client";
import type { PipelineStatus } from "../api/types";
import Badge from "./ui/Badge";

/* ── Helpers ── */

const ADMIN_KEY_STORAGE = "trader_koo_admin_key";

function getAdminKey(): string {
  try {
    return localStorage.getItem(ADMIN_KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

function setAdminKey(key: string): void {
  try {
    localStorage.setItem(ADMIN_KEY_STORAGE, key);
  } catch {
    /* localStorage unavailable */
  }
}

function hasAdminKey(): boolean {
  return getAdminKey().trim().length > 0;
}

type PipelineState = "idle" | "running" | "completed" | "warning" | "error";

type MLValidationRun = {
  run_id: string;
  created_ts: string;
  source: string;
  status: string;
  target_mode: string | null;
  model_path?: string | null;
  artifact_path?: string | null;
  avg_auc: number | null;
  backtest_return_pct: number | null;
  spy_return_pct: number | null;
  alpha_vs_spy_pct: number | null;
  rule_baseline_return_pct: number | null;
  alpha_vs_rule_baseline_pct: number | null;
  max_drawdown_pct: number | null;
  total_trades: number | null;
  profit_factor: number | null;
  champion_eligible: boolean;
  promotion_status: string;
  eligibility_reasons: string[];
};

type MLVersionCard = {
  deployment_state: string;
  summary: string;
  candidate: MLValidationRun | null;
  latest_eligible: MLValidationRun | null;
  promotion_gates: Record<string, number | boolean | string | null>;
};

type MLValidationPayload = {
  ok: boolean;
  runs: MLValidationRun[];
  champion: {
    latest_run: MLValidationRun | null;
    latest_eligible_run: MLValidationRun | null;
    promotion_gates: Record<string, number | boolean | string | null>;
  };
  model_card?: MLVersionCard;
};

function derivePipelineState(data: PipelineStatus | undefined): PipelineState {
  if (!data) return "idle";

  const run = data.latest_run;
  if (data.pipeline_active || data.pipeline?.active) return "running";

  if (run) {
    const status = (run.status ?? "").toLowerCase();
    if (status === "failed" || status === "error") return "error";
    if (status === "partial_failed" || status === "warning") return "warning";
    if (status === "completed" || status === "success" || status === "ok") {
      const failed = run.tickers_failed ?? 0;
      const ok = run.tickers_ok ?? 0;
      if (failed > 0 && ok > 0) return "warning";
      if (failed > 0 && ok === 0) return "error";
      return "completed";
    }
  }

  if (typeof data.warning_count === "number" && data.warning_count > 0) return "warning";
  if (data.errors?.latest_error_message) return "error";

  return "idle";
}

function stateColor(state: PipelineState): string {
  switch (state) {
    case "completed":
    case "idle":
      return "var(--green)";
    case "running":
    case "warning":
      return "var(--amber)";
    case "error":
      return "var(--red)";
  }
}

function stateBadgeVariant(state: PipelineState): "green" | "amber" | "red" | "muted" {
  switch (state) {
    case "completed":
    case "idle":
      return "green";
    case "running":
    case "warning":
      return "amber";
    case "error":
      return "red";
  }
}

function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return "\u2014";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function formatPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "\u2014";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatNum(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return "\u2014";
  return value.toFixed(digits);
}

function fileName(path: string | null | undefined): string {
  if (!path) return "none";
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}

function versionCardVariant(state: string): "green" | "amber" | "red" | "muted" {
  if (state === "latest_candidate_eligible") return "green";
  if (state === "using_previous_eligible") return "amber";
  if (state === "no_validation_runs") return "muted";
  return "red";
}

function parseFailedTickers(errorMsg: string | null | undefined): string[] {
  if (!errorMsg) return [];
  // Try to extract ticker symbols like ^VIX, ^VIX3M from error messages
  const matches = errorMsg.match(/\^[A-Z0-9]+|[A-Z]{1,5}(?=\s*(?:failed|error|timeout))/gi);
  return matches ?? [];
}

function freshnessLabel(days: number | null | undefined, unit: string): string {
  if (days == null) return "\u2014";
  if (unit === "days") {
    if (days <= 1) return `${days.toFixed(1)}d (fresh)`;
    if (days <= 3) return `${days.toFixed(1)}d`;
    return `${days.toFixed(1)}d (stale)`;
  }
  // hours
  if (days <= 24) return `${days.toFixed(1)}h (fresh)`;
  if (days <= 72) return `${days.toFixed(1)}h`;
  return `${days.toFixed(1)}h (stale)`;
}

function freshnessColor(age: number | null | undefined, thresholdFresh: number, thresholdStale: number): string {
  if (age == null) return "var(--muted)";
  if (age <= thresholdFresh) return "var(--green)";
  if (age <= thresholdStale) return "var(--amber)";
  return "var(--red)";
}

/* ── API Key Input ── */

function ApiKeyInput({ onKeyChange }: { onKeyChange?: () => void }) {
  const [key, setKey] = useState(getAdminKey);
  const hasKey = key.length > 0;
  const [visible, setVisible] = useState(hasKey);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span
          className="text-sm"
          title={hasKey ? "API key is set" : "No API key configured"}
          aria-hidden="true"
        >
          {hasKey ? "\uD83D\uDD13" : "\uD83D\uDD12"}
        </span>
        <button
          type="button"
          onClick={() => setVisible((p) => !p)}
          className="rounded-md border border-[var(--line)] bg-[var(--bg)] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text)]"
        >
          {visible ? "Hide Admin Key" : hasKey ? "Admin Unlocked" : "Unlock Admin"}
        </button>
        <span className="text-[10px] text-[var(--muted)]">
          {hasKey ? "Key stored locally" : "Required for actions"}
        </span>
      </div>
      {visible && (
        <input
          type="password"
          value={key}
          onChange={(e) => {
            setKey(e.target.value);
            setAdminKey(e.target.value);
            onKeyChange?.();
          }}
          placeholder="Admin API key"
          className="h-8 w-full max-w-xs rounded-md border border-[var(--line)] bg-[var(--bg)] px-2 text-xs text-[var(--text)] placeholder:text-[var(--muted)] focus:border-[var(--accent)] focus:outline-none"
        />
      )}
    </div>
  );
}

function useMLValidationRuns(enabled: boolean) {
  return useQuery({
    queryKey: ["ml-validation-runs"],
    queryFn: () =>
      apiFetch<MLValidationPayload>("/api/admin/ml/validation-runs?limit=5", {
        headers: { "X-API-Key": getAdminKey() },
      }),
    enabled,
    refetchInterval: 60_000,
    staleTime: 30_000,
    retry: false,
  });
}

/* ── Action Button ── */

interface ActionButtonProps {
  label: string;
  mode: "full" | "yolo" | "report";
  disabled: boolean;
}

function ActionButton({ label, mode, disabled }: ActionButtonProps) {
  const trigger = useTriggerUpdate();
  const [feedback, setFeedback] = useState<{ ok: boolean; msg: string } | null>(null);

  const handleClick = useCallback(() => {
    setFeedback(null);
    trigger.mutate(
      { mode, apiKey: getAdminKey() },
      {
        onSuccess: (msg) => setFeedback({ ok: true, msg }),
        onError: (err) => setFeedback({ ok: false, msg: err instanceof Error ? err.message : String(err) }),
      },
    );
  }, [trigger, mode]);

  const isLoading = trigger.isPending;

  return (
    <div className="flex flex-col gap-1">
      <button
        onClick={handleClick}
        disabled={disabled || isLoading || !getAdminKey()}
        className="rounded-md border border-[var(--line)] bg-[var(--panel-hover)] px-3 py-1.5 text-xs font-medium text-[var(--text)] transition-colors hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        {isLoading ? "Running\u2026" : label}
      </button>
      {feedback && (
        <span
          className={`text-[10px] ${feedback.ok ? "text-[var(--green)]" : "text-[var(--red)]"}`}
        >
          {feedback.msg.slice(0, 120)}
        </span>
      )}
    </div>
  );
}

/* ── Status Section ── */

function StatusSection({ data, state }: { data: PipelineStatus; state: PipelineState }) {
  const run = data.latest_run;
  const pipeline = data.pipeline;
  const freshness = data.freshness;
  const tickersOk = run?.tickers_ok ?? 0;
  const tickersFailed = run?.tickers_failed ?? 0;
  const tickersTotal = run?.tickers_total ?? 0;
  const issueMessage =
    (typeof run?.error_message === "string" && run.error_message.length > 0
      ? run.error_message
      : data.errors?.latest_error_message) ?? null;
  const failedNames = parseFailedTickers(issueMessage);

  return (
    <div className="space-y-3">
      {/* State badge + stage */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: stateColor(state) }}
          />
          <Badge variant={stateBadgeVariant(state)}>
            {state.toUpperCase()}
          </Badge>
        </div>
        {pipeline?.stage && (
          <span className="text-xs text-[var(--muted)]">
            Stage: <strong className="text-[var(--text)]">{pipeline.stage}</strong>
          </span>
        )}
      </div>

      {/* Latest run details */}
      {run && (
        <div className="grid gap-2 text-xs text-[var(--muted)] sm:grid-cols-2 lg:grid-cols-3">
          <div>
            Started: <strong className="text-[var(--text)]">{formatTimestamp(run.started_ts)}</strong>
          </div>
          <div>
            Finished: <strong className="text-[var(--text)]">{formatTimestamp(run.finished_ts)}</strong>
          </div>
          <div>
            Status: <strong className="text-[var(--text)]">{run.status}</strong>
          </div>
        </div>
      )}

      {/* Ticker stats */}
      {tickersTotal > 0 && (
        <div className="text-xs text-[var(--muted)]">
          Tickers:{" "}
          <strong className="text-[var(--text)]">{tickersTotal}</strong> total,{" "}
          <strong className="text-[var(--green)]">{tickersOk}</strong> ok,{" "}
          <strong className={tickersFailed > 0 ? "text-[var(--amber)]" : "text-[var(--green)]"}>
            {tickersFailed}
          </strong>{" "}
          failed
          {tickersFailed > 0 && tickersOk > 0 && failedNames.length > 0 && (
            <span className="ml-1 text-[var(--amber)]">
              ({failedNames.join(", ")})
            </span>
          )}
        </div>
      )}

      {/* Error message */}
      {issueMessage && state !== "completed" && (
        <div className="rounded-md border border-[var(--red)]/30 bg-[var(--red)]/5 px-3 py-2 text-xs text-[var(--red)]">
          {issueMessage.slice(0, 300)}
          {data.errors.latest_error_ts && (
            <span className="ml-2 text-[var(--muted)]">
              ({formatTimestamp(data.errors.latest_error_ts)})
            </span>
          )}
        </div>
      )}

      {/* Report generation time */}
      {pipeline?.last_completed_ts && (
        <div className="text-xs text-[var(--muted)]">
          Last completed:{" "}
          <strong className="text-[var(--text)]">
            {pipeline.last_completed_stage ?? "unknown"}
          </strong>{" "}
          at {formatTimestamp(pipeline.last_completed_ts)}
        </div>
      )}

      {/* Data freshness */}
      {freshness && (
        <div className="flex flex-wrap gap-4 text-xs">
          <div>
            Price age:{" "}
            <strong style={{ color: freshnessColor(freshness.price_age_days, 1, 3) }}>
              {freshnessLabel(freshness.price_age_days, "days")}
            </strong>
          </div>
          <div>
            Fundamentals age:{" "}
            <strong style={{ color: freshnessColor(freshness.fund_age_hours, 24, 72) }}>
              {freshnessLabel(freshness.fund_age_hours, "hours")}
            </strong>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Events / Issue Log ── */

function EventsLog({ data }: { data: PipelineStatus }) {
  const events: Array<{ text: string; variant: "amber" | "red" | "muted" }> = [];

  // Warnings
  if (Array.isArray(data.warnings)) {
    for (const w of data.warnings) {
      if (typeof w === "string" && w.length > 0) {
        events.push({ text: w, variant: "amber" });
      }
    }
  }

  // Latest error
  if (typeof data.errors?.latest_error_message === "string" && data.errors.latest_error_message.length > 0) {
    events.push({
      text: `Error: ${data.errors.latest_error_message.slice(0, 200)}`,
      variant: "red",
    });
  }

  // Last completed stage
  if (typeof data.pipeline?.last_completed_stage === "string") {
    events.push({
      text: `Completed: ${data.pipeline.last_completed_stage} [${data.pipeline.last_completed_status ?? "?"}] at ${formatTimestamp(data.pipeline.last_completed_ts)}`,
      variant: "muted",
    });
  }

  if (events.length === 0) {
    return (
      <p className="text-xs text-[var(--muted)]">No recent events.</p>
    );
  }

  return (
    <ul className="space-y-1.5">
      {events.map((ev, i) => (
        <li key={i} className="flex items-start gap-2 text-xs">
          <span
            className="mt-1 inline-block h-1.5 w-1.5 shrink-0 rounded-full"
            style={{
              backgroundColor:
                ev.variant === "red"
                  ? "var(--red)"
                  : ev.variant === "amber"
                    ? "var(--amber)"
                    : "var(--muted)",
            }}
          />
          <span className="text-[var(--muted)]">{ev.text}</span>
        </li>
      ))}
    </ul>
  );
}

function ValidationMetric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "green" | "amber" | "red";
}) {
  const color =
    tone === "green"
      ? "text-[var(--green)]"
      : tone === "amber"
        ? "text-[var(--amber)]"
        : tone === "red"
          ? "text-[var(--red)]"
          : "text-[var(--text)]";
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-[var(--muted)]">{label}</div>
      <div className={`mt-1 text-sm font-semibold ${color}`}>{value}</div>
    </div>
  );
}

function MLValidationStatus({ enabled }: { enabled: boolean }) {
  const { data, isLoading, error } = useMLValidationRuns(enabled);

  if (!enabled) return null;

  if (isLoading) {
    return (
      <div className="rounded-md border border-[var(--line)] bg-[var(--bg)]/45 px-3 py-2 text-xs text-[var(--muted)]">
        Loading model validation status...
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-[var(--red)]/30 bg-[var(--red)]/5 px-3 py-2 text-xs text-[var(--red)]">
        Model validation status unavailable: {error instanceof Error ? error.message : String(error)}
      </div>
    );
  }

  const latest = data?.champion?.latest_run ?? null;
  const eligible = data?.champion?.latest_eligible_run ?? null;
  const modelCard = data?.model_card ?? null;
  const reasons = latest?.eligibility_reasons ?? [];

  if (!latest) {
    return (
      <div className="rounded-md border border-[var(--line)] bg-[var(--bg)]/45 px-3 py-3 text-xs text-[var(--muted)]">
        No model validation runs recorded yet.
      </div>
    );
  }

  const eligibleNow = latest.champion_eligible;
  const alphaRule = latest.alpha_vs_rule_baseline_pct;
  const alphaSpy = latest.alpha_vs_spy_pct;

  return (
    <div className="rounded-md border border-[var(--line)] bg-[var(--bg)]/35 p-3">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h4 className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          Model Promotion Gate
        </h4>
        <Badge variant={eligibleNow ? "green" : "amber"}>
          {eligibleNow ? "ELIGIBLE" : "BLOCKED"}
        </Badge>
        {eligible && !eligibleNow && (
          <span className="text-[10px] text-[var(--muted)]">
            Latest eligible: {eligible.run_id}
          </span>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <ValidationMetric label="Avg AUC" value={formatNum(latest.avg_auc, 3)} />
        <ValidationMetric
          label="Model Return"
          value={formatPct(latest.backtest_return_pct)}
          tone={(latest.backtest_return_pct ?? 0) >= 0 ? "green" : "red"}
        />
        <ValidationMetric
          label="Alpha vs SPY"
          value={formatPct(alphaSpy)}
          tone={(alphaSpy ?? 0) >= 0 ? "green" : "red"}
        />
        <ValidationMetric label="Rule Baseline" value={formatPct(latest.rule_baseline_return_pct)} />
        <ValidationMetric
          label="Alpha vs Rules"
          value={formatPct(alphaRule)}
          tone={(alphaRule ?? 0) > 0 ? "green" : "red"}
        />
        <ValidationMetric label="Trades / PF" value={`${latest.total_trades ?? 0} / ${formatNum(latest.profit_factor)}`} />
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-[10px] text-[var(--muted)]">
        <span>Run: {latest.run_id}</span>
        <span>Source: {latest.source}</span>
        <span>Target: {latest.target_mode ?? "unknown"}</span>
        <span>Model: {fileName(latest.model_path)}</span>
        <span>Artifact: {fileName(latest.artifact_path)}</span>
        <span>Drawdown: {formatPct(latest.max_drawdown_pct)}</span>
        <span>{formatTimestamp(latest.created_ts)}</span>
      </div>

      {modelCard && (
        <div className="mt-3 rounded-md border border-[var(--line)] bg-[var(--panel)]/60 px-3 py-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
              Version card
            </span>
            <Badge variant={versionCardVariant(modelCard.deployment_state)}>
              {modelCard.deployment_state.replaceAll("_", " ")}
            </Badge>
          </div>
          <p className="mt-1 text-xs text-[var(--muted)]">{modelCard.summary}</p>
          <div className="mt-2 flex flex-wrap gap-2 text-[10px] text-[var(--muted)]">
            <span>Candidate: {modelCard.candidate?.run_id ?? "none"}</span>
            <span>Latest eligible: {modelCard.latest_eligible?.run_id ?? "none"}</span>
            <span>Min AUC: {formatNum(Number(modelCard.promotion_gates.min_avg_auc), 2)}</span>
            <span>Min trades: {String(modelCard.promotion_gates.min_total_trades ?? "n/a")}</span>
            <span>Rule baseline required</span>
          </div>
        </div>
      )}

      {reasons.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {reasons.map((reason) => (
            <span
              key={reason}
              className="rounded-md border border-[var(--amber)]/30 bg-[var(--amber)]/10 px-2 py-1 text-[10px] text-[var(--amber)]"
            >
              {reason.replaceAll("_", " ")}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Full Panel (for GuidePage) ── */

export default function PipelineOpsPanel() {
  const { data, isLoading } = usePipelineStatus();
  const queryClient = useQueryClient();
  const [, setAdminKeyRevision] = useState(0);

  const state = derivePipelineState(data);
  const isRunning = state === "running";
  const adminUnlocked = hasAdminKey();

  if (isLoading || !data) {
    return (
      <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
        <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-[var(--line)] border-t-[var(--accent)]" />
        Loading pipeline status...
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* API Key input */}
      <ApiKeyInput onKeyChange={() => setAdminKeyRevision((value) => value + 1)} />

      {/* Status */}
      <StatusSection data={data} state={state} />

      {/* Action buttons */}
      {adminUnlocked ? (
        <div className="space-y-2">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            Admin Controls
          </div>
          <div className="flex flex-wrap gap-3">
            <ActionButton label="Run Full Update" mode="full" disabled={isRunning} />
            <ActionButton label="Rerun YOLO + Report" mode="yolo" disabled={isRunning} />
            <ActionButton label="Rebuild Report Only" mode="report" disabled={isRunning} />
            <button
              onClick={() => {
                void queryClient.invalidateQueries({ queryKey: ["pipeline-status"] });
                void queryClient.invalidateQueries({ queryKey: ["ml-validation-runs"] });
              }}
              className="rounded-md border border-[var(--line)] bg-[var(--panel-hover)] px-3 py-1.5 text-xs font-medium text-[var(--muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text)]"
            >
              Refresh Status
            </button>
          </div>
        </div>
      ) : (
        <div className="rounded-md border border-[var(--line)] bg-[var(--bg)]/45 px-3 py-2 text-xs text-[var(--muted)]">
          Status auto-refreshes every 30 seconds. Admin actions stay hidden until a valid API key is set locally.
        </div>
      )}

      {/* Recent Events */}
      {adminUnlocked && <MLValidationStatus enabled={adminUnlocked} />}

      <div>
        <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          Recent Events
        </h4>
        <EventsLog data={data} />
      </div>
    </div>
  );
}

/* ── Inline Mini Panel (for ReportPage) ── */

export function PipelineStatusInline() {
  const { data, isLoading } = usePipelineStatus();

  const state = derivePipelineState(data);

  if (isLoading || !data) {
    return null;
  }

  const errorMsg =
    (typeof data.latest_run?.error_message === "string" && data.latest_run.error_message.length > 0
      ? data.latest_run.error_message
      : data.errors?.latest_error_message) ?? null;

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-4 py-2">
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-2 w-2 rounded-full"
          style={{ backgroundColor: stateColor(state) }}
        />
        <span className="text-xs font-medium text-[var(--text)]">
          Pipeline: {state}
        </span>
      </div>
      {typeof errorMsg === "string" && errorMsg.length > 0 && state !== "completed" && (
        <span className="text-xs text-[var(--red)]">
          {errorMsg.slice(0, 100)}{errorMsg.length > 100 ? "\u2026" : ""}
        </span>
      )}
      {data.latest_data?.price_date && (
        <span className="text-xs text-[var(--muted)]">
          Raw DB price date: {data.latest_data.price_date}
        </span>
      )}
      <span className="ml-auto text-[10px] uppercase tracking-wider text-[var(--muted)]">
        Auto-refreshing
      </span>
    </div>
  );
}

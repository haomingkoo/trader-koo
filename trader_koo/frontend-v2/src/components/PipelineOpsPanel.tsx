import { useState, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { usePipelineStatus, useTriggerUpdate } from "../api/hooks";
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

type PipelineState = "idle" | "running" | "completed" | "warning" | "error";

function derivePipelineState(data: PipelineStatus | undefined): PipelineState {
  if (!data) return "idle";

  const run = data.latest_run;
  if (data.pipeline_active || data.pipeline?.active) return "running";

  if (run) {
    const status = (run.status ?? "").toLowerCase();
    if (status === "failed" || status === "error") return "error";
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

function ApiKeyInput() {
  const [key, setKey] = useState(getAdminKey);
  const hasKey = key.length > 0;

  return (
    <div className="flex items-center gap-2">
      <span
        className="text-sm"
        title={hasKey ? "API key is set" : "No API key configured"}
        aria-hidden="true"
      >
        {hasKey ? "\uD83D\uDD13" : "\uD83D\uDD12"}
      </span>
      <input
        type="password"
        value={key}
        onChange={(e) => {
          setKey(e.target.value);
          setAdminKey(e.target.value);
        }}
        placeholder="Admin API key"
        className="h-7 w-48 rounded-md border border-[var(--line)] bg-[var(--bg)] px-2 text-xs text-[var(--text)] placeholder:text-[var(--muted)] focus:border-[var(--accent)] focus:outline-none"
      />
      <span className="text-[10px] text-[var(--muted)]">
        {hasKey ? "Key set" : "Required for actions"}
      </span>
    </div>
  );
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
  const failedNames = parseFailedTickers(data.errors?.latest_error_message);

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
      {data.errors?.latest_error_message && state !== "completed" && (
        <div className="rounded-md border border-[var(--red)]/30 bg-[var(--red)]/5 px-3 py-2 text-xs text-[var(--red)]">
          {data.errors.latest_error_message.slice(0, 300)}
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

/* ── Full Panel (for GuidePage) ── */

export default function PipelineOpsPanel() {
  const { data, isLoading } = usePipelineStatus();
  const queryClient = useQueryClient();

  const state = derivePipelineState(data);
  const isRunning = state === "running";

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
      <ApiKeyInput />

      {/* Status */}
      <StatusSection data={data} state={state} />

      {/* Action buttons */}
      <div className="flex flex-wrap gap-3">
        <ActionButton label="Run Full Update" mode="full" disabled={isRunning} />
        <ActionButton label="Rerun YOLO + Report" mode="yolo" disabled={isRunning} />
        <ActionButton label="Rebuild Report Only" mode="report" disabled={isRunning} />
        <button
          onClick={() => queryClient.invalidateQueries({ queryKey: ["pipeline-status"] })}
          className="rounded-md border border-[var(--line)] bg-[var(--panel-hover)] px-3 py-1.5 text-xs font-medium text-[var(--muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text)]"
        >
          Refresh Status
        </button>
      </div>

      {/* Recent Events */}
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
  const queryClient = useQueryClient();

  const state = derivePipelineState(data);

  if (isLoading || !data) {
    return null;
  }

  const errorMsg = data.errors?.latest_error_message;

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
      <button
        onClick={() => queryClient.invalidateQueries({ queryKey: ["pipeline-status"] })}
        className="ml-auto text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
      >
        Refresh
      </button>
    </div>
  );
}

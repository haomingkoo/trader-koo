import type { PipelineStatus } from "../../api/types";
import { derivePipelineStates, type StageState } from "./pipelineStageStates";

function PipelineDot({ state }: { state: StageState }) {
  const colors: Record<string, string> = {
    idle: "bg-[var(--line)]",
    active: "bg-[var(--amber)] animate-pulse",
    done: "bg-[var(--green)]",
    warning: "bg-[var(--amber)]",
    error: "bg-[var(--red)]",
  };

  return (
    <div
      className={`h-2.5 w-2.5 rounded-full ${colors[state]}`}
      title={state}
      aria-label={`Pipeline stage: ${state}`}
    />
  );
}

export default function PipelineStatusBadge({
  data,
}: {
  data: PipelineStatus | undefined;
}) {
  let states: { ingest: StageState; yolo: StageState; report: StageState } = {
    ingest: "idle",
    yolo: "idle",
    report: "idle",
  };
  try {
    if (data && typeof data === "object" && data.pipeline && typeof data.pipeline === "object") {
      states = derivePipelineStates(data);
    }
  } catch {
    // Shape mismatch — use idle defaults.
  }

  const pipelineHint = data?.errors?.latest_error_message
    ? `Latest pipeline issue: ${data.errors.latest_error_message}`
    : "Pipeline status: Ingest, YOLO, Report";

  return (
    <div
      className="flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5"
      title={pipelineHint}
    >
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
        Pipeline
      </span>
      <div className="flex items-center gap-1.5" aria-label="Pipeline status: Ingest, YOLO, Report">
        <span className="flex items-center gap-1 text-[11px] text-[var(--muted)]">
          <PipelineDot state={states.ingest} /> Ingest
        </span>
        <span className="text-[var(--muted)]" aria-hidden="true">&rarr;</span>
        <span className="flex items-center gap-1 text-[11px] text-[var(--muted)]">
          <PipelineDot state={states.yolo} /> YOLO
        </span>
        <span className="text-[var(--muted)]" aria-hidden="true">&rarr;</span>
        <span className="flex items-center gap-1 text-[11px] text-[var(--muted)]">
          <PipelineDot state={states.report} /> Report
        </span>
      </div>
    </div>
  );
}

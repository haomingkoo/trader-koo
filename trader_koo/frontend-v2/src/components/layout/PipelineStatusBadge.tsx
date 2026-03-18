import type { PipelineStatus } from "../../api/types";

type StageState = "idle" | "active" | "done" | "warning" | "error";

function PipelineDot({ state }: { state: StageState }) {
  const colors: Record<string, string> = {
    idle: "bg-[var(--line)]",
    active: "bg-[var(--amber)] animate-pulse",
    done: "bg-[var(--green)]",
    warning: "bg-[var(--amber)]",
    error: "bg-[var(--red)]",
  };
  const labels: Record<string, string> = {
    idle: "idle",
    active: "active",
    done: "done",
    warning: "warning",
    error: "error",
  };

  return (
    <div
      className={`h-2.5 w-2.5 rounded-full ${colors[state]}`}
      title={labels[state]}
      aria-label={`Pipeline stage: ${labels[state]}`}
    />
  );
}

function derivePipelineStates(data: Pick<PipelineStatus, "pipeline" | "latest_run" | "errors">) {
  const stage = (data.pipeline?.stage ?? "idle").toLowerCase();
  const runStatus = (data.latest_run?.status ?? "").toLowerCase();
  const lastCompletedStage = (data.pipeline?.last_completed_stage ?? "").toLowerCase();
  const lastCompletedStatus = (data.pipeline?.last_completed_status ?? "").toLowerCase();
  const pipelineActive = Boolean(data.pipeline?.active);
  const runningStale = Boolean(data.pipeline?.running_stale);
  const tickersOk = Number(data.latest_run?.tickers_ok ?? 0);
  const tickersFailed = Number(data.latest_run?.tickers_failed ?? 0);
  const partialFailure = runStatus === "failed" && tickersOk > 0 && tickersFailed > 0;

  const ingestStages = ["price_daily", "price_seed", "fundamentals", "ingest"];
  const yoloStages = ["yolo", "yolo_batch", "patterns"];
  const reportStages = ["report", "narrative", "scoring", "daily_report"];

  let ingest: StageState = "idle";
  let yolo: StageState = "idle";
  let report: StageState = "idle";

  const markCompletedThrough = (completedStage: string) => {
    if (
      reportStages.some((item) => completedStage.includes(item)) ||
      completedStage.includes("daily_update")
    ) {
      ingest = "done";
      yolo = "done";
      report = "done";
    } else if (yoloStages.some((item) => completedStage.includes(item))) {
      ingest = "done";
      yolo = "done";
    } else if (ingestStages.some((item) => completedStage.includes(item))) {
      ingest = "done";
    }
  };

  if (runningStale || stage === "stale_running") {
    ingest = "error";
    return { ingest, yolo, report };
  }

  if (pipelineActive || runStatus === "running" || runStatus === "in_progress") {
    if (ingestStages.some((item) => stage.includes(item))) {
      ingest = "active";
    } else if (yoloStages.some((item) => stage.includes(item))) {
      ingest = "done";
      yolo = "active";
    } else if (reportStages.some((item) => stage.includes(item))) {
      ingest = "done";
      yolo = "done";
      report = "active";
    } else {
      ingest = "active";
    }
    return { ingest, yolo, report };
  }

  const lastCompletedOk =
    lastCompletedStatus === "ok" ||
    lastCompletedStatus === "completed" ||
    lastCompletedStatus === "done";

  const latestRunOk =
    runStatus === "ok" ||
    runStatus === "completed" ||
    runStatus === "done";

  if (lastCompletedOk) {
    markCompletedThrough(lastCompletedStage);
  } else if (latestRunOk) {
    ingest = "done";
  } else if (runStatus === "failed" && lastCompletedStatus === "failed") {
    if (reportStages.some((item) => lastCompletedStage.includes(item))) {
      ingest = "done";
      yolo = "done";
      report = "error";
    } else if (yoloStages.some((item) => lastCompletedStage.includes(item))) {
      ingest = "done";
      yolo = partialFailure ? "warning" : "error";
    } else {
      ingest = partialFailure ? "warning" : "error";
    }
  }

  return { ingest, yolo, report };
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
      <div className="flex items-center gap-1" aria-label="Pipeline status: Ingest, YOLO, Report">
        <PipelineDot state={states.ingest} />
        <span className="text-[var(--line)]" aria-hidden="true">&rarr;</span>
        <PipelineDot state={states.yolo} />
        <span className="text-[var(--line)]" aria-hidden="true">&rarr;</span>
        <PipelineDot state={states.report} />
      </div>
    </div>
  );
}

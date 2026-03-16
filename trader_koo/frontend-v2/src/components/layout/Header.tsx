import ClockStrip from "./ClockStrip";
import { usePipelineStatus } from "../../api/hooks";

function PipelineDot({ state }: { state: "idle" | "active" | "done" | "error" }) {
  const colors: Record<string, string> = {
    idle: "bg-[var(--line)]",
    active: "bg-[var(--amber)] animate-pulse",
    done: "bg-[var(--green)]",
    error: "bg-[var(--red)]",
  };
  return <div className={`h-2.5 w-2.5 rounded-full ${colors[state]}`} />;
}

function derivePipelineStates(data: { pipeline: { stage: string }; latest_run: { status: string } }) {
  const stage = (data.pipeline?.stage ?? "idle").toLowerCase();
  const runStatus = (data.latest_run?.status ?? "").toLowerCase();

  const ingestStages = ["price_daily", "price_seed", "fundamentals", "ingest"];
  const yoloStages = ["yolo", "yolo_batch", "patterns"];
  const reportStages = ["report", "narrative", "scoring", "daily_report"];

  type StageState = "idle" | "active" | "done" | "error";
  let ingest: StageState = "idle";
  let yolo: StageState = "idle";
  let report: StageState = "idle";

  if (runStatus === "running" || runStatus === "in_progress") {
    if (ingestStages.some((s) => stage.includes(s))) {
      ingest = "active";
    } else if (yoloStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "active";
    } else if (reportStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "done";
      report = "active";
    } else {
      ingest = "active";
    }
  } else if (runStatus === "completed" || runStatus === "done") {
    ingest = "done";
    yolo = "done";
    report = "done";
  } else if (runStatus === "failed") {
    if (reportStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "done";
      report = "error";
    } else if (yoloStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "error";
    } else {
      ingest = "error";
    }
  }

  return { ingest, yolo, report };
}

export default function Header() {
  const { data } = usePipelineStatus();
  const states = data
    ? derivePipelineStates(data)
    : { ingest: "idle" as const, yolo: "idle" as const, report: "idle" as const };

  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-3">
      <div className="flex items-center gap-4">
        <h1 className="text-base font-bold tracking-wide text-[var(--text)]">
          trader_koo
        </h1>
        <ClockStrip />
      </div>
      <div className="flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Pipeline
        </span>
        <div className="flex items-center gap-1">
          <PipelineDot state={states.ingest} />
          <span className="text-[var(--line)]">&rarr;</span>
          <PipelineDot state={states.yolo} />
          <span className="text-[var(--line)]">&rarr;</span>
          <PipelineDot state={states.report} />
        </div>
      </div>
    </header>
  );
}

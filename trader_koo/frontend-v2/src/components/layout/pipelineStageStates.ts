import type { PipelineStatus } from "../../api/types";

export type StageState = "idle" | "active" | "done" | "warning" | "error";

export function derivePipelineStates(data: PipelineStatus) {
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

  // daily_update.sh's last line carries every stage's exit code: [DONE], [WARN] (partial) or [FAIL].
  const summary = /ingest=(\d+) yolo=(\d+) report=(\d+)/.exec(data.pipeline?.last_completed_line ?? "");
  if (lastCompletedStage.includes("daily_update") && summary) {
    const fromRc = (rc: string): StageState => (rc === "0" ? "done" : "error");
    ingest = fromRc(summary[1]);
    yolo = fromRc(summary[2]);
    report = fromRc(summary[3]);
  } else if (lastCompletedOk) {
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

  if (!pipelineActive && data.freshness?.report_fresh === false) {
    report = "error";
  }

  return { ingest, yolo, report };
}

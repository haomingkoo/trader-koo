import type { PipelineStatus } from "../api/types";

export type PipelineState = "idle" | "running" | "completed" | "warning" | "error";

export function derivePipelineState(data: PipelineStatus | undefined): PipelineState {
  if (!data) return "idle";
  if (data.pipeline_active || data.pipeline?.active) return "running";

  const run = data.latest_run;
  if (run) {
    const status = (run.status ?? "").toLowerCase();
    if (status === "failed" || status === "error") return "error";
    if (status === "partial_failed" || status === "warning") return "warning";
    if (status === "completed" || status === "success" || status === "ok") {
      const failed = run.tickers_failed ?? 0;
      const ok = run.tickers_ok ?? 0;
      if (failed > 0 && ok === 0) return "error";
      if (failed > 0 && ok > 0) return "warning";
      if ((data.operational_warning_count ?? 0) > 0) return "warning";
      return "completed";
    }
  }

  const operationalWarnings =
    data.operational_warning_count ?? data.warning_count ?? 0;
  if (operationalWarnings > 0) return "warning";
  if (data.errors?.latest_error_message) return "error";
  return "idle";
}

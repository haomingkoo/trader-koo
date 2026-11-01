import test from "node:test";
import assert from "node:assert/strict";

import type { PipelineStatus } from "../../api/types.ts";
import { derivePipelineStates } from "./pipelineStageStates.ts";

function status(lastStage: string, lastStatus: string, lastLine: string, runStatus = "ok"): PipelineStatus {
  return {
    pipeline: {
      active: false,
      stage: "idle",
      last_completed_stage: lastStage,
      last_completed_status: lastStatus,
      last_completed_line: lastLine,
    },
    latest_run: { status: runStatus },
    freshness: { report_fresh: true },
  } as unknown as PipelineStatus;
}

test("a clean run shows every stage done", () => {
  const line = "2026-10-22T22:30:00+0000 [DONE]  daily_update.sh mode=full ingest=0 yolo=0 report=0";
  assert.deepEqual(derivePipelineStates(status("daily_update", "ok", line)), { ingest: "done", yolo: "done", report: "done" });
});

test("a partial run marks only the stage that failed", () => {
  const line = "2026-10-22T22:30:00+0000 [WARN]  daily_update.sh mode=full ingest=0 yolo=1 report=0 (partial failures, report OK)";
  assert.deepEqual(derivePipelineStates(status("daily_update", "partial", line)), { ingest: "done", yolo: "error", report: "done" });
});

test("a failed report shows red even when ingest succeeded", () => {
  const line = "2026-10-22T22:30:00+0000 [FAIL]  daily_update.sh mode=full ingest=0 yolo=0 report=1";
  assert.deepEqual(derivePipelineStates(status("daily_update", "failed", line, "ok")), { ingest: "done", yolo: "done", report: "error" });
});

test("a finished YOLO step counts through YOLO", () => {
  const line = "2026-10-22T22:20:00+0000 [YOLO]  Daily pattern detection done. rc=0 sec=300";
  assert.deepEqual(derivePipelineStates(status("yolo", "ok", line)), { ingest: "done", yolo: "done", report: "idle" });
});

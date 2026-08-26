import assert from "node:assert/strict";
import test from "node:test";

import type { PipelineStatus } from "../api/types.ts";
import { derivePipelineState } from "./pipelineState.ts";

function status(overrides: Partial<PipelineStatus>): PipelineStatus {
  return {
    ok: true,
    service: "trader_koo-api",
    now_utc: "2026-08-26T00:00:00Z",
    db_exists: true,
    warnings: [],
    warning_count: 0,
    pipeline_active: false,
    pipeline_stage: "idle",
    pipeline: { active: false } as PipelineStatus["pipeline"],
    latest_run: null,
    counts: { tracked_tickers: 0, price_rows: 0, fundamentals_rows: 0, options_rows: 0 },
    ...overrides,
  } as PipelineStatus;
}

test("research-only warnings do not turn operational pipeline amber", () => {
  const data = status({
    warnings: ["price basis unresolved"],
    warning_count: 1,
    operational_warning_count: 0,
    research_warning_count: 1,
  });

  assert.equal(derivePipelineState(data), "idle");
});

test("operational warnings remain visible after a completed ingest", () => {
  const data = status({
    operational_warning_count: 1,
    latest_run: {
      status: "ok",
      tickers_ok: 536,
      tickers_failed: 0,
    } as PipelineStatus["latest_run"],
  });

  assert.equal(derivePipelineState(data), "warning");
});

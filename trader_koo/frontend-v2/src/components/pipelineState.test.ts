import assert from "node:assert/strict";
import test from "node:test";

import type { PipelineStatus } from "../api/types.ts";
import { derivePipelineState, priceBasisStatusCopy } from "./pipelineState.ts";

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

test("price basis copy distinguishes canonical seals from retained history", () => {
  const priceBasis = {
    cohort_available: true,
    cohort_source: "latest_canonical_sp500_price_ingest",
    cohort_run_id: "run-1",
    cohort_finished_ts: "2026-08-26T00:00:00Z",
    cohort_tickers: 536,
    verified_tickers: 495,
    unresolved_tickers: 41,
    bases: [],
    retained_history: {
      ticker_count: 552,
      verified_tickers: 495,
      unresolved_tickers: 57,
      revision_tickers: 537,
      missing_revision_tickers: 15,
      bases: [],
    },
  } as PipelineStatus["price_basis"];

  assert.deepEqual(priceBasisStatusCopy(priceBasis), {
    cohort: "Current canonical cohort with verified persisted seals: 495 / 536",
    unresolved: "Current unresolved or unsealed: 41",
    retained: "Retained history: 552 symbols · 15 missing seal(s)",
  });
  assert.deepEqual(priceBasisStatusCopy({ ...priceBasis, cohort_available: false }), {
    cohort: "Current canonical cohort seal status unavailable",
    unresolved: "Current unresolved count unavailable",
    retained: "Retained history: 552 symbols · 15 missing seal(s)",
  });
});

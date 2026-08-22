import test from "node:test";
import assert from "node:assert/strict";

import {
  isResearchChartEligible,
  resampleToMonthly,
  resampleToWeekly,
} from "./buildEquityChartData.ts";

const row = (date: string, close: number) => ({
  date,
  open: close,
  high: close + 1,
  low: close - 1,
  close,
  volume: 100,
});

test("completed weekly bars exclude the current partial week", () => {
  const rows = [
    row("2026-08-14", 100),
    row("2026-08-17", 101),
    row("2026-08-18", 102),
    row("2026-08-19", 103),
  ];

  const bars = resampleToWeekly(rows, true, new Date("2026-08-19T12:00:00Z"));

  assert.deepEqual(bars.map((bar) => bar.date), ["2026-08-14"]);
});

test("weekly grouping starts a new week when Monday is a market holiday", () => {
  const rows = [
    row("2026-09-04", 100),
    row("2026-09-08", 101),
    row("2026-09-09", 102),
    row("2026-09-11", 103),
  ];

  const bars = resampleToWeekly(rows);

  assert.deepEqual(bars.map((bar) => bar.date), ["2026-09-04", "2026-09-11"]);
});

test("completed monthly bars exclude the current partial month", () => {
  const rows = [
    row("2026-07-31", 100),
    row("2026-08-03", 101),
    row("2026-08-19", 102),
  ];

  const bars = resampleToMonthly(rows, true, new Date("2026-08-19T12:00:00Z"));

  assert.deepEqual(bars.map((bar) => bar.date), ["2026-07-31"]);
});

test("research chart fails closed without a verified eligible contract", () => {
  assert.equal(isResearchChartEligible(undefined), false);
  assert.equal(isResearchChartEligible({ research_eligible: false }), false);
  assert.equal(isResearchChartEligible({ research_eligible: true }), true);
});

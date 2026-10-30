import test from "node:test";
import assert from "node:assert/strict";

import {
  isResearchChartEligible,
  liveDailyChangePct,
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

test("Friday weekly bar stays partial before the New York close", () => {
  const rows = [row("2026-08-14", 100), row("2026-08-21", 101)];
  const bars = resampleToWeekly(rows, true, new Date("2026-08-21T12:00:00Z"));
  assert.deepEqual(bars.map((bar) => bar.date), ["2026-08-14"]);
});

test("month-end bar stays partial before the New York close", () => {
  const rows = [row("2026-07-31", 100), row("2026-08-31", 101)];
  const bars = resampleToMonthly(rows, true, new Date("2026-08-31T12:00:00Z"));
  assert.deepEqual(bars.map((bar) => bar.date), ["2026-07-31"]);
});

test("server session watermark completes a month ending on a weekend", () => {
  const rows = [row("2026-01-30", 100), row("2026-02-02", 101)];
  const bars = resampleToMonthly(
    rows,
    true,
    new Date("2026-01-30T21:30:00Z"),
    "2026-01-30",
  );
  assert.deepEqual(bars.map((bar) => bar.date), ["2026-01-30"]);
});

test("research chart fails closed without a verified eligible contract", () => {
  assert.equal(isResearchChartEligible(undefined), false);
  assert.equal(isResearchChartEligible({ research_eligible: false }), false);
  assert.equal(isResearchChartEligible({ research_eligible: true }), true);
});

test("live day change uses the last daily close before the tick's NY session", () => {
  const chart = [row("2026-10-28", 100), row("2026-10-29", 104)];
  // prev_price is the previous trade, so it must not drive the day change.
  const tick = { symbol: "X", price: 99.84, volume: 1, timestamp: "2026-10-30T18:21:09Z", prev_price: 99.84 };
  assert.equal(liveDailyChangePct(tick, chart)?.toFixed(2), "-4.00");
  // An after-hours tick on a session the chart already stores compares with the day before.
  const sameSession = { ...tick, price: 105.04, timestamp: "2026-10-29T21:30:00Z" };
  assert.equal(liveDailyChangePct(sameSession, chart)?.toFixed(2), "5.04");
  assert.equal(liveDailyChangePct({ ...tick, timestamp: "" }, chart), null);
  assert.equal(liveDailyChangePct(tick, []), null);
  assert.equal(liveDailyChangePct(null, chart), null);
});

import assert from "node:assert/strict";
import test from "node:test";

import { macroQuoteView, resolveEquityQuote } from "./quoteView.ts";

// Closes and ticks observed in production on 2026-10-30.
const SPY = {
  price: 769.36,
  change_pct_1d: -0.38,
  history: [
    { date: "2026-10-27", close: 769.66 },
    { date: "2026-10-28", close: 772.31 },
    { date: "2026-10-29", close: 769.36 },
  ],
};

test("live tick is compared with the previous session close, not the previous tick", () => {
  const view = resolveEquityQuote(
    { symbol: "SPY", price: 752.62, volume: 80, timestamp: "2026-10-30T18:21:09Z", prev_price: 752.62 },
    SPY,
    "2026-10-29",
  );
  assert.equal(view?.mode, "live");
  assert.equal(view?.price, 752.62);
  assert.equal(view?.changePct?.toFixed(2), "-2.18");
});

test("a tick no newer than the stored close yields to that close", () => {
  const view = resolveEquityQuote(
    { symbol: "SPY", price: 770, volume: 1, timestamp: "2026-10-29T23:30:00Z" },
    SPY,
    "2026-10-29",
  );
  assert.deepEqual(view, { price: 769.36, changePct: -0.38, mode: "close", asOf: "2026-10-29" });
});

test("live tick without a usable previous close shows no change", () => {
  const view = resolveEquityQuote(
    { symbol: "SPY", price: 752.62, volume: 1, timestamp: "2026-10-30T18:21:09Z" },
    undefined,
    null,
  );
  assert.equal(view?.mode, "live");
  assert.equal(view?.changePct, null);
});

test("without a live tick the strip falls back to the dated last close", () => {
  const view = resolveEquityQuote(undefined, SPY, "2026-10-29");
  assert.deepEqual(view, { price: 769.36, changePct: -0.38, mode: "close", asOf: "2026-10-29" });
});

test("nothing to show returns null", () => {
  assert.equal(resolveEquityQuote(undefined, null, "2026-10-29"), null);
});

test("macro instruments without a live quote show their dated close", () => {
  const base = { ticker: "^VIX", name: "VIX", emoji: "", threshold_pct: 10, exceeded: false };
  assert.deepEqual(
    macroQuoteView({ ...base, current: null, prev_close: 18.2, prev_close_date: "2026-10-29", change_pct: null }),
    { price: 18.2, changePct: null, mode: "close", asOf: "2026-10-29" },
  );
  // A dated close carries its own day change against the session before it.
  assert.deepEqual(
    macroQuoteView({ ...base, current: null, prev_close: 19.42, prev_close_date: "2026-10-30", change_pct: 12.91 }),
    { price: 19.42, changePct: 12.91, mode: "close", asOf: "2026-10-30" },
  );
  assert.deepEqual(
    macroQuoteView({ ...base, current: 19.1, prev_close: 18.2, change_pct: 4.95 }),
    { price: 19.1, changePct: 4.95, mode: "live", asOf: null },
  );
  assert.equal(macroQuoteView({ ...base, current: null, prev_close: null, change_pct: null }), null);
});

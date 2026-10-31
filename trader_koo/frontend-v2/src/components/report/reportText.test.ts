import assert from "node:assert/strict";
import test from "node:test";

import {
  isIndexTicker,
  parseSetupHeadline,
  presentKeyChanges,
  SECTORS_UNAVAILABLE_TEXT,
  shortSetupReason,
  splitReportSentences,
} from "./reportText.ts";

// Text from the production report published 2026-10-31 (run 60351a9e).
const TOP_SETUP =
  "Highest-rated: PANW scored 100.0 (B) with move -3.86%, discount 9.65%, PEG 5.54, ATR 5.48%. " +
  "Read: Bearish continuation candidate. high PEG. lost the 50-day average. price rejected at resistance. " +
  "fresh bearish gap is still influencing price. stale YOLO context: W_Bottom. " +
  "Action: Directional conflict: latest bullish YOLO context disagrees with the current bearish read. " +
  "Treat this as watch-only until candles and levels resolve in one direction.";

const UNKNOWN_HEATMAP = [{
  sector: "Unknown",
  tickers: 495,
  advancers: 216,
  decliners: 271,
  unchanged: 8,
  near_high_count: 63,
  near_low_count: 10,
  avg_pct_change: -0.8,
  median_pct_change: -0.33,
  pct_advancing: 43.64,
}];

test("sentences split on full stops but keep decimals intact", () => {
  assert.deepEqual(splitReportSentences("scored 100.0 here. lost the 50-day average."), [
    "Scored 100.0 here",
    "Lost the 50-day average",
  ]);
});

test("top setup line becomes a summary, three reasons, the rest and the plan", () => {
  const view = parseSetupHeadline(TOP_SETUP);
  assert.equal(view.summary, "Highest-rated: PANW scored 100.0 (B) with move -3.86%, discount 9.65%, PEG 5.54, ATR 5.48%.");
  assert.deepEqual(view.reasons, ["Bearish continuation candidate", "High PEG", "Lost the 50-day average"]);
  assert.deepEqual(view.more, [
    "Price rejected at resistance",
    "Fresh bearish gap is still influencing price",
    "Stale YOLO context: W_Bottom",
  ]);
  assert.match(view.plan ?? "", /^Directional conflict/);
});

test("detail without a Read section is shown unchanged", () => {
  assert.deepEqual(parseSetupHeadline("No setups tonight."), {
    summary: "No setups tonight.",
    reasons: [],
    more: [],
    plan: null,
  });
});

test("a sector line comparing Unknown with itself is reported as unavailable", () => {
  const views = presentKeyChanges(
    [{ slug: "sectors", title: "Sector Rotation", detail: "Leader: Unknown (-0.8%) | Laggard: Unknown (-0.8%)." }],
    UNKNOWN_HEATMAP,
  );
  assert.equal(views[0].detail, SECTORS_UNAVAILABLE_TEXT);
  assert.doesNotMatch(views[0].detail, /Leader: Unknown/);
});

test("filler slots are dropped and real sector lines kept", () => {
  const views = presentKeyChanges(
    [
      { slug: "sectors", title: "Sector Rotation", detail: "Leader: Technology (1.2%) | Laggard: Energy (-0.4%)." },
      { slug: "placeholder_5", title: "Signal", detail: "No material change detected for this slot." },
    ],
    [{ ...UNKNOWN_HEATMAP[0], sector: "Technology" }, UNKNOWN_HEATMAP[0]],
  );
  assert.equal(views.length, 1);
  assert.match(views[0].detail, /Leader: Technology/);
});

test("short setup reason keeps the first three clauses", () => {
  assert.equal(
    shortSetupReason("Bearish continuation candidate. lost the 50-day average. price rejected at resistance. no candle confirmation."),
    "Bearish continuation candidate. Lost the 50-day average. Price rejected at resistance.",
  );
  assert.equal(shortSetupReason(null), null);
});

test("caret symbols are indexes", () => {
  assert.equal(isIndexTicker("^GSPC"), true);
  assert.equal(isIndexTicker("SPY"), false);
});

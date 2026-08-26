import assert from "node:assert/strict";
import test from "node:test";

import type { ReportSuggestion } from "../../api/types.ts";
import {
  hasResolvedProbability,
  presentSuggestionReason,
  presentSuggestionRisk,
} from "./suggestionPresentation.ts";

const suggestion = (overrides: Partial<ReportSuggestion> = {}): ReportSuggestion => ({
  ticker: "AFL",
  action: "Paper Short",
  direction: "short",
  conviction: "Medium",
  quality_score: 90.9,
  probability_pct: 81.9,
  sample_size: 0,
  persona: "Trend continuation",
  title: "AFL paper short setup",
  why: [],
  risk: "Wait for confirmation.",
  invalidation: "Invalid above resistance.",
  data_gaps: [],
  source_tier: "A",
  setup_family: "bearish_continuation",
  ...overrides,
});

test("does not present a rule prior as a resolved probability", () => {
  assert.equal(hasResolvedProbability(suggestion()), false);
  assert.equal(hasResolvedProbability(suggestion({ sample_size: 8 })), true);
});

test("fails legacy agent wording closed to deterministic provenance", () => {
  assert.equal(
    presentSuggestionReason("Agent agreement 90% with bearish consensus.", 8),
    "Rule consensus 90% with bearish consensus.",
  );
});

test("removes unsupported probability language when the sample is empty", () => {
  assert.equal(
    presentSuggestionReason(
      "Evidence score maps to 82% prior probability; history is still thin.",
      0,
    ),
    "Rule score only; no resolved outcome sample yet.",
  );
});

test("labels legacy debate state as deterministic rule review", () => {
  assert.equal(
    presentSuggestionRisk("macro pulse is risk-off; debate=ready (90% agreement)"),
    "macro pulse is risk-off; rule review=ready (90% agreement)",
  );
});

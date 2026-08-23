import test from "node:test";
import assert from "node:assert/strict";

import type { SetupRow } from "../../api/types.ts";
import { buildSetupQualityView, type DebateData } from "./setupQualityView.ts";

function setup(overrides: Partial<SetupRow>): SetupRow {
  return {
    ticker: "IP",
    score: 87.3,
    setup_tier: "A",
    setup_family: null,
    signal_bias: "bullish",
    pct_change: null,
    observation: "Neutral trend and no candle confirmation.",
    action: "Watch-only. Wait for clearer signals from trend, levels, and participation.",
    risk_note: null,
    technical_read: null,
    actionability: "watch_only",
    yolo_pattern: null,
    yolo_recency: null,
    yolo_bias: null,
    yolo_direction_conflict: false,
    level_context: null,
    debate_consensus_state: null,
    debate_agreement_score: null,
    debate_consensus_bias: null,
    debate_disagreement_count: null,
    narrative_source: "deterministic",
    discount_pct: null,
    peg: null,
    sector: null,
    ...overrides,
  };
}

test("builds one presentation contract for compact and wide setup views", () => {
  const debate = {
    version: "v1",
    consensus: {
      consensus_state: "conditional",
      consensus_bias: "bullish",
      agreement_score: 75,
      disagreement_count: 1,
    },
    roles: [],
  } satisfies DebateData;
  const rows = [
    setup({
      calibrated_hit_prob: 0.55,
      probability_sample_size: 2116,
      options_positioning_signal: "elevated_iv_event_risk",
      options_iv_rank_pct: 71,
      options_oi_rank_pct: 29,
      news_sentiment_score: 25,
      macro_news_score: 25,
    }),
  ];

  const [item] = buildSetupQualityView(rows, new Map([["IP", debate]]), {
    tier: "all",
    sortKey: "score",
    ascending: false,
  });

  assert.equal(item.decisionNote, rows[0].action);
  assert.equal(item.probability?.compactText, "55%, n=2116");
  assert.equal(item.options?.label, "elevated iv event risk");
  assert.equal(item.options?.ranksText, "IV 71% / OI 29%");
  assert.equal(item.news?.macroText, "Macro 25");
  assert.equal(item.ruleReviewLabel, "Rules 75%");
});

test("filters tiers and sorts without mutating the API rows", () => {
  const rows = [setup({ ticker: "IP", score: 87.3 }), setup({ ticker: "PHM", score: 81.8 })];

  const result = buildSetupQualityView(rows, new Map(), {
    tier: "A",
    sortKey: "score",
    ascending: true,
  });

  assert.deepEqual(result.map((item) => item.ticker), ["PHM", "IP"]);
  assert.deepEqual(rows.map((row) => row.ticker), ["IP", "PHM"]);
});

test("keeps missing evidence unavailable instead of turning null into zero", () => {
  const [item] = buildSetupQualityView(
    [setup({ calibrated_hit_prob: null, news_sentiment_score: null })],
    new Map(),
    { tier: "all", sortKey: "score", ascending: false },
  );

  assert.equal(item.probability, null);
  assert.equal(item.news, null);
  assert.equal(item.options, null);
  assert.equal(item.ruleReviewLabel, "View");
});

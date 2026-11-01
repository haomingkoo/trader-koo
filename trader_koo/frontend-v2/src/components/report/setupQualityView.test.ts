import test from "node:test";
import assert from "node:assert/strict";

import type { SetupRow, SetupSecEvents } from "../../api/types.ts";
import {
  buildSetupQualityView,
  presentCompanyFilings,
  summarizeSharedSetupFields,
  type DebateData,
} from "./setupQualityView.ts";

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

  assert.equal(item.decisionNote, "Neutral trend and no candle confirmation.");
  assert.equal(item.planNote, rows[0].action);
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

// Production 2026-10-31: 22 of 40 setups carried this exact plan sentence and
// every setup carried macro news score 33.3.
const BOILERPLATE =
  "Watch-only. Multi-angle evidence is mixed; wait for clearer confirmation from trend, levels, and participation.";

test("a plan sentence repeated across setups is shown once, not on each card", () => {
  const rows = [
    setup({ ticker: "PEG", action: BOILERPLATE, macro_news_score: 33.3, news_sentiment_score: 33.3 }),
    setup({ ticker: "AIZ", action: BOILERPLATE, macro_news_score: 33.3, news_sentiment_score: 33.3 }),
    setup({ ticker: "FIS", action: BOILERPLATE, macro_news_score: 33.3, news_sentiment_score: 33.3 }),
    setup({ ticker: "PANW", action: "Directional conflict: wait.", macro_news_score: 33.3, news_sentiment_score: 33.3 }),
  ];

  const shared = summarizeSharedSetupFields(rows);
  assert.deepEqual(shared.sharedPlans, [{ text: BOILERPLATE, count: 3 }]);
  assert.equal(shared.macroScoreText, "33");

  const items = buildSetupQualityView(rows, new Map(), { tier: "all", sortKey: "ticker", ascending: true });
  const byTicker = new Map(items.map((item) => [item.ticker, item]));
  assert.equal(byTicker.get("PEG")?.planNote, null);
  assert.equal(byTicker.get("PANW")?.planNote, "Directional conflict: wait.");
  assert.ok(items.every((item) => item.news?.macroText === null));
});

test("differing macro scores stay on each setup", () => {
  const rows = [
    setup({ ticker: "A", macro_news_score: 20, news_sentiment_score: 40 }),
    setup({ ticker: "B", macro_news_score: 60, news_sentiment_score: 40 }),
  ];
  assert.equal(summarizeSharedSetupFields(rows).macroScoreText, null);
  const items = buildSetupQualityView(rows, new Map(), { tier: "all", sortKey: "ticker", ascending: true });
  assert.deepEqual(items.map((item) => item.news?.macroText), ["Macro 20", "Macro 60"]);
});

function secEvents(overrides: Partial<SetupSecEvents> = {}): SetupSecEvents {
  return {
    enabled: true,
    available: true,
    source: "SEC EDGAR",
    lookback_days: 30,
    tickers: {
      AAPL: [{
        form: "8-K",
        filed_date: "2026-07-30",
        accepted_at: "2026-07-30T20:30:28+00:00",
        items: ["Earnings results"],
        url: "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm",
      }],
      MSFT: [],
    },
    unmapped: ["SPY"],
    errors: { NVDA: "HTTPError" },
    ...overrides,
  };
}

test("company filings list a setup's 8-K filings", () => {
  const view = presentCompanyFilings("AAPL", secEvents());
  assert.equal(view?.message, null);
  assert.deepEqual(view?.filings.map((f) => f.items), [["Earnings results"]]);
});

test("company filings explain empty, unmapped and failed lookups", () => {
  assert.match(presentCompanyFilings("MSFT", secEvents())?.message ?? "", /No 8-K filings in the 30 days/);
  assert.match(presentCompanyFilings("SPY", secEvents())?.message ?? "", /No SEC filer/);
  assert.match(presentCompanyFilings("NVDA", secEvents())?.message ?? "", /could not be loaded/);
  assert.match(
    presentCompanyFilings("AAPL", secEvents({ error: "cik_map_failed: Timeout", tickers: {} }))?.message ?? "",
    /could not be loaded/,
  );
});

test("company filings stay hidden for indexes, old reports and unchecked setups", () => {
  assert.equal(presentCompanyFilings("^GSPC", secEvents()), null);
  assert.equal(presentCompanyFilings("AAPL", undefined), null);
  assert.equal(presentCompanyFilings("AAPL", secEvents({ enabled: false })), null);
  assert.equal(presentCompanyFilings("ZZZZ", secEvents()), null);
});

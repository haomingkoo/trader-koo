import type { ChartCommentary, SetupRow } from "../../api/types";
import type { BadgeVariant } from "../ui/Badge";
import { formatReportNumber } from "./reportShared.ts";

export type DebateData = NonNullable<ChartCommentary["debate_v1"]>;

export interface SetupQualityQuery {
  tier: string;
  sortKey: string;
  ascending: boolean;
}

export interface SetupQualityItem {
  row: SetupRow;
  debate?: DebateData;
  ticker: string;
  scoreText: string;
  probability: null | {
    pctText: string;
    sampleText: string | null;
    compactText: string;
    tone: BadgeVariant;
  };
  options: null | {
    label: string;
    ranksText: string | null;
    tone: BadgeVariant;
  };
  news: null | {
    scoreText: string;
    macroText: string | null;
    tone: BadgeVariant;
  };
  decisionNote: string;
  ruleReviewLabel: string;
}

function numberOrNull(value: unknown): number | null {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeDebate(
  raw: SetupRow["debate_v1"] | DebateData | null | undefined,
): DebateData | undefined {
  const debate = raw as DebateData | null | undefined;
  if (!debate || typeof debate !== "object" || !debate.consensus) return undefined;
  return {
    version: String(debate.version ?? "v1"),
    consensus: {
      consensus_state: String(debate.consensus.consensus_state ?? "unknown"),
      consensus_bias: String(debate.consensus.consensus_bias ?? "neutral"),
      agreement_score: Number(debate.consensus.agreement_score ?? 0),
      disagreement_count: Number(debate.consensus.disagreement_count ?? 0),
    },
    roles: Array.isArray(debate.roles)
      ? debate.roles.map((role: DebateData["roles"][number]) => ({
          role: String(role.role ?? "unknown"),
          stance: String(role.stance ?? "neutral"),
          confidence: Number(role.confidence ?? 0),
          evidence: Array.isArray(role.evidence)
            ? role.evidence.map((item: string) => String(item))
            : [],
        }))
      : [],
  };
}

function probabilityTone(probability: number): BadgeVariant {
  if (probability >= 0.62) return "green";
  if (probability <= 0.45) return "red";
  if (probability >= 0.53) return "blue";
  return "muted";
}

function optionsTone(signal: unknown): BadgeVariant {
  const value = String(signal ?? "");
  if (value === "underpriced_positioning" || value === "subdued_iv") return "green";
  if (value === "elevated_iv_event_risk" || value === "crowded_open_interest") return "amber";
  return "muted";
}

function presentSetup(
  row: SetupRow,
  debateMap: Map<string, DebateData>,
): SetupQualityItem {
  const ticker = String(row.ticker ?? "");
  const probability = numberOrNull(row.calibrated_hit_prob);
  const sampleSize = numberOrNull(row.probability_sample_size);
  const optionSignal = row.options_positioning_signal;
  const ivRank = numberOrNull(row.options_iv_rank_pct);
  const oiRank = numberOrNull(row.options_oi_rank_pct);
  const newsScore = numberOrNull(row.news_sentiment_score);
  const macroScore = numberOrNull(row.macro_news_score);
  const debate = debateMap.get(ticker) ?? normalizeDebate(row.debate_v1);
  const agreement = debate?.consensus?.agreement_score;

  return {
    row,
    debate,
    ticker,
    scoreText: formatReportNumber(row.score, 1),
    probability: probability == null
      ? null
      : {
          pctText: `${(probability * 100).toFixed(0)}%`,
          sampleText: sampleSize == null ? null : `n=${sampleSize.toFixed(0)}`,
          compactText: `${(probability * 100).toFixed(0)}%${
            sampleSize == null ? "" : `, n=${sampleSize.toFixed(0)}`
          }`,
          tone: probabilityTone(probability),
        },
    options: !optionSignal && ivRank == null && oiRank == null
      ? null
      : {
          label: String(optionSignal ?? "neutral").replace(/_/g, " "),
          ranksText: ivRank == null && oiRank == null
            ? null
            : `IV ${ivRank?.toFixed(0) ?? "\u2014"}% / OI ${oiRank?.toFixed(0) ?? "\u2014"}%`,
          tone: optionsTone(optionSignal),
        },
    news: newsScore == null
      ? null
      : {
          scoreText: newsScore.toFixed(0),
          macroText: macroScore == null ? null : `Macro ${macroScore.toFixed(0)}`,
          tone: newsScore >= 60 ? "green" : newsScore <= 40 ? "red" : "muted",
        },
    decisionNote: String(row.action ?? row.observation ?? "\u2014"),
    ruleReviewLabel: agreement == null ? "View" : `Rules ${agreement.toFixed(0)}%`,
  };
}

export function buildSetupQualityView(
  rows: SetupRow[],
  debateMap: Map<string, DebateData>,
  query: SetupQualityQuery,
): SetupQualityItem[] {
  const direction = query.ascending ? 1 : -1;
  return rows
    .filter(
      (row) => query.tier === "all" ||
        String(row.setup_tier ?? "").toUpperCase() === query.tier,
    )
    .sort((a, b) => {
      const aValue = a[query.sortKey as keyof SetupRow];
      const bValue = b[query.sortKey as keyof SetupRow];
      const aNumber = Number(aValue);
      const bNumber = Number(bValue);
      if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) {
        return (aNumber - bNumber) * direction;
      }
      return String(aValue ?? "").localeCompare(String(bValue ?? "")) * direction;
    })
    .map((row) => presentSetup(row, debateMap));
}

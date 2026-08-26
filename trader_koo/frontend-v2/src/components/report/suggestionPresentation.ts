import type { ReportSuggestion } from "../../api/types";

export function hasResolvedProbability(item: ReportSuggestion) {
  return (item.sample_size ?? 0) > 0 && item.probability_pct != null;
}

export function presentSuggestionReason(reason: string, sampleSize: number | null | undefined) {
  if ((sampleSize ?? 0) === 0 && /^Evidence score maps to .*prior probability/i.test(reason)) {
    return "Rule score only; no resolved outcome sample yet.";
  }
  return reason.replace(/^Agent agreement/i, "Rule consensus");
}

export function presentSuggestionRisk(risk: string) {
  return risk.replace(/\bdebate=/gi, "rule review=");
}

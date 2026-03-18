import { useState } from "react";
import { useFearGreed } from "../api/hooks";
import {
  ComponentRow,
  GaugeSvg,
  HistoryItem,
  NewsPulseCard,
  noteLooksBlocked,
  SocialPulseCard,
} from "./sentiment/SentimentSections";

export default function FearGreedGauge() {
  const { data, isLoading, error } = useFearGreed();
  const [expanded, setExpanded] = useState(false);

  if (isLoading) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6">
        <div className="animate-pulse space-y-3">
          <div className="h-4 w-48 rounded bg-[var(--line)]" />
          <div className="h-32 rounded bg-[var(--line)]" />
        </div>
      </div>
    );
  }

  if (error || !data?.ok || data.score === null) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
        <div className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">
          Market Sentiment
        </div>
        <p className="mt-2 text-sm text-[var(--muted)]">
          {error
            ? `Market sentiment unavailable: ${String((error as Error)?.message ?? "Unknown error")}`
            : "Market sentiment unavailable"}
        </p>
      </div>
    );
  }

  const {
    score,
    color,
    label,
    previous_close,
    one_week_ago,
    one_month_ago,
    components,
    summary,
    basis,
    uses_social_sentiment,
    external_news,
    social_sentiment,
    blended_score,
    blended_label,
    blended_color,
    blended_summary,
    methodology_meta,
  } = data;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:gap-8">
        <div className="flex flex-col items-center lg:min-w-[300px]">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            Market Sentiment
          </div>
          <GaugeSvg score={score} scoreColor={color} />
          <div
            className="mt-2 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
            style={{ color, backgroundColor: `${color}18` }}
          >
            {label}
          </div>
        </div>

        <div className="flex flex-1 flex-col justify-center gap-4">
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full border border-[var(--line)] bg-[var(--bg)]/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
                Internal Composite
              </span>
              <span className="rounded-full border border-[var(--line)] bg-[var(--bg)]/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
                {uses_social_sentiment
                  ? "Social Pulse Live"
                  : noteLooksBlocked(social_sentiment.note)
                    ? "Social Pulse Blocked"
                    : "Social Pulse Unavailable"}
              </span>
              <span className="rounded-full border border-[var(--line)] bg-[var(--bg)]/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
                {external_news.available
                  ? "News Source Live"
                  : external_news.article_count > 0
                    ? "News Source Cached"
                    : "News Source Unavailable"}
              </span>
              <span className="rounded-full border border-[var(--line)] bg-[var(--bg)]/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
                {methodology_meta.version}
              </span>
            </div>
            <p className="text-sm text-[var(--muted)]">{summary}</p>
            <div className="flex flex-wrap gap-1.5">
              {basis.map((item) => (
                <span
                  key={item}
                  className="rounded-full border border-[var(--line)] bg-[var(--bg)]/60 px-2 py-1 text-[10px] text-[var(--muted)]"
                >
                  {item}
                </span>
              ))}
            </div>
          </div>

          <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            Historical
          </div>
          <div className="grid grid-cols-3 gap-4">
            <HistoryItem label="Previous Close" value={previous_close} />
            <HistoryItem label="1 Week Ago" value={one_week_ago} />
            <HistoryItem label="1 Month Ago" value={one_month_ago} />
          </div>

          <NewsPulseCard
            news={external_news}
            blendedScore={blended_score}
            blendedLabel={blended_label}
            blendedColor={blended_color}
            blendedSummary={blended_summary}
          />

          <SocialPulseCard social={social_sentiment} />

          <button
            onClick={() => setExpanded((prev) => !prev)}
            className="mt-1 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
          >
            {expanded ? "Hide" : "Show"} indicators ({components.length})
          </button>
        </div>
      </div>

      {expanded && components.length > 0 && (
        <div className="mt-4 space-y-3 border-t border-[var(--line)] pt-4">
          {components.map((component) => (
            <ComponentRow key={component.name} component={component} />
          ))}
        </div>
      )}

      <p className="mt-3 text-[10px] text-[var(--muted)]">
        This market sentiment gauge is for educational purposes only. The core
        score is still our internal market-data composite. External news and
        social sentiment overlays are optional context and should be treated as
        context, not trade instructions.
      </p>
    </div>
  );
}

import { useState } from "react";
import { useFearGreed } from "../api/hooks";
import {
  ComponentRow,
  GaugeSvg,
  HistoryItem,
  NewsPulseCard,
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
    uses_social_sentiment,
    external_news,
    social_sentiment,
    blended_score,
    blended_label,
    blended_color,
    blended_summary,
  } = data;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      {/* Compact header: gauge + score + label inline */}
      <div className="flex items-center gap-4">
        <div className="shrink-0">
          <GaugeSvg score={score} scoreColor={color} />
        </div>
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
              Market Sentiment
            </span>
            <span
              className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
              style={{ color, backgroundColor: `${color}18` }}
            >
              {label}
            </span>
            <span className="text-lg font-bold tabular-nums" style={{ color }}>
              {score}
            </span>
          </div>

          {/* Source status pills — compact row */}
          <div className="flex flex-wrap gap-1.5">
            <span className="rounded-full border border-[var(--line)] px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-[var(--accent)]">
              Internal
            </span>
            <span className={`rounded-full border border-[var(--line)] px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${
              uses_social_sentiment ? "text-[var(--green)]" : "text-[var(--muted)]"
            }`}>
              Social {uses_social_sentiment ? "Live" : "Off"}
            </span>
            <span className={`rounded-full border border-[var(--line)] px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${
              external_news.available ? "text-[var(--green)]" : "text-[var(--muted)]"
            }`}>
              News {external_news.available ? "Live" : "Off"}
            </span>
          </div>

          {/* Historical — inline */}
          <div className="flex gap-4 text-xs">
            <HistoryItem label="Prev Close" value={previous_close} />
            <HistoryItem label="1W Ago" value={one_week_ago} />
            <HistoryItem label="1M Ago" value={one_month_ago} />
          </div>
        </div>
      </div>

      {/* Blended + pulse cards — collapsible */}
      <button
        onClick={() => setExpanded((prev) => !prev)}
        className="mt-3 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
      >
        {expanded ? "Hide" : "Show"} details ({components.length} indicators)
      </button>

      {expanded && (
        <div className="mt-3 space-y-3 border-t border-[var(--line)] pt-3">
          <NewsPulseCard
            news={external_news}
            blendedScore={blended_score}
            blendedLabel={blended_label}
            blendedColor={blended_color}
            blendedSummary={blended_summary}
          />
          <SocialPulseCard social={social_sentiment} />
          {components.length > 0 && (
            <div className="space-y-2">
              {components.map((component) => (
                <ComponentRow key={component.name} component={component} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

import { useState } from "react";
import { useFearGreed } from "../api/hooks";
import type { ExternalNewsSentiment, FearGreedComponent } from "../api/types";

/* ── Zone definitions matching backend ── */
const ZONES: Array<[number, number, string, string]> = [
  [0, 25, "Extreme Fear", "#ff6b6b"],
  [25, 45, "Fear", "#ff9800"],
  [45, 55, "Neutral", "#fdd835"],
  [55, 75, "Greed", "#4caf50"],
  [75, 100, "Extreme Greed", "#1b5e20"],
];

/* ── SVG arc helpers ── */
function fgPolarToCartesian(
  cx: number,
  cy: number,
  r: number,
  angleDeg: number,
) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function fgDescribeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number,
) {
  const start = fgPolarToCartesian(cx, cy, r, endAngle);
  const end = fgPolarToCartesian(cx, cy, r, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
}

/* ── SVG Semicircular Gauge ── */
function GaugeSvg({ score, scoreColor }: { score: number; scoreColor: string }) {
  const cx = 150;
  const cy = 155;
  const r = 120;
  const arcWidth = 14;

  // Semicircle: 180 deg (left) to 360 deg (right)
  // Score 0 -> 180 deg, score 100 -> 360 deg
  const scoreToAngle = (v: number) => 180 + (v / 100) * 180;

  // Build zone arc paths
  const zoneArcs = ZONES.map(([lo, hi, , color]) => {
    const a1 = scoreToAngle(lo);
    const a2 = scoreToAngle(hi);
    return { d: fgDescribeArc(cx, cy, r, a1, a2), color };
  });

  // SVG rotation is anchored on the vertical needle where 0deg points up.
  // Convert the left->right semicircle into that coordinate system.
  const clamped = Math.max(0, Math.min(score, 100));
  const needleRotation = scoreToAngle(clamped) - 270;

  // Current zone
  const currentZone = ZONES.find(
    ([lo, hi]) => score >= lo && score < hi,
  ) ?? ZONES[ZONES.length - 1];
  const zoneLabel = currentZone[2];
  const zoneColor = currentZone[3];

  const needleLen = r - arcWidth / 2 - 6;

  return (
    <svg viewBox="0 0 300 180" className="w-full max-w-[340px]">
      {/* Dim background track */}
      <path
        d={fgDescribeArc(cx, cy, r, 180, 360)}
        fill="none"
        stroke="var(--panel-hover)"
        strokeWidth={arcWidth + 4}
        strokeLinecap="round"
        opacity={0.3}
      />

      {/* Colored zone arcs */}
      {zoneArcs.map((arc, i) => (
        <path
          key={i}
          d={arc.d}
          fill="none"
          stroke={arc.color}
          strokeWidth={arcWidth}
          strokeLinecap="butt"
        />
      ))}

      {/* Round caps at the ends of the full arc */}
      {(() => {
        const leftCap = fgPolarToCartesian(cx, cy, r, 180);
        const rightCap = fgPolarToCartesian(cx, cy, r, 360);
        return (
          <>
            <circle
              cx={leftCap.x}
              cy={leftCap.y}
              r={arcWidth / 2}
              fill={ZONES[0][3]}
            />
            <circle
              cx={rightCap.x}
              cy={rightCap.y}
              r={arcWidth / 2}
              fill={ZONES[ZONES.length - 1][3]}
            />
          </>
        );
      })()}

      {/* Scale labels: 0 and 100 */}
      <text
        x={cx - r}
        y={cy + 18}
        textAnchor="middle"
        fontSize={9}
        fill="var(--muted)"
        opacity={0.6}
      >
        0
      </text>
      <text
        x={cx + r}
        y={cy + 18}
        textAnchor="middle"
        fontSize={9}
        fill="var(--muted)"
        opacity={0.6}
      >
        100
      </text>

      {/* Needle -- rotated from center */}
      <line
        x1={cx}
        y1={cy}
        x2={cx}
        y2={cy - needleLen}
        stroke="var(--text)"
        strokeWidth={2}
        strokeLinecap="round"
        transform={`rotate(${needleRotation}, ${cx}, ${cy})`}
      />

      {/* Center dot */}
      <circle cx={cx} cy={cy} r={6} fill="var(--panel-hover)" />
      <circle cx={cx} cy={cy} r={3} fill="var(--text)" />

      {/* Score -- large centered */}
      <text
        x={cx}
        y={cy - 24}
        textAnchor="middle"
        fontSize={34}
        fontWeight={800}
        fill={scoreColor}
        style={{ fontFamily: "'Inter', system-ui, sans-serif" }}
      >
        {score}
      </text>

      {/* Zone label below score */}
      <text
        x={cx}
        y={cy - 2}
        textAnchor="middle"
        fontSize={11}
        fontWeight={700}
        fill={zoneColor}
        style={{
          textTransform: "uppercase" as const,
          letterSpacing: "0.12em",
        }}
      >
        {zoneLabel}
      </text>
    </svg>
  );
}

/* ── Historical comparison item ── */
function HistoryItem({
  label,
  value,
}: {
  label: string;
  value: number | null;
}) {
  if (value === null || typeof value !== "number") {
    return (
      <div className="text-center">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          {label}
        </div>
        <div className="text-sm text-[var(--muted)]">{"\u2014"}</div>
      </div>
    );
  }

  const zone = ZONES.find(([lo, hi]) => value >= lo && value < hi) ?? ZONES[ZONES.length - 1];
  const zoneLabel = zone[2];
  const zoneColor = zone[3];

  return (
    <div className="text-center">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
        {label}
      </div>
      <div className="text-lg font-bold tabular-nums" style={{ color: zoneColor }}>
        {value}
      </div>
      <div className="text-[10px]" style={{ color: zoneColor }}>
        {zoneLabel}
      </div>
    </div>
  );
}

/* ── Component score row ── */
function ComponentRow({ component }: { component: FearGreedComponent }) {
  const score = component.score;
  const barPct = score !== null ? Math.min(100, Math.max(0, score)) : 0;
  const zone = score !== null
    ? ZONES.find(([lo, hi]) => score >= lo && score < hi) ?? ZONES[ZONES.length - 1]
    : null;
  const barColor = zone ? zone[3] : "var(--muted)";

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-[var(--text)]">{String(component.name ?? "")}</span>
        <div className="flex items-center gap-2">
          {score !== null ? (
            <span className="font-bold tabular-nums" style={{ color: barColor }}>
              {score.toFixed(0)}
            </span>
          ) : (
            <span className="text-[var(--muted)]">{"\u2014"}</span>
          )}
          <span
            className="inline-block rounded px-1.5 py-0.5 text-[10px] font-semibold"
            style={{
              color: barColor,
              backgroundColor: `${barColor}18`,
            }}
          >
            {String(component.signal ?? "")}
          </span>
        </div>
      </div>
      {/* Bar */}
      <div className="h-1.5 w-full rounded-full bg-[var(--line)]">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${barPct}%`, backgroundColor: barColor }}
        />
      </div>
      <div className="text-[10px] text-[var(--muted)]">{String(component.detail ?? "")}</div>
    </div>
  );
}

function formatPublished(ts: string | null): string {
  if (!ts) return "Unknown time";
  const compact = ts.trim();
  if (/^\d{8}T\d{6}$/.test(compact)) {
    const normalized = `${compact.slice(0, 4)}-${compact.slice(4, 6)}-${compact.slice(6, 8)}T${compact.slice(9, 11)}:${compact.slice(11, 13)}:${compact.slice(13, 15)}Z`;
    const date = new Date(normalized);
    if (!Number.isNaN(date.getTime())) {
      return date.toLocaleString();
    }
  }
  const parsed = new Date(compact);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleString();
  }
  return compact;
}

function NewsPulseCard({
  news,
  blendedScore,
  blendedLabel,
  blendedColor,
  blendedSummary,
}: {
  news: ExternalNewsSentiment;
  blendedScore: number | null;
  blendedLabel: string | null;
  blendedColor: string | null;
  blendedSummary: string | null;
}) {
  const scoreColor = blendedColor ?? "var(--accent)";

  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--bg)]/55 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--muted)]">
            External News Pulse
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-[var(--line)] bg-[var(--panel)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--accent)]">
              {news.provider.replaceAll("_", " ")}
            </span>
            <span className="rounded-full border border-[var(--line)] bg-[var(--panel)] px-2 py-1 text-[10px] text-[var(--muted)]">
              {news.lookback_hours}h window
            </span>
            <span className="rounded-full border border-[var(--line)] bg-[var(--panel)] px-2 py-1 text-[10px] text-[var(--muted)]">
              {news.article_count} articles
            </span>
          </div>
        </div>
        {news.available && news.score !== null ? (
          <div className="text-right">
            <div className="text-2xl font-bold tabular-nums text-[var(--text)]">{news.score}</div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              {news.label ?? "News pulse"}
            </div>
          </div>
        ) : null}
      </div>

      <p className="mt-3 text-sm text-[var(--muted)]">{news.note}</p>

      {news.available && blendedScore !== null && blendedLabel && blendedSummary ? (
        <div className="mt-3 rounded-xl border border-[var(--line)] bg-[var(--panel)]/80 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
            Blended View
          </div>
          <div className="mt-1 flex items-end justify-between gap-3">
            <div>
              <div className="text-xs text-[var(--muted)]">{blendedSummary}</div>
              <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: scoreColor }}>
                {blendedLabel}
              </div>
            </div>
            <div className="text-2xl font-bold tabular-nums" style={{ color: scoreColor }}>
              {blendedScore}
            </div>
          </div>
        </div>
      ) : null}

      {news.headlines.length > 0 ? (
        <div className="mt-3 space-y-2">
          {news.headlines.slice(0, 3).map((headline) => (
            <a
              key={`${headline.title}-${headline.time_published ?? ""}`}
              href={headline.url ?? undefined}
              target={headline.url ? "_blank" : undefined}
              rel={headline.url ? "noreferrer" : undefined}
              className="block rounded-xl border border-[var(--line)] bg-[var(--panel)]/70 p-3 transition-colors hover:border-[var(--accent)]"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-[var(--text)]">{headline.title}</div>
                  <div className="mt-1 text-[11px] text-[var(--muted)]">
                    {[headline.source, formatPublished(headline.time_published)].filter(Boolean).join(" · ")}
                  </div>
                </div>
                {headline.score !== null ? (
                  <div className="shrink-0 text-right">
                    <div className="text-sm font-semibold tabular-nums text-[var(--text)]">
                      {headline.score}
                    </div>
                    <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
                      {headline.label ?? "News"}
                    </div>
                  </div>
                ) : null}
              </div>
            </a>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/* ── Main component ── */
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
    blended_score,
    blended_label,
    blended_color,
    blended_summary,
  } = data;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:gap-8">
        {/* Left: Gauge */}
        <div className="flex flex-col items-center lg:min-w-[300px]">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            Market Sentiment
          </div>
          <GaugeSvg score={score!} scoreColor={color} />
          <div
            className="mt-2 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em]"
            style={{ color, backgroundColor: `${color}18` }}
          >
            {label}
          </div>
        </div>

        {/* Right: Historical comparisons */}
        <div className="flex flex-1 flex-col justify-center gap-4">
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full border border-[var(--line)] bg-[var(--bg)]/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
                Internal Composite
              </span>
              <span className="rounded-full border border-[var(--line)] bg-[var(--bg)]/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
                {uses_social_sentiment ? "Social Signals" : "No Social Scraping"}
              </span>
              <span className="rounded-full border border-[var(--line)] bg-[var(--bg)]/70 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
                {external_news.available ? "News Source Live" : "News Source Optional"}
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

          {/* Expand/collapse button for components */}
          <button
            onClick={() => setExpanded((p) => !p)}
            className="mt-1 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
          >
            {expanded ? "Hide" : "Show"} indicators ({components.length})
          </button>
        </div>
      </div>

      {/* Expandable component breakdown */}
      {expanded && components.length > 0 && (
        <div className="mt-4 space-y-3 border-t border-[var(--line)] pt-4">
          {components.map((comp) => (
            <ComponentRow key={comp.name} component={comp} />
          ))}
        </div>
      )}

      {/* NFA disclaimer */}
      <p className="mt-3 text-[10px] text-[var(--muted)]">
        This market sentiment gauge is for educational purposes only. The core score is still our internal market-data composite. External news sentiment is optional, and Twitter/Reddit-style social scraping is not included yet.
      </p>
    </div>
  );
}

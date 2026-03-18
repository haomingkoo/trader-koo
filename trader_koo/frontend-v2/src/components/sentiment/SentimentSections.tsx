import type {
  ExternalNewsSentiment,
  FearGreedComponent,
  SocialSentiment,
} from "../../api/types";

export const ZONES: Array<[number, number, string, string]> = [
  [0, 25, "Extreme Fear", "#ff6b6b"],
  [25, 45, "Fear", "#ff9800"],
  [45, 55, "Neutral", "#fdd835"],
  [55, 75, "Greed", "#4caf50"],
  [75, 100, "Extreme Greed", "#1b5e20"],
];

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number,
) {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
}

export function GaugeSvg({ score, scoreColor }: { score: number; scoreColor: string }) {
  const cx = 150;
  const cy = 155;
  const r = 120;
  const arcWidth = 14;
  const scoreToAngle = (value: number) => 180 + (value / 100) * 180;
  const zoneArcs = ZONES.map(([lo, hi, , color]) => ({
    d: describeArc(cx, cy, r, scoreToAngle(lo), scoreToAngle(hi)),
    color,
  }));
  const clamped = Math.max(0, Math.min(score, 100));
  const needleRotation = scoreToAngle(clamped) - 270;
  const currentZone = ZONES.find(([lo, hi]) => score >= lo && score < hi) ?? ZONES[ZONES.length - 1];
  const zoneLabel = currentZone[2];
  const zoneColor = currentZone[3];
  const needleLen = r - arcWidth / 2 - 6;

  return (
    <svg viewBox="0 0 300 180" className="w-full max-w-[340px]">
      <path
        d={describeArc(cx, cy, r, 180, 360)}
        fill="none"
        stroke="var(--panel-hover)"
        strokeWidth={arcWidth + 4}
        strokeLinecap="round"
        opacity={0.3}
      />
      {zoneArcs.map((arc, index) => (
        <path
          key={index}
          d={arc.d}
          fill="none"
          stroke={arc.color}
          strokeWidth={arcWidth}
          strokeLinecap="butt"
        />
      ))}
      {(() => {
        const leftCap = polarToCartesian(cx, cy, r, 180);
        const rightCap = polarToCartesian(cx, cy, r, 360);
        return (
          <>
            <circle cx={leftCap.x} cy={leftCap.y} r={arcWidth / 2} fill={ZONES[0][3]} />
            <circle
              cx={rightCap.x}
              cy={rightCap.y}
              r={arcWidth / 2}
              fill={ZONES[ZONES.length - 1][3]}
            />
          </>
        );
      })()}
      <text x={cx - r} y={cy + 18} textAnchor="middle" fontSize={9} fill="var(--muted)" opacity={0.6}>
        0
      </text>
      <text x={cx + r} y={cy + 18} textAnchor="middle" fontSize={9} fill="var(--muted)" opacity={0.6}>
        100
      </text>
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
      <circle cx={cx} cy={cy} r={6} fill="var(--panel-hover)" />
      <circle cx={cx} cy={cy} r={3} fill="var(--text)" />
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
      <text
        x={cx}
        y={cy - 2}
        textAnchor="middle"
        fontSize={11}
        fontWeight={700}
        fill={zoneColor}
        style={{ textTransform: "uppercase", letterSpacing: "0.12em" }}
      >
        {zoneLabel}
      </text>
    </svg>
  );
}

export function HistoryItem({ label, value }: { label: string; value: number | null }) {
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

export function ComponentRow({ component }: { component: FearGreedComponent }) {
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
            style={{ color: barColor, backgroundColor: `${barColor}18` }}
          >
            {String(component.signal ?? "")}
          </span>
        </div>
      </div>
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

export function noteLooksBlocked(note: string | null | undefined): boolean {
  const raw = String(note ?? "").toLowerCase();
  return raw.includes("403") || raw.includes("blocked") || raw.includes("forbidden");
}

function statusBadgeTone(
  available: boolean,
  count: number,
  note: string | null | undefined,
): { label: string; color: string; bg: string } {
  if (available && count > 0) {
    return { label: "Live", color: "var(--green)", bg: "rgba(34,197,94,0.12)" };
  }
  if (noteLooksBlocked(note)) {
    return { label: "Blocked", color: "var(--red)", bg: "rgba(248,113,113,0.12)" };
  }
  if (count === 0) {
    return { label: "No data", color: "var(--amber)", bg: "rgba(245,158,11,0.12)" };
  }
  return { label: "Unavailable", color: "var(--muted)", bg: "rgba(148,163,184,0.12)" };
}

export function NewsPulseCard({
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
  const status = statusBadgeTone(news.available, news.article_count, news.note);

  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--bg)]/55 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--muted)]">
            External News Pulse
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="rounded-full border border-[var(--line)] bg-[var(--panel)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em]"
              style={{ color: news.available ? "var(--accent)" : "var(--muted)" }}
            >
              {news.provider.replaceAll("_", " ")}
            </span>
            <span
              className="rounded-full border border-[var(--line)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em]"
              style={{ color: status.color, backgroundColor: status.bg }}
            >
              {status.label}
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

      {!news.available && news.headlines.length === 0 ? (
        <div className="mt-3 rounded-xl border border-[var(--line)] bg-[var(--panel)]/70 p-3 text-xs text-[var(--muted)]">
          No live news headlines are being blended right now. The internal composite is still real, but this overlay is currently unavailable.
        </div>
      ) : null}

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

export function SocialPulseCard({ social }: { social: SocialSentiment }) {
  const status = statusBadgeTone(social.available, social.post_count, social.note);

  return (
    <div className="rounded-2xl border border-[var(--line)] bg-[var(--bg)]/55 p-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--muted)]">
            Reddit Social Pulse
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="rounded-full border border-[var(--line)] bg-[var(--panel)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em]"
              style={{ color: social.available ? "var(--accent)" : "var(--muted)" }}
            >
              {social.provider.replaceAll("_", " ")}
            </span>
            <span
              className="rounded-full border border-[var(--line)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em]"
              style={{ color: status.color, backgroundColor: status.bg }}
            >
              {status.label}
            </span>
            <span className="rounded-full border border-[var(--line)] bg-[var(--panel)] px-2 py-1 text-[10px] text-[var(--muted)]">
              {social.lookback_hours}h window
            </span>
            <span className="rounded-full border border-[var(--line)] bg-[var(--panel)] px-2 py-1 text-[10px] text-[var(--muted)]">
              {social.post_count} posts
            </span>
          </div>
        </div>
        {social.available && social.score !== null ? (
          <div className="text-right">
            <div className="text-2xl font-bold tabular-nums text-[var(--text)]">{social.score}</div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              {social.label ?? "Social pulse"}
            </div>
          </div>
        ) : null}
      </div>

      <p className="mt-3 text-sm text-[var(--muted)]">{social.note}</p>

      {!social.available && social.posts.length === 0 ? (
        <div className="mt-3 rounded-xl border border-[var(--line)] bg-[var(--panel)]/70 p-3 text-xs text-[var(--muted)]">
          No Reddit posts were ingested for the current window. If the provider is blocked or rate-limited, this overlay should be treated as unavailable rather than neutral.
        </div>
      ) : null}

      <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)]/75 p-3">
          <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">Subreddits</div>
          <div className="mt-1 font-semibold text-[var(--text)]">{social.subreddit_count}</div>
        </div>
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)]/75 p-3">
          <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">Bull Terms</div>
          <div className="mt-1 font-semibold text-[var(--green)]">{social.bullish_terms_total}</div>
        </div>
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)]/75 p-3">
          <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">Bear Terms</div>
          <div className="mt-1 font-semibold text-[var(--red)]">{social.bearish_terms_total}</div>
        </div>
      </div>

      {social.posts.length > 0 ? (
        <div className="mt-3 space-y-2">
          {social.posts.slice(0, 3).map((post) => (
            <a
              key={`${post.subreddit}-${post.title}`}
              href={post.url ?? undefined}
              target={post.url ? "_blank" : undefined}
              rel={post.url ? "noreferrer" : undefined}
              className="block rounded-xl border border-[var(--line)] bg-[var(--panel)]/70 p-3 transition-colors hover:border-[var(--accent)]"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-[var(--text)]">{post.title}</div>
                  <div className="mt-1 text-[11px] text-[var(--muted)]">
                    {`r/${post.subreddit} · ${post.upvotes} upvotes · ${post.num_comments} comments`}
                  </div>
                  {post.excerpt ? (
                    <div className="mt-1 text-[11px] text-[var(--muted)]/90">{post.excerpt}</div>
                  ) : null}
                </div>
                {post.sentiment_score !== null ? (
                  <div className="shrink-0 text-right">
                    <div className="text-sm font-semibold tabular-nums text-[var(--text)]">
                      {post.sentiment_score}
                    </div>
                    <div className="text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
                      {post.label ?? "Pulse"}
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

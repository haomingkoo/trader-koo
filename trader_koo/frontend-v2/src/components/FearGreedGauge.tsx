import { useState } from "react";
import { useFearGreed } from "../api/hooks";
import type { FearGreedComponent } from "../api/types";

/* ── Zone definitions matching backend ── */
const ZONES: Array<[number, number, string, string]> = [
  [0, 25, "Extreme Fear", "#ff6b6b"],
  [25, 45, "Fear", "#ff9800"],
  [45, 55, "Neutral", "#fdd835"],
  [55, 75, "Greed", "#4caf50"],
  [75, 100, "Extreme Greed", "#1b5e20"],
];

/* ── SVG Semicircular Gauge ── */
function GaugeSvg({ score }: { score: number }) {
  const cx = 150;
  const cy = 140;
  const r = 110;
  const startAngle = Math.PI; // left (180 deg)
  const totalArc = Math.PI;

  // Build zone arcs
  const zoneArcs = ZONES.map(([lo, hi, , color]) => {
    const a1 = startAngle - (lo / 100) * totalArc;
    const a2 = startAngle - (hi / 100) * totalArc;
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy - r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2);
    const y2 = cy - r * Math.sin(a2);
    const largeArc = Math.abs(a1 - a2) > Math.PI ? 1 : 0;
    const d = `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 0 ${x2} ${y2}`;
    return { d, color, lo };
  });

  // Needle position
  const clamped = Math.max(0, Math.min(score, 100));
  const needleAngle = startAngle - (clamped / 100) * totalArc;
  const needleLen = r - 15;
  const nx = cx + needleLen * Math.cos(needleAngle);
  const ny = cy - needleLen * Math.sin(needleAngle);

  // Zone labels along the arc
  const labelR = r + 20;
  const zoneLabels = ZONES.map(([lo, hi, label]) => {
    const mid = (lo + hi) / 2;
    const angle = startAngle - (mid / 100) * totalArc;
    const lx = cx + labelR * Math.cos(angle);
    const ly = cy - labelR * Math.sin(angle);
    return { lx, ly, label, angle };
  });

  return (
    <svg viewBox="0 0 300 175" className="w-full max-w-[340px]">
      {/* Zone arcs */}
      {zoneArcs.map((arc) => (
        <path
          key={arc.lo}
          d={arc.d}
          fill="none"
          stroke={arc.color}
          strokeWidth={18}
          strokeLinecap="butt"
        />
      ))}
      {/* Zone labels */}
      {zoneLabels.map(({ lx, ly, label, angle }) => (
        <text
          key={label}
          x={lx}
          y={ly}
          textAnchor="middle"
          fontSize={7}
          fill="var(--muted)"
          transform={`rotate(${(-angle * 180) / Math.PI + 90}, ${lx}, ${ly})`}
        >
          {label}
        </text>
      ))}
      {/* Needle */}
      <line
        x1={cx}
        y1={cy}
        x2={nx}
        y2={ny}
        stroke="var(--text)"
        strokeWidth={2.5}
        strokeLinecap="round"
      />
      {/* Center dot */}
      <circle cx={cx} cy={cy} r={5} fill="var(--text)" />
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
  if (value === null) {
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
        <span className="font-medium text-[var(--text)]">{component.name}</span>
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
            {component.signal}
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
      <div className="text-[10px] text-[var(--muted)]">{component.detail}</div>
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
          Fear & Greed Index
        </div>
        <p className="mt-2 text-sm text-[var(--muted)]">
          {error
            ? `Fear & Greed unavailable: ${String((error as Error)?.message ?? "Unknown error")}`
            : "Fear & Greed unavailable"}
        </p>
      </div>
    );
  }

  const { score, label, color, previous_close, one_week_ago, one_month_ago, components } = data;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:gap-8">
        {/* Left: Gauge */}
        <div className="flex flex-col items-center lg:min-w-[300px]">
          <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            Fear & Greed Index
          </div>
          <GaugeSvg score={score!} />
          {/* Score + label below gauge */}
          <div className="mt-2 text-center">
            <div
              className="text-4xl font-bold tabular-nums"
              style={{ color }}
            >
              {score}
            </div>
            <div
              className="mt-0.5 text-sm font-semibold"
              style={{ color }}
            >
              {label}
            </div>
          </div>
        </div>

        {/* Right: Historical comparisons */}
        <div className="flex flex-1 flex-col justify-center gap-4">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
            Historical
          </div>
          <div className="grid grid-cols-3 gap-4">
            <HistoryItem label="Previous Close" value={previous_close} />
            <HistoryItem label="1 Week Ago" value={one_week_ago} />
            <HistoryItem label="1 Month Ago" value={one_month_ago} />
          </div>

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
    </div>
  );
}

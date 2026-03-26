import type { SectorHeatmapRow } from "../../api/types";
import GlassCard from "../ui/GlassCard";

/**
 * Map avg_pct_change to a CSS background color.
 * Green for positive, red for negative, intensity scales with magnitude.
 */
function cellColor(pct: number | null): string {
  if (pct == null) return "transparent";
  // Clamp to [-5, +5] for color mapping
  const clamped = Math.max(-5, Math.min(5, pct));
  const intensity = Math.abs(clamped) / 5;
  // Scale alpha from 0.08 (near zero) to 0.7 (at extremes)
  const alpha = 0.08 + intensity * 0.62;
  if (clamped >= 0) {
    return `rgba(34, 197, 94, ${alpha.toFixed(2)})`;
  }
  return `rgba(239, 68, 68, ${alpha.toFixed(2)})`;
}

function changeText(pct: number | null): string {
  if (pct == null) return "--";
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

function changeColor(pct: number | null): string {
  if (pct == null) return "var(--muted)";
  if (pct > 0.05) return "var(--green)";
  if (pct < -0.05) return "var(--red)";
  return "var(--muted)";
}

interface Props {
  rows: SectorHeatmapRow[];
}

export default function SectorHeatmap({ rows }: Props) {
  if (!rows || rows.length === 0) {
    return (
      <GlassCard label="Sector Heatmap">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No sector data available.
        </p>
      </GlassCard>
    );
  }

  return (
    <GlassCard>
      <div className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
        Sector Heatmap
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {rows.map((row) => (
          <div
            key={row.sector}
            className="relative overflow-hidden rounded-lg border border-[var(--line)] p-3"
            style={{ backgroundColor: cellColor(row.avg_pct_change) }}
          >
            {/* Sector name */}
            <div className="text-xs font-semibold text-[var(--text)] truncate">
              {row.sector}
            </div>

            {/* Avg change - prominent */}
            <div
              className="mt-1 text-lg font-bold tabular-nums"
              style={{ color: changeColor(row.avg_pct_change) }}
            >
              {changeText(row.avg_pct_change)}
            </div>

            {/* Stats row */}
            <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-[var(--muted)]">
              <span>
                {row.tickers} tickers
              </span>
              {row.pct_advancing != null && (
                <span>
                  {row.pct_advancing.toFixed(0)}% advancing
                </span>
              )}
            </div>

            {/* Near-high / near-low indicators */}
            {(row.near_high_count > 0 || row.near_low_count > 0) && (
              <div className="mt-1 flex gap-2 text-[10px]">
                {row.near_high_count > 0 && (
                  <span className="text-[var(--green)]">
                    {row.near_high_count} near 52W high
                  </span>
                )}
                {row.near_low_count > 0 && (
                  <span className="text-[var(--red)]">
                    {row.near_low_count} near 52W low
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </GlassCard>
  );
}

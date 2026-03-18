import type { ReactNode } from "react";

export function formatPrice(price: number): string {
  if (price >= 1000) {
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    });
  }
  if (price >= 1) {
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  return price.toLocaleString("en-US", {
    minimumFractionDigits: 4,
    maximumFractionDigits: 6,
  });
}

export function formatVolume(vol: number): string {
  if (vol >= 1_000_000_000) return `${(vol / 1_000_000_000).toFixed(2)}B`;
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(2)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(1)}K`;
  return vol.toFixed(0);
}

export function formatPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

export function formatLevelContext(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function GlassCard({
  label,
  children,
  className = "",
}: {
  label?: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl backdrop-blur-sm bg-[var(--panel)]/80 border border-[var(--line)] p-4 ${className}`}
    >
      {label && (
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          {label}
        </div>
      )}
      {children}
    </div>
  );
}

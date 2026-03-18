import type { ReactNode } from "react";

interface GlassCardProps {
  label?: string;
  value?: string | number;
  children?: ReactNode;
  className?: string;
}

export default function GlassCard({
  label,
  value,
  children,
  className = "",
}: GlassCardProps) {
  return (
    <div
      className={`rounded-xl backdrop-blur-sm bg-[var(--panel)]/80 border border-[var(--line)] p-4 ${className}`}
    >
      {label && (
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          {label}
        </div>
      )}
      {value !== undefined && (
        <div className="text-lg font-bold tabular-nums text-[var(--text)]">
          {typeof value === "string" || typeof value === "number" ? (value ?? "\u2014") : String(value ?? "\u2014")}
        </div>
      )}
      {children}
    </div>
  );
}

import type { ReactNode } from "react";

interface GlassCardProps {
  label?: string;
  value?: string | number;
  children?: ReactNode;
  className?: string;
}

export const GLASS_BASE =
  "rounded-xl backdrop-blur-sm bg-[var(--panel)]/86 border border-[var(--line)] p-4 shadow-[var(--shadow-soft)]";

export default function GlassCard({
  label,
  value,
  children,
  className = "",
}: GlassCardProps) {
  return (
    <div className={`${GLASS_BASE} ${className}`}>
      {label && (
        <div className="label-xs mb-1">
          {label}
        </div>
      )}
      {value !== undefined && (
        <div className="text-lg font-bold tabular-nums text-[var(--text)]">
          {value}
        </div>
      )}
      {children}
    </div>
  );
}

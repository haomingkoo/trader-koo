import type { ReactNode } from "react";

interface GlassCardProps {
  label?: string;
  value?: string | number;
  children?: ReactNode;
  className?: string;
  onClick?: () => void;
  header?: ReactNode;
}

export const GLASS_BASE =
  "rounded-xl backdrop-blur-sm bg-[var(--panel)]/86 border border-[var(--line)] p-4 shadow-[var(--shadow-soft)]";

export default function GlassCard({
  label,
  value,
  children,
  className = "",
  onClick,
  header,
}: GlassCardProps) {
  return (
    <div
      className={`${GLASS_BASE} ${className}`}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") onClick();
            }
          : undefined
      }
    >
      {header}
      {label && (
        <div className="label-xs mb-1">
          {label}
        </div>
      )}
      {value !== undefined && (
        <div className="text-lg font-bold tabular-nums text-[var(--text)]">
          {typeof value === "string" || typeof value === "number"
            ? (value ?? "\u2014")
            : String(value ?? "\u2014")}
        </div>
      )}
      {children}
    </div>
  );
}

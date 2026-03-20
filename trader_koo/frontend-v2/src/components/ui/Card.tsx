import { GLASS_BASE } from "./GlassCard";

interface CardProps {
  label?: string;
  value?: string | number;
  children?: React.ReactNode;
  className?: string;
  glass?: boolean;
}

const SOLID_BASE = "rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4";

export default function Card({ label, value, children, className = "", glass = false }: CardProps) {
  const base = glass ? GLASS_BASE : SOLID_BASE;
  return (
    <div
      className={`${base} ${className}`}
    >
      {label && (
        <div className="mb-1 text-xs font-medium uppercase tracking-wide text-[var(--muted)]">
          {label}
        </div>
      )}
      {value !== undefined && (
        <div className="text-lg font-semibold text-[var(--text)]">
          {typeof value === "string" || typeof value === "number" ? (value ?? "\u2014") : String(value ?? "\u2014")}
        </div>
      )}
      {children}
    </div>
  );
}

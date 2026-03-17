interface CardProps {
  label?: string;
  value?: string | number;
  children?: React.ReactNode;
  className?: string;
  glass?: boolean;
}

export default function Card({ label, value, children, className = "", glass = false }: CardProps) {
  const base = glass
    ? "rounded-xl backdrop-blur-sm bg-[var(--panel)]/80 border border-[var(--line)] p-4"
    : "rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4";
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
          {value ?? "\u2014"}
        </div>
      )}
      {children}
    </div>
  );
}

interface CardProps {
  label?: string;
  value?: string | number;
  children?: React.ReactNode;
  className?: string;
}

export default function Card({ label, value, children, className = "" }: CardProps) {
  return (
    <div
      className={`rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 ${className}`}
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

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "green" | "red" | "amber" | "blue" | "muted";
  className?: string;
}

const variantStyles: Record<string, string> = {
  default:
    "bg-[var(--panel-hover)] text-[var(--text)] border-[var(--line)]",
  green:
    "bg-[rgba(56,211,159,0.15)] text-[var(--green)] border-[rgba(56,211,159,0.25)]",
  red:
    "bg-[rgba(255,107,107,0.12)] text-[var(--red)] border-[rgba(255,107,107,0.25)]",
  amber:
    "bg-[rgba(248,194,78,0.15)] text-[var(--amber)] border-[rgba(248,194,78,0.25)]",
  blue:
    "bg-[rgba(106,169,255,0.15)] text-[var(--blue)] border-[rgba(106,169,255,0.25)]",
  muted:
    "bg-[var(--panel)] text-[var(--muted)] border-[var(--line)]",
};

export function tierVariant(tier: string | null | undefined): BadgeProps["variant"] {
  if (!tier) return "muted";
  const t = tier.toUpperCase();
  if (t === "A") return "green";
  if (t === "B") return "amber";
  if (t === "C") return "red";
  return "default";
}

export default function Badge({ children, variant = "default", className = "" }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${variantStyles[variant]} ${className}`}
    >
      {children}
    </span>
  );
}

import type { BadgeVariant } from "./Badge";

export function tierVariant(tier: string | null | undefined): BadgeVariant {
  if (!tier) return "muted";
  const normalized = tier.toUpperCase();
  if (normalized === "A") return "green";
  if (normalized === "B") return "amber";
  if (normalized === "C") return "red";
  return "default";
}

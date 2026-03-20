export function formatReportTimestamp(ts: string | null): {
  local: string;
  ny: string;
} {
  if (!ts) return { local: "\u2014", ny: "\u2014" };
  try {
    const d = new Date(ts);
    const local = d.toLocaleString();
    const ny = d.toLocaleString("en-US", { timeZone: "America/New_York" });
    return { local, ny };
  } catch {
    return { local: ts, ny: ts };
  }
}

export function formatReportNumber(
  n: number | null | undefined,
  decimals = 2,
): string {
  if (n == null || !Number.isFinite(n)) return "\u2014";
  return n.toFixed(decimals);
}

export function severityVariant(
  severity: string,
): "red" | "amber" | "green" | "muted" {
  const s = severity.toLowerCase();
  if (s === "hard" || s === "block" || s === "critical") return "red";
  if (s === "soft" || s === "warning" || s === "elevated") return "amber";
  if (s === "ok" || s === "normal" || s === "low") return "green";
  return "muted";
}

export function biasVariant(
  bias: string | null,
): "green" | "red" | "amber" | "muted" {
  if (!bias) return "muted";
  const b = bias.toLowerCase();
  if (b.includes("bull") || b === "long") return "green";
  if (b.includes("bear") || b === "short") return "red";
  if (b.includes("neutral") || b === "flat") return "amber";
  return "muted";
}


export function commentarySourceLabel(source: unknown): string {
  const normalized = typeof source === "string" ? source.trim().toLowerCase() : "";
  if (normalized === "llm") return "LLM (generated from current regime snapshot)";
  if (normalized === "rule") return "Deterministic regime rules";
  return "Provenance not recorded";
}

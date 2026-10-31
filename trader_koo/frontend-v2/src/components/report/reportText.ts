import type { KeyChange, SectorHeatmapRow } from "../../api/types.ts";

/**
 * Presentation-only readers for the prose the report generator writes. The
 * report artifact is immutable lineage, so these split and shorten its text
 * for display and never change what it says.
 */

/** "lost the 50-day average. price rejected at resistance." -> two clauses. */
export function splitReportSentences(text: string | null | undefined): string[] {
  return String(text ?? "")
    .split(/(?<=\.)\s+/)
    .map((part) => part.trim().replace(/\.$/, "").trim())
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1));
}

/** The first few clauses of a setup observation, as one short line. */
export function shortSetupReason(observation: string | null | undefined, maxClauses = 3): string | null {
  const clauses = splitReportSentences(observation);
  if (!clauses.length) return null;
  return clauses.slice(0, maxClauses).join(". ") + ".";
}

export interface SetupHeadline {
  summary: string;
  reasons: string[];
  more: string[];
  plan: string | null;
}

/**
 * Splits the "Top Setup Candidate" detail ("Highest-rated: ... Read: ...
 * Action: ...") into a summary, the leading reasons, and the rest.
 */
export function parseSetupHeadline(detail: string, visibleReasons = 3): SetupHeadline {
  const readAt = detail.indexOf(" Read: ");
  const actionAt = detail.lastIndexOf(" Action: ");
  if (readAt < 0) return { summary: detail, reasons: [], more: [], plan: null };
  const summary = detail.slice(0, readAt).trim();
  const read = detail.slice(readAt + " Read: ".length, actionAt > readAt ? actionAt : undefined);
  const plan = actionAt > readAt ? detail.slice(actionAt + " Action: ".length).trim() || null : null;
  const clauses = splitReportSentences(read);
  return {
    summary,
    reasons: clauses.slice(0, visibleReasons),
    more: clauses.slice(visibleReasons),
    plan,
  };
}

const UNKNOWN_SECTOR = "unknown";

export function isUnknownSector(sector: string | null | undefined): boolean {
  return String(sector ?? "").trim().toLowerCase() === UNKNOWN_SECTOR;
}

/** True when the report has heatmap rows but none of them is a real sector. */
export function sectorsUnclassified(rows: SectorHeatmapRow[]): boolean {
  return rows.length > 0 && rows.every((row) => isUnknownSector(row.sector));
}

export const SECTORS_UNAVAILABLE_TEXT =
  "Unavailable. No ticker in this report has a sector classification, so there is no sector leader or laggard.";

export interface KeyChangeView {
  slug: string;
  title: string;
  detail: string;
  setup: SetupHeadline | null;
}

/**
 * Drops the generator's filler slots ("No material change detected for this
 * slot"), replaces a sector line that compares "Unknown" with itself, and
 * splits the top-setup line into reasons.
 */
export function presentKeyChanges(changes: KeyChange[], sectorRows: SectorHeatmapRow[]): KeyChangeView[] {
  const noSectors = sectorsUnclassified(sectorRows);
  return changes
    .filter((change) => !String(change.slug ?? "").startsWith("placeholder_"))
    .map((change) => {
      const slug = String(change.slug ?? "");
      const detail = String(change.detail ?? "");
      if (slug === "sectors" && noSectors) {
        return { slug, title: String(change.title), detail: SECTORS_UNAVAILABLE_TEXT, setup: null };
      }
      return {
        slug,
        title: String(change.title),
        detail,
        setup: slug === "setup" ? parseSetupHeadline(detail) : null,
      };
    });
}

/** Index symbols such as ^GSPC are a calculation, not an instrument anyone can buy or sell. */
export function isIndexTicker(ticker: string | null | undefined): boolean {
  return String(ticker ?? "").startsWith("^");
}

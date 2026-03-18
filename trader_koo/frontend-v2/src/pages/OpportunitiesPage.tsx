import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useOpportunities } from "../api/hooks";
import type { OpportunityRow } from "../api/types";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";

type View = "all" | "undervalued" | "deep_value" | "overvalued";

const viewLabels: Record<View, string> = {
  all: "All",
  undervalued: "Undervalued",
  deep_value: "Deep Value",
  overvalued: "Overvalued",
};

const valuationBadgeVariant = (
  label: string | null | undefined,
): "green" | "amber" | "red" | "muted" => {
  const lower = (label ?? "").toLowerCase();
  if (lower.includes("under")) return "green";
  if (lower.includes("fair")) return "amber";
  if (lower.includes("over")) return "red";
  return "muted";
};

const formatTimestamp = (ts: string | null | undefined): string => {
  if (!ts) return "\u2014";
  try {
    const d = new Date(ts);
    const local = d.toLocaleString();
    const ny = d.toLocaleString("en-US", { timeZone: "America/New_York" });
    return `${local} (NY: ${ny})`;
  } catch {
    return ts;
  }
};

const opportunityColumns = [
  {
    key: "ticker" as const,
    label: "Ticker",
    render: (v: unknown) => {
      const ticker = String(v ?? "");
      if (!ticker) return "\u2014";
      return (
        <Link
          to={`/chart?t=${ticker}`}
          className="font-mono font-bold text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          {ticker}
        </Link>
      );
    },
  },
  {
    key: "price" as const,
    label: "Price",
    render: (v: unknown) => {
      const n = typeof v === "number" ? v : null;
      return n != null ? `$${n.toFixed(2)}` : "\u2014";
    },
  },
  {
    key: "pe" as const,
    label: "P/E",
    render: (v: unknown) => {
      const n = typeof v === "number" ? v : null;
      return n != null ? n.toFixed(1) : "\u2014";
    },
  },
  {
    key: "peg" as const,
    label: "PEG",
    render: (v: unknown) => {
      const n = typeof v === "number" ? v : null;
      if (n == null) return "\u2014";
      const color =
        n <= 1 ? "text-[var(--green)]" : n <= 2 ? "text-[var(--amber)]" : "text-[var(--red)]";
      return <span className={color}>{n.toFixed(2)}</span>;
    },
  },
  {
    key: "target_price" as const,
    label: "Target",
    render: (v: unknown) => {
      const n = typeof v === "number" ? v : null;
      return n != null ? `$${n.toFixed(2)}` : "\u2014";
    },
  },
  {
    key: "discount_pct" as const,
    label: "Discount %",
    render: (v: unknown) => {
      const n = typeof v === "number" ? v : null;
      if (n == null) return "\u2014";
      const color =
        n > 0 ? "text-[var(--green)]" : n < 0 ? "text-[var(--red)]" : "";
      return (
        <span className={`font-medium ${color}`}>
          {n > 0 ? "+" : ""}
          {n.toFixed(1)}%
        </span>
      );
    },
  },
  {
    key: "valuation_label" as const,
    label: "Valuation",
    render: (v: unknown) => {
      const label = typeof v === "string" ? v : null;
      if (!label) return "\u2014";
      return (
        <Badge variant={valuationBadgeVariant(label)}>
          {label}
        </Badge>
      );
    },
  },
  {
    key: "eps_growth_5y" as const,
    label: "EPS Growth 5Y",
    render: (v: unknown) => {
      const n = typeof v === "number" ? v : null;
      if (n == null) return "\u2014";
      const color = n > 0 ? "text-[var(--green)]" : n < 0 ? "text-[var(--red)]" : "";
      return (
        <span className={color}>
          {n > 0 ? "+" : ""}
          {n.toFixed(1)}%
        </span>
      );
    },
  },
];

export default function OpportunitiesPage() {
  const [view, setView] = useState<View>("all");
  const apiView = view === "deep_value" ? "undervalued" : view;
  const { data, isLoading, error } = useOpportunities({
    view: apiView,
    limit: 1000,
  });

  const rows = useMemo(() => {
    const allRows = data?.rows ?? [];
    if (view === "deep_value") {
      return allRows.filter(
        (r: OpportunityRow) =>
          r.peg != null && r.peg <= 1 && r.discount_pct != null && r.discount_pct > 20,
      );
    }
    return allRows;
  }, [data?.rows, view]);

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load opportunities: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold tracking-tight">
          PEG Screening &middot; Opportunities
        </h2>
        <div className="flex gap-1">
          {(Object.keys(viewLabels) as View[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                view === v
                  ? "bg-[var(--blue)] text-white"
                  : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {viewLabels[v]}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] px-4 py-3 text-xs text-[var(--muted)]">
        Snapshot: {formatTimestamp(data?.snapshot_ts)}
        {typeof rows.length === "number" && (
          <span className="ml-2">• {rows.length} names shown</span>
        )}
      </div>

      {/* Main table */}
      <Table
        columns={opportunityColumns}
        data={rows}
        sortable
      />
    </div>
  );
}

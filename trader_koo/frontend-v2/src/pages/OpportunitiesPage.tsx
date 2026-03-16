import { useState } from "react";
import { useOpportunities } from "../api/hooks";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";

const opportunityColumns = [
  { key: "ticker" as const, label: "Ticker" },
  { key: "price" as const, label: "Price" },
  { key: "pe" as const, label: "P/E" },
  { key: "peg" as const, label: "PEG" },
  { key: "target_price" as const, label: "Target" },
  { key: "discount_pct" as const, label: "Discount %" },
  { key: "valuation_label" as const, label: "Valuation" },
  { key: "eps_growth_5y" as const, label: "EPS Growth 5Y" },
];

type View = "all" | "undervalued" | "deep_value" | "overvalued";

const viewLabels: Record<View, string> = {
  all: "All",
  undervalued: "Undervalued",
  deep_value: "Deep Value",
  overvalued: "Overvalued",
};

export default function OpportunitiesPage() {
  const [view, setView] = useState<View>("all");
  const apiView = view === "deep_value" ? "undervalued" : view;
  const { data, isLoading, error } = useOpportunities({ view: apiView });

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load opportunities: {(error as Error).message}
      </div>
    );
  }

  const rows = data?.rows ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold tracking-tight">Opportunities</h2>
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

      <div className="grid gap-3 sm:grid-cols-3">
        <Card label="Universe" value={data?.universe_count ?? "\u2014"} />
        <Card label="Eligible" value={data?.eligible_count ?? "\u2014"} />
        <Card label="Showing" value={rows.length} />
      </div>

      <Table
        columns={opportunityColumns}
        data={rows as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}

import { useState } from "react";
import { useEarnings } from "../api/hooks";
import Card from "../components/ui/Card";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";

const earningsColumns = [
  { key: "earnings_date" as const, label: "Date" },
  { key: "earnings_session" as const, label: "Session" },
  { key: "schedule_quality" as const, label: "Timing" },
  { key: "days_until" as const, label: "Days" },
  { key: "ticker" as const, label: "Ticker" },
  { key: "recommendation_state" as const, label: "State" },
  { key: "score" as const, label: "Score" },
  { key: "signal_bias" as const, label: "Bias" },
  { key: "earnings_risk" as const, label: "Risk" },
  { key: "sector" as const, label: "Sector" },
  { key: "price" as const, label: "Price" },
  { key: "discount_pct" as const, label: "Discount %" },
];

export default function EarningsPage() {
  const [days, setDays] = useState(21);
  const { data, isLoading, error } = useEarnings(days);

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load earnings: {(error as Error).message}
      </div>
    );
  }

  const rows = data?.rows ?? [];
  const summary = data?.summary ?? { setup_ready: 0, watch: 0, calendar_only: 0, unverified: 0 };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold tracking-tight">Earnings Calendar</h2>
        <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
          <label htmlFor="earningsDays">Lookahead (days):</label>
          <input
            id="earningsDays"
            type="number"
            min={1}
            max={90}
            value={days}
            onChange={(e) => setDays(Math.max(1, Math.min(90, Number(e.target.value) || 21)))}
            className="w-16 rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-[var(--text)]"
          />
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
        <Card label="Window" value={`${days}d`} />
        <Card label="Matches" value={data?.count ?? 0} />
        <Card label="Setup Ready" value={summary.setup_ready} />
        <Card label="Watch" value={summary.watch} />
        <Card label="Calendar Only" value={summary.calendar_only} />
        <Card label="Unverified" value={summary.unverified} />
        <Card label="Market Date" value={data?.market_date ?? "\u2014"} />
      </div>

      {data?.detail && (
        <div className="text-xs text-[var(--muted)]">{data.detail}</div>
      )}

      <Table
        columns={earningsColumns}
        data={rows as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}

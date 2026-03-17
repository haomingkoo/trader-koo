import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useEarnings } from "../api/hooks";
import type { EarningsRow, EarningsGroup, EarningsGroupSession } from "../api/types";
import Card from "../components/ui/Card";
import Badge, { tierVariant } from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";

type ViewMode = "calendar" | "table";

const recStateBadgeVariant = (
  state: string | null | undefined,
): "green" | "amber" | "red" | "muted" => {
  const s = (state ?? "").toLowerCase().replace(/_/g, " ");
  if (s.includes("setup ready")) return "green";
  if (s.includes("watch")) return "amber";
  if (s.includes("calendar")) return "muted";
  return "muted";
};

const biasBadgeVariant = (
  bias: string | null | undefined,
): "green" | "red" | "amber" | "muted" => {
  const b = (bias ?? "").toLowerCase();
  if (b.includes("bull")) return "green";
  if (b.includes("bear")) return "red";
  return "muted";
};

const riskBadgeVariant = (
  risk: string | null | undefined,
): "green" | "red" | "amber" | "muted" => {
  const r = (risk ?? "").toLowerCase();
  if (r.includes("high") || r.includes("elevated")) return "red";
  if (r.includes("normal")) return "green";
  return "amber";
};

const formatState = (s: string): string =>
  s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const earningsColumns = [
  {
    key: "ticker" as const,
    label: "Ticker",
    render: (v: unknown) => {
      const ticker = String(v ?? "");
      if (!ticker) return "\u2014";
      return (
        <Link
          to={`/v2/chart?t=${ticker}`}
          className="font-mono font-bold text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          {ticker}
        </Link>
      );
    },
  },
  { key: "earnings_date" as const, label: "Date" },
  {
    key: "earnings_session" as const,
    label: "Session",
    render: (v: unknown) => (
      <span className="uppercase">{String(v ?? "\u2014")}</span>
    ),
  },
  {
    key: "schedule_quality" as const,
    label: "Timing",
    render: (v: unknown) => {
      const val = v as string | null;
      return val ? (
        <Badge variant={val === "confirmed" ? "green" : "muted"}>
          {formatState(val)}
        </Badge>
      ) : (
        "\u2014"
      );
    },
  },
  {
    key: "days_until" as const,
    label: "Days",
    render: (v: unknown) => {
      const n = v as number | null;
      return n != null ? `${n}d` : "\u2014";
    },
  },
  {
    key: "recommendation_state" as const,
    label: "State",
    render: (v: unknown) => {
      const val = v as string | null;
      return val ? (
        <Badge variant={recStateBadgeVariant(val)}>
          {formatState(val)}
        </Badge>
      ) : (
        "\u2014"
      );
    },
  },
  {
    key: "score" as const,
    label: "Score",
    render: (v: unknown) => {
      const n = v as number | null;
      return n != null ? String(n) : "\u2014";
    },
  },
  {
    key: "signal_bias" as const,
    label: "Bias",
    render: (v: unknown) => {
      const val = v as string | null;
      return val ? (
        <Badge variant={biasBadgeVariant(val)}>
          {formatState(val)}
        </Badge>
      ) : (
        "\u2014"
      );
    },
  },
  {
    key: "earnings_risk" as const,
    label: "Risk",
    render: (v: unknown) => {
      const val = v as string | null;
      return val ? (
        <Badge variant={riskBadgeVariant(val)}>
          {formatState(val)}
        </Badge>
      ) : (
        "\u2014"
      );
    },
  },
  {
    key: "setup_tier" as const,
    label: "Tier",
    render: (_v: unknown, row: unknown) => {
      const r = row as EarningsRow;
      return r.setup_tier ? (
        <Badge variant={tierVariant(r.setup_tier)}>
          {r.setup_tier}
        </Badge>
      ) : (
        "\u2014"
      );
    },
  },
  { key: "sector" as const, label: "Sector" },
  {
    key: "price" as const,
    label: "Price",
    render: (v: unknown) => {
      const n = v as number | null;
      return n != null ? `$${n.toFixed(2)}` : "\u2014";
    },
  },
  {
    key: "discount_pct" as const,
    label: "Discount %",
    render: (v: unknown) => {
      const n = v as number | null;
      if (n == null) return "\u2014";
      const color =
        n > 0 ? "text-[var(--green)]" : n < 0 ? "text-[var(--red)]" : "";
      return (
        <span className={color}>
          {n > 0 ? "+" : ""}
          {n.toFixed(1)}%
        </span>
      );
    },
  },
];

function CalendarCard({ row }: { row: EarningsRow }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="cursor-pointer rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3 transition-colors hover:bg-[var(--panel-hover)]"
      onClick={() => setExpanded((prev) => !prev)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") setExpanded((prev) => !prev);
      }}
    >
      <div className="flex items-center justify-between gap-2">
        <Link
          to={`/v2/chart?t=${row.ticker}`}
          className="font-mono font-bold text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
          onClick={(e) => e.stopPropagation()}
        >
          {row.ticker}
        </Link>
        <div className="flex flex-wrap gap-1">
          {row.recommendation_state && (
            <Badge variant={recStateBadgeVariant(row.recommendation_state)}>
              {formatState(row.recommendation_state)}
            </Badge>
          )}
          {row.setup_tier && (
            <Badge variant={tierVariant(row.setup_tier)}>
              {row.setup_tier}
            </Badge>
          )}
        </div>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[10px] text-[var(--muted)]">
        <span>
          {row.earnings_session
            ? row.earnings_session.toUpperCase()
            : "TBD"}{" "}
          &middot; {row.days_until != null ? `${row.days_until}d` : "\u2014"}
        </span>
        {row.score != null && <span>Score: {row.score}</span>}
        {row.signal_bias && (
          <span>
            Bias:{" "}
            <span
              className={
                row.signal_bias.toLowerCase().includes("bull")
                  ? "text-[var(--green)]"
                  : row.signal_bias.toLowerCase().includes("bear")
                    ? "text-[var(--red)]"
                    : ""
              }
            >
              {formatState(row.signal_bias)}
            </span>
          </span>
        )}
        {row.earnings_risk && (
          <span>
            Risk:{" "}
            <span
              className={
                row.earnings_risk.toLowerCase().includes("high")
                  ? "text-[var(--red)]"
                  : ""
              }
            >
              {formatState(row.earnings_risk)}
            </span>
          </span>
        )}
      </div>
      {expanded && (
        <div className="mt-3 space-y-2 border-t border-[var(--line)] pt-3 text-xs text-[var(--muted)]">
          {row.observation && (
            <p>
              <strong className="text-[var(--text)]">Observation:</strong>{" "}
              {row.observation}
            </p>
          )}
          {row.action && (
            <p>
              <strong className="text-[var(--text)]">Action:</strong>{" "}
              {row.action}
            </p>
          )}
          {!row.observation && !row.action && (
            <p>No observation or action available for this ticker.</p>
          )}
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px]">
            {row.sector && <span>Sector: {row.sector}</span>}
            {row.price != null && <span>Price: ${row.price.toFixed(2)}</span>}
            {row.peg != null && <span>PEG: {row.peg.toFixed(2)}</span>}
            {row.yolo_pattern && <span>YOLO: {row.yolo_pattern}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

function SessionLane({ session }: { session: EarningsGroupSession }) {
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-2 text-xs text-[var(--muted)]">
        <span className="font-semibold text-[var(--text)]">
          {session.label || session.session || "TBD"}
        </span>
        <Badge variant="muted">{session.rows.length}</Badge>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {session.rows.map((row, idx) => (
          <CalendarCard key={`${row.ticker}-${idx}`} row={row} />
        ))}
      </div>
    </div>
  );
}

function CalendarView({ groups }: { groups: EarningsGroup[] }) {
  if (groups.length === 0) {
    return (
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
        No earnings found for the selected window.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {groups.map((group) => {
        const allRows = group.sessions.flatMap((s) => s.rows);
        return (
          <div key={group.date}>
            <div className="mb-3 flex items-center gap-3">
              <h3 className="text-sm font-semibold text-[var(--text)]">
                {group.display_date || group.date}
              </h3>
              <Badge variant="muted">
                {group.count ?? allRows.length} event(s)
              </Badge>
            </div>
            {group.sessions.length > 0 ? (
              <div className="space-y-4">
                {group.sessions.map((session) => (
                  <SessionLane
                    key={`${group.date}-${session.session}`}
                    session={session}
                  />
                ))}
              </div>
            ) : (
              <p className="text-xs text-[var(--muted)]">
                No events for this date.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function EarningsPage() {
  const [days, setDays] = useState(14);
  const [viewMode, setViewMode] = useState<ViewMode>("calendar");
  const { data, isLoading, error } = useEarnings(days);

  const groups = useMemo((): EarningsGroup[] => {
    if (data?.groups?.length) return data.groups;
    if (data?.rows?.length) {
      const grouped: Record<string, EarningsRow[]> = {};
      for (const row of data.rows) {
        const date = row.earnings_date ?? "Unknown";
        if (!grouped[date]) grouped[date] = [];
        grouped[date].push(row);
      }
      return Object.entries(grouped)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([date, rows]) => ({
          date,
          display_date: rows[0]?.display_date ?? date,
          count: rows.length,
          sessions: [{ session: "all", label: "All Sessions", rows }],
        }));
    }
    return [];
  }, [data]);

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load earnings: {(error as Error).message}
      </div>
    );
  }

  const rows = data?.rows ?? [];
  const summary = data?.summary ?? {
    window_days: days,
    total_events: 0,
    high_risk: 0,
    elevated_risk: 0,
    setup_ready: 0,
    watch: 0,
    calendar_only: 0,
    unverified: 0,
    by_session: {},
    scored_rows: 0,
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold tracking-tight">
          Earnings Calendar
        </h2>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
            <label htmlFor="earningsDays">Lookahead (days):</label>
            <input
              id="earningsDays"
              type="number"
              min={1}
              max={90}
              value={days}
              onChange={(e) =>
                setDays(
                  Math.max(1, Math.min(90, Number(e.target.value) || 14)),
                )
              }
              className="w-16 rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-[var(--text)]"
            />
          </div>
          <div className="flex gap-1">
            {(["calendar", "table"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                  viewMode === mode
                    ? "bg-[var(--blue)] text-white"
                    : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Summary KPI cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
        <Card
          glass
          label="Matches"
          value={data?.count ?? 0}
        />
        <Card
          glass
          label="Setup Ready"
          value={summary.setup_ready}
        />
        <Card
          glass
          label="Watch"
          value={summary.watch}
        />
        <Card
          glass
          label="Calendar Only"
          value={summary.calendar_only}
        />
        <Card
          glass
          label="Unverified"
          value={summary.unverified}
        />
        <Card
          glass
          label="Market Date"
          value={data?.market_date ?? "\u2014"}
        />
        <Card
          glass
          label="Provider"
          value={data?.provider ?? "\u2014"}
        />
      </div>

      {data?.detail && (
        <div className="text-xs text-[var(--muted)]">{data.detail}</div>
      )}

      {/* View toggle */}
      {viewMode === "calendar" ? (
        <CalendarView groups={groups} />
      ) : (
        <Table
          columns={earningsColumns}
          data={rows}
          sortable
        />
      )}
    </div>
  );
}

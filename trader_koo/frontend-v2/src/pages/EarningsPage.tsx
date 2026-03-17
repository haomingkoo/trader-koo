import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useEarnings } from "../api/hooks";
import type { EarningsRow, EarningsGroup } from "../api/types";
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
          to={`/chart?t=${ticker}`}
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

/** Session label mapping for compact badges */
const sessionBadgeLabel = (session: string | null | undefined): string => {
  const s = (session ?? "").toLowerCase();
  if (s.includes("bmo") || s.includes("pre")) return "BMO";
  if (s.includes("amc") || s.includes("after")) return "AMC";
  return "TBD";
};

const sessionBadgeVariant = (
  session: string | null | undefined,
): "blue" | "amber" | "muted" => {
  const s = (session ?? "").toLowerCase();
  if (s.includes("bmo") || s.includes("pre")) return "blue";
  if (s.includes("amc") || s.includes("after")) return "amber";
  return "muted";
};

function CalendarTickerRow({ row }: { row: EarningsRow }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <tr
        className="cursor-pointer border-b border-[var(--line)] last:border-b-0 transition-colors hover:bg-[var(--panel-hover)]"
        onClick={() => setExpanded((prev) => !prev)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") setExpanded((prev) => !prev);
        }}
      >
        {/* Ticker */}
        <td className="py-2 pl-3 pr-2">
          <Link
            to={`/chart?t=${row.ticker}`}
            className="font-mono text-sm font-bold text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
            onClick={(e) => e.stopPropagation()}
          >
            {row.ticker}
          </Link>
        </td>
        {/* Session badge */}
        <td className="px-2 py-2">
          <Badge variant={sessionBadgeVariant(row.earnings_session)}>
            {sessionBadgeLabel(row.earnings_session)}
          </Badge>
        </td>
        {/* Recommendation state */}
        <td className="px-2 py-2">
          {row.recommendation_state ? (
            <Badge variant={recStateBadgeVariant(row.recommendation_state)}>
              {formatState(row.recommendation_state)}
            </Badge>
          ) : (
            <span className="text-xs text-[var(--muted)]">{"\u2014"}</span>
          )}
        </td>
        {/* Tier */}
        <td className="px-2 py-2">
          {row.setup_tier ? (
            <Badge variant={tierVariant(row.setup_tier)}>
              {row.setup_tier}
            </Badge>
          ) : (
            <span className="text-xs text-[var(--muted)]">{"\u2014"}</span>
          )}
        </td>
        {/* Score */}
        <td className="px-2 py-2 text-right tabular-nums text-xs font-semibold text-[var(--text)]">
          {row.score != null ? row.score : "\u2014"}
        </td>
        {/* Bias */}
        <td className="px-2 py-2">
          {row.signal_bias ? (
            <Badge variant={biasBadgeVariant(row.signal_bias)}>
              {formatState(row.signal_bias)}
            </Badge>
          ) : (
            <span className="text-xs text-[var(--muted)]">{"\u2014"}</span>
          )}
        </td>
        {/* Risk */}
        <td className="px-2 py-2">
          {row.earnings_risk ? (
            <Badge variant={riskBadgeVariant(row.earnings_risk)}>
              {formatState(row.earnings_risk)}
            </Badge>
          ) : (
            <span className="text-xs text-[var(--muted)]">{"\u2014"}</span>
          )}
        </td>
        {/* Expand indicator */}
        <td className="px-2 py-2 text-right text-[var(--muted)]">
          <span
            className={`inline-block text-xs transition-transform ${expanded ? "rotate-90" : ""}`}
          >
            &#9656;
          </span>
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-[var(--line)] last:border-b-0">
          <td colSpan={8} className="px-3 pb-3 pt-1">
            <div className="rounded-lg bg-[var(--bg)] p-3 text-xs text-[var(--muted)] space-y-2">
              {row.observation && (
                <p>
                  <strong className="text-[var(--text)]">Observation:</strong>{" "}
                  {String(row.observation)}
                </p>
              )}
              {row.action && (
                <p>
                  <strong className="text-[var(--text)]">Action:</strong>{" "}
                  {String(row.action)}
                </p>
              )}
              {!row.observation && !row.action && (
                <p>No observation or action available for this ticker.</p>
              )}
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px]">
                {row.sector && <span>Sector: {String(row.sector)}</span>}
                {row.price != null && (
                  <span>Price: ${row.price.toFixed(2)}</span>
                )}
                {row.peg != null && <span>PEG: {row.peg.toFixed(2)}</span>}
                {row.discount_pct != null && (
                  <span
                    className={
                      row.discount_pct > 0
                        ? "text-[var(--green)]"
                        : row.discount_pct < 0
                          ? "text-[var(--red)]"
                          : ""
                    }
                  >
                    Discount: {row.discount_pct > 0 ? "+" : ""}
                    {row.discount_pct.toFixed(1)}%
                  </span>
                )}
                {row.yolo_pattern && (
                  <span>YOLO: {String(row.yolo_pattern)}</span>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
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
    <div className="space-y-4">
      {groups.map((group) => {
        const allRows = group.sessions.flatMap((s) => s.rows);
        const rowCount = group.count ?? allRows.length;
        return (
          <Card key={group.date} className="overflow-hidden">
            {/* Date header */}
            <div className="flex items-center gap-3 pb-3 border-b border-[var(--line)]">
              <h3 className="text-sm font-bold text-[var(--text)]">
                {group.display_date || group.date}
              </h3>
              <Badge variant="muted">
                {rowCount} {rowCount === 1 ? "event" : "events"}
              </Badge>
            </div>

            {allRows.length > 0 ? (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-[var(--muted)]">
                      <th className="py-1.5 pl-3 pr-2 font-semibold">Ticker</th>
                      <th className="px-2 py-1.5 font-semibold">Session</th>
                      <th className="px-2 py-1.5 font-semibold">State</th>
                      <th className="px-2 py-1.5 font-semibold">Tier</th>
                      <th className="px-2 py-1.5 text-right font-semibold">Score</th>
                      <th className="px-2 py-1.5 font-semibold">Bias</th>
                      <th className="px-2 py-1.5 font-semibold">Risk</th>
                      <th className="px-2 py-1.5 w-6" />
                    </tr>
                  </thead>
                  <tbody>
                    {allRows.map((row, idx) => (
                      <CalendarTickerRow
                        key={`${row.ticker}-${idx}`}
                        row={row}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="mt-3 text-xs text-[var(--muted)]">
                No events for this date.
              </p>
            )}
          </Card>
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
        Failed to load earnings: {String((error as Error)?.message ?? "Unknown error")}
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
        <div className="text-xs text-[var(--muted)]">{String(data.detail)}</div>
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

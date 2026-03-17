import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useEarnings } from "../api/hooks";
import type { EarningsRow } from "../api/types";
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
      const val = typeof v === "string" ? v : null;
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
      const n = typeof v === "number" ? v : null;
      return n != null ? `${n}d` : "\u2014";
    },
  },
  {
    key: "recommendation_state" as const,
    label: "State",
    render: (v: unknown) => {
      const val = typeof v === "string" ? v : null;
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
      return typeof v === "number" ? String(v) : "\u2014";
    },
  },
  {
    key: "signal_bias" as const,
    label: "Bias",
    render: (v: unknown) => {
      const val = typeof v === "string" ? v : null;
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
      const val = typeof v === "string" ? v : null;
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
        <span className={color}>
          {n > 0 ? "+" : ""}
          {n.toFixed(1)}%
        </span>
      );
    },
  },
];

/* ── Week Grid Calendar helpers ── */

type SessionKey = "pre" | "tbd" | "amc";

const SESSION_ORDER: SessionKey[] = ["pre", "tbd", "amc"];

const SESSION_LABELS: Record<SessionKey, string> = {
  pre: "PRE",
  tbd: "TBD",
  amc: "AMC",
};

const SESSION_COLORS: Record<SessionKey, string> = {
  pre: "text-[var(--blue)]",
  tbd: "text-[var(--muted)]",
  amc: "text-[var(--amber)]",
};

const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"] as const;

interface DayBucket {
  date: Date;
  dateStr: string;
  label: string;
  sessions: Record<SessionKey, EarningsRow[]>;
}

/** Classify a row's session into pre/tbd/amc */
function classifySession(session: string | null | undefined): SessionKey {
  const s = (session ?? "").toLowerCase();
  if (s.includes("bmo") || s.includes("pre")) return "pre";
  if (s.includes("amc") || s.includes("after")) return "amc";
  return "tbd";
}

/** Get Monday of the week containing `date` */
function getMonday(date: Date): Date {
  const d = new Date(date);
  const day = d.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  d.setDate(d.getDate() + diff);
  d.setHours(0, 0, 0, 0);
  return d;
}

/** Format date as "Mon Mar 17" */
function formatDayHeader(date: Date): string {
  const dayName = DAY_NAMES[date.getDay() === 0 ? 4 : date.getDay() - 1] ?? "?";
  const month = date.toLocaleDateString("en-US", { month: "short" });
  return `${dayName} ${month} ${date.getDate()}`;
}

/** Format YYYY-MM-DD string for comparison */
function toDateKey(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/** Get the today date key in local time */
function getTodayKey(): string {
  return toDateKey(new Date());
}

/** Tier dot color mapping */
function tierDotColor(tier: string | null | undefined): string {
  if (!tier) return "bg-[var(--muted)]";
  const t = tier.toUpperCase();
  if (t === "A") return "bg-[var(--green)]";
  if (t === "B") return "bg-[var(--amber)]";
  if (t === "C") return "bg-[var(--red)]";
  return "bg-[var(--muted)]";
}

function TickerTile({ row }: { row: EarningsRow }) {
  return (
    <Link
      to={`/chart?t=${row.ticker}`}
      className="group flex items-center gap-1.5 rounded px-1.5 py-0.5 transition-colors hover:bg-[var(--panel-hover)]"
      title={[
        row.ticker,
        row.setup_tier ? `Tier ${row.setup_tier}` : null,
        row.signal_bias ? formatState(row.signal_bias) : null,
        row.score != null ? `Score ${row.score}` : null,
      ]
        .filter(Boolean)
        .join(" \u2022 ")}
    >
      <span
        className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${tierDotColor(row.setup_tier)}`}
        aria-label={row.setup_tier ? `Tier ${row.setup_tier}` : "No tier"}
      />
      <span className="text-sm font-mono font-bold text-[var(--accent)] group-hover:text-[var(--blue)] transition-colors">
        {row.ticker}
      </span>
    </Link>
  );
}

function WeekGridCalendar({ rows }: { rows: EarningsRow[] }) {
  const [weekOffset, setWeekOffset] = useState(0);

  const todayKey = useMemo(() => getTodayKey(), []);

  const weekDays = useMemo((): DayBucket[] => {
    const today = new Date();
    const monday = getMonday(today);
    monday.setDate(monday.getDate() + weekOffset * 7);

    const buckets: DayBucket[] = [];
    for (let i = 0; i < 5; i++) {
      const d = new Date(monday);
      d.setDate(monday.getDate() + i);
      buckets.push({
        date: d,
        dateStr: toDateKey(d),
        label: formatDayHeader(d),
        sessions: { pre: [], tbd: [], amc: [] },
      });
    }

    // Index buckets by dateStr for fast lookup
    const bucketMap = new Map<string, DayBucket>();
    for (const bucket of buckets) {
      bucketMap.set(bucket.dateStr, bucket);
    }

    // Fill buckets with rows
    for (const row of rows) {
      const dateStr = row.earnings_date;
      const bucket = bucketMap.get(dateStr);
      if (bucket) {
        const session = classifySession(row.earnings_session);
        bucket.sessions[session].push(row);
      }
    }

    return buckets;
  }, [rows, weekOffset]);

  const weekLabel = useMemo(() => {
    if (weekDays.length === 0) return "";
    const first = weekDays[0];
    const last = weekDays[weekDays.length - 1];
    if (!first || !last) return "";
    const fmtOpts: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
    };
    return `${first.date.toLocaleDateString("en-US", fmtOpts)} \u2013 ${last.date.toLocaleDateString("en-US", fmtOpts)}`;
  }, [weekDays]);

  const totalThisWeek = useMemo(
    () =>
      weekDays.reduce(
        (sum, day) =>
          sum +
          day.sessions.pre.length +
          day.sessions.tbd.length +
          day.sessions.amc.length,
        0,
      ),
    [weekDays],
  );

  return (
    <div className="space-y-3">
      {/* Week navigation */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setWeekOffset((p) => p - 1)}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs font-semibold text-[var(--muted)] hover:text-[var(--text)] transition-colors"
            aria-label="Previous week"
          >
            &larr;
          </button>
          <button
            onClick={() => setWeekOffset(0)}
            className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
              weekOffset === 0
                ? "bg-[var(--blue)] text-white"
                : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            This Week
          </button>
          <button
            onClick={() => setWeekOffset(1)}
            className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
              weekOffset === 1
                ? "bg-[var(--blue)] text-white"
                : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            Next Week
          </button>
          <button
            onClick={() => setWeekOffset((p) => p + 1)}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs font-semibold text-[var(--muted)] hover:text-[var(--text)] transition-colors"
            aria-label="Next week"
          >
            &rarr;
          </button>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-[var(--text)]">
            {weekLabel}
          </span>
          <Badge variant="muted">
            {totalThisWeek} {totalThisWeek === 1 ? "event" : "events"}
          </Badge>
        </div>
      </div>

      {/* 5-day grid — responsive: stack on narrow, 3 cols on md, 5 on lg */}
      <div className="grid grid-cols-1 gap-px rounded-xl border border-[var(--line)] bg-[var(--line)] overflow-hidden md:grid-cols-3 lg:grid-cols-5">
        {weekDays.map((day) => {
          const isToday = day.dateStr === todayKey;
          const dayTotal =
            day.sessions.pre.length +
            day.sessions.tbd.length +
            day.sessions.amc.length;

          return (
            <div
              key={day.dateStr}
              className={`flex flex-col bg-[var(--panel)] ${
                isToday
                  ? "bg-[rgba(74,158,255,0.05)] ring-1 ring-inset ring-[rgba(74,158,255,0.3)]"
                  : ""
              }`}
            >
              {/* Day header */}
              <div
                className={`flex items-center justify-between border-b px-3 py-2 ${
                  isToday
                    ? "border-[rgba(74,158,255,0.3)]"
                    : "border-[var(--line)]"
                }`}
              >
                <span
                  className={`text-xs font-bold ${
                    isToday ? "text-[var(--accent)]" : "text-[var(--text)]"
                  }`}
                >
                  {day.label}
                </span>
                {dayTotal > 0 && (
                  <span className="text-[10px] font-medium text-[var(--muted)]">
                    {dayTotal}
                  </span>
                )}
              </div>

              {/* Session sub-columns */}
              <div className="grid grid-cols-3 flex-1">
                {SESSION_ORDER.map((sessionKey) => {
                  const sessionRows = day.sessions[sessionKey];
                  return (
                    <div
                      key={sessionKey}
                      className="flex flex-col border-r border-[var(--line)] last:border-r-0"
                    >
                      {/* Session header badge */}
                      <div className="border-b border-[var(--line)] px-1.5 py-1 text-center">
                        <span
                          className={`text-[10px] font-bold uppercase tracking-wider ${SESSION_COLORS[sessionKey]}`}
                        >
                          {SESSION_LABELS[sessionKey]}
                        </span>
                      </div>

                      {/* Ticker list */}
                      <div className="flex flex-col gap-0.5 px-1 py-1.5 min-h-[2.5rem]">
                        {sessionRows.length > 0 ? (
                          sessionRows.map((row) => (
                            <TickerTile
                              key={`${row.ticker}-${day.dateStr}-${sessionKey}`}
                              row={row}
                            />
                          ))
                        ) : (
                          <span className="text-[10px] text-[var(--muted)] text-center py-1">
                            &mdash;
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function EarningsPage() {
  const [days, setDays] = useState(30);
  const [viewMode, setViewMode] = useState<ViewMode>("calendar");
  const { data, isLoading, error } = useEarnings(days);

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
            <label htmlFor="earningsDays">Data window (days):</label>
            <input
              id="earningsDays"
              type="number"
              min={7}
              max={90}
              value={days}
              onChange={(e) =>
                setDays(
                  Math.max(7, Math.min(90, Number(e.target.value) || 30)),
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
        <WeekGridCalendar rows={rows} />
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

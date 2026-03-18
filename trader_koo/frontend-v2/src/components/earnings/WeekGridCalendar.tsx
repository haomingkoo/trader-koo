import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { EarningsRow } from "../../api/types";
import Badge from "../ui/Badge";

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

function classifySession(session: string | null | undefined): SessionKey {
  const value = (session ?? "").toLowerCase();
  if (value.includes("bmo") || value.includes("pre")) return "pre";
  if (value.includes("amc") || value.includes("after")) return "amc";
  return "tbd";
}

function getMonday(date: Date): Date {
  const next = new Date(date);
  const day = next.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  next.setDate(next.getDate() + diff);
  next.setHours(0, 0, 0, 0);
  return next;
}

function formatDayHeader(date: Date): string {
  const dayName = DAY_NAMES[date.getDay() === 0 ? 4 : date.getDay() - 1] ?? "?";
  const month = date.toLocaleDateString("en-US", { month: "short" });
  return `${dayName} ${month} ${date.getDate()}`;
}

function toDateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function getTodayKey(): string {
  return toDateKey(new Date());
}

function formatState(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function tierDotColor(tier: string | null | undefined): string {
  if (!tier) return "bg-[var(--muted)]";
  const value = tier.toUpperCase();
  if (value === "A") return "bg-[var(--green)]";
  if (value === "B") return "bg-[var(--amber)]";
  if (value === "C") return "bg-[var(--red)]";
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
      <span className="font-mono text-sm font-bold text-[var(--accent)] transition-colors group-hover:text-[var(--blue)]">
        {row.ticker}
      </span>
    </Link>
  );
}

export default function WeekGridCalendar({ rows }: { rows: EarningsRow[] }) {
  const [weekOffset, setWeekOffset] = useState(0);
  const todayKey = useMemo(() => getTodayKey(), []);

  const weekDays = useMemo((): DayBucket[] => {
    const monday = getMonday(new Date());
    monday.setDate(monday.getDate() + weekOffset * 7);

    const buckets: DayBucket[] = [];
    for (let index = 0; index < 5; index++) {
      const date = new Date(monday);
      date.setDate(monday.getDate() + index);
      buckets.push({
        date,
        dateStr: toDateKey(date),
        label: formatDayHeader(date),
        sessions: { pre: [], tbd: [], amc: [] },
      });
    }

    const bucketMap = new Map<string, DayBucket>();
    for (const bucket of buckets) {
      bucketMap.set(bucket.dateStr, bucket);
    }

    for (const row of rows) {
      const bucket = bucketMap.get(row.earnings_date);
      if (bucket) {
        bucket.sessions[classifySession(row.earnings_session)].push(row);
      }
    }

    return buckets;
  }, [rows, weekOffset]);

  const weekLabel = useMemo(() => {
    if (weekDays.length === 0) return "";
    const first = weekDays[0];
    const last = weekDays[weekDays.length - 1];
    if (!first || !last) return "";
    const options: Intl.DateTimeFormatOptions = { month: "short", day: "numeric" };
    return `${first.date.toLocaleDateString("en-US", options)} \u2013 ${last.date.toLocaleDateString("en-US", options)}`;
  }, [weekDays]);

  const totalThisWeek = useMemo(
    () =>
      weekDays.reduce(
        (sum, day) =>
          sum + day.sessions.pre.length + day.sessions.tbd.length + day.sessions.amc.length,
        0,
      ),
    [weekDays],
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setWeekOffset((prev) => prev - 1)}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs font-semibold text-[var(--muted)] transition-colors hover:text-[var(--text)]"
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
            onClick={() => setWeekOffset((prev) => prev + 1)}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs font-semibold text-[var(--muted)] transition-colors hover:text-[var(--text)]"
            aria-label="Next week"
          >
            &rarr;
          </button>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-[var(--text)]">{weekLabel}</span>
          <Badge variant="muted">
            {totalThisWeek} {totalThisWeek === 1 ? "event" : "events"}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-px overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--line)] md:grid-cols-3 lg:grid-cols-5">
        {weekDays.map((day) => {
          const isToday = day.dateStr === todayKey;
          const dayTotal =
            day.sessions.pre.length + day.sessions.tbd.length + day.sessions.amc.length;

          return (
            <div
              key={day.dateStr}
              className={`flex flex-col bg-[var(--panel)] ${
                isToday
                  ? "bg-[rgba(74,158,255,0.05)] ring-1 ring-inset ring-[rgba(74,158,255,0.3)]"
                  : ""
              }`}
            >
              <div
                className={`flex items-center justify-between border-b px-3 py-2 ${
                  isToday ? "border-[rgba(74,158,255,0.3)]" : "border-[var(--line)]"
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
                  <span className="text-[10px] font-medium text-[var(--muted)]">{dayTotal}</span>
                )}
              </div>

              <div className="grid flex-1 grid-cols-3">
                {SESSION_ORDER.map((sessionKey) => {
                  const sessionRows = day.sessions[sessionKey];
                  return (
                    <div
                      key={sessionKey}
                      className="flex flex-col border-r border-[var(--line)] last:border-r-0"
                    >
                      <div className="border-b border-[var(--line)] px-1.5 py-1 text-center">
                        <span
                          className={`text-[10px] font-bold uppercase tracking-wider ${SESSION_COLORS[sessionKey]}`}
                        >
                          {SESSION_LABELS[sessionKey]}
                        </span>
                      </div>
                      <div className="flex min-h-[2.5rem] flex-col gap-0.5 px-1 py-1.5">
                        {sessionRows.length > 0 ? (
                          sessionRows.map((row) => (
                            <TickerTile
                              key={`${row.ticker}-${day.dateStr}-${sessionKey}`}
                              row={row}
                            />
                          ))
                        ) : (
                          <span className="py-1 text-center text-[10px] text-[var(--muted)]">
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

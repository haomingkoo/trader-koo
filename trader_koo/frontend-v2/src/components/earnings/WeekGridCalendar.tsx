import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { EarningsRow } from "../../api/types";
import Badge from "../ui/Badge";
import TickerLogo from "./TickerLogo";

/* ── Types ── */

type SessionKey = "pre" | "tbd" | "amc";
type RangePreset = "5d" | "7d" | "14d";

interface WeekGridCalendarProps {
  rows: EarningsRow[];
}

interface DayBucket {
  date: Date;
  dateStr: string;
  dayName: string;
  dateLabel: string;
  sessions: Record<SessionKey, EarningsRow[]>;
}

/* ── Constants ── */

const SESSION_ORDER: SessionKey[] = ["pre", "tbd", "amc"];

const SESSION_LABELS: Record<SessionKey, string> = {
  pre: "PRE",
  tbd: "TBD",
  amc: "AFT",
};

const SESSION_FULL_LABELS: Record<SessionKey, string> = {
  pre: "PREMARKET",
  tbd: "TBD",
  amc: "AFTERHOURS",
};

const SESSION_COLORS: Record<SessionKey, string> = {
  pre: "text-[var(--amber)]",
  tbd: "text-[var(--muted)]",
  amc: "text-[var(--blue)]",
};

const SESSION_BG: Record<SessionKey, string> = {
  pre: "bg-[rgba(248,194,78,0.04)]",
  tbd: "",
  amc: "bg-[rgba(106,169,255,0.04)]",
};

const DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] as const;

const RANGE_OPTIONS: { value: RangePreset; label: string; days: number }[] = [
  { value: "5d", label: "5D", days: 5 },
  { value: "7d", label: "7D", days: 7 },
  { value: "14d", label: "14D", days: 14 },
];

/* ── Helpers ── */

function classifySession(session: string | null | undefined): SessionKey {
  const value = (session ?? "").toLowerCase();
  if (value.includes("bmo") || value.includes("pre")) return "pre";
  if (value.includes("amc") || value.includes("after")) return "amc";
  return "tbd";
}

function getMonday(date: Date): Date {
  const d = new Date(date);
  const day = d.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  d.setDate(d.getDate() + diff);
  d.setHours(0, 0, 0, 0);
  return d;
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
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function tierColor(tier: string | null | undefined): string {
  if (!tier) return "border-[var(--line)] bg-[var(--panel)]";
  const t = tier.toUpperCase();
  if (t === "A")
    return "border-[rgba(56,211,159,0.4)] bg-[rgba(56,211,159,0.08)]";
  if (t === "B")
    return "border-[rgba(248,194,78,0.4)] bg-[rgba(248,194,78,0.08)]";
  if (t === "C")
    return "border-[rgba(255,107,107,0.3)] bg-[rgba(255,107,107,0.06)]";
  return "border-[var(--line)] bg-[var(--panel)]";
}

function tierTextColor(tier: string | null | undefined): string {
  if (!tier) return "";
  const t = tier.toUpperCase();
  if (t === "A") return "text-[var(--green)]";
  if (t === "B") return "text-[var(--amber)]";
  if (t === "C") return "text-[var(--red)]";
  return "";
}

function biasIcon(bias: string | null | undefined): string | null {
  const b = (bias ?? "").toLowerCase();
  if (b.includes("bull")) return "\u25B2";
  if (b.includes("bear")) return "\u25BC";
  return null;
}

function biasColor(bias: string | null | undefined): string {
  const b = (bias ?? "").toLowerCase();
  if (b.includes("bull")) return "text-[var(--green)]";
  if (b.includes("bear")) return "text-[var(--red)]";
  return "";
}

const TIER_SORT_ORDER: Record<string, number> = { A: 0, B: 1, C: 2 };

function tierSortValue(tier: string | null | undefined): number {
  if (!tier) return 9;
  return TIER_SORT_ORDER[tier.toUpperCase()] ?? 3;
}

/* ── Ticker Tile ── */

function TickerTile({ row }: { row: EarningsRow }) {
  const icon = biasIcon(row.signal_bias);
  const colorCls = biasColor(row.signal_bias);

  return (
    <Link
      to={`/chart?t=${row.ticker}`}
      className={`group flex items-center gap-1.5 rounded-md border px-2 py-1 transition-all hover:scale-[1.02] hover:brightness-125 ${tierColor(row.setup_tier)}`}
      title={[
        row.ticker,
        row.company_name,
        row.sector,
        row.setup_tier ? `Tier ${row.setup_tier}` : null,
        row.signal_bias ? formatState(row.signal_bias) : null,
        row.score != null ? `Score ${row.score}` : null,
        row.price != null ? `$${row.price.toFixed(2)}` : null,
      ]
        .filter(Boolean)
        .join(" \u2022 ")}
    >
      <TickerLogo ticker={row.ticker} size={20} />
      {row.setup_tier && (
        <span
          className={`text-[9px] font-black leading-none ${tierTextColor(row.setup_tier)}`}
        >
          {row.setup_tier.toUpperCase()}
        </span>
      )}
      <span className="font-mono text-xs font-bold text-[var(--accent)] transition-colors group-hover:text-[var(--blue)]">
        {row.ticker}
      </span>
      {icon && <span className={`text-[9px] leading-none ${colorCls}`}>{icon}</span>}
    </Link>
  );
}

/* ── Desktop Grid Row (one session across all days) ── */

interface SessionRowProps {
  sessionKey: SessionKey;
  week: DayBucket[];
  todayKey: string;
}

function SessionRow({ sessionKey, week, todayKey }: SessionRowProps) {
  return (
    <div className="flex border-b border-[var(--line)] last:border-b-0">
      {/* Session label cell */}
      <div className="flex w-16 shrink-0 flex-col items-center justify-center border-r border-[var(--line)] bg-[var(--panel)] py-3">
        <span
          className={`text-[9px] font-bold uppercase tracking-widest ${SESSION_COLORS[sessionKey]}`}
        >
          {SESSION_LABELS[sessionKey]}
        </span>
      </div>

      {/* Day cells */}
      <div className="grid flex-1" style={{ gridTemplateColumns: `repeat(${week.length}, 1fr)` }}>
        {week.map((day) => {
          const isToday = day.dateStr === todayKey;
          const sessionRows = day.sessions[sessionKey];

          return (
            <div
              key={day.dateStr}
              className={`flex min-h-[3rem] flex-col border-r border-[var(--line)] last:border-r-0 ${
                isToday ? "bg-[rgba(74,158,255,0.04)]" : SESSION_BG[sessionKey]
              }`}
            >
              <div className="flex flex-wrap gap-1 px-2 py-1.5">
                {sessionRows.length > 0 ? (
                  sessionRows.map((row) => (
                    <TickerTile
                      key={`${row.ticker}-${day.dateStr}-${sessionKey}`}
                      row={row}
                    />
                  ))
                ) : (
                  <span className="flex w-full items-center justify-center py-1 text-[10px] text-[var(--line)]">
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
}

/* ── Mobile Day Card (stacked vertical layout) ── */

function MobileDayCard({
  day,
  isToday,
}: {
  day: DayBucket;
  isToday: boolean;
}) {
  const total =
    day.sessions.pre.length +
    day.sessions.tbd.length +
    day.sessions.amc.length;

  return (
    <div
      className={`overflow-hidden rounded-xl border ${
        isToday
          ? "border-[rgba(74,158,255,0.4)] bg-[rgba(74,158,255,0.04)]"
          : "border-[var(--line)] bg-[var(--bg)]"
      }`}
    >
      {/* Day header */}
      <div
        className={`flex items-center justify-between px-3 py-2 ${
          isToday ? "bg-[rgba(74,158,255,0.08)]" : "bg-[var(--panel)]"
        }`}
      >
        <div className="flex items-baseline gap-2">
          <span
            className={`text-sm font-bold ${
              isToday ? "text-[var(--accent)]" : "text-[var(--text)]"
            }`}
          >
            {day.dayName}
          </span>
          <span
            className={`text-xs ${
              isToday ? "text-[var(--accent)]" : "text-[var(--muted)]"
            }`}
          >
            {day.dateLabel}
          </span>
        </div>
        {total > 0 && (
          <span className="text-[10px] font-medium text-[var(--muted)]">
            {total}
          </span>
        )}
      </div>

      {/* Sessions */}
      {SESSION_ORDER.map((sk) => {
        const sessionRows = day.sessions[sk];
        if (sessionRows.length === 0) return null;
        return (
          <div key={sk} className="border-t border-[var(--line)]">
            <div className="flex items-center gap-2 px-3 py-1">
              <span
                className={`text-[9px] font-bold uppercase tracking-widest ${SESSION_COLORS[sk]}`}
              >
                {SESSION_FULL_LABELS[sk]}
              </span>
              <span className="text-[9px] text-[var(--muted)]">
                {sessionRows.length}
              </span>
            </div>
            <div className="flex flex-wrap gap-1 px-2 pb-2">
              {sessionRows.map((row) => (
                <TickerTile
                  key={`${row.ticker}-${day.dateStr}-${sk}`}
                  row={row}
                />
              ))}
            </div>
          </div>
        );
      })}

      {/* Empty state */}
      {total === 0 && (
        <div className="py-4 text-center text-[10px] text-[var(--muted)]">
          No earnings
        </div>
      )}
    </div>
  );
}

/* ── Main Component ── */

export default function WeekGridCalendar({ rows }: WeekGridCalendarProps) {
  const [weekOffset, setWeekOffset] = useState(0);
  const [range, setRange] = useState<RangePreset>("5d");
  const todayKey = useMemo(() => getTodayKey(), []);
  const autoAdvanced = useRef(false);

  // Auto-advance to next week with events if current week is empty
  useEffect(() => {
    if (autoAdvanced.current || weekOffset !== 0 || rows.length === 0) return;
    const thisMonday = getMonday(new Date());
    const thisFriday = new Date(thisMonday);
    thisFriday.setDate(thisMonday.getDate() + 4);
    const thisWeekEnd = toDateKey(thisFriday);
    const thisWeekStart = toDateKey(thisMonday);
    const hasThisWeek = rows.some((r) => r.earnings_date >= thisWeekStart && r.earnings_date <= thisWeekEnd);
    if (!hasThisWeek) {
      // Find the earliest future event and jump to its week
      const future = rows
        .filter((r) => r.earnings_date > thisWeekEnd)
        .sort((a, b) => a.earnings_date.localeCompare(b.earnings_date));
      if (future.length > 0) {
        const targetDate = new Date(future[0].earnings_date + "T00:00:00");
        const targetMonday = getMonday(targetDate);
        const diffMs = targetMonday.getTime() - thisMonday.getTime();
        const diffWeeks = Math.round(diffMs / (7 * 24 * 60 * 60 * 1000));
        if (diffWeeks > 0) {
          setWeekOffset(diffWeeks);
          autoAdvanced.current = true;
        }
      }
    }
  }, [rows, weekOffset]);

  const rangeDays = RANGE_OPTIONS.find((o) => o.value === range)?.days ?? 5;

  /* Build day buckets for the selected range */
  const dayBuckets = useMemo((): DayBucket[] => {
    const monday = getMonday(new Date());
    monday.setDate(monday.getDate() + weekOffset * 7);

    const buckets: DayBucket[] = [];
    let daysAdded = 0;
    let offset = 0;

    while (daysAdded < rangeDays) {
      const date = new Date(monday);
      date.setDate(monday.getDate() + offset);
      offset++;

      const dayOfWeek = date.getDay();
      if (dayOfWeek === 0 || dayOfWeek === 6) continue;

      buckets.push({
        date,
        dateStr: toDateKey(date),
        dayName: DAY_NAMES[dayOfWeek] ?? "?",
        dateLabel: `${date.getMonth() + 1}/${date.getDate()}`,
        sessions: { pre: [], tbd: [], amc: [] },
      });
      daysAdded++;
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

    const tierSort = (a: EarningsRow, b: EarningsRow): number =>
      tierSortValue(a.setup_tier) - tierSortValue(b.setup_tier);

    for (const bucket of buckets) {
      bucket.sessions.pre.sort(tierSort);
      bucket.sessions.tbd.sort(tierSort);
      bucket.sessions.amc.sort(tierSort);
    }

    return buckets;
  }, [rows, weekOffset, rangeDays]);

  /* Group into weeks (5-day chunks) for multi-week ranges */
  const weeks = useMemo((): DayBucket[][] => {
    if (rangeDays <= 5) return [dayBuckets];
    const result: DayBucket[][] = [];
    for (let i = 0; i < dayBuckets.length; i += 5) {
      result.push(dayBuckets.slice(i, i + 5));
    }
    return result;
  }, [dayBuckets, rangeDays]);

  const dateRangeLabel = useMemo(() => {
    if (dayBuckets.length === 0) return "";
    const first = dayBuckets[0];
    const last = dayBuckets[dayBuckets.length - 1];
    if (!first || !last) return "";
    const opts: Intl.DateTimeFormatOptions = {
      month: "short",
      day: "numeric",
    };
    return `${first.date.toLocaleDateString("en-US", opts)} \u2013 ${last.date.toLocaleDateString("en-US", opts)}`;
  }, [dayBuckets]);

  const totalEvents = useMemo(
    () =>
      dayBuckets.reduce(
        (sum, day) =>
          sum +
          day.sessions.pre.length +
          day.sessions.tbd.length +
          day.sessions.amc.length,
        0,
      ),
    [dayBuckets],
  );

  return (
    <div className="space-y-4">
      {/* ── Controls bar ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Left: navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setWeekOffset((p) => p - 1)}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs font-semibold text-[var(--muted)] transition-colors hover:text-[var(--text)]"
            aria-label="Previous week"
          >
            &#8592;
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
            onClick={() => setWeekOffset((p) => p + 1)}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs font-semibold text-[var(--muted)] transition-colors hover:text-[var(--text)]"
            aria-label="Next week"
          >
            &#8594;
          </button>
        </div>

        {/* Center: date label + count */}
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-[var(--text)]">
            {dateRangeLabel}
          </span>
          <Badge variant="muted">
            {totalEvents} {totalEvents === 1 ? "event" : "events"}
          </Badge>
        </div>

        {/* Right: range preset selector */}
        <div className="flex gap-1 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-0.5">
          {RANGE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setRange(opt.value)}
              className={`rounded-md px-3 py-1 text-xs font-bold transition-colors ${
                range === opt.value
                  ? "bg-[var(--accent)] text-white"
                  : "text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Desktop Grid (hidden on mobile) ── */}
      <div className="hidden sm:block">
        {weeks.map((week, weekIdx) => (
          <div
            key={weekIdx}
            className="mb-4 overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--bg)]"
          >
            {/* Day header row */}
            <div className="flex border-b border-[var(--line)]">
              <div className="w-16 shrink-0 border-r border-[var(--line)] bg-[var(--panel)]" />
              <div
                className="grid flex-1"
                style={{
                  gridTemplateColumns: `repeat(${week.length}, 1fr)`,
                }}
              >
                {week.map((day) => {
                  const isToday = day.dateStr === todayKey;
                  const dayTotal =
                    day.sessions.pre.length +
                    day.sessions.tbd.length +
                    day.sessions.amc.length;
                  return (
                    <div
                      key={day.dateStr}
                      className={`border-r border-[var(--line)] px-3 py-2.5 text-center last:border-r-0 ${
                        isToday
                          ? "bg-[rgba(74,158,255,0.08)]"
                          : "bg-[var(--panel)]"
                      }`}
                    >
                      <div className="flex items-center justify-center gap-2">
                        <span
                          className={`text-xs font-bold ${
                            isToday
                              ? "text-[var(--accent)]"
                              : "text-[var(--text)]"
                          }`}
                        >
                          {day.dayName}
                        </span>
                        <span
                          className={`text-[11px] ${
                            isToday
                              ? "font-semibold text-[var(--accent)]"
                              : "text-[var(--muted)]"
                          }`}
                        >
                          {day.dateLabel}
                        </span>
                      </div>
                      {dayTotal > 0 && (
                        <div className="mt-0.5 text-[10px] text-[var(--muted)]">
                          {dayTotal} {dayTotal === 1 ? "event" : "events"}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Session rows */}
            {SESSION_ORDER.map((sk) => (
              <SessionRow
                key={sk}
                sessionKey={sk}
                week={week}
                todayKey={todayKey}
              />
            ))}
          </div>
        ))}
      </div>

      {/* ── Mobile stacked cards (visible on mobile only) ── */}
      <div className="flex flex-col gap-3 sm:hidden">
        {dayBuckets.map((day) => (
          <MobileDayCard
            key={day.dateStr}
            day={day}
            isToday={day.dateStr === todayKey}
          />
        ))}
      </div>

      {/* ── Legend ── */}
      <div className="flex flex-wrap items-center gap-4 px-1 text-[10px] text-[var(--muted)]">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm border border-[rgba(56,211,159,0.4)] bg-[rgba(56,211,159,0.08)]" />
          Tier A
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm border border-[rgba(248,194,78,0.4)] bg-[rgba(248,194,78,0.08)]" />
          Tier B
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm border border-[rgba(255,107,107,0.3)] bg-[rgba(255,107,107,0.06)]" />
          Tier C
        </span>
        <span className="ml-2 flex items-center gap-1">
          <span className="text-[var(--green)]">&#9650;</span> Bullish
        </span>
        <span className="flex items-center gap-1">
          <span className="text-[var(--red)]">&#9660;</span> Bearish
        </span>
      </div>
    </div>
  );
}

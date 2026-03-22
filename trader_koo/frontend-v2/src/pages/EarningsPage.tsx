import { useState, useEffect, useMemo } from "react";
import { Link } from "react-router-dom";
import { useEarnings } from "../api/hooks";
import type { EarningsRow } from "../api/types";
import Badge, { tierVariant } from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";
import Table from "../components/ui/Table";
import TickerLogo from "../components/earnings/TickerLogo";
import WeekGridCalendar from "../components/earnings/WeekGridCalendar";

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
          className="flex items-center gap-1.5 font-mono font-bold text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          <TickerLogo ticker={ticker} size={24} />
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

export default function EarningsPage() {
  useEffect(() => {
    document.title = "Market Calendar - Trader Koo";
  }, []);

  const [days, setDays] = useState(30);
  const [viewMode, setViewMode] = useState<ViewMode>("calendar");
  const [sp500Only, setSp500Only] = useState(true);
  const [search, setSearch] = useState("");
  const { data, isLoading, error } = useEarnings(days, undefined, sp500Only);

  const filteredRows = useMemo(() => {
    const allRows = data?.rows ?? [];
    const query = search.trim().toUpperCase();
    if (!query) return allRows;
    return allRows.filter((r) => r.ticker.toUpperCase().includes(query));
  }, [data?.rows, search]);

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load earnings: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

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
      {/* ── Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-bold tracking-tight">
          Market Calendar
        </h2>
        <div className="flex flex-wrap items-center gap-3">
          {/* Search box */}
          <div className="relative">
            <input
              type="text"
              placeholder="Search ticker..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-36 rounded-lg border border-[var(--line)] bg-[var(--bg)] px-3 py-1.5 pl-8 text-xs text-[var(--text)] placeholder:text-[var(--muted)] focus:border-[var(--accent)] focus:outline-none"
            />
            <svg
              className="absolute left-2.5 top-1/2 h-3 w-3 -translate-y-1/2 text-[var(--muted)]"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              viewBox="0 0 24 24"
            >
              <circle cx={11} cy={11} r={8} />
              <path d="M21 21l-4.35-4.35" />
            </svg>
          </div>
          {/* S&P 500 toggle */}
          <button
            onClick={() => setSp500Only((prev) => !prev)}
            className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors ${
              sp500Only
                ? "border-[var(--accent)] bg-[rgba(74,158,255,0.1)] text-[var(--accent)]"
                : "border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            {sp500Only ? "S&P 500" : "All Companies"}
          </button>
          {/* View toggle */}
          <div className="flex gap-1 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-0.5">
            {(["calendar", "table"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`rounded-md px-3 py-1 text-xs font-bold transition-colors ${
                  viewMode === mode
                    ? "bg-[var(--accent)] text-white"
                    : "text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {mode === "calendar" ? "Grid" : "Table"}
              </button>
            ))}
          </div>
          {/* Data window */}
          <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
            <label htmlFor="earningsDays">Window:</label>
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
              className="w-14 rounded border border-[var(--line)] bg-[var(--bg)] px-2 py-1 text-center text-[var(--text)]"
            />
            <span>days</span>
          </div>
        </div>
      </div>

      {/* ── Summary bar ── */}
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-[var(--line)] bg-[var(--panel)] px-4 py-3">
        <Badge variant="blue">{data?.provider ?? "Provider unknown"}</Badge>
        <Badge variant="muted">{filteredRows.length} events</Badge>
        {summary.setup_ready > 0 && (
          <Badge variant="green">{summary.setup_ready} setup-ready</Badge>
        )}
        {summary.watch > 0 && (
          <Badge variant="amber">{summary.watch} watch</Badge>
        )}
        {summary.high_risk > 0 && (
          <Badge variant="red">{summary.high_risk} high-risk</Badge>
        )}
        {data?.economic_events && data.economic_events.length > 0 && (
          <Badge variant="blue">
            {data.economic_events.length} macro events
          </Badge>
        )}
        {data?.market_date && (
          <span className="ml-auto text-[10px] text-[var(--muted)]">
            Market date: {data.market_date}
          </span>
        )}
      </div>

      {data?.detail && (
        <div className="text-xs text-[var(--muted)]">{String(data.detail)}</div>
      )}

      {/* ── Content ── */}
      {viewMode === "calendar" ? (
        <WeekGridCalendar
          rows={filteredRows}
          economicEvents={data?.economic_events ?? []}
        />
      ) : (
        <Table
          columns={earningsColumns}
          data={filteredRows}
          sortable
        />
      )}
    </div>
  );
}

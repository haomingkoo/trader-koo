import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Spinner from "../components/ui/Spinner";
import { apiFetch } from "../api/client";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface SubMarket {
  question: string;
  outcomes: string[];
  prices_pct: (number | null)[];
  volume: number;
  end_date: string | null;
  active: boolean;
  resolved: boolean;
}

interface PolyEvent {
  title: string;
  slug: string;
  market_count: number;
  total_volume: number;
  end_date: string | null;
  url: string;
  top_market: SubMarket | null;
  markets: SubMarket[];
  active_count: number;
  resolved_count: number;
  event_type: "simple" | "timeline" | "multi_outcome";
}

interface PolyResponse {
  ok: boolean;
  count: number;
  events: PolyEvent[];
}

/* ------------------------------------------------------------------ */
/* Data hook                                                           */
/* ------------------------------------------------------------------ */

function usePolymarket() {
  return useQuery({
    queryKey: ["polymarket"],
    queryFn: () => apiFetch<PolyResponse>("/api/polymarket?limit=15"),
    refetchInterval: 300_000,
  });
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const MAX_VISIBLE_MARKETS = 5;

function formatVolume(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}

function formatShortDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

function formatEndDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/** Bar color based on probability value. */
function probColor(pct: number): string {
  if (pct > 50) return "var(--green)";
  if (pct >= 10) return "var(--amber)";
  return "var(--muted)";
}

/** Extract the leading probability for a sub-market (first YES price or first price). */
function leadingProb(market: SubMarket): number {
  const outcomes = market.outcomes ?? [];
  const prices = market.prices_pct ?? [];
  for (let i = 0; i < outcomes.length; i++) {
    if (String(outcomes[i]).toLowerCase() === "yes" && prices[i] != null) {
      return prices[i] as number;
    }
  }
  return (prices[0] ?? 0) as number;
}

/** Extract a readable label from a sub-market question.
 *  Shows the full question (wrapped) — never truncate to unreadable "...".
 *  For timeline events, prepend the date badge if available.
 */
function shortLabel(market: SubMarket): string {
  const q = market.question.trim();
  // Strip common prefixes that repeat the event title
  const cleaned = q
    .replace(/^Will\s+/i, "")
    .replace(/\?$/, "");
  return cleaned || q;
}

/** Format a date badge for timeline rows. */
function dateBadge(market: SubMarket): string | null {
  if (!market.end_date) return null;
  return formatShortDate(market.end_date);
}

/* ------------------------------------------------------------------ */
/* Chevron SVG                                                         */
/* ------------------------------------------------------------------ */

interface ChevronProps {
  open: boolean;
  className?: string;
}

function Chevron({ open, className = "" }: ChevronProps) {
  return (
    <svg
      className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""} ${className}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* SubMarketRow — single probability bar row                           */
/* ------------------------------------------------------------------ */

interface SubMarketRowProps {
  market: SubMarket;
  eventType: "simple" | "timeline" | "multi_outcome";
  muted?: boolean;
}

function SubMarketRow({ market, eventType, muted = false }: SubMarketRowProps) {
  const pct = leadingProb(market);
  const label = shortLabel(market);
  const date = dateBadge(market);
  const color = probColor(pct);
  const opacity = muted ? "opacity-40" : "";

  return (
    <div className={`rounded-lg border border-[var(--line)]/30 px-2.5 py-1.5 ${opacity} ${!muted ? "bg-[var(--bg)]/50" : ""}`}>
      {/* Question text — full, wrapped */}
      <div className="flex items-start justify-between gap-2">
        <p
          className={`text-[11px] leading-snug ${muted ? "line-through text-[var(--muted)]" : "text-[var(--text)] font-medium"}`}
        >
          {label}
        </p>
        <div className="flex shrink-0 items-center gap-1.5">
          {date && (
            <span className="rounded bg-[var(--line)] px-1.5 py-0.5 text-[8px] font-mono text-[var(--muted)]">
              {date}
            </span>
          )}
          <span className="text-[10px] font-bold tabular-nums" style={{ color }}>
            {pct.toFixed(0)}%
          </span>
        </div>
      </div>
      {/* Probability bar */}
      <div className="mt-1 flex items-center gap-2">
        <div className="relative h-2.5 flex-1 rounded-full bg-[var(--line)]">
          <div
            className="absolute left-0 top-0 h-full rounded-full transition-all"
            style={{
              width: `${Math.min(100, Math.max(2, pct))}%`,
              backgroundColor: color,
            }}
          />
        </div>
        <span className="w-11 shrink-0 text-right text-[8px] tabular-nums text-[var(--muted)]">
          {formatVolume(market.volume)}
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* SimpleYesNo — for single YES/NO events                              */
/* ------------------------------------------------------------------ */

interface SimpleYesNoProps {
  market: SubMarket;
}

function SimpleYesNo({ market }: SimpleYesNoProps) {
  const outcomes = market.outcomes ?? [];
  const prices = market.prices_pct ?? [];

  return (
    <div className="mt-2 space-y-1">
      {outcomes.map((outcome, i) => {
        const value = (prices[i] ?? 0) as number;
        const isYes = String(outcome).toLowerCase() === "yes";
        const color = isYes
          ? value > 60 ? "var(--green)" : value > 40 ? "var(--amber)" : "var(--red)"
          : value > 60 ? "var(--red)" : value > 40 ? "var(--amber)" : "var(--green)";

        return (
          <div key={outcome} className="flex items-center gap-2">
            <span className="w-7 text-[9px] font-bold uppercase tracking-wider text-[var(--muted)]">
              {outcome}
            </span>
            <div className="relative h-4 flex-1 rounded bg-[var(--line)]">
              <div
                className="absolute left-0 top-0 h-full rounded transition-all"
                style={{
                  width: `${Math.min(100, Math.max(2, value))}%`,
                  backgroundColor: color,
                }}
              />
              <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold tabular-nums text-white drop-shadow">
                {value.toFixed(0)}%
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* SubMarketList — active markets with show-more                       */
/* ------------------------------------------------------------------ */

interface SubMarketListProps {
  markets: SubMarket[];
  eventType: "simple" | "timeline" | "multi_outcome";
}

function SubMarketList({ markets, eventType }: SubMarketListProps) {
  const [showAll, setShowAll] = useState(false);

  // Sort: timeline by end_date asc, multi_outcome by probability desc
  const sorted = [...markets].sort((a, b) => {
    if (eventType === "timeline") {
      const aDate = a.end_date ?? "";
      const bDate = b.end_date ?? "";
      return aDate.localeCompare(bDate);
    }
    return leadingProb(b) - leadingProb(a);
  });

  const visible = showAll ? sorted : sorted.slice(0, MAX_VISIBLE_MARKETS);
  const hiddenCount = sorted.length - MAX_VISIBLE_MARKETS;

  return (
    <div className="mt-2 space-y-1">
      {visible.map((mkt, i) => (
        <SubMarketRow
          key={`${mkt.question}-${i}`}
          market={mkt}
          eventType={eventType}
        />
      ))}
      {!showAll && hiddenCount > 0 && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="mt-1 w-full text-center text-[9px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          Show {hiddenCount} more
        </button>
      )}
      {showAll && hiddenCount > 0 && (
        <button
          type="button"
          onClick={() => setShowAll(false)}
          className="mt-1 w-full text-center text-[9px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          Show less
        </button>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* ResolvedToggle — collapsed resolved markets                         */
/* ------------------------------------------------------------------ */

interface ResolvedToggleProps {
  markets: SubMarket[];
  eventType: "simple" | "timeline" | "multi_outcome";
}

function ResolvedToggle({ markets, eventType }: ResolvedToggleProps) {
  const [open, setOpen] = useState(false);

  if (markets.length === 0) return null;

  return (
    <div className="mt-2 border-t border-[var(--line)]/30 pt-1.5">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center gap-1.5 text-[9px] text-[var(--muted)] hover:text-[var(--text)] transition-colors"
      >
        <Chevron open={open} />
        <span>{markets.length} resolved</span>
      </button>
      {open && (
        <div className="mt-1.5 space-y-1">
          {markets.map((mkt, i) => (
            <SubMarketRow
              key={`resolved-${mkt.question}-${i}`}
              market={mkt}
              eventType={eventType}
              muted
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* EventHeader — title, volume, end date, link                         */
/* ------------------------------------------------------------------ */

interface EventHeaderProps {
  event: PolyEvent;
}

function EventHeader({ event }: EventHeaderProps) {
  return (
    <div>
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-sm font-semibold text-[var(--text)] leading-snug">
          {event.title}
        </h3>
        <a
          href={event.url}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 text-[var(--muted)] hover:text-[var(--accent)] transition-colors"
          aria-label={`View ${event.title} on Polymarket`}
        >
          <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </a>
      </div>
      <div className="mt-1.5 flex items-center gap-3 text-[10px] text-[var(--muted)]">
        <span>
          Vol: <strong className="text-[var(--text)]">{formatVolume(event.total_volume)}</strong>
        </span>
        <span>
          {event.active_count ?? event.market_count} active
        </span>
        {event.end_date && (
          <span>
            Ends <strong className="text-[var(--text)]">{formatEndDate(event.end_date)}</strong>
          </span>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* EventCard — smart display based on event_type                       */
/* ------------------------------------------------------------------ */

interface EventCardProps {
  event: PolyEvent;
}

function EventCard({ event }: EventCardProps) {
  const allMarkets = event.markets ?? [];
  const activeMarkets = allMarkets.filter((m) => m.active !== false);
  const resolvedMarkets = allMarkets.filter((m) => m.resolved === true);

  const eventType = event.event_type ?? "simple";

  return (
    <article className="flex max-h-80 flex-col rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 transition-colors hover:border-[var(--accent)]/40">
      <EventHeader event={event} />

      {/* Content area — scrollable if it overflows */}
      <div className="relative mt-2 min-h-0 flex-1 overflow-hidden">
        <div className="h-full overflow-y-auto scrollbar-none">
          {eventType === "simple" && event.top_market ? (
            <SimpleYesNo market={event.top_market} />
          ) : (
            <SubMarketList
              markets={activeMarkets}
              eventType={eventType}
            />
          )}

          <ResolvedToggle markets={resolvedMarkets} eventType={eventType} />
        </div>

        {/* Gradient fade when content is clipped */}
        <div
          className="pointer-events-none absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t from-[var(--panel)] to-transparent"
          aria-hidden="true"
        />
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function PolymarketPage() {
  useEffect(() => {
    document.title = "Polymarket \u2014 Trader Koo";
  }, []);

  const { data, isLoading, error } = usePolymarket();

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load prediction markets: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  const events = data?.events ?? [];

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        <strong>For informational and educational purposes only.</strong> Prediction market
        probabilities reflect crowd consensus, not certainty.
      </div>

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold tracking-tight">Prediction Markets</h2>
          <p className="mt-1 text-xs text-[var(--muted)]">
            Finance-relevant events from Polymarket, sorted by trading volume.
            Useful as a macro regime signal — when odds shift, markets follow.
          </p>
        </div>
        <a
          href="https://polymarket.com"
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          Polymarket &rarr;
        </a>
      </div>

      {events.length === 0 ? (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-8 text-center text-sm text-[var(--muted)]">
          No finance-relevant prediction markets found. Check back later.
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {events.map((event, i) => (
            <EventCard key={event.slug || i} event={event} />
          ))}
        </div>
      )}
    </div>
  );
}

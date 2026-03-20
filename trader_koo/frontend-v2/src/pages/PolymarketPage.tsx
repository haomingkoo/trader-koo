import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Spinner from "../components/ui/Spinner";
import { apiFetch } from "../api/client";

interface SubMarket {
  question: string;
  outcomes: string[];
  prices_pct: (number | null)[];
  volume: number;
  end_date: string | null;
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
}

interface PolyResponse {
  ok: boolean;
  count: number;
  events: PolyEvent[];
}

function usePolymarket() {
  return useQuery({
    queryKey: ["polymarket"],
    queryFn: () => apiFetch<PolyResponse>("/api/polymarket?limit=15"),
    refetchInterval: 300_000,
  });
}

function formatVolume(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}

function formatEndDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

interface PriceBarProps {
  label: string;
  pct: number | null;
}

function PriceBar({ label, pct }: PriceBarProps) {
  const value = pct ?? 0;
  const isYes = label.toLowerCase() === "yes";
  const color = isYes
    ? value > 60 ? "var(--green)" : value > 40 ? "var(--amber)" : "var(--red)"
    : value > 60 ? "var(--red)" : value > 40 ? "var(--amber)" : "var(--green)";

  return (
    <div className="flex items-center gap-2">
      <span className="w-7 text-[9px] font-bold uppercase tracking-wider text-[var(--muted)]">
        {label}
      </span>
      <div className="relative h-4 flex-1 rounded bg-[var(--line)]">
        <div
          className="absolute left-0 top-0 h-full rounded transition-all"
          style={{ width: `${Math.min(100, Math.max(2, value))}%`, backgroundColor: color }}
        />
        <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold tabular-nums text-white drop-shadow">
          {value.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

interface SubMarketRowProps {
  market: SubMarket;
}

function SubMarketRow({ market }: SubMarketRowProps) {
  const outcomes = market.outcomes ?? [];
  const prices = market.prices_pct ?? [];

  return (
    <div className="rounded-lg border border-[var(--line)]/50 bg-[var(--bg)] px-3 py-2">
      <div className="flex items-start justify-between gap-2">
        <p className="text-[11px] font-medium text-[var(--text)] leading-snug">
          {market.question}
        </p>
        {market.end_date && (
          <span className="shrink-0 text-[9px] text-[var(--muted)] tabular-nums">
            {formatEndDate(market.end_date)}
          </span>
        )}
      </div>
      {outcomes.length > 0 && (
        <div className="mt-1.5 space-y-0.5">
          {outcomes.map((outcome, i) => (
            <PriceBar key={outcome} label={String(outcome)} pct={prices[i] ?? null} />
          ))}
        </div>
      )}
      <div className="mt-1 text-[9px] text-[var(--muted)]">
        Vol: <strong className="text-[var(--text)]">{formatVolume(market.volume)}</strong>
      </div>
    </div>
  );
}

interface EventCardProps {
  event: PolyEvent;
}

function EventCard({ event }: EventCardProps) {
  const [expanded, setExpanded] = useState(false);
  const top = event.top_market;
  const subMarkets = event.markets ?? [];
  const hasMultipleMarkets = subMarkets.length > 1;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 transition-colors hover:border-[var(--accent)]/40">
      <a
        href={event.url}
        target="_blank"
        rel="noopener noreferrer"
        className="block"
      >
        <h3 className="text-sm font-semibold text-[var(--text)] leading-snug">
          {event.title}
        </h3>

        {top && Array.isArray(top.outcomes) && top.outcomes.length > 0 && (
          <div className="mt-2 space-y-1">
            {top.outcomes.map((outcome, i) => (
              <PriceBar key={outcome} label={String(outcome)} pct={Array.isArray(top.prices_pct) ? top.prices_pct[i] : null} />
            ))}
          </div>
        )}

        <div className="mt-2 flex items-center gap-3 text-[10px] text-[var(--muted)]">
          <span>Vol: <strong className="text-[var(--text)]">{formatVolume(event.total_volume)}</strong></span>
          <span>{event.market_count} market{event.market_count !== 1 ? "s" : ""}</span>
          {event.end_date && (
            <span>Ends <strong className="text-[var(--text)]">{formatEndDate(event.end_date)}</strong></span>
          )}
        </div>
      </a>

      {hasMultipleMarkets && (
        <div className="mt-3 border-t border-[var(--line)]/50 pt-2">
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="flex w-full items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
          >
            <span>{subMarkets.length} sub-markets</span>
            <svg
              className={`h-3 w-3 transition-transform ${expanded ? "rotate-180" : ""}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {expanded && (
            <div className="mt-2 space-y-2">
              {subMarkets.map((mkt, i) => (
                <SubMarketRow key={`${mkt.question}-${i}`} market={mkt} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

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

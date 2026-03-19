import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import Spinner from "../components/ui/Spinner";
import { apiFetch } from "../api/client";

interface TopMarket {
  question: string;
  outcomes: string[];
  prices_pct: (number | null)[];
  volume: number;
}

interface PolyEvent {
  title: string;
  slug: string;
  market_count: number;
  total_volume: number;
  end_date: string | null;
  url: string;
  top_market: TopMarket | null;
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

function PriceBar({ label, pct }: { label: string; pct: number | null }) {
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

function EventCard({ event }: { event: PolyEvent }) {
  const top = event.top_market;

  return (
    <a
      href={event.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 transition-colors hover:border-[var(--accent)]/40"
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
          <span>Ends <strong className="text-[var(--text)]">{new Date(event.end_date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}</strong></span>
        )}
      </div>
    </a>
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

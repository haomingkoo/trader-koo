import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Spinner from "../components/ui/Spinner";
import { apiFetch } from "../api/client";

interface PolymarketMarket {
  question: string;
  slug: string;
  outcomes: string[];
  prices_pct: (number | null)[];
  volume: number;
  liquidity: number;
  end_date: string | null;
  image: string | null;
  url: string;
}

interface PolymarketResponse {
  ok: boolean;
  count: number;
  markets: PolymarketMarket[];
}

function usePolymarket() {
  return useQuery({
    queryKey: ["polymarket"],
    queryFn: () => apiFetch<PolymarketResponse>("/api/polymarket?limit=20"),
    refetchInterval: 300_000,
  });
}

function formatVolume(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
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
      <span className="w-8 text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
        {label}
      </span>
      <div className="relative h-5 flex-1 rounded-full bg-[var(--line)]">
        <div
          className="absolute left-0 top-0 h-full rounded-full transition-all"
          style={{ width: `${Math.min(100, Math.max(2, value))}%`, backgroundColor: color }}
        />
        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold tabular-nums text-white">
          {value.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

function MarketCard({ market }: { market: PolymarketMarket }) {
  return (
    <a
      href={market.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 transition-colors hover:border-[var(--accent)]/40"
    >
      <h3 className="text-sm font-semibold text-[var(--text)] leading-snug">
        {market.question}
      </h3>

      <div className="mt-3 space-y-1.5">
        {market.outcomes.map((outcome, i) => (
          <PriceBar key={outcome} label={outcome} pct={market.prices_pct[i]} />
        ))}
      </div>

      <div className="mt-3 flex items-center gap-4 text-[10px] text-[var(--muted)]">
        <span>Vol: <strong className="text-[var(--text)]">{formatVolume(market.volume)}</strong></span>
        <span>Liq: <strong className="text-[var(--text)]">{formatVolume(market.liquidity)}</strong></span>
        {market.end_date && (
          <span>Ends: <strong className="text-[var(--text)]">{new Date(market.end_date).toLocaleDateString()}</strong></span>
        )}
      </div>
    </a>
  );
}

const FILTER_CATEGORIES: Record<string, string[]> = {
  "All": [],
  "Macro": ["fed", "rate", "recession", "inflation", "gdp", "economy", "fiscal", "monetary", "fomc", "powell"],
  "Crypto": ["bitcoin", "btc", "eth", "crypto", "microstrategy"],
  "Geopolitical": ["trump", "china", "war", "tariff", "sanctions", "election", "trade"],
  "Commodities": ["oil", "gold", "opec"],
};

export default function PolymarketPage() {
  const { data, isLoading, error } = usePolymarket();
  const [activeFilter, setActiveFilter] = useState("All");

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load Polymarket data: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  const allMarkets = data?.markets ?? [];
  const filterKeywords = FILTER_CATEGORIES[activeFilter] ?? [];
  const markets = filterKeywords.length === 0
    ? allMarkets
    : allMarkets.filter((m) =>
        filterKeywords.some((kw) => m.question.toLowerCase().includes(kw))
      );

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        <strong>For informational and educational purposes only.</strong> Prediction market
        probabilities reflect crowd consensus, not certainty. Do not treat these as investment advice.
      </div>

      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold tracking-tight">Prediction Markets</h2>
        <a
          href="https://polymarket.com"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          Polymarket.com &rarr;
        </a>
      </div>

      <p className="text-sm text-[var(--muted)]">
        Live prediction market odds from Polymarket. Real-money bets on future events —
        useful as a regime signal. Filtered for finance-relevant markets.
      </p>

      <div className="flex flex-wrap gap-1.5">
        {Object.keys(FILTER_CATEGORIES).map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveFilter(cat)}
            className={`rounded-lg px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition-colors ${
              activeFilter === cat
                ? "bg-[var(--accent)] text-white"
                : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            {cat} {cat !== "All" && `(${
              (data?.markets ?? []).filter((m) =>
                (FILTER_CATEGORIES[cat] ?? []).some((kw) => m.question.toLowerCase().includes(kw))
              ).length
            })`}
          </button>
        ))}
      </div>

      {markets.length === 0 ? (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-8 text-center text-sm text-[var(--muted)]">
          No active markets available. Check back later.
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {markets.map((market, i) => (
            <MarketCard key={market.slug || i} market={market} />
          ))}
        </div>
      )}
    </div>
  );
}

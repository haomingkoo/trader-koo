import { useCryptoWs } from "../../hooks/useCryptoWs";
import { useEquityWs } from "../../hooks/useEquityWs";
import { useMarketSummary, useStreamingPrices } from "../../api/hooks";
import type { CryptoPrice } from "../../api/types";
import { resolveEquityQuote, shortCloseDate } from "./quoteView";
import type { QuoteView } from "./quoteView";

function formatPrice(price: number): string {
  if (price >= 1000) {
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    });
  }
  return price.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function CryptoPriceChip({ label, tick }: { label: string; tick: CryptoPrice | undefined | null }) {
  if (!tick || typeof tick.price !== "number" || typeof tick.change_pct_24h !== "number") {
    return null;
  }
  const isPositive = tick.change_pct_24h >= 0;
  const changeColor = isPositive ? "text-[var(--green)]" : "text-[var(--red)]";
  const sign = isPositive ? "+" : "";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
        {label}
      </span>
      <span className="tabular-nums text-[var(--text)]">${formatPrice(tick.price)}</span>
      <span className={`tabular-nums text-[10px] font-semibold ${changeColor}`}>
        {sign}{tick.change_pct_24h.toFixed(2)}%
      </span>
    </div>
  );
}

export function QuoteChip({ label, quote, formatValue, exceeded = false }: {
  label: string;
  quote: QuoteView;
  formatValue: (price: number) => string;
  exceeded?: boolean;
}) {
  const change = quote.changePct;
  const changeColor = change == null || change === 0
    ? "text-[var(--muted)]"
    : change > 0 ? "text-[var(--green)]" : "text-[var(--red)]";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
        {label}
      </span>
      <span className="tabular-nums text-[var(--text)]">{formatValue(quote.price)}</span>
      {change != null && (
        <span className={`tabular-nums text-[10px] font-semibold ${changeColor}`}>
          {change >= 0 ? "+" : ""}{change.toFixed(2)}%
          {exceeded && <span className="ml-0.5" title="Threshold exceeded">!</span>}
        </span>
      )}
      {quote.mode === "close" && quote.asOf && (
        <span className="text-[10px] text-[var(--muted)]" title={`Last close on ${quote.asOf}; no live quote`}>
          {shortCloseDate(quote.asOf)} close
        </span>
      )}
    </div>
  );
}

const CRYPTO_SYMBOLS = [
  { key: "BTC-USD", label: "BTC" },
  { key: "ETH-USD", label: "ETH" },
  { key: "SOL-USD", label: "SOL" },
  { key: "XRP-USD", label: "XRP" },
  { key: "DOGE-USD", label: "DOGE" },
] as const;

const EQUITY_SYMBOLS = [
  { key: "SPY", label: "SPY" },
  { key: "QQQ", label: "QQQ" },
  { key: "DIA", label: "DIA" },
] as const;

export function EquityPriceStrip() {
  const { prices: wsPrices, connected } = useEquityWs();
  const { data: streamed } = useStreamingPrices();
  const { data: marketSummary } = useMarketSummary(7);
  // The socket only pushes new trades, so seed with the latest streamed quotes.
  const prices = { ...streamed?.prices, ...wsPrices };
  const quotes = EQUITY_SYMBOLS.flatMap((symbol) => {
    const quote = resolveEquityQuote(
      prices[symbol.key],
      marketSummary?.tickers?.[symbol.key],
      marketSummary?.as_of,
    );
    return quote ? [{ ...symbol, quote }] : [];
  });
  const hasData = quotes.length > 0;

  if (!hasData) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Equities
        </span>
        <span className="text-[10px] text-[var(--amber)]">Quotes unavailable</span>
      </div>
    );
  }

  return (
    <div className="flex w-full flex-wrap items-center gap-x-4 gap-y-1 rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[11.5px]">
      {quotes.map(({ key, label, quote }) => (
        <span key={key} className="flex items-center whitespace-nowrap">
          <QuoteChip label={label} quote={quote} formatValue={(price) => `$${formatPrice(price)}`} />
        </span>
      ))}
      {!connected && (
        <span className="ml-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--red)]" title="Reconnecting..." />
      )}
    </div>
  );
}

export function CryptoPriceStrip() {
  const { prices, connected } = useCryptoWs();

  if (!connected && Object.keys(prices).length === 0) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Crypto
        </span>
        <span className="text-[10px] text-[var(--red)]">Connecting...</span>
      </div>
    );
  }

  const availableChips = CRYPTO_SYMBOLS.filter((symbol) => prices[symbol.key]);

  return (
    <div className="flex w-full flex-wrap items-center gap-x-4 gap-y-1 rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[11.5px]">
      {availableChips.map((symbol) => (
        <span key={symbol.key} className="flex items-center whitespace-nowrap">
          <CryptoPriceChip label={symbol.label} tick={prices[symbol.key]} />
        </span>
      ))}
      {availableChips.length === 0 && (
        <span className="text-[10px] text-[var(--muted)]">No data</span>
      )}
      {!connected && (
        <span className="ml-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--red)]" title="Reconnecting..." />
      )}
    </div>
  );
}

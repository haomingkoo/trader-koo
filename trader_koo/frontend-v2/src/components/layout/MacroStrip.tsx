import { useMacroLive } from "../../api/hooks";
import { QuoteChip } from "./HeaderMarketStrips";
import { macroQuoteView } from "./quoteView";

/** Display order for the macro strip — most important first. */
const DISPLAY_ORDER = [
  "^TNX",  // 10Y Yield
  "^VIX",  // VIX
  "GLD",   // Gold
  "USO",   // Oil
  "UUP",   // Dollar
  "^TYX",  // 30Y Yield
  "^IRX",  // 3M T-Bill
  "SLV",   // Silver
] as const;

/** Short labels for compact display. */
const SHORT_LABELS: Record<string, string> = {
  "^TNX": "10Y",
  "^TYX": "30Y",
  "^IRX": "3M",
  "GLD": "Gold",
  "USO": "Oil",
  "SLV": "Silver",
  "UUP": "DXY",
  "^VIX": "VIX",
};

function formatMacroPrice(ticker: string, price: number): string {
  // Yields are displayed as percentages (e.g. 4.301)
  if (ticker === "^TNX" || ticker === "^TYX" || ticker === "^IRX") {
    return price.toFixed(3);
  }
  if (price >= 1000) {
    return `$${price.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  }
  return `$${price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

const REGIME_STYLES: Record<string, { bg: string; text: string }> = {
  RISK_OFF: { bg: "bg-red-500/10", text: "text-[var(--red)]" },
  RISK_ON: { bg: "bg-green-500/10", text: "text-[var(--green)]" },
  MIXED: { bg: "bg-amber-500/10", text: "text-[var(--amber)]" },
  UNKNOWN: { bg: "", text: "text-[var(--muted)]" },
};

export default function MacroStrip() {
  const { data, isLoading } = useMacroLive();

  if (isLoading || !data?.ok) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Macro
        </span>
        <span className="text-[10px] text-[var(--muted)]">
          {isLoading ? "Loading..." : "Unavailable"}
        </span>
      </div>
    );
  }

  // Build a lookup for quick access
  const instrumentMap = new Map(
    data.instruments.map((inst) => [inst.ticker, inst]),
  );

  // Instruments without a live quote (indices and yields) show their dated close.
  const orderedInstruments = DISPLAY_ORDER.flatMap((ticker) => {
    const inst = instrumentMap.get(ticker);
    const quote = inst ? macroQuoteView(inst) : null;
    return inst && quote ? [{ inst, quote }] : [];
  });

  if (orderedInstruments.length === 0) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Macro
        </span>
        <span className="text-[10px] text-[var(--muted)]">No data</span>
      </div>
    );
  }

  const regime = data.regime;
  const regimeStyle = REGIME_STYLES[regime.regime] ?? REGIME_STYLES.UNKNOWN;

  return (
    <div className="flex w-full flex-wrap items-center gap-x-4 gap-y-1 rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[11.5px]">
      {/* Regime badge */}
      {regime.regime !== "UNKNOWN" && regime.confidence >= 40 && (
        <>
          <span
            className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${regimeStyle.bg} ${regimeStyle.text}`}
            title={regime.reasoning}
          >
            {regime.regime.replace("_", " ")}
          </span>
        </>
      )}

      {/* Instrument chips */}
      {orderedInstruments.map(({ inst, quote }) => (
        <span key={inst.ticker} className="flex items-center whitespace-nowrap">
          <QuoteChip
            label={SHORT_LABELS[inst.ticker] ?? inst.name}
            exceeded={inst.exceeded}
            quote={quote}
            formatValue={(price) => formatMacroPrice(inst.ticker, price)}
          />
        </span>
      ))}
    </div>
  );
}

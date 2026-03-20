import { useMacroLive } from "../../api/hooks";
import type { MacroInstrument } from "../../api/types";

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

function MacroPriceChip({ instrument }: { instrument: MacroInstrument }) {
  if (instrument.current == null || instrument.change_pct == null) {
    return null;
  }

  const isPositive = instrument.change_pct >= 0;
  const changeColor = isPositive ? "text-[var(--green)]" : "text-[var(--red)]";
  const sign = isPositive ? "+" : "";
  const label = SHORT_LABELS[instrument.ticker] ?? instrument.name;

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
        {label}
      </span>
      <span className="tabular-nums text-[var(--text)]">
        {formatMacroPrice(instrument.ticker, instrument.current)}
      </span>
      <span className={`tabular-nums text-[10px] font-semibold ${changeColor}`}>
        {sign}{instrument.change_pct.toFixed(2)}%
        {instrument.exceeded && (
          <span className="ml-0.5" title="Threshold exceeded">!</span>
        )}
      </span>
    </div>
  );
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

  // Filter to instruments with data, in display order
  const orderedInstruments = DISPLAY_ORDER
    .map((ticker) => instrumentMap.get(ticker))
    .filter((inst): inst is MacroInstrument => inst != null && inst.current != null);

  if (orderedInstruments.length === 0) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Macro
        </span>
        <span className="text-[10px] text-[var(--muted)]">Market closed</span>
      </div>
    );
  }

  const regime = data.regime;
  const regimeStyle = REGIME_STYLES[regime.regime] ?? REGIME_STYLES.UNKNOWN;

  return (
    <div className="flex w-full items-center gap-3 overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[11.5px] scrollbar-none">
      {/* Regime badge */}
      {regime.regime !== "UNKNOWN" && regime.confidence >= 40 && (
        <>
          <span
            className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${regimeStyle.bg} ${regimeStyle.text}`}
            title={regime.reasoning}
          >
            {regime.regime.replace("_", " ")}
          </span>
          <span className="text-[var(--line)]">|</span>
        </>
      )}

      {/* Instrument chips */}
      {orderedInstruments.map((inst, index) => (
        <span key={inst.ticker} className="flex items-center gap-3 whitespace-nowrap">
          {index > 0 && <span className="text-[var(--line)]">|</span>}
          <MacroPriceChip instrument={inst} />
        </span>
      ))}
    </div>
  );
}

type IntervalValue = "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "12h" | "1d" | "1w";

type IntervalOption = {
  value: IntervalValue;
  label: string;
  limit: number;
  targetWindow: string;
};

type OverlayOption = {
  key: string;
  label: string;
  minBars: number;
};

type OverlayState = Record<string, boolean>;

interface CryptoToolbarProps {
  allSymbols: readonly string[];
  intervals: readonly IntervalOption[];
  overlayOptions: readonly OverlayOption[];
  selectedSymbol: string;
  selectedInterval: IntervalValue;
  overlays: OverlayState;
  availableBarCount: number;
  shortHistory: boolean;
  currentInterval: IntervalOption;
  formatVisibleWindow: (interval: IntervalOption["value"], barCount: number) => string;
  onSelectSymbol: (symbol: string) => void;
  onSelectInterval: (interval: IntervalValue) => void;
  onToggleOverlay: (overlayKey: string) => void;
}

export default function CryptoToolbar({
  allSymbols,
  intervals,
  overlayOptions,
  selectedSymbol,
  selectedInterval,
  overlays,
  availableBarCount,
  shortHistory,
  currentInterval,
  formatVisibleWindow,
  onSelectSymbol,
  onSelectInterval,
  onToggleOverlay,
}: CryptoToolbarProps) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex gap-1">
            {allSymbols.map((sym) => (
              <button
                key={sym}
                onClick={() => onSelectSymbol(sym)}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                  selectedSymbol === sym
                    ? "bg-[var(--accent)] text-white"
                    : "border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {sym.split("-")[0]}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-1">
            {intervals.map((iv) => (
              <button
                key={iv.label}
                onClick={() => onSelectInterval(iv.value)}
                className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                  selectedInterval === iv.value
                    ? "bg-[var(--blue)] text-white"
                    : "border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
                }`}
              >
                {iv.label}
              </button>
            ))}
          </div>
        </div>
        <div className="text-right text-xs text-[var(--muted)]">
          <div>
            {availableBarCount > 0
              ? `${availableBarCount} bars · ${formatVisibleWindow(selectedInterval, availableBarCount)} visible`
              : `Target window ${currentInterval.targetWindow}`}
          </div>
          <div>{selectedSymbol.replace("-USD", "")} from native Binance history with live 1-minute patching</div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
          Indicators
        </span>
        {overlayOptions.map((option) => {
          const unavailable = availableBarCount > 0 && availableBarCount < option.minBars;
          const active = overlays[option.key];
          return (
            <button
              key={option.key}
              type="button"
              disabled={unavailable}
              title={unavailable ? `Needs ${option.minBars} bars on this timeframe` : undefined}
              onClick={() => onToggleOverlay(option.key)}
              className={`rounded-full border px-3 py-1 text-xs font-semibold transition-colors ${
                unavailable
                  ? "cursor-not-allowed border-[var(--line)] bg-[var(--panel)]/50 text-[var(--muted)] opacity-45"
                  : active
                    ? "border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--text)]"
                    : "border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
      {shortHistory && (
        <div className="text-xs text-[var(--muted)]">
          Showing {availableBarCount} {currentInterval.label} bars cached so far. Missing history is backfilled from Binance and then stored locally.
        </div>
      )}
    </div>
  );
}

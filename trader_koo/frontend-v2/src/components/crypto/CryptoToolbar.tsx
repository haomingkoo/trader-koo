import { useState } from "react";
import type { CryptoOverlayState } from "../../lib/crypto/buildCandlestickChart";

type IntervalValue = "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "12h" | "1d" | "1w";
type OverlayKey = keyof CryptoOverlayState;

type IntervalOption = {
  value: IntervalValue;
  label: string;
  limit: number;
  targetWindow: string;
};

type OverlayOption = {
  key: OverlayKey;
  label: string;
  minBars: number;
};

type OverlayState = CryptoOverlayState;

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
  onToggleOverlay: (overlayKey: OverlayKey) => void;
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
  const [expanded, setExpanded] = useState(false);

  const activeOverlayCount = Object.values(overlays).filter(Boolean).length;

  return (
    <div className="space-y-3">
      {/* ── Mobile collapsed summary (< sm) ── */}
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-2 sm:hidden"
      >
        <div className="flex items-center gap-2 text-xs font-semibold">
          <span className="rounded bg-[var(--accent)] px-2 py-0.5 text-white">
            {selectedSymbol.split("-")[0]}
          </span>
          <span className="rounded bg-[var(--blue)] px-2 py-0.5 text-white">
            {intervals.find((iv) => iv.value === selectedInterval)?.label ?? selectedInterval}
          </span>
          {activeOverlayCount > 0 && (
            <span className="text-[var(--muted)]">
              +{activeOverlayCount} indicator{activeOverlayCount > 1 ? "s" : ""}
            </span>
          )}
        </div>
        <svg
          className={`h-4 w-4 text-[var(--muted)] transition-transform ${expanded ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* ── Mobile expanded controls (< sm) ── */}
      {expanded && (
        <div className="space-y-3 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3 sm:hidden">
          {/* Symbol pills — horizontal scroll */}
          <div>
            <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              Symbol
            </span>
            <div className="-mx-3 flex gap-1.5 overflow-x-auto px-3 pb-1 scrollbar-none">
              {allSymbols.map((sym) => (
                <button
                  key={sym}
                  onClick={() => {
                    onSelectSymbol(sym);
                    setExpanded(false);
                  }}
                  className={`shrink-0 rounded-full px-3 py-1 text-xs font-semibold transition-colors ${
                    selectedSymbol === sym
                      ? "bg-[var(--accent)] text-white"
                      : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)]"
                  }`}
                >
                  {sym.split("-")[0]}
                </button>
              ))}
            </div>
          </div>

          {/* Interval pills — horizontal scroll */}
          <div>
            <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              Interval
            </span>
            <div className="-mx-3 flex gap-1.5 overflow-x-auto px-3 pb-1 scrollbar-none">
              {intervals.map((iv) => (
                <button
                  key={iv.label}
                  onClick={() => {
                    onSelectInterval(iv.value);
                    setExpanded(false);
                  }}
                  className={`shrink-0 rounded-full px-3 py-1 text-xs font-semibold transition-colors ${
                    selectedInterval === iv.value
                      ? "bg-[var(--blue)] text-white"
                      : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)]"
                  }`}
                >
                  {iv.label}
                </button>
              ))}
            </div>
          </div>

          {/* Overlay toggles */}
          <div>
            <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              Indicators
            </span>
            <div className="flex flex-wrap gap-1.5">
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
                          : "border-[var(--line)] bg-[var(--panel)] text-[var(--muted)]"
                    }`}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Bar count info */}
          <div className="text-xs text-[var(--muted)]">
            {availableBarCount > 0
              ? `${availableBarCount} bars \u00b7 ${formatVisibleWindow(selectedInterval, availableBarCount)} visible`
              : `Target window ${currentInterval.targetWindow}`}
          </div>
        </div>
      )}

      {/* ── Desktop layout (>= sm) — unchanged ── */}
      <div className="hidden sm:block sm:space-y-3">
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
                ? `${availableBarCount} bars \u00b7 ${formatVisibleWindow(selectedInterval, availableBarCount)} visible`
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
      </div>

      {shortHistory && (
        <div className="text-xs text-[var(--muted)]">
          Showing {availableBarCount} {currentInterval.label} bars cached so far. Missing history is backfilled from Binance and then stored locally.
        </div>
      )}
    </div>
  );
}

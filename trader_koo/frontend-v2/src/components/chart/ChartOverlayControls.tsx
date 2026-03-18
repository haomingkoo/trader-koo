interface OverlayOption {
  key: string;
  label: string;
  minBars: number;
}

interface ChartOverlayControlsProps {
  overlayOptions: readonly OverlayOption[];
  barCount: number;
  overlays: Record<string, boolean>;
  onToggleOverlay: (overlayKey: string) => void;
}

export default function ChartOverlayControls({
  overlayOptions,
  barCount,
  overlays,
  onToggleOverlay,
}: ChartOverlayControlsProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
        Indicators
      </span>
      {overlayOptions.map((option) => {
        const unavailable = barCount > 0 && barCount < option.minBars;
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
      {barCount > 0 && (
        <span className="text-xs text-[var(--muted)]">
          Default view opens on the latest 3 months.
        </span>
      )}
    </div>
  );
}

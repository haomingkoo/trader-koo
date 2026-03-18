import type { CryptoPrice } from "../../api/types";
import { formatPrice, formatVolume, GlassCard } from "./CryptoCardPrimitives";

export function CryptoPriceCard({
  tick,
  selected,
  onSelect,
}: {
  tick: CryptoPrice | undefined;
  selected: boolean;
  onSelect: () => void;
}) {
  if (!tick) {
    return (
      <GlassCard label="--" className="cursor-pointer opacity-50">
        <div className="text-sm text-[var(--muted)]">No data</div>
      </GlassCard>
    );
  }

  const isPositive = tick.change_pct_24h >= 0;
  const changeColor = isPositive ? "text-[var(--green)]" : "text-[var(--red)]";
  const sign = isPositive ? "+" : "";
  const borderClass = selected
    ? "border-[var(--accent)] ring-1 ring-[var(--accent)]/30"
    : isPositive
      ? "border-[rgba(56,211,159,0.3)]"
      : "border-[rgba(255,107,107,0.3)]";

  return (
    <button onClick={onSelect} className="w-full text-left">
      <GlassCard label={tick.symbol} className={`${borderClass} transition-all`}>
        <div className="text-2xl font-bold tabular-nums text-[var(--text)]">
          ${formatPrice(tick.price)}
        </div>
        <div className="mt-1 flex items-center gap-3 text-xs">
          <span className={`font-semibold tabular-nums ${changeColor}`}>
            {sign}
            {tick.change_pct_24h.toFixed(2)}% 24h
          </span>
          <span className="text-[var(--muted)]">
            Vol: {formatVolume(tick.volume_24h)}
          </span>
        </div>
      </GlassCard>
    </button>
  );
}

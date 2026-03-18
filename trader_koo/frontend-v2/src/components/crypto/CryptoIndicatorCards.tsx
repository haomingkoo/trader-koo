import type { CryptoIndicators } from "../../api/types";
import { formatPrice, GlassCard } from "./CryptoCardPrimitives";

export function RsiGauge({ value }: { value: number | null }) {
  if (value === null) {
    return (
      <GlassCard label="RSI 14">
        <div className="text-sm text-[var(--muted)]">Insufficient data</div>
      </GlassCard>
    );
  }

  const rounded = Math.round(value * 100) / 100;
  let color = "text-[var(--text)]";
  let label = "Neutral";
  if (rounded >= 70) {
    color = "text-[var(--red)]";
    label = "Overbought";
  } else if (rounded <= 30) {
    color = "text-[var(--green)]";
    label = "Oversold";
  }

  const barPct = Math.min(100, Math.max(0, rounded));

  return (
    <GlassCard label="RSI 14">
      <div className={`text-2xl font-bold tabular-nums ${color}`}>
        {rounded.toFixed(1)}
      </div>
      <div className="mt-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--muted)]">
        {label}
      </div>
      <div className="mt-2 h-1.5 w-full rounded-full bg-[var(--line)]">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${barPct}%`,
            background:
              rounded >= 70
                ? "var(--red)"
                : rounded <= 30
                  ? "var(--green)"
                  : "var(--blue)",
          }}
        />
      </div>
      <div className="mt-0.5 flex justify-between text-[9px] text-[var(--muted)]">
        <span>0</span>
        <span>30</span>
        <span>70</span>
        <span>100</span>
      </div>
    </GlassCard>
  );
}

export function MacdCard({
  macd,
}: {
  macd: CryptoIndicators["macd"];
}) {
  const hasData =
    macd.macd !== null || macd.signal !== null || macd.histogram !== null;

  if (!hasData) {
    return (
      <GlassCard label="MACD (12, 26, 9)">
        <div className="text-sm text-[var(--muted)]">Insufficient data</div>
      </GlassCard>
    );
  }

  const histColor =
    macd.histogram !== null && macd.histogram >= 0
      ? "text-[var(--green)]"
      : "text-[var(--red)]";

  return (
    <GlassCard label="MACD (12, 26, 9)">
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            MACD
          </div>
          <div className="tabular-nums text-[var(--text)]">
            {macd.macd !== null ? macd.macd.toFixed(4) : "--"}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            Signal
          </div>
          <div className="tabular-nums text-[var(--text)]">
            {macd.signal !== null ? macd.signal.toFixed(4) : "--"}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            Histogram
          </div>
          <div className={`tabular-nums font-semibold ${histColor}`}>
            {macd.histogram !== null ? macd.histogram.toFixed(4) : "--"}
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

export function BollingerCard({
  bollinger,
}: {
  bollinger: CryptoIndicators["bollinger"];
}) {
  if (bollinger.width === null) {
    return (
      <GlassCard label="Bollinger Bands (20, 2)">
        <div className="text-sm text-[var(--muted)]">Insufficient data</div>
      </GlassCard>
    );
  }

  return (
    <GlassCard label="Bollinger Bands (20, 2)">
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            Width
          </div>
          <div className="text-lg font-bold tabular-nums text-[var(--text)]">
            {bollinger.width !== null
              ? `${(bollinger.width * 100).toFixed(2)}%`
              : "--"}
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-[9px] uppercase text-[var(--muted)]">
              Upper
            </span>
            <span className="tabular-nums text-[var(--text)]">
              {bollinger.upper !== null ? formatPrice(bollinger.upper) : "--"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-[9px] uppercase text-[var(--muted)]">
              Mid
            </span>
            <span className="tabular-nums text-[var(--text)]">
              {bollinger.middle !== null ? formatPrice(bollinger.middle) : "--"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-[9px] uppercase text-[var(--muted)]">
              Lower
            </span>
            <span className="tabular-nums text-[var(--text)]">
              {bollinger.lower !== null ? formatPrice(bollinger.lower) : "--"}
            </span>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

export function VwapSmaCard({
  vwap,
  sma20,
  sma50,
}: {
  vwap: number | null;
  sma20: number | null;
  sma50: number | null;
}) {
  return (
    <GlassCard label="VWAP & SMA">
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            24h VWAP
          </div>
          <div className="tabular-nums text-[var(--text)]">
            {vwap !== null ? formatPrice(vwap) : "--"}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            SMA 20
          </div>
          <div className="tabular-nums text-[var(--text)]">
            {sma20 !== null ? formatPrice(sma20) : "--"}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            SMA 50
          </div>
          <div className="tabular-nums text-[var(--text)]">
            {sma50 !== null ? formatPrice(sma50) : "--"}
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

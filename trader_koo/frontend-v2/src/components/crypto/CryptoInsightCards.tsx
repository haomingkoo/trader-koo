import type { ReactNode } from "react";
import type {
  CryptoCorrelationPayload,
  CryptoIndicators,
  CryptoMarketStructurePayload,
  CryptoPrice,
  CryptoStructurePayload,
} from "../../api/types";

function formatPrice(price: number): string {
  if (price >= 1000) {
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    });
  }
  if (price >= 1) {
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }
  return price.toLocaleString("en-US", {
    minimumFractionDigits: 4,
    maximumFractionDigits: 6,
  });
}

function formatVolume(vol: number): string {
  if (vol >= 1_000_000_000) return `${(vol / 1_000_000_000).toFixed(2)}B`;
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(2)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(1)}K`;
  return vol.toFixed(0);
}

function formatPct(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatLevelContext(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function GlassCard({
  label,
  children,
  className = "",
}: {
  label?: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl backdrop-blur-sm bg-[var(--panel)]/80 border border-[var(--line)] p-4 ${className}`}
    >
      {label && (
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          {label}
        </div>
      )}
      {children}
    </div>
  );
}

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

export function StructureCard({
  structure,
}: {
  structure: CryptoStructurePayload | null | undefined;
}) {
  if (!structure) {
    return (
      <GlassCard label="Structure Engine">
        <div className="text-sm text-[var(--muted)]">
          Waiting for enough bars to map support and resistance.
        </div>
      </GlassCard>
    );
  }

  const context = structure.context;
  const regime = structure.hmm_regime;
  const regimeLabel = regime?.current_state?.replaceAll("_", " ") ?? "unavailable";
  const regimeConf = regime?.current_probs?.[regime.current_state] ?? null;

  return (
    <GlassCard label="Structure Engine">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            Level Context
          </div>
          <div className="mt-1 text-lg font-bold text-[var(--text)]">
            {formatLevelContext(context.level_context)}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            Trend: {formatLevelContext(context.ma_trend)}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            Nearest Support
          </div>
          <div className="mt-1 text-lg font-bold tabular-nums text-[var(--blue)]">
            {context.support_level !== null ? `$${formatPrice(context.support_level)}` : "--"}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            Distance {formatPct(context.pct_to_support)}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            Nearest Resistance
          </div>
          <div className="mt-1 text-lg font-bold tabular-nums text-[var(--red)]">
            {context.resistance_level !== null ? `$${formatPrice(context.resistance_level)}` : "--"}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            Distance {formatPct(context.pct_to_resistance)}
          </div>
        </div>
        <div>
          <div className="text-[9px] font-semibold uppercase text-[var(--muted)]">
            HMM Regime
          </div>
          <div className="mt-1 text-lg font-bold text-[var(--text)]">
            {formatLevelContext(regimeLabel)}
          </div>
          <div className="mt-1 text-[11px] text-[var(--muted)]">
            {regimeConf != null
              ? `${(regimeConf * 100).toFixed(0)}% confidence · ${regime?.days_in_current ?? 0} bars`
              : "Insufficient bars for stable regime fit"}
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Range Position</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {context.range_position != null ? `${(context.range_position * 100).toFixed(0)}%` : "--"}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">ATR %</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(context.atr_pct)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Momentum 20</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(context.momentum_20)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Realized Vol 20</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {context.realized_vol_20 != null ? `${context.realized_vol_20.toFixed(2)}%` : "--"}
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
        <div className="mb-2 text-[9px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Auto Levels
        </div>
        <div className="space-y-2">
          {structure.levels.length > 0 ? structure.levels.map((level) => (
            <div
              key={`${level.type}-${level.level}`}
              className="flex items-center justify-between gap-3 text-xs"
            >
              <div>
                <span className={level.type === "support" ? "text-[var(--blue)]" : "text-[var(--red)]"}>
                  {level.type.toUpperCase()}
                </span>
                <span className="ml-2 text-[var(--muted)]">
                  {level.tier.toUpperCase()} · {level.source ?? "pivot_cluster"}
                </span>
              </div>
              <div className="font-semibold tabular-nums text-[var(--text)]">
                ${formatPrice(level.level)}
              </div>
            </div>
          )) : (
            <div className="text-sm text-[var(--muted)]">No nearby levels detected.</div>
          )}
        </div>
      </div>
    </GlassCard>
  );
}

export function BtcSpyCorrelationCard({
  correlation,
}: {
  correlation: CryptoCorrelationPayload | null | undefined;
}) {
  if (!correlation) {
    return (
      <GlassCard label="BTC vs SPY">
        <div className="text-sm text-[var(--muted)]">
          Waiting for aligned daily BTC and SPY closes.
        </div>
      </GlassCard>
    );
  }

  const windows = Object.entries(correlation.windows).sort(
    ([left], [right]) => Number.parseInt(left, 10) - Number.parseInt(right, 10),
  );

  return (
    <GlassCard label="BTC vs SPY Cross-Asset">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-lg font-bold text-[var(--text)]">
            {formatLevelContext(correlation.relationship_label)}
          </div>
          <div className="mt-1 text-xs text-[var(--muted)]">
            {correlation.sample_size} aligned sessions
            {correlation.as_of ? ` · as of ${correlation.as_of}` : ""}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 text-right text-xs">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)]">BTC</div>
            <div className="font-semibold tabular-nums text-[var(--text)]">
              ${correlation.latest.asset_close !== null ? formatPrice(correlation.latest.asset_close) : "--"}
            </div>
            <div
              className={
                correlation.latest.asset_change_1d_pct !== null &&
                correlation.latest.asset_change_1d_pct >= 0
                  ? "text-[var(--green)]"
                  : "text-[var(--red)]"
              }
            >
              {formatPct(correlation.latest.asset_change_1d_pct)}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)]">SPY</div>
            <div className="font-semibold tabular-nums text-[var(--text)]">
              ${correlation.latest.benchmark_close !== null ? formatPrice(correlation.latest.benchmark_close) : "--"}
            </div>
            <div
              className={
                correlation.latest.benchmark_change_1d_pct !== null &&
                correlation.latest.benchmark_change_1d_pct >= 0
                  ? "text-[var(--green)]"
                  : "text-[var(--red)]"
              }
            >
              {formatPct(correlation.latest.benchmark_change_1d_pct)}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {windows.map(([label, window]) => (
          <div
            key={label}
            className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/45 p-3 text-xs"
          >
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              {label}
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2">
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">Corr</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {window.correlation !== null ? window.correlation.toFixed(2) : "--"}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">Beta</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {window.beta !== null ? window.beta.toFixed(2) : "--"}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">BTC Ret</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {formatPct(window.asset_return_pct)}
                </div>
              </div>
              <div>
                <div className="text-[9px] uppercase text-[var(--muted)]">Spread</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {formatPct(window.relative_performance_pct)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 text-xs text-[var(--muted)]">{correlation.note}</div>
    </GlassCard>
  );
}

export function CryptoBreadthCard({
  market,
}: {
  market: CryptoMarketStructurePayload | null | undefined;
}) {
  if (!market) {
    return (
      <GlassCard label="Crypto Breadth">
        <div className="text-sm text-[var(--muted)]">
          Waiting for enough 1h bars across tracked crypto pairs.
        </div>
      </GlassCard>
    );
  }

  const overview = market.overview;

  return (
    <GlassCard label="Crypto Breadth (1h)">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-lg font-bold text-[var(--text)]">
            {formatLevelContext(overview.market_posture)}
          </div>
          <div className="mt-1 text-xs text-[var(--muted)]">
            Volatility regime: {formatLevelContext(overview.volatility_regime)}
          </div>
        </div>
        <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-[0.16em]">
          {market.leaders.slice(0, 2).map((row) => (
            <span
              key={`leader-${row.symbol}`}
              className="rounded-full border border-[var(--line)] bg-[rgba(56,211,159,0.1)] px-2 py-1 text-[var(--green)]"
            >
              Lead {row.symbol.replace("-USD", "")}
            </span>
          ))}
          {market.laggards.slice(0, 1).map((row) => (
            <span
              key={`laggard-${row.symbol}`}
              className="rounded-full border border-[var(--line)] bg-[rgba(255,107,107,0.1)] px-2 py-1 text-[var(--red)]"
            >
              Lag {row.symbol.replace("-USD", "")}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-4 grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Bullish Trend Breadth</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {overview.bullish_trend_count}/{overview.tracked_symbols}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Avg 24h Change</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(overview.avg_change_pct_24h)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Avg Momentum 20</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {formatPct(overview.avg_momentum_20)}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
          <div className="text-[9px] uppercase text-[var(--muted)]">Support / Resistance</div>
          <div className="mt-1 font-semibold tabular-nums text-[var(--text)]">
            {overview.at_support_count} / {overview.at_resistance_count}
          </div>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        {market.symbols.map((row) => (
          <div
            key={row.symbol}
            className="grid grid-cols-[1.1fr_0.8fr_0.9fr_0.9fr] gap-3 rounded-lg border border-[var(--line)] bg-[var(--bg)]/35 p-3 text-xs"
          >
            <div>
              <div className="font-semibold text-[var(--text)]">{row.symbol.replace("-USD", "")}</div>
              <div className="text-[var(--muted)]">{formatLevelContext(row.level_context)}</div>
            </div>
            <div className="tabular-nums">
              <div className="text-[9px] uppercase text-[var(--muted)]">24h</div>
              <div className="font-semibold text-[var(--text)]">{formatPct(row.change_pct_24h)}</div>
            </div>
            <div className="tabular-nums">
              <div className="text-[9px] uppercase text-[var(--muted)]">Trend</div>
              <div className="font-semibold text-[var(--text)]">{formatLevelContext(row.ma_trend)}</div>
            </div>
            <div className="tabular-nums">
              <div className="text-[9px] uppercase text-[var(--muted)]">ATR %</div>
              <div className="font-semibold text-[var(--text)]">{formatPct(row.atr_pct)}</div>
            </div>
          </div>
        ))}
      </div>
    </GlassCard>
  );
}

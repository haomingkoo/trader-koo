import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import {
  useCryptoSummary,
  useCryptoHistory,
  useCryptoIndicators,
  useCryptoStructure,
  useCryptoCorrelation,
  useCryptoMarketStructure,
} from "../api/hooks";
import type {
  CryptoPrice,
  CryptoBar,
  CryptoIndicators,
  CryptoStructurePayload,
  CryptoCorrelationPayload,
  CryptoMarketStructurePayload,
  LevelRow,
} from "../api/types";
import PlotlyWrapper from "../components/PlotlyWrapper";
import Spinner from "../components/ui/Spinner";

/* ── Constants ── */

const ALL_SYMBOLS = [
  "BTC-USD",
  "ETH-USD",
  "SOL-USD",
  "XRP-USD",
  "DOGE-USD",
] as const;

const INTERVALS = [
  { value: "1m", label: "1m", limit: 1440, targetWindow: "~1d" },
  { value: "5m", label: "5m", limit: 2016, targetWindow: "~1w" },
  { value: "15m", label: "15m", limit: 2880, targetWindow: "~30d" },
  { value: "30m", label: "30m", limit: 2160, targetWindow: "~45d" },
  { value: "1h", label: "1h", limit: 2160, targetWindow: "~90d" },
  { value: "4h", label: "4h", limit: 1440, targetWindow: "~240d" },
  { value: "12h", label: "12h", limit: 1095, targetWindow: "~1.5y" },
  { value: "1d", label: "1D", limit: 1825, targetWindow: "~5y" },
  { value: "1w", label: "1W", limit: 260, targetWindow: "~5y" },
] as const;

const INTERVAL_TO_MINUTES = {
  "1m": 1,
  "5m": 5,
  "15m": 15,
  "30m": 30,
  "1h": 60,
  "4h": 240,
  "12h": 720,
  "1d": 1440,
  "1w": 10080,
} as const;

const OVERLAY_OPTIONS = [
  { key: "sma20", label: "20 MA", minBars: 20 },
  { key: "sma50", label: "50 MA", minBars: 50 },
  { key: "sma200", label: "200 MA", minBars: 200 },
  { key: "bollinger", label: "Bollinger", minBars: 20 },
] as const;

type IntervalValue = (typeof INTERVALS)[number]["value"];
type OverlayKey = (typeof OVERLAY_OPTIONS)[number]["key"];
type OverlayState = Record<OverlayKey, boolean>;

const DEFAULT_OVERLAYS: OverlayState = {
  sma20: true,
  sma50: true,
  sma200: false,
  bollinger: false,
};

/* ── Helpers ── */

function formatPrice(price: number): string {
  if (price >= 1000)
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    });
  if (price >= 1)
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
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

function formatVisibleWindow(interval: IntervalValue, barCount: number): string {
  if (!barCount) return "--";
  const totalMinutes = barCount * INTERVAL_TO_MINUTES[interval];
  if (totalMinutes < 60) return `${totalMinutes}m`;
  if (totalMinutes < 1440) {
    const hours = totalMinutes / 60;
    return `${hours % 1 === 0 ? hours.toFixed(0) : hours.toFixed(1)}h`;
  }
  if (totalMinutes < 43200) {
    const days = totalMinutes / 1440;
    return `${days % 1 === 0 ? days.toFixed(0) : days.toFixed(1)}d`;
  }
  if (totalMinutes < 525600) {
    const months = totalMinutes / 43200;
    return `${months % 1 === 0 ? months.toFixed(0) : months.toFixed(1)}mo`;
  }
  const years = totalMinutes / 525600;
  return `${years % 1 === 0 ? years.toFixed(0) : years.toFixed(1)}y`;
}

/* ── Glassmorphism card ── */

function GlassCard({
  label,
  children,
  className = "",
}: {
  label?: string;
  children?: React.ReactNode;
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

/* ── Price card ── */

function CryptoPriceCard({
  tick,
  selected,
  onSelect,
}: {
  tick: CryptoPrice | undefined;
  symbol: string;
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

/* ── RSI gauge card ── */

function RsiGauge({ value }: { value: number | null }) {
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

  // Bar width as percentage (0..100)
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

/* ── MACD card ── */

function MacdCard({
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

/* ── Bollinger width card ── */

function BollingerCard({
  bollinger,
}: {
  bollinger: CryptoIndicators["bollinger"];
}) {
  const hasData = bollinger.width !== null;

  if (!hasData) {
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
              ? (bollinger.width * 100).toFixed(2) + "%"
              : "--"}
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-[9px] uppercase text-[var(--muted)]">
              Upper
            </span>
            <span className="tabular-nums text-[var(--text)]">
              {bollinger.upper !== null
                ? formatPrice(bollinger.upper)
                : "--"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-[9px] uppercase text-[var(--muted)]">
              Mid
            </span>
            <span className="tabular-nums text-[var(--text)]">
              {bollinger.middle !== null
                ? formatPrice(bollinger.middle)
                : "--"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-[9px] uppercase text-[var(--muted)]">
              Lower
            </span>
            <span className="tabular-nums text-[var(--text)]">
              {bollinger.lower !== null
                ? formatPrice(bollinger.lower)
                : "--"}
            </span>
          </div>
        </div>
      </div>
    </GlassCard>
  );
}

/* ── VWAP + SMA card ── */

function VwapSmaCard({
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

function StructureCard({
  structure,
}: {
  structure: CryptoStructurePayload | null | undefined;
}) {
  if (!structure) {
    return (
      <GlassCard label="Structure Engine">
        <div className="text-sm text-[var(--muted)]">Waiting for enough bars to map support and resistance.</div>
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

      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4 text-xs">
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
            <div key={`${level.type}-${level.level}`} className="flex items-center justify-between gap-3 text-xs">
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

function BtcSpyCorrelationCard({
  correlation,
}: {
  correlation: CryptoCorrelationPayload | null | undefined;
}) {
  if (!correlation) {
    return (
      <GlassCard label="BTC vs SPY">
        <div className="text-sm text-[var(--muted)]">Waiting for aligned daily BTC and SPY closes.</div>
      </GlassCard>
    );
  }

  const windows = Object.entries(correlation.windows).sort(
    ([a], [b]) => Number.parseInt(a, 10) - Number.parseInt(b, 10),
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
            <div className={correlation.latest.asset_change_1d_pct !== null && correlation.latest.asset_change_1d_pct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}>
              {formatPct(correlation.latest.asset_change_1d_pct)}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wide text-[var(--muted)]">SPY</div>
            <div className="font-semibold tabular-nums text-[var(--text)]">
              ${correlation.latest.benchmark_close !== null ? formatPrice(correlation.latest.benchmark_close) : "--"}
            </div>
            <div className={correlation.latest.benchmark_change_1d_pct !== null && correlation.latest.benchmark_change_1d_pct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}>
              {formatPct(correlation.latest.benchmark_change_1d_pct)}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {windows.map(([label, window]) => (
          <div key={label} className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/45 p-3 text-xs">
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

function CryptoBreadthCard({
  market,
}: {
  market: CryptoMarketStructurePayload | null | undefined;
}) {
  if (!market) {
    return (
      <GlassCard label="Crypto Breadth">
        <div className="text-sm text-[var(--muted)]">Waiting for enough 1h bars across tracked crypto pairs.</div>
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

      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4 text-xs">
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

/* ── Candlestick chart builder with overlays ── */

function addLevelOverlays(
  levels: LevelRow[],
  annotations: Record<string, unknown>[],
  shapes: Record<string, unknown>[],
) {
  levels.forEach((level) => {
    if (!Number.isFinite(level.level)) return;
    const color = level.type === "support" ? "#3f8cff" : "#ff7b5b";
    const dash = level.tier === "primary" ? "solid" : level.tier === "secondary" ? "dot" : "dash";

    if (Number.isFinite(level.zone_low) && Number.isFinite(level.zone_high)) {
      shapes.push({
        type: "rect",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: Math.min(level.zone_low, level.zone_high),
        y1: Math.max(level.zone_low, level.zone_high),
        fillcolor: level.type === "support" ? "rgba(63,140,255,0.09)" : "rgba(255,123,91,0.09)",
        line: { width: 0 },
      });
    }

    shapes.push({
      type: "line",
      xref: "paper",
      yref: "y",
      x0: 0,
      x1: 1,
      y0: level.level,
      y1: level.level,
      line: { color, width: level.tier === "primary" ? 2 : 1, dash },
    });

    annotations.push({
      xref: "paper",
      yref: "y",
      x: 1.0,
      y: level.level,
      text: `${level.type.toUpperCase()} ${formatPrice(level.level)}`,
      showarrow: false,
      xanchor: "left",
      yanchor: "middle",
      xshift: 4,
      borderpad: 2,
      bgcolor: "rgba(18,25,39,0.9)",
      bordercolor: color,
      font: { color, size: 10 },
    });
  });
}

function buildCandlestickChart(
  bars: CryptoBar[],
  symbol: string,
  structure: CryptoStructurePayload | null,
  overlays: OverlayState,
  formingCandle?: FormingCandleData | null,
) {
  const timestamps = bars.map((b) => b.timestamp);
  const open = bars.map((b) => b.open);
  const high = bars.map((b) => b.high);
  const low = bars.map((b) => b.low);
  const close = bars.map((b) => b.close);
  const volume = bars.map((b) => b.volume);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: Record<string, any>[] = [
    {
      type: "candlestick",
      x: timestamps,
      open,
      high,
      low,
      close,
      name: symbol,
      xaxis: "x",
      yaxis: "y",
      increasing: {
        line: { color: "#38d39f" },
        fillcolor: "rgba(56,211,159,0.85)",
      },
      decreasing: {
        line: { color: "#ff6b6b" },
        fillcolor: "rgba(255,107,107,0.85)",
      },
    },
    {
      type: "bar",
      x: timestamps,
      y: volume,
      name: "Volume",
      marker: {
        color: bars.map((b) =>
          b.close >= b.open
            ? "rgba(56,211,159,0.5)"
            : "rgba(255,107,107,0.5)",
        ),
      },
      xaxis: "x",
      yaxis: "y2",
    },
  ];

  // Forming candle as a semi-transparent dashed-outline bar
  if (formingCandle && formingCandle.timestamp) {
    const fColor = formingCandle.close >= formingCandle.open
      ? "rgba(56,211,159,0.35)"
      : "rgba(255,107,107,0.35)";
    const fLine = formingCandle.close >= formingCandle.open
      ? "rgba(56,211,159,0.8)"
      : "rgba(255,107,107,0.8)";
    traces.push({
      type: "candlestick",
      x: [formingCandle.timestamp],
      open: [formingCandle.open],
      high: [formingCandle.high],
      low: [formingCandle.low],
      close: [formingCandle.close],
      name: `Forming (${Math.round(formingCandle.progress_pct)}%)`,
      xaxis: "x",
      yaxis: "y",
      increasing: {
        line: { color: fLine, width: 1, dash: "dot" },
        fillcolor: fColor,
      },
      decreasing: {
        line: { color: fLine, width: 1, dash: "dot" },
        fillcolor: fColor,
      },
    });
    // Forming candle volume bar
    traces.push({
      type: "bar",
      x: [formingCandle.timestamp],
      y: [formingCandle.volume],
      name: "Forming Vol",
      marker: { color: fColor },
      xaxis: "x",
      yaxis: "y2",
      showlegend: false,
    });
  }

  // Compute SMA overlays (rolling averages from available bars)
  if (overlays.sma20 && bars.length >= 20) {
    const sma20Values = computeRollingSma(close, 20);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: timestamps.slice(19),
      y: sma20Values,
      name: "SMA 20",
      line: { color: "#f0c040", width: 1.2 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.sma50 && bars.length >= 50) {
    const sma50Values = computeRollingSma(close, 50);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: timestamps.slice(49),
      y: sma50Values,
      name: "SMA 50",
      line: { color: "#6baed6", width: 1.2 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.sma200 && bars.length >= 200) {
    const sma200Values = computeRollingSma(close, 200);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: timestamps.slice(199),
      y: sma200Values,
      name: "SMA 200",
      line: { color: "#d291ff", width: 1.3 },
      xaxis: "x",
      yaxis: "y",
    });
  }

  // Bollinger Band overlays
  if (overlays.bollinger && bars.length >= 20) {
    const bbands = computeRollingBollinger(close, 20, 2);
    const bbTimestamps = timestamps.slice(19);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: bbTimestamps,
      y: bbands.upper,
      name: "BB Upper",
      line: { color: "rgba(186,130,255,0.6)", width: 1, dash: "dash" },
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x: bbTimestamps,
      y: bbands.middle,
      name: "BB Middle",
      line: { color: "rgba(186,130,255,0.8)", width: 1 },
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x: bbTimestamps,
      y: bbands.lower,
      name: "BB Lower",
      line: { color: "rgba(186,130,255,0.6)", width: 1, dash: "dash" },
      xaxis: "x",
      yaxis: "y",
    });
  }

  const shapes: Record<string, unknown>[] = [];
  const annotations: Record<string, unknown>[] = [];
  addLevelOverlays(structure?.levels ?? [], annotations, shapes);

  // "FORMING" badge annotation near the forming candle
  if (formingCandle && formingCandle.timestamp) {
    const pctLabel = typeof formingCandle.progress_pct === "number"
      ? `${Math.round(formingCandle.progress_pct)}%`
      : "";
    annotations.push({
      xref: "x",
      yref: "y",
      x: formingCandle.timestamp,
      y: formingCandle.high,
      text: pctLabel ? `FORMING ${pctLabel}` : "FORMING",
      showarrow: false,
      xanchor: "center",
      yanchor: "bottom",
      yshift: 6,
      bgcolor: "rgba(56,211,159,0.18)",
      bordercolor: "rgba(56,211,159,0.5)",
      borderpad: 2,
      font: { color: "#38d39f", size: 9 },
    });
  }

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#8ea0bd", size: 11 },
    margin: { t: 30, r: 60, b: 50, l: 60 },
    dragmode: "zoom" as const,
    legend: {
      orientation: "h" as const,
      y: -0.04,
      x: 0,
      xanchor: "left" as const,
    },
    xaxis: {
      gridcolor: "rgba(255,255,255,0.04)",
      rangeslider: { visible: false },
    },
    yaxis: {
      gridcolor: "rgba(255,255,255,0.06)",
      domain: [0.28, 1],
      title: "Price",
    },
    yaxis2: {
      gridcolor: "rgba(255,255,255,0.04)",
      domain: [0, 0.22],
      title: "Volume",
    },
    shapes,
    annotations,
    height: 500,
  };

  return { traces, layout };
}

/* ── Client-side rolling computations for chart overlays ── */

function computeRollingSma(values: number[], period: number): number[] {
  const result: number[] = [];
  for (let i = period - 1; i < values.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) sum += values[j];
    result.push(sum / period);
  }
  return result;
}

function computeRollingBollinger(
  values: number[],
  period: number,
  stdDev: number,
): { upper: number[]; middle: number[]; lower: number[] } {
  const upper: number[] = [];
  const middle: number[] = [];
  const lower: number[] = [];

  for (let i = period - 1; i < values.length; i++) {
    const window = values.slice(i - period + 1, i + 1);
    const mean = window.reduce((a, b) => a + b, 0) / period;
    const variance = window.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
    const sd = Math.sqrt(variance);
    middle.push(mean);
    upper.push(mean + stdDev * sd);
    lower.push(mean - stdDev * sd);
  }

  return { upper, middle, lower };
}

/* ── WebSocket subscription hook for symbol+interval fan-out ── */

interface FormingCandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  progress_pct: number;
}

interface WsMessage {
  type?: string;
  symbol?: string;
  interval?: string;
  price?: number;
  volume_24h?: number;
  change_pct_24h?: number;
  timestamp?: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
  progress_pct?: number;
  bar?: CryptoBar;
}

function useCryptoSubscription(
  symbol: string,
  interval: IntervalValue,
): {
  formingCandle: FormingCandleData | null;
  closedBar: CryptoBar | null;
  wsConnected: boolean;
} {
  const [formingCandle, setFormingCandle] = useState<FormingCandleData | null>(null);
  const [closedBar, setClosedBar] = useState<CryptoBar | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);
  const disposedRef = useRef(false);
  const currentSub = useRef({ symbol, interval });

  // Keep ref in sync so callbacks can access latest values
  currentSub.current = { symbol, interval };

  const sendSubscribe = useCallback((ws: WebSocket, sym: string, iv: string) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "subscribe", symbol: sym, interval: iv }));
    }
  }, []);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/crypto`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setWsConnected(true);
      backoff.current = 1000;
      sendSubscribe(ws, currentSub.current.symbol, currentSub.current.interval);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage;
        if (
          msg.type === "forming" &&
          msg.symbol === currentSub.current.symbol &&
          msg.interval === currentSub.current.interval
        ) {
          setFormingCandle({
            timestamp: msg.timestamp ?? "",
            open: msg.open ?? 0,
            high: msg.high ?? 0,
            low: msg.low ?? 0,
            close: msg.close ?? 0,
            volume: msg.volume ?? 0,
            progress_pct: msg.progress_pct ?? 0,
          });
          return;
        }
        if (
          msg.type === "candle_close" &&
          msg.symbol === currentSub.current.symbol &&
          msg.interval === currentSub.current.interval &&
          msg.bar
        ) {
          setClosedBar(msg.bar);
          setFormingCandle(null);
        }
      } catch { /* malformed message */ }
    };

    ws.onclose = () => {
      setWsConnected(false);
      wsRef.current = null;
      if (disposedRef.current) {
        return;
      }
      reconnectTimer.current = setTimeout(() => {
        if (disposedRef.current) {
          return;
        }
        backoff.current = Math.min(backoff.current * 2, 30000);
        connect();
      }, backoff.current);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, [sendSubscribe]);

  useEffect(() => {
    disposedRef.current = false;
    connect();
    return () => {
      disposedRef.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // Re-subscribe when symbol or interval changes
  useEffect(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      sendSubscribe(ws, symbol, interval);
      setFormingCandle(null); // Clear stale forming candle
      setClosedBar(null);
    }
  }, [symbol, interval, sendSubscribe]);

  return { formingCandle, closedBar, wsConnected };
}

function mergeClosedBarIntoHistory(
  bars: CryptoBar[],
  closedBar: CryptoBar | null,
): CryptoBar[] {
  if (!closedBar) {
    return bars;
  }
  if (bars.length === 0) {
    return [closedBar];
  }

  const nextBars = [...bars];
  const existingIndex = nextBars.findIndex((bar) => bar.timestamp === closedBar.timestamp);
  if (existingIndex >= 0) {
    nextBars[existingIndex] = closedBar;
    return nextBars;
  }

  nextBars.push(closedBar);
  nextBars.sort((left, right) => left.timestamp.localeCompare(right.timestamp));
  return nextBars;
}

/* ── Main page ── */

export default function CryptoPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC-USD");
  const [selectedInterval, setSelectedInterval] = useState<IntervalValue>("1h");
  const [overlays, setOverlays] = useState<OverlayState>(DEFAULT_OVERLAYS);

  const { data: summary } = useCryptoSummary();
  const { data: indicatorsData } = useCryptoIndicators(selectedSymbol);

  const interval =
    INTERVALS.find((item) => item.value === selectedInterval) ?? INTERVALS[1];
  const { data: historyData, isLoading: historyLoading } = useCryptoHistory(
    selectedSymbol,
    interval.value,
    interval.limit,
  );
  const { data: structureData } = useCryptoStructure(
    selectedSymbol,
    interval.value,
    interval.limit,
  );
  const { data: btcSpyCorrelation } = useCryptoCorrelation("BTC-USD", "SPY", 40);
  const { data: cryptoMarketStructure } = useCryptoMarketStructure("1h", 168);

  // Subscribe to real-time forming candle updates via WebSocket
  const {
    formingCandle: wsFormingCandle,
    closedBar: wsClosedBar,
  } = useCryptoSubscription(selectedSymbol, selectedInterval);

  // Merge WS forming candle with API forming_candle: WS takes priority if present
  const effectiveFormingCandle = useMemo((): FormingCandleData | null => {
    // WS forming candle is realtime and takes precedence
    if (wsFormingCandle) return wsFormingCandle;

    // Fall back to API forming_candle from /api/crypto/history response
    const apiForming =
      typeof historyData?.forming_candle === "object" &&
      historyData.forming_candle !== null
        ? historyData.forming_candle
        : null;
    if (!apiForming) return null;

    // Convert CryptoBar to FormingCandleData (progress_pct unknown from API, default 0)
    return {
      timestamp: apiForming.timestamp,
      open: apiForming.open,
      high: apiForming.high,
      low: apiForming.low,
      close: apiForming.close,
      volume: apiForming.volume,
      progress_pct: 0,
    };
  }, [wsFormingCandle, historyData?.forming_candle]);

  const indicators: CryptoIndicators | null =
    indicatorsData?.indicators ?? null;

  const effectiveBars = useMemo(
    () => mergeClosedBarIntoHistory(historyData?.bars ?? [], wsClosedBar),
    [historyData?.bars, wsClosedBar],
  );

  const chartResult = useMemo(() => {
    if (effectiveBars.length === 0)
      return null;
    return buildCandlestickChart(
      effectiveBars,
      selectedSymbol,
      structureData ?? null,
      overlays,
      effectiveFormingCandle,
    );
  }, [effectiveBars, selectedSymbol, structureData, overlays, effectiveFormingCandle]);

  const availableBarCount = effectiveBars.length;
  const shortHistory = historyData != null && availableBarCount < interval.limit;

  const connected = summary?.connected ?? false;

  return (
    <div className="space-y-6">
      {/* NFA disclaimer */}
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        Cryptocurrency trading carries significant risk. Prices are from Binance and may differ from other exchanges.
      </div>

      <h2 className="text-xl font-bold tracking-tight">Crypto</h2>

      {/* Connection status */}
      {!connected && (
        <div className="rounded-lg border border-[rgba(255,107,107,0.3)] bg-[rgba(255,107,107,0.08)] px-4 py-2 text-sm text-[var(--red)]">
          Crypto feed disconnected — prices may be stale or unavailable.
        </div>
      )}

      {/* Price cards — all 5 pairs */}
      <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
        {ALL_SYMBOLS.map((sym) => (
          <CryptoPriceCard
            key={sym}
            tick={summary?.prices?.[sym]}
            symbol={sym}
            selected={selectedSymbol === sym}
            onSelect={() => setSelectedSymbol(sym)}
          />
        ))}
      </div>

      {/* Chart controls */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex gap-1">
              {ALL_SYMBOLS.map((sym) => (
                <button
                  key={sym}
                  onClick={() => setSelectedSymbol(sym)}
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
              {INTERVALS.map((iv) => (
                <button
                  key={iv.label}
                  onClick={() => setSelectedInterval(iv.value)}
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
                ? `${availableBarCount} bars · ${formatVisibleWindow(interval.value, availableBarCount)} visible`
                : `Target window ${interval.targetWindow}`}
            </div>
            <div>{selectedSymbol.replace("-USD", "")} from native Binance history with live 1-minute patching</div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
            Indicators
          </span>
          {OVERLAY_OPTIONS.map((option) => {
            const unavailable = availableBarCount > 0 && availableBarCount < option.minBars;
            const active = overlays[option.key];
            return (
              <button
                key={option.key}
                type="button"
                disabled={unavailable}
                title={unavailable ? `Needs ${option.minBars} bars on this timeframe` : undefined}
                onClick={() =>
                  setOverlays((current) => ({
                    ...current,
                    [option.key]: !current[option.key],
                  }))
                }
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
            Showing {availableBarCount} {interval.label} bars cached so far. Missing history is backfilled from Binance and then stored locally.
          </div>
        )}
      </div>

      {/* Candlestick chart */}
      {historyLoading && <Spinner className="mt-8" />}
      {!historyLoading && chartResult && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
          <PlotlyWrapper
            data={chartResult.traces as unknown as Record<string, unknown>[]}
            layout={chartResult.layout as unknown as Record<string, unknown>}
            config={{
              responsive: true,
              displayModeBar: true,
              scrollZoom: true,
            }}
            style={{ width: "100%", height: 500 }}
          />
        </div>
      )}
      {!historyLoading && !chartResult && connected && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--muted)]">
          No chart data available yet. The app will backfill Binance history for this symbol and timeframe on demand.
        </div>
      )}
      {!historyLoading && !chartResult && !connected && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--red)]">
          Crypto feed disconnected — no chart data available.
        </div>
      )}

      <StructureCard structure={structureData} />

      <div className="grid gap-4 xl:grid-cols-2">
        <BtcSpyCorrelationCard correlation={btcSpyCorrelation} />
        <CryptoBreadthCard market={cryptoMarketStructure} />
      </div>

      {/* Technical indicator cards */}
      {indicators && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <RsiGauge value={indicators.rsi_14} />
          <MacdCard macd={indicators.macd} />
          <BollingerCard bollinger={indicators.bollinger} />
          <VwapSmaCard
            vwap={indicators.vwap}
            sma20={indicators.sma_20}
            sma50={indicators.sma_50}
          />
        </div>
      )}

      {/* Info footer */}
      <div className="text-xs text-[var(--muted)]">
        Data source: Binance WebSocket (public, no API key) &middot; multi-timeframe
        aggregation from persisted 1-minute bars &middot; longer higher-timeframe windows
        depend on retained history &middot; prices in USDT
      </div>
    </div>
  );
}

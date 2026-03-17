import { useState, useMemo, lazy, Suspense } from "react";
import { useCryptoSummary, useCryptoHistory, useCryptoIndicators } from "../api/hooks";
import type { CryptoPrice, CryptoBar, CryptoIndicators } from "../api/types";
import Spinner from "../components/ui/Spinner";

const Plot = lazy(() => import("react-plotly.js"));

/* ── Constants ── */

const ALL_SYMBOLS = [
  "BTC-USD",
  "ETH-USD",
  "SOL-USD",
  "XRP-USD",
  "DOGE-USD",
] as const;

const INTERVALS = [
  { value: "1m", label: "1m", limit: 60 },
  { value: "1m", label: "5m", limit: 300 },
  { value: "1m", label: "1h", limit: 720 },
  { value: "1m", label: "24h", limit: 1440 },
] as const;

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

/* ── Candlestick chart builder with overlays ── */

function buildCandlestickChart(
  bars: CryptoBar[],
  symbol: string,
  indicators: CryptoIndicators | null,
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

  // Compute SMA overlays (rolling averages from available bars)
  if (bars.length >= 20) {
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
  if (bars.length >= 50) {
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

  // Bollinger Band overlays
  if (indicators && indicators.bollinger.upper !== null && bars.length >= 20) {
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

/* ── Main page ── */

export default function CryptoPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC-USD");
  const [selectedInterval, setSelectedInterval] = useState(0);

  const { data: summary } = useCryptoSummary();
  const { data: indicatorsData } = useCryptoIndicators(selectedSymbol);

  const interval = INTERVALS[selectedInterval];
  const { data: historyData, isLoading: historyLoading } = useCryptoHistory(
    selectedSymbol,
    interval.value,
    interval.limit,
  );

  const indicators: CryptoIndicators | null =
    indicatorsData?.indicators ?? null;

  const chartResult = useMemo(() => {
    if (!historyData || !historyData.bars || historyData.bars.length === 0)
      return null;
    return buildCandlestickChart(historyData.bars, selectedSymbol, indicators);
  }, [historyData, selectedSymbol, indicators]);

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
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1">
          {ALL_SYMBOLS.map((sym) => (
            <button
              key={sym}
              onClick={() => setSelectedSymbol(sym)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                selectedSymbol === sym
                  ? "bg-[var(--accent)] text-white"
                  : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {sym.split("-")[0]}
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {INTERVALS.map((iv, idx) => (
            <button
              key={iv.label}
              onClick={() => setSelectedInterval(idx)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                selectedInterval === idx
                  ? "bg-[var(--blue)] text-white"
                  : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {iv.label}
            </button>
          ))}
        </div>
      </div>

      {/* Candlestick chart */}
      {historyLoading && <Spinner className="mt-8" />}
      {!historyLoading && chartResult && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
          <Suspense fallback={<Spinner className="py-24" />}>
            <Plot
              data={chartResult.traces as unknown as Record<string, unknown>[]}
              layout={chartResult.layout as unknown as Record<string, unknown>}
              config={{
                responsive: true,
                displayModeBar: true,
                scrollZoom: true,
              }}
              style={{ width: "100%", height: 500 }}
            />
          </Suspense>
        </div>
      )}
      {!historyLoading && !chartResult && connected && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--muted)]">
          No chart data available yet. Bars accumulate as the feed runs (1 bar
          per minute).
        </div>
      )}
      {!historyLoading && !chartResult && !connected && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--red)]">
          Crypto feed disconnected — no chart data available.
        </div>
      )}

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
        Data source: Binance WebSocket (public, no API key) &middot; 1-minute
        bars &middot; Prices in USDT
      </div>
    </div>
  );
}

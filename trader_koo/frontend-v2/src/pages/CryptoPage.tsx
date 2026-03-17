import { useState, useMemo, lazy, Suspense } from "react";
import { useCryptoSummary, useCryptoHistory } from "../api/hooks";
import type { CryptoPrice, CryptoBar } from "../api/types";
import Spinner from "../components/ui/Spinner";

const Plot = lazy(() => import("react-plotly.js"));

/* ── Helpers ── */

function formatPrice(price: number): string {
  if (price >= 1000)
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    });
  return price.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
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

function CryptoPriceCard({ tick }: { tick: CryptoPrice | undefined; symbol: string }) {
  if (!tick) {
    return (
      <GlassCard label="--">
        <div className="text-sm text-[var(--muted)]">No data</div>
      </GlassCard>
    );
  }

  const isPositive = tick.change_pct_24h >= 0;
  const changeColor = isPositive ? "text-[var(--green)]" : "text-[var(--red)]";
  const sign = isPositive ? "+" : "";
  const borderClass = isPositive
    ? "border-[rgba(56,211,159,0.3)]"
    : "border-[rgba(255,107,107,0.3)]";

  return (
    <GlassCard label={tick.symbol} className={borderClass}>
      <div className="text-2xl font-bold tabular-nums text-[var(--text)]">
        ${formatPrice(tick.price)}
      </div>
      <div className="mt-1 flex items-center gap-3 text-xs">
        <span className={`font-semibold tabular-nums ${changeColor}`}>
          {sign}{tick.change_pct_24h.toFixed(2)}% 24h
        </span>
        <span className="text-[var(--muted)]">
          Vol: {formatVolume(tick.volume_24h)}
        </span>
      </div>
    </GlassCard>
  );
}

/* ── Candlestick chart builder ── */

function buildCandlestickChart(bars: CryptoBar[], symbol: string) {
  const timestamps = bars.map((b) => b.timestamp);
  const open = bars.map((b) => b.open);
  const high = bars.map((b) => b.high);
  const low = bars.map((b) => b.low);
  const close = bars.map((b) => b.close);
  const volume = bars.map((b) => b.volume);

  const traces = [
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

/* ── Interval selector ── */

const INTERVALS = [
  { value: "1m", label: "1m", limit: 60 },
  { value: "1m", label: "5m", limit: 300 },
  { value: "1m", label: "1h", limit: 720 },
  { value: "1m", label: "24h", limit: 1440 },
] as const;

/* ── Main page ── */

export default function CryptoPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC-USD");
  const [selectedInterval, setSelectedInterval] = useState(0);

  const { data: summary } = useCryptoSummary();

  const interval = INTERVALS[selectedInterval];
  const { data: historyData, isLoading: historyLoading } = useCryptoHistory(
    selectedSymbol,
    interval.value,
    interval.limit,
  );

  const chartResult = useMemo(() => {
    if (!historyData || !historyData.bars || historyData.bars.length === 0)
      return null;
    return buildCandlestickChart(historyData.bars, selectedSymbol);
  }, [historyData, selectedSymbol]);

  const btc = summary?.prices?.["BTC-USD"];
  const eth = summary?.prices?.["ETH-USD"];
  const connected = summary?.connected ?? false;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">Crypto</h2>

      {/* Connection status */}
      {!connected && (
        <div className="rounded-lg border border-[rgba(255,107,107,0.3)] bg-[rgba(255,107,107,0.08)] px-4 py-2 text-sm text-[var(--red)]">
          Crypto feed disconnected — prices may be stale or unavailable.
        </div>
      )}

      {/* Price cards */}
      <div className="grid gap-3 sm:grid-cols-2">
        <CryptoPriceCard tick={btc} symbol="BTC-USD" />
        <CryptoPriceCard tick={eth} symbol="ETH-USD" />
      </div>

      {/* Chart controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1">
          {(["BTC-USD", "ETH-USD"] as const).map((sym) => (
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
              data={
                chartResult.traces as unknown as Record<string, unknown>[]
              }
              layout={
                chartResult.layout as unknown as Record<string, unknown>
              }
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

      {/* Info footer */}
      <div className="text-xs text-[var(--muted)]">
        Data source: Binance WebSocket (public, no API key) &middot; 1-minute
        bars &middot; Prices in USDT
      </div>
    </div>
  );
}

import { useState, useMemo, useEffect, useCallback } from "react";
import {
  useCryptoSummary,
  useCryptoHistory,
  useCryptoIndicators,
  useCryptoStructure,
  useCryptoCorrelation,
  useCryptoMarketStructure,
  useCryptoOpenInterest,
} from "../api/hooks";
import type { CryptoIndicators } from "../api/types";
import {
  CryptoPriceCard,
} from "../components/crypto/CryptoInsightCards";
import CryptoAnalyticsPanels from "../components/crypto/CryptoAnalyticsPanels";
import CryptoToolbar from "../components/crypto/CryptoToolbar";
import CryptoChartPanel from "../components/crypto/CryptoChartPanel";
import {
  mergeClosedBarIntoHistory,
  type FormingCandleData,
  useCryptoSubscription,
} from "../hooks/useCryptoSubscription";
import {
  buildCandlestickChart,
  type CryptoOverlayState,
} from "../lib/crypto/buildCandlestickChart";

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
type OverlayState = CryptoOverlayState;
type AxisRangeValue = string | number;

interface PersistedZoomState {
  xRange: [AxisRangeValue, AxisRangeValue] | null;
  yRange: [number, number] | null;
}

const DEFAULT_OVERLAYS: OverlayState = {
  sma20: true,
  sma50: true,
  sma200: false,
  bollinger: false,
};

const EMPTY_ZOOM_STATE: PersistedZoomState = {
  xRange: null,
  yRange: null,
};

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

function cryptoZoomStorageKey(symbol: string, interval: IntervalValue): string {
  return `trader_koo_crypto_zoom:${symbol}:${interval}`;
}

function readZoomState(storageKey: string): PersistedZoomState {
  if (typeof window === "undefined") {
    return EMPTY_ZOOM_STATE;
  }
  try {
    const raw = window.sessionStorage.getItem(storageKey);
    if (!raw) {
      return EMPTY_ZOOM_STATE;
    }
    const parsed = JSON.parse(raw) as Partial<PersistedZoomState>;
    const xRange = Array.isArray(parsed.xRange) && parsed.xRange.length === 2
      ? [parsed.xRange[0], parsed.xRange[1]] as [AxisRangeValue, AxisRangeValue]
      : null;
    const yRange = Array.isArray(parsed.yRange) &&
      parsed.yRange.length === 2 &&
      Number.isFinite(parsed.yRange[0]) &&
      Number.isFinite(parsed.yRange[1])
      ? [Number(parsed.yRange[0]), Number(parsed.yRange[1])] as [number, number]
      : null;
    return { xRange, yRange };
  } catch {
    return EMPTY_ZOOM_STATE;
  }
}

function writeZoomState(storageKey: string, value: PersistedZoomState): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.sessionStorage.setItem(storageKey, JSON.stringify(value));
  } catch {
    // Ignore storage failures and keep the zoom only in memory.
  }
}

function readAxisRange(
  eventData: Record<string, unknown>,
  axis: "xaxis" | "yaxis",
): [AxisRangeValue, AxisRangeValue] | null {
  const direct = eventData[`${axis}.range`];
  if (Array.isArray(direct) && direct.length === 2) {
    return [direct[0] as AxisRangeValue, direct[1] as AxisRangeValue];
  }

  const left = eventData[`${axis}.range[0]`];
  const right = eventData[`${axis}.range[1]`];
  if ((typeof left === "string" || typeof left === "number") &&
      (typeof right === "string" || typeof right === "number")) {
    return [left, right];
  }

  return null;
}

function readNumericRange(
  eventData: Record<string, unknown>,
  axis: "xaxis" | "yaxis",
): [number, number] | null {
  const range = readAxisRange(eventData, axis);
  if (!range) {
    return null;
  }
  const left = Number(range[0]);
  const right = Number(range[1]);
  if (!Number.isFinite(left) || !Number.isFinite(right)) {
    return null;
  }
  return [left, right];
}

/* ── Main page ── */

export default function CryptoPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC-USD");
  const [selectedInterval, setSelectedInterval] = useState<IntervalValue>("1h");
  const [overlays, setOverlays] = useState<OverlayState>(DEFAULT_OVERLAYS);
  const zoomKey = useMemo(
    () => cryptoZoomStorageKey(selectedSymbol, selectedInterval),
    [selectedSymbol, selectedInterval],
  );
  const [zoomState, setZoomState] = useState<PersistedZoomState>(() => readZoomState(zoomKey));

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
  const { data: openInterestData } = useCryptoOpenInterest(selectedSymbol, "1h", 100);

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

  useEffect(() => {
    setZoomState(readZoomState(zoomKey));
  }, [zoomKey]);

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
      `${selectedSymbol}-${selectedInterval}`,
    );
  }, [effectiveBars, selectedSymbol, selectedInterval, structureData, overlays, effectiveFormingCandle]);

  const chartLayout = useMemo(() => {
    if (!chartResult) {
      return null;
    }

    const baseLayout = chartResult.layout as Record<string, unknown>;
    const baseXAxis = (baseLayout.xaxis as Record<string, unknown> | undefined) ?? {};
    const baseYAxis = (baseLayout.yaxis as Record<string, unknown> | undefined) ?? {};

    return {
      ...baseLayout,
      xaxis: zoomState.xRange
        ? {
            ...baseXAxis,
            range: zoomState.xRange,
            autorange: false,
          }
        : {
            ...baseXAxis,
            autorange: true,
          },
      yaxis: zoomState.yRange
        ? {
            ...baseYAxis,
            range: zoomState.yRange,
            autorange: false,
          }
        : {
            ...baseYAxis,
            autorange: true,
          },
    };
  }, [chartResult, zoomState]);

  const handleChartRelayout = useCallback((eventData: Record<string, unknown>) => {
    setZoomState((current) => {
      let next = current;

      if (eventData["xaxis.autorange"] === true) {
        next = { ...next, xRange: null };
      } else {
        const xRange = readAxisRange(eventData, "xaxis");
        if (xRange) {
          next = { ...next, xRange };
        }
      }

      if (eventData["yaxis.autorange"] === true) {
        next = { ...next, yRange: null };
      } else {
        const yRange = readNumericRange(eventData, "yaxis");
        if (yRange) {
          next = { ...next, yRange };
        }
      }

      writeZoomState(zoomKey, next);
      return next;
    });
  }, [zoomKey]);

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
            selected={selectedSymbol === sym}
            onSelect={() => setSelectedSymbol(sym)}
          />
        ))}
      </div>

      {/* Chart controls */}
      <CryptoToolbar
        allSymbols={ALL_SYMBOLS}
        intervals={INTERVALS}
        overlayOptions={OVERLAY_OPTIONS}
        selectedSymbol={selectedSymbol}
        selectedInterval={selectedInterval}
        overlays={overlays}
        availableBarCount={availableBarCount}
        shortHistory={shortHistory}
        currentInterval={interval}
        formatVisibleWindow={formatVisibleWindow}
        onSelectSymbol={setSelectedSymbol}
        onSelectInterval={(value) => setSelectedInterval(value as IntervalValue)}
        onToggleOverlay={(overlayKey) => {
          const key = overlayKey as OverlayKey;
          setOverlays((current) => ({
            ...current,
            [key]: !current[key],
          }));
        }}
      />

      {/* Candlestick chart */}
      <CryptoChartPanel
        historyLoading={historyLoading}
        hasChart={Boolean(chartResult && chartLayout)}
        connected={connected}
        chartData={chartResult ? (chartResult.traces as unknown as Record<string, unknown>[]) : null}
        chartLayout={chartLayout as Record<string, unknown> | null}
        onRelayout={handleChartRelayout}
      />

      <CryptoAnalyticsPanels
        structure={structureData}
        btcSpyCorrelation={btcSpyCorrelation}
        cryptoMarketStructure={cryptoMarketStructure}
        indicators={indicators}
      />

      {/* Open Interest Panel */}
      {openInterestData && openInterestData.oi_bars.length > 0 && (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-[var(--text)]">
              Open Interest ({selectedSymbol})
            </h3>
            <div className="flex items-center gap-3 text-xs text-[var(--muted)]">
              {openInterestData.current_oi && (
                <span>
                  Current: <span className="font-medium text-[var(--text)]">
                    {(openInterestData.current_oi.open_interest).toLocaleString()} contracts
                  </span>
                </span>
              )}
              {openInterestData.oi_change_24h_pct != null && (
                <span className={openInterestData.oi_change_24h_pct >= 0 ? "text-[var(--green)]" : "text-[var(--red)]"}>
                  {openInterestData.oi_change_24h_pct > 0 ? "+" : ""}
                  {openInterestData.oi_change_24h_pct.toFixed(2)}% 24h
                </span>
              )}
            </div>
          </div>
          <div className="h-[180px]">
            <CryptoChartPanel
              chartData={[
                {
                  type: "scatter",
                  mode: "lines",
                  fill: "tozeroy",
                  x: openInterestData.oi_bars.map((b) => b.timestamp),
                  y: openInterestData.oi_bars.map((b) => b.open_interest_value),
                  name: "Open Interest (USD)",
                  line: { color: "#f59e0b", width: 1.5 },
                  fillcolor: "rgba(245,158,11,0.08)",
                },
              ]}
              chartLayout={{
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#8ea0bd", size: 10 },
                margin: { t: 10, r: 10, b: 30, l: 60 },
                xaxis: { gridcolor: "rgba(255,255,255,0.04)" },
                yaxis: {
                  gridcolor: "rgba(255,255,255,0.06)",
                  title: { text: "OI Value (USD)", font: { size: 10 } },
                },
                showlegend: false,
                height: 180,
              }}
              onRelayout={() => {}}
            />
          </div>
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

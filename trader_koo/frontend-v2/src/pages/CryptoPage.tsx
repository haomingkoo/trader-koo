import { useState, useMemo, useEffect, useCallback } from "react";
import {
  useCryptoSummary,
  useCryptoHistory,
  useCryptoIndicators,
  useCryptoStructure,
  useCryptoCorrelation,
  useCryptoMarketStructure,
} from "../api/hooks";
import type {
  CryptoBar,
  CryptoIndicators,
  CryptoStructurePayload,
  LevelRow,
} from "../api/types";
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
  uirevisionKey?: string,
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
      name: "BOLL Upper",
      line: { color: "#ffd21f", width: 1.8 },
      showlegend: false,
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x: bbTimestamps,
      y: bbands.middle,
      name: "BOLL Mid",
      line: { color: "#f6bed8", width: 1.6 },
      showlegend: false,
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x: bbTimestamps,
      y: bbands.lower,
      name: "BOLL Lower",
      line: { color: "#16c7ff", width: 1.8 },
      showlegend: false,
      xaxis: "x",
      yaxis: "y",
    });
  }

  const shapes: Record<string, unknown>[] = [];
  const annotations: Record<string, unknown>[] = [];
  addLevelOverlays(structure?.levels ?? [], annotations, shapes);

  if (overlays.bollinger && bars.length >= 20) {
    const bbands = computeRollingBollinger(close, 20, 2);
    const bbUpper = bbands.upper[bbands.upper.length - 1];
    const bbMiddle = bbands.middle[bbands.middle.length - 1];
    const bbLower = bbands.lower[bbands.lower.length - 1];
    if (
      Number.isFinite(bbUpper) &&
      Number.isFinite(bbMiddle) &&
      Number.isFinite(bbLower)
    ) {
      annotations.push(
        {
          xref: "paper",
          yref: "paper",
          x: 0.015,
          y: 0.985,
          text: "BOLL",
          showarrow: false,
          xanchor: "left",
          yanchor: "top",
          font: { color: "#e7edf7", size: 12 },
        },
        {
          xref: "paper",
          yref: "paper",
          x: 0.075,
          y: 0.985,
          text: `MID: ${formatPrice(bbMiddle)}`,
          showarrow: false,
          xanchor: "left",
          yanchor: "top",
          font: { color: "#f6bed8", size: 12 },
        },
        {
          xref: "paper",
          yref: "paper",
          x: 0.17,
          y: 0.985,
          text: `UPPER: ${formatPrice(bbUpper)}`,
          showarrow: false,
          xanchor: "left",
          yanchor: "top",
          font: { color: "#ffd21f", size: 12 },
        },
        {
          xref: "paper",
          yref: "paper",
          x: 0.315,
          y: 0.985,
          text: `LOWER: ${formatPrice(bbLower)}`,
          showarrow: false,
          xanchor: "left",
          yanchor: "top",
          font: { color: "#16c7ff", size: 12 },
        },
      );
    }
  }

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
    uirevision: uirevisionKey ?? symbol,
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
        onToggleOverlay={(overlayKey) =>
          setOverlays((current) => ({
            ...current,
            [overlayKey]: !current[overlayKey as OverlayKey],
          }))
        }
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

      {/* Info footer */}
      <div className="text-xs text-[var(--muted)]">
        Data source: Binance WebSocket (public, no API key) &middot; multi-timeframe
        aggregation from persisted 1-minute bars &middot; longer higher-timeframe windows
        depend on retained history &middot; prices in USDT
      </div>
    </div>
  );
}

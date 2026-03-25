import type {
  DashboardPayload,
  LiveCandle,
  OhlcvRow,
  YoloPatternRow,
  EquityTick,
} from "../../api/types";

export function formatChartNumber(
  n: number | null | undefined,
  decimals = 2,
): string {
  if (n == null || !Number.isFinite(n)) return "\u2014";
  return n.toFixed(decimals);
}

function ma(arr: number[], n: number): (number | null)[] {
  return arr.map((_, i) => {
    if (i < n - 1) return null;
    let s = 0;
    for (let j = i - n + 1; j <= i; j++) s += arr[j];
    return s / n;
  });
}

function ema(arr: number[], n: number): (number | null)[] {
  if (arr.length < n) return arr.map(() => null);
  const multiplier = 2 / (n + 1);
  const result: (number | null)[] = [];
  // Seed with SMA of first n values
  let seed = 0;
  for (let i = 0; i < n; i++) {
    seed += arr[i];
    result.push(null);
  }
  seed /= n;
  result[n - 1] = seed;
  for (let i = n; i < arr.length; i++) {
    seed = (arr[i] - seed) * multiplier + seed;
    result.push(seed);
  }
  return result;
}

function lastFinite(values: Array<number | null | undefined>): number | null {
  for (let i = values.length - 1; i >= 0; i -= 1) {
    const value = values[i];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

export function resampleToWeekly(rows: OhlcvRow[]): OhlcvRow[] {
  if (rows.length === 0) return [];
  const weeks: OhlcvRow[] = [];
  let current: OhlcvRow | null = null;

  for (const row of rows) {
    const d = new Date(row.date);
    const dayOfWeek = d.getDay();
    if (!current || dayOfWeek === 1) {
      if (current) weeks.push(current);
      current = { ...row };
    } else {
      current.high = Math.max(current.high, row.high);
      current.low = Math.min(current.low, row.low);
      current.close = row.close;
      current.volume += row.volume;
      current.date = row.date;
    }
  }
  if (current) weeks.push(current);
  return weeks;
}

interface PlotlyTrace {
  type: string;
  x: string[];
  y?: (number | null)[];
  open?: number[];
  high?: number[];
  low?: number[];
  close?: number[];
  name: string;
  mode?: string;
  line?: Record<string, unknown>;
  marker?: Record<string, unknown>;
  fill?: string;
  fillcolor?: string;
  hovertext?: string[];
  hoverinfo?: string;
  showlegend?: boolean;
  xaxis: string;
  yaxis: string;
  increasing?: Record<string, unknown>;
  decreasing?: Record<string, unknown>;
}

interface PlotlyShape {
  type: string;
  xref: string;
  yref: string;
  x0: string | number;
  x1: string | number;
  y0: number;
  y1: number;
  line?: Record<string, unknown>;
  fillcolor?: string;
}

interface PlotlyAnnotation {
  xref: string;
  yref: string;
  x: string | number;
  y: number;
  text: string;
  showarrow: boolean;
  xanchor?: string;
  yanchor?: string;
  xshift?: number;
  yshift?: number;
  borderpad?: number;
  bgcolor?: string;
  bordercolor?: string;
  font?: Record<string, unknown>;
  align?: string;
}

export const CHART_OVERLAY_OPTIONS = [
  { key: "ma5", label: "5 MA", minBars: 5 },
  { key: "ma20", label: "20 MA", minBars: 20 },
  { key: "ma50", label: "50 MA", minBars: 50 },
  { key: "ma200", label: "200 MA", minBars: 200 },
  { key: "ema20", label: "20 EMA", minBars: 20 },
  { key: "ema50", label: "50 EMA", minBars: 50 },
  { key: "ema200", label: "200 EMA", minBars: 200 },
  { key: "bollinger", label: "Bollinger", minBars: 20 },
] as const;

export type ChartOverlayKey = (typeof CHART_OVERLAY_OPTIONS)[number]["key"];
export type ChartOverlayState = Record<ChartOverlayKey, boolean>;

export const DEFAULT_CHART_OVERLAYS: ChartOverlayState = {
  ma5: false,
  ma20: true,
  ma50: true,
  ma200: false,
  ema20: false,
  ema50: false,
  ema200: false,
  bollinger: false,
};

function computeRollingBollinger(
  values: number[],
  period: number,
  stdDev: number,
): { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] } {
  return values
    .map((_, idx) => {
      if (idx < period - 1) {
        return { upper: null, middle: null, lower: null };
      }
      const window = values.slice(idx - period + 1, idx + 1);
      const mean = window.reduce((a, b) => a + b, 0) / period;
      const variance = window.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
      const sd = Math.sqrt(variance);
      return {
        upper: mean + stdDev * sd,
        middle: mean,
        lower: mean - stdDev * sd,
      };
    })
    .reduce(
      (acc, row) => {
        acc.upper.push(row.upper);
        acc.middle.push(row.middle);
        acc.lower.push(row.lower);
        return acc;
      },
      {
        upper: [] as (number | null)[],
        middle: [] as (number | null)[],
        lower: [] as (number | null)[],
      },
    );
}

function defaultThreeMonthRange(dates: string[]): [string, string] | undefined {
  if (dates.length === 0) return undefined;
  const end = new Date(dates[dates.length - 1]);
  if (Number.isNaN(end.getTime())) return undefined;
  const start = new Date(end);
  start.setMonth(start.getMonth() - 3);
  const earliest = new Date(dates[0]);

  // Add ~3 trading days of padding on each side so edge candles aren't clipped
  const PAD_DAYS = 3;
  const padStart = new Date(start);
  padStart.setDate(padStart.getDate() - PAD_DAYS);
  const padEnd = new Date(end);
  padEnd.setDate(padEnd.getDate() + PAD_DAYS);

  if (!Number.isNaN(earliest.getTime()) && padStart < earliest) {
    const clampedStart = new Date(earliest);
    clampedStart.setDate(clampedStart.getDate() - PAD_DAYS);
    return [clampedStart.toISOString(), padEnd.toISOString()];
  }
  return [padStart.toISOString(), padEnd.toISOString()];
}

function isoDateOnly(value: string): string | null {
  const raw = value.length <= 10 ? `${value}T00:00:00Z` : value;
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString().slice(0, 10);
}

function tradingDayRangeBreaks(dates: string[]): string[] {
  if (dates.length < 2) return [];
  const normalized = dates
    .map(isoDateOnly)
    .filter((value): value is string => value !== null);
  if (normalized.length < 2) return [];

  const missing = new Set<string>();
  for (let i = 1; i < normalized.length; i += 1) {
    const prev = new Date(`${normalized[i - 1]}T00:00:00Z`);
    const next = new Date(`${normalized[i]}T00:00:00Z`);
    if (Number.isNaN(prev.getTime()) || Number.isNaN(next.getTime())) continue;

    prev.setUTCDate(prev.getUTCDate() + 1);
    while (prev < next) {
      const weekday = prev.getUTCDay();
      if (weekday >= 1 && weekday <= 5) {
        missing.add(prev.toISOString().slice(0, 10));
      }
      prev.setUTCDate(prev.getUTCDate() + 1);
    }
  }
  return [...missing];
}

function nyDateStringFromIso(value: string | null | undefined): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const year = parts.find((part) => part.type === "year")?.value;
  const month = parts.find((part) => part.type === "month")?.value;
  const day = parts.find((part) => part.type === "day")?.value;
  if (!year || !month || !day) return null;
  return `${year}-${month}-${day}`;
}

function mondayWeekKey(value: string | null | undefined): string | null {
  if (!value) return null;
  const raw = value.length <= 10 ? `${value}T00:00:00Z` : value;
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return null;
  const utcDay = date.getUTCDay();
  const deltaToMonday = utcDay === 0 ? -6 : 1 - utcDay;
  date.setUTCDate(date.getUTCDate() + deltaToMonday);
  return date.toISOString().slice(0, 10);
}

export function applyLivePriceToPayload(
  payload: DashboardPayload | undefined,
  livePrice: EquityTick | null,
): DashboardPayload | undefined {
  if (!payload || !livePrice || livePrice.symbol !== payload.ticker) {
    return payload;
  }
  const chart = payload.chart ?? [];
  if (chart.length === 0) return payload;

  const sessionDate =
    nyDateStringFromIso(livePrice.timestamp) ?? chart[chart.length - 1]?.date;
  if (!sessionDate) return payload;

  const nextChart = [...chart];
  const lastBar = nextChart[nextChart.length - 1];
  if (lastBar.date === sessionDate) {
    nextChart[nextChart.length - 1] = {
      ...lastBar,
      high: Math.max(lastBar.high, livePrice.price),
      low: Math.min(lastBar.low, livePrice.price),
      close: livePrice.price,
    };
  } else {
    const open = livePrice.prev_price ?? lastBar.close;
    nextChart.push({
      date: sessionDate,
      open,
      high: Math.max(open, livePrice.price),
      low: Math.min(open, livePrice.price),
      close: livePrice.price,
      volume: 0,
      ma20: null,
      ma50: null,
      ma100: null,
      ma200: null,
      atr: null,
      atr_pct: null,
    });
  }

  return {
    ...payload,
    chart: nextChart,
    fundamentals: {
      ...payload.fundamentals,
      price: livePrice.price,
    },
  };
}

export function buildChartData(
  payload: DashboardPayload,
  isWeekly: boolean,
  overlays: ChartOverlayState,
  liveCandle?: LiveCandle | null,
  compactMode = false,
) {
  const rawChart = payload.chart ?? [];
  const baseChart = isWeekly ? resampleToWeekly(rawChart) : rawChart;
  const levels = payload.levels ?? [];
  const gaps = payload.gaps ?? [];
  const yoloPatterns = payload.yolo_patterns ?? [];
  const earningsMarkers = payload.earnings_markers ?? [];
  const candlePatterns = payload.candlestick_patterns ?? [];
  const hmmRegime = payload.hmm_regime ?? null;
  const ticker = payload.ticker ?? "N/A";

  const hasLiveCandle =
    typeof liveCandle === "object" &&
    liveCandle !== null &&
    typeof liveCandle.timestamp === "string" &&
    typeof liveCandle.open === "number" &&
    typeof liveCandle.high === "number" &&
    typeof liveCandle.low === "number" &&
    typeof liveCandle.close === "number" &&
    Number.isFinite(liveCandle.open) &&
    Number.isFinite(liveCandle.close);

  const liveDate = hasLiveCandle
    ? nyDateStringFromIso(liveCandle.timestamp) ?? liveCandle.timestamp
    : null;

  const chart = [...baseChart];
  let ghostLiveCandle: LiveCandle | null = hasLiveCandle ? liveCandle : null;

  if (hasLiveCandle && chart.length > 0) {
    const lastRow = chart[chart.length - 1];
    const sameWeeklyBucket =
      isWeekly &&
      mondayWeekKey(lastRow.date) !== null &&
      mondayWeekKey(lastRow.date) === mondayWeekKey(liveCandle.timestamp);

    if (sameWeeklyBucket) {
      chart[chart.length - 1] = {
        ...lastRow,
        date: liveDate ?? lastRow.date,
        high: Math.max(lastRow.high, liveCandle.high),
        low: Math.min(lastRow.low, liveCandle.low),
        close: liveCandle.close,
        volume:
          lastRow.volume + (Number.isFinite(liveCandle.volume) ? liveCandle.volume : 0),
      };
      ghostLiveCandle = null;
    }
  }

  const x = chart.map((r) => r.date);
  const open = chart.map((r) => r.open);
  const high = chart.map((r) => r.high);
  const low = chart.map((r) => r.low);
  const close = chart.map((r) => r.close);
  const vol = chart.map((r) => r.volume);

  const lastX = x[x.length - 1] ?? null;
  const shouldAppendLive = ghostLiveCandle !== null && liveDate !== null && liveDate !== lastX;
  const liveBadgeX = shouldAppendLive ? liveDate : lastX;
  const liveBadgeY = shouldAppendLive
    ? ghostLiveCandle?.high ?? null
    : high.length > 0
      ? high[high.length - 1]
      : null;

  const traces: PlotlyTrace[] = [
    {
      type: "candlestick",
      x,
      open,
      high,
      low,
      close,
      name: ticker,
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
      x,
      y: vol,
      name: "Volume",
      marker: {
        color: chart.map((r) =>
          r.close >= r.open
            ? "rgba(56,211,159,0.5)"
            : "rgba(255,107,107,0.5)",
        ),
      },
      xaxis: "x",
      yaxis: "y2",
    },
  ];

  if (shouldAppendLive && liveDate !== null) {
    const lc = ghostLiveCandle;
    if (lc) {
      traces.push({
        type: "candlestick",
        x: [liveDate],
        open: [lc.open],
        high: [lc.high],
        low: [lc.low],
        close: [lc.close],
        name: "Live",
        xaxis: "x",
        yaxis: "y",
        increasing: {
          line: { color: "rgba(56,211,159,0.4)" },
          fillcolor: "rgba(56,211,159,0.25)",
        },
        decreasing: {
          line: { color: "rgba(255,107,107,0.4)" },
          fillcolor: "rgba(255,107,107,0.25)",
        },
      });
      traces.push({
        type: "bar",
        x: [liveDate],
        y: [typeof lc.volume === "number" ? lc.volume : 0],
        name: "Live Vol",
        marker: {
          color:
            lc.close >= lc.open
              ? "rgba(56,211,159,0.25)"
              : "rgba(255,107,107,0.25)",
        },
        xaxis: "x",
        yaxis: "y2",
      });
    }
  }

  if (overlays.ma5) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: ma(close, 5),
      name: "MA5",
      line: { color: "#f97316", width: 1.1 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.ma20) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: ma(close, 20),
      name: "MA20",
      line: { color: "#6aa9ff", width: 1.2 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.ma50) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: ma(close, 50),
      name: "MA50",
      line: { color: "#f8c24e", width: 1.2 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.ma200) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: ma(close, 200),
      name: "MA200",
      line: { color: "#c07bff", width: 1.2 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.ema20) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: ema(close, 20),
      name: "EMA20",
      line: { color: "#38bdf8", width: 1.2, dash: "dot" },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.ema50) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: ema(close, 50),
      name: "EMA50",
      line: { color: "#facc15", width: 1.2, dash: "dot" },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.ema200) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: ema(close, 200),
      name: "EMA200",
      line: { color: "#a78bfa", width: 1.2, dash: "dot" },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.bollinger) {
    const bb = computeRollingBollinger(close, 20, 2);
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: bb.upper,
      name: "BOLL Upper",
      line: { color: "#ffd21f", width: 1.8 },
      showlegend: false,
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: bb.middle,
      name: "BOLL Mid",
      line: { color: "#f6bed8", width: 1.6 },
      showlegend: false,
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: bb.lower,
      name: "BOLL Lower",
      line: { color: "#16c7ff", width: 1.8 },
      showlegend: false,
      xaxis: "x",
      yaxis: "y",
    });
  }

  if (candlePatterns.length > 0) {
    const bullish = candlePatterns.filter(
      (p) => (p.bias ?? "").toLowerCase() === "bullish",
    );
    const bearish = candlePatterns.filter(
      (p) => (p.bias ?? "").toLowerCase() === "bearish",
    );
    if (bullish.length > 0) {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: bullish.map((p) => p.date),
        y: bullish.map((p) => {
          const i = x.indexOf(p.date);
          return i >= 0 ? low[i] * 0.994 : null;
        }),
        name: "Bullish Signal",
        marker: { symbol: "triangle-up", size: 9, color: "#38d39f" },
        hovertext: bullish.map(
          (p) => `${p.pattern} (${formatChartNumber(p.confidence, 2)})`,
        ),
        hoverinfo: "text+x",
        xaxis: "x",
        yaxis: "y",
      });
    }
    if (bearish.length > 0) {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: bearish.map((p) => p.date),
        y: bearish.map((p) => {
          const i = x.indexOf(p.date);
          return i >= 0 ? high[i] * 1.006 : null;
        }),
        name: "Bearish Signal",
        marker: { symbol: "triangle-down", size: 9, color: "#ff6b6b" },
        hovertext: bearish.map(
          (p) => `${p.pattern} (${formatChartNumber(p.confidence, 2)})`,
        ),
        hoverinfo: "text+x",
        xaxis: "x",
        yaxis: "y",
      });
    }
  }

  const shapes: PlotlyShape[] = [];
  const annotations: PlotlyAnnotation[] = [];

  if (overlays.bollinger && !compactMode) {
    const bb = computeRollingBollinger(close, 20, 2);
    const bbUpper = lastFinite(bb.upper);
    const bbMiddle = lastFinite(bb.middle);
    const bbLower = lastFinite(bb.lower);
    if (bbUpper !== null && bbMiddle !== null && bbLower !== null) {
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
          text: `MID: ${formatChartNumber(bbMiddle, 2)}`,
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
          text: `UPPER: ${formatChartNumber(bbUpper, 2)}`,
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
          text: `LOWER: ${formatChartNumber(bbLower, 2)}`,
          showarrow: false,
          xanchor: "left",
          yanchor: "top",
          font: { color: "#16c7ff", size: 12 },
        },
      );
    }
  }

  levels.forEach((r) => {
    const lvl = Number(r.level);
    if (Number.isNaN(lvl)) return;
    const color = r.type === "support" ? "#3f8cff" : "#ff7b5b";
    const tier = (r.tier ?? "primary").toLowerCase();
    const dash =
      tier === "primary" ? "solid" : tier === "secondary" ? "dot" : "dash";
    const width = tier === "primary" ? 2 : 1;

    const z0 = Number(r.zone_low);
    const z1 = Number(r.zone_high);
    if (!Number.isNaN(z0) && !Number.isNaN(z1)) {
      shapes.push({
        type: "rect",
        xref: "paper",
        yref: "y",
        x0: 0,
        x1: 1,
        y0: Math.min(z0, z1),
        y1: Math.max(z0, z1),
        fillcolor:
          r.type === "support"
            ? "rgba(63,140,255,0.10)"
            : "rgba(255,123,91,0.10)",
        line: { width: 0 },
      });
    }
    shapes.push({
      type: "line",
      xref: "paper",
      yref: "y",
      x0: 0,
      x1: 1,
      y0: lvl,
      y1: lvl,
      line: { color, width, dash },
    });
    const tierLabel = tier.toUpperCase();
    const shortTierLabel =
      tierLabel === "PRIMARY"
        ? "P"
        : tierLabel === "SECONDARY"
          ? "S"
          : tierLabel === "FALLBACK"
            ? "FB"
            : tierLabel;
    const typeLabel = r.type === "support" ? "SUP" : "RES";
    if (compactMode) {
      annotations.push({
        xref: "paper",
        yref: "y",
        x: 1.0,
        y: lvl,
        text: `${shortTierLabel} ${typeLabel} ${formatChartNumber(lvl)}`,
        showarrow: false,
        xanchor: "left",
        yanchor: "middle",
        align: "left",
        xshift: 2,
        borderpad: 1,
        bgcolor: "rgba(18,25,39,0.9)",
        bordercolor: color,
        font: { color, size: 9 },
      });
    } else {
      annotations.push({
        xref: "paper",
        yref: "y",
        x: 1.0,
        y: lvl,
        text: `${tierLabel} ${typeLabel} ${formatChartNumber(lvl)} (${r.touches ?? "-"})`,
        showarrow: false,
        xanchor: "left",
        yanchor: "middle",
        align: "left",
        xshift: 4,
        borderpad: 2,
        bgcolor: "rgba(18,25,39,0.9)",
        bordercolor: color,
        font: { color, size: 11 },
      });
    }
  });

  earningsMarkers.forEach((marker, idx) => {
    const markerDate = (marker.date ?? "").trim();
    if (!markerDate) return;
    const session = (marker.session ?? "TBD").toUpperCase();
    const color =
      session === "BMO" ? "#38bdf8" : session === "AMC" ? "#a855f7" : "#94a3b8";
    shapes.push({
      type: "line",
      xref: "x",
      yref: "paper",
      x0: markerDate,
      x1: markerDate,
      y0: 0,
      y1: 1,
      line: { color, width: 1.6, dash: "dot" },
    });
    if (!compactMode) {
      annotations.push({
        xref: "x",
        yref: "paper",
        x: markerDate,
        y: 1,
        yshift: -18 - idx * 18,
        text: `E ${session}`,
        showarrow: false,
        xanchor: "left",
        bgcolor: "rgba(18,25,39,0.92)",
        bordercolor: color,
        borderpad: 3,
        font: { color, size: 11 },
      });
    }
  });

  gaps.forEach((g) => {
    const y0 = Number(g.gap_low);
    const y1 = Number(g.gap_high);
    if (Number.isNaN(y0) || Number.isNaN(y1)) return;
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "y",
      x0: g.date,
      x1: x[x.length - 1],
      y0,
      y1,
      fillcolor:
        g.type === "bull_gap" ? "rgba(248,194,78,0.22)" : "rgba(106,169,255,0.20)",
      line: { width: 1, color: "rgba(142,160,189,0.6)" },
    });
  });

  const yoloColorMap: Record<string, { stroke: string; fill: string }> = {
    bull: { stroke: "rgba(56,211,159,0.5)", fill: "rgba(56,211,159,0.03)" },
    bear: { stroke: "rgba(255,107,107,0.5)", fill: "rgba(255,107,107,0.03)" },
    neutral: { stroke: "rgba(248,194,78,0.5)", fill: "rgba(248,194,78,0.03)" },
  };

  const yoloStyle = (name: string) => {
    const n = name.toLowerCase();
    if (
      n.includes("bottom") ||
      n.includes("w_bottom") ||
      n.includes("shoulders bottom")
    ) {
      return yoloColorMap.bull;
    }
    if (
      n.includes("top") ||
      n.includes("m_head") ||
      n.includes("shoulders top")
    ) {
      return yoloColorMap.bear;
    }
    return yoloColorMap.neutral;
  };

  const dailyYolo = yoloPatterns.filter(
    (p) => String(p.timeframe ?? "daily") === "daily",
  );
  const weeklyYolo = yoloPatterns.filter(
    (p) => String(p.timeframe ?? "") === "weekly",
  );

  const renderYoloGroup = (group: Array<YoloPatternRow>, isWeeklyGroup: boolean) => {
    let labelCount = 0;
    group.forEach((p) => {
      const y0 = Number(p.y0);
      const y1 = Number(p.y1);
      if (Number.isNaN(y0) || Number.isNaN(y1)) return;
      const patternName = String(p.pattern ?? "");
      const baseStyle = yoloStyle(patternName);
      const age = Number(p.age_days);
      const staleCutoff = isWeeklyGroup ? 120 : 45;
      const isStale = Number.isFinite(age) && age > staleCutoff;
      const borderWidth = isStale ? 1.0 : isWeeklyGroup ? 2.2 : 1.5;
      const borderDash = isStale ? "dot" : isWeeklyGroup ? "dash" : "dot";
      const fillOpacity = isStale ? "0.02" : isWeeklyGroup ? "0.05" : "0.07";
      const stroke = isStale
        ? `${baseStyle.stroke}99`
        : isWeeklyGroup
          ? `${baseStyle.stroke}cc`
          : baseStyle.stroke;
      const fill = baseStyle.fill.replace(/[\d.]+\)$/, `${fillOpacity})`);

      shapes.push({
        type: "rect",
        xref: "x",
        yref: "y",
        x0: String(p.x0_date ?? ""),
        x1: String(p.x1_date ?? ""),
        y0: Math.min(y0, y1),
        y1: Math.max(y0, y1),
        line: { color: stroke, width: borderWidth, dash: borderDash },
        fillcolor: fill,
      });

      if (!compactMode && labelCount < 5) {
        const conf = Number(p.confidence);
        const confPct = Number.isFinite(conf) ? `${(conf * 100).toFixed(0)}%` : "";
        const streak = Number(p.current_streak);
        const streakText =
          Number.isFinite(streak) && streak > 1 ? ` ${Math.round(streak)}x` : "";
        const ageText = Number.isFinite(age)
          ? isStale
            ? ` old ${Math.round(age)}d`
            : age > 0
              ? ` ${Math.round(age)}d`
              : " fresh"
          : "";
        const label = isWeeklyGroup ? "W" : "D";
        const humanName = patternName.replace(/_/g, " ");
        const text = `[${label}] ${humanName} ${confPct}${streakText}${ageText}`;

        annotations.push({
          xref: "x",
          yref: "y",
          x: String(p.x1_date ?? ""),
          y: Math.max(y0, y1),
          text,
          showarrow: false,
          xanchor: "left",
          yanchor: "bottom",
          xshift: 4,
          bgcolor: "rgba(18,25,39,0.85)",
          bordercolor: baseStyle.stroke,
          font: { color: baseStyle.stroke, size: isWeeklyGroup ? 9 : 10 },
        });
        labelCount += 1;
      }
    });
  };

  if (isWeekly) {
    renderYoloGroup(weeklyYolo, true);
  } else {
    renderYoloGroup(dailyYolo, false);
    renderYoloGroup(weeklyYolo, true);
  }

  const hasHmm = hmmRegime !== null && hmmRegime.regimes.length > 0;
  const showHmm = hasHmm && !compactMode;

  if (showHmm) {
    const regimes = hmmRegime.regimes;
    // HMM regime is shown via the stacked area chart in the y3 panel
    // and the badge in the commentary sidebar — no background rectangles
    // needed in the price chart (they add too much visual noise).

    const regimeDates = regimes.map((r) => r.date);
    const probLow = regimes.map((r) => r.prob_low);
    const probNormal = regimes.map((r) => r.prob_normal);
    const probHigh = regimes.map((r) => r.prob_high);

    traces.push({
      type: "scatter",
      mode: "lines",
      x: regimeDates,
      y: probLow,
      name: "P(Low Vol)",
      showlegend: false,
      line: { color: "#38d39f", width: 0.5 },
      fill: "tozeroy",
      fillcolor: "rgba(56,211,159,0.25)",
      xaxis: "x",
      yaxis: "y3",
      hoverinfo: "text+x",
      hovertext: probLow.map((p) => `Low Vol: ${(p * 100).toFixed(1)}%`),
    });

    const probLowPlusNormal = probLow.map((p, i) => p + probNormal[i]);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: regimeDates,
      y: probLowPlusNormal,
      name: "P(Normal)",
      showlegend: false,
      line: { color: "#f8c24e", width: 0.5 },
      fill: "tonexty",
      fillcolor: "rgba(248,194,78,0.25)",
      xaxis: "x",
      yaxis: "y3",
      hoverinfo: "text+x",
      hovertext: probNormal.map((p) => `Normal: ${(p * 100).toFixed(1)}%`),
    });

    traces.push({
      type: "scatter",
      mode: "lines",
      x: regimeDates,
      y: probLowPlusNormal.map((p, i) => p + probHigh[i]),
      name: "P(High Vol)",
      showlegend: false,
      line: { color: "#ff6b6b", width: 0.5 },
      fill: "tonexty",
      fillcolor: "rgba(255,107,107,0.25)",
      xaxis: "x",
      yaxis: "y3",
      hoverinfo: "text+x",
      hovertext: probHigh.map((p) => `High Vol: ${(p * 100).toFixed(1)}%`),
    });
  }

  if (hasLiveCandle && liveBadgeX !== null && typeof liveBadgeY === "number") {
    annotations.push({
      xref: "x",
      yref: "y",
      x: liveBadgeX,
      y: liveBadgeY,
      text: "LIVE",
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

  const priceDomain: [number, number] = showHmm ? [0.38, 1] : [0.28, 1];
  const volumeDomain: [number, number] = showHmm ? [0.20, 0.33] : [0, 0.22];
  const regimeDomain: [number, number] = [0, 0.15];
  const chartHeight = compactMode ? 460 : showHmm ? 700 : 580;
  const missingTradingDays = tradingDayRangeBreaks(x);
  const rangeBreaks = isWeekly
    ? []
    : [
        { bounds: ["sat", "mon"] },
        ...(missingTradingDays.length > 0 ? [{ values: missingTradingDays }] : []),
      ];

  const initialRange = defaultThreeMonthRange(x);
  const visibleRows = chart.filter((row) => {
    if (!initialRange) return true;
    const rowTime = new Date(row.date).getTime();
    const start = new Date(initialRange[0]).getTime();
    const end = new Date(initialRange[1]).getTime();
    return Number.isFinite(rowTime) && rowTime >= start && rowTime <= end;
  });
  const rangeRows = visibleRows.length > 0 ? visibleRows : chart;
  const minVisible = Math.min(...rangeRows.map((row) => row.low));
  const maxVisible = Math.max(...rangeRows.map((row) => row.high));
  const pricePadding =
    Number.isFinite(minVisible) && Number.isFinite(maxVisible)
      ? Math.max((maxVisible - minVisible) * 0.08, maxVisible * 0.015)
      : 0;

  const layout: Record<string, unknown> = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    uirevision: `${ticker}-${isWeekly ? "weekly" : "daily"}`,
    font: { color: "#8ea0bd", size: 11 },
    margin: compactMode ? { t: 18, r: 100, b: 44, l: 52 } : { t: 40, r: 200, b: 50, l: 60 },
    dragmode: "zoom" as const,
    legend: {
      orientation: "h" as const,
      y: 1.0,
      x: 0.5,
      xanchor: "center" as const,
      yanchor: "bottom" as const,
      bgcolor: "rgba(11,15,22,0.7)",
      bordercolor: "rgba(255,255,255,0.08)",
      borderwidth: 1,
      font: { size: 10 },
      traceorder: "normal" as const,
    },
    xaxis: {
      gridcolor: "rgba(255,255,255,0.04)",
      rangeslider: { visible: false },
      range: initialRange,
      rangebreaks: rangeBreaks,
      rangeselector: {
        bgcolor: "#e9eef8",
        activecolor: "#6aa9ff",
        bordercolor: "#0b0f16",
        borderwidth: 1,
        font: { color: "#0b0f16", size: compactMode ? 11 : 12 },
        buttons: compactMode
          ? [
              { count: 3, step: "month", stepmode: "backward", label: "3M" },
              { count: 1, step: "year", stepmode: "backward", label: "1Y" },
              { step: "all", label: "ALL" },
            ]
          : [
              { count: 3, step: "month", stepmode: "backward", label: "3M" },
              { count: 6, step: "month", stepmode: "backward", label: "6M" },
              { count: 1, step: "year", stepmode: "todate", label: "YTD" },
              { count: 1, step: "year", stepmode: "backward", label: "1Y" },
              { count: 2, step: "year", stepmode: "backward", label: "2Y" },
              { step: "all", label: "ALL" },
            ],
      },
    },
    yaxis: {
      gridcolor: "rgba(255,255,255,0.06)",
      domain: priceDomain,
      title: compactMode ? undefined : "Price",
      range:
        Number.isFinite(minVisible) && Number.isFinite(maxVisible)
          ? [minVisible - pricePadding, maxVisible + pricePadding]
          : undefined,
    },
    yaxis2: {
      gridcolor: "rgba(255,255,255,0.04)",
      domain: volumeDomain,
      title: compactMode ? undefined : "Volume",
    },
    shapes,
    annotations,
    height: chartHeight,
  };

  if (showHmm) {
    (layout as Record<string, unknown>).yaxis3 = {
      gridcolor: "rgba(255,255,255,0.04)",
      domain: regimeDomain,
      title: "Regime",
      range: [0, 1.05],
      tickvals: [0, 0.5, 1],
      ticktext: ["0%", "50%", "100%"],
    };
  }

  return { traces, layout };
}

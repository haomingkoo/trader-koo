import { useState, useCallback, useEffect, useMemo } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { useChart } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import { useLiveEquityPrice } from "../hooks/useLiveEquityPrice";
import type {
  DashboardPayload,
  LiveCandle,
  LevelRow,
  GapRow,
  OhlcvRow,
  PatternRow,
  HybridPatternRow,
  CandlestickPatternRow,
  YoloAuditRow,
  YoloPatternRow,
  ChartCommentary,
  HmmRegime,
  EquityTick,
} from "../api/types";
import Card from "../components/ui/Card";
import PlotlyWrapper from "../components/PlotlyWrapper";
import Spinner from "../components/ui/Spinner";
import Badge, { tierVariant } from "../components/ui/Badge";
import Table from "../components/ui/Table";

/* ── Helpers ── */

function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null || !Number.isFinite(n)) return "\u2014";
  return n.toFixed(decimals);
}

function biasVariant(
  bias: string | null,
): "green" | "red" | "amber" | "muted" {
  if (!bias) return "muted";
  const b = bias.toLowerCase();
  if (b.includes("bull") || b === "long") return "green";
  if (b.includes("bear") || b === "short") return "red";
  if (b.includes("neutral") || b === "flat") return "amber";
  return "muted";
}

function ma(arr: number[], n: number): (number | null)[] {
  return arr.map((_, i) => {
    if (i < n - 1) return null;
    let s = 0;
    for (let j = i - n + 1; j <= i; j++) s += arr[j];
    return s / n;
  });
}

function resampleToWeekly(
  rows: OhlcvRow[],
): OhlcvRow[] {
  if (rows.length === 0) return [];
  const weeks: OhlcvRow[] = [];
  let current: OhlcvRow | null = null;

  for (const row of rows) {
    const d = new Date(row.date);
    const dayOfWeek = d.getDay();
    // Start new week on Monday (or first day of data)
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

/* ── Glassmorphism card wrapper ── */

function GlassCard({
  label,
  value,
  children,
  className = "",
}: {
  label?: string;
  value?: string | number;
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
      {value !== undefined && (
        <div className="text-lg font-bold tabular-nums text-[var(--text)]">
          {typeof value === "string" || typeof value === "number" ? (value ?? "\u2014") : String(value ?? "\u2014")}
        </div>
      )}
      {children}
    </div>
  );
}

/* ── Chart building ── */

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

const CHART_OVERLAY_OPTIONS = [
  { key: "ma20", label: "20 MA", minBars: 20 },
  { key: "ma50", label: "50 MA", minBars: 50 },
  { key: "ma200", label: "200 MA", minBars: 200 },
  { key: "bollinger", label: "Bollinger", minBars: 20 },
] as const;

type ChartOverlayKey = (typeof CHART_OVERLAY_OPTIONS)[number]["key"];
type ChartOverlayState = Record<ChartOverlayKey, boolean>;

const DEFAULT_CHART_OVERLAYS: ChartOverlayState = {
  ma20: true,
  ma50: true,
  ma200: false,
  bollinger: false,
};

function computeRollingBollinger(
  values: number[],
  period: number,
  stdDev: number,
): { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] } {
  return values.map((_, idx) => {
    if (idx < period - 1) {
      return { upper: null, middle: null, lower: null };
    }
    const window = values.slice(idx - period + 1, idx + 1);
    const mean = window.reduce((a, b) => a + b, 0) / period;
    const variance =
      window.reduce((a, b) => a + (b - mean) ** 2, 0) / period;
    const sd = Math.sqrt(variance);
    return {
      upper: mean + stdDev * sd,
      middle: mean,
      lower: mean - stdDev * sd,
    };
  }).reduce(
    (acc, row) => {
      acc.upper.push(row.upper);
      acc.middle.push(row.middle);
      acc.lower.push(row.lower);
      return acc;
    },
    { upper: [] as (number | null)[], middle: [] as (number | null)[], lower: [] as (number | null)[] },
  );
}

function defaultThreeMonthRange(dates: string[]): [string, string] | undefined {
  if (dates.length === 0) return undefined;
  const end = new Date(dates[dates.length - 1]);
  if (Number.isNaN(end.getTime())) return undefined;
  const start = new Date(end);
  start.setMonth(start.getMonth() - 3);
  const earliest = new Date(dates[0]);
  if (!Number.isNaN(earliest.getTime()) && start < earliest) {
    return [dates[0], dates[dates.length - 1]];
  }
  return [start.toISOString(), end.toISOString()];
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

function applyLivePriceToPayload(
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

function buildChartData(
  payload: DashboardPayload,
  isWeekly: boolean,
  overlays: ChartOverlayState,
  liveCandle?: LiveCandle | null,
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
        volume: lastRow.volume + (Number.isFinite(liveCandle.volume) ? liveCandle.volume : 0),
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
  const shouldAppendLive =
    ghostLiveCandle !== null && liveDate !== null && liveDate !== lastX;
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

  // Render the live candle as a separate semi-transparent candlestick trace
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
  if (overlays.bollinger) {
    const bb = computeRollingBollinger(close, 20, 2);
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: bb.upper,
      name: "BB Upper",
      line: { color: "rgba(186,130,255,0.6)", width: 1, dash: "dash" },
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: bb.middle,
      name: "BB Mid",
      line: { color: "rgba(186,130,255,0.85)", width: 1 },
      xaxis: "x",
      yaxis: "y",
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: bb.lower,
      name: "BB Lower",
      line: { color: "rgba(186,130,255,0.6)", width: 1, dash: "dash" },
      xaxis: "x",
      yaxis: "y",
    });
  }

  // Candlestick pattern markers
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
          (p) => `${p.pattern} (${fmt(p.confidence, 2)})`,
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
          (p) => `${p.pattern} (${fmt(p.confidence, 2)})`,
        ),
        hoverinfo: "text+x",
        xaxis: "x",
        yaxis: "y",
      });
    }
  }

  const shapes: PlotlyShape[] = [];
  const annotations: PlotlyAnnotation[] = [];

  // Support / Resistance levels
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
    annotations.push({
      xref: "paper",
      yref: "y",
      x: 1.0,
      y: lvl,
      text: `${tierLabel} ${String(r.type).toUpperCase()} ${fmt(lvl)} (${r.touches ?? "-"})`,
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
  });

  // Earnings markers
  earningsMarkers.forEach((marker, idx) => {
    const markerDate = (marker.date ?? "").trim();
    if (!markerDate) return;
    const session = (marker.session ?? "TBD").toUpperCase();
    const color =
      session === "BMO"
        ? "#38bdf8"
        : session === "AMC"
          ? "#a855f7"
          : "#94a3b8";
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
  });

  // Gap zones
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
        g.type === "bull_gap"
          ? "rgba(248,194,78,0.22)"
          : "rgba(106,169,255,0.20)",
      line: { width: 1, color: "rgba(142,160,189,0.6)" },
    });
  });

  // YOLO pattern bounding boxes
  const yoloColorMap: Record<string, { stroke: string; fill: string }> = {
    bull: { stroke: "#38d39f", fill: "rgba(56,211,159,0.07)" },
    bear: { stroke: "#ff6b6b", fill: "rgba(255,107,107,0.07)" },
    neutral: { stroke: "#f8c24e", fill: "rgba(248,194,78,0.07)" },
  };

  const yoloStyle = (name: string) => {
    const n = name.toLowerCase();
    if (n.includes("bottom") || n.includes("w_bottom") || n.includes("shoulders bottom"))
      return yoloColorMap.bull;
    if (n.includes("top") || n.includes("m_head") || n.includes("shoulders top"))
      return yoloColorMap.bear;
    return yoloColorMap.neutral;
  };

  const dailyYolo = yoloPatterns.filter(
    (p) => (String(p.timeframe ?? "daily")) === "daily",
  );
  const weeklyYolo = yoloPatterns.filter(
    (p) => String(p.timeframe ?? "") === "weekly",
  );

  const renderYoloGroup = (
    group: Array<YoloPatternRow>,
    isWeeklyGroup: boolean,
  ) => {
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
        ? baseStyle.stroke + "99"
        : isWeeklyGroup
          ? baseStyle.stroke + "cc"
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

      if (labelCount < 5) {
        const conf = Number(p.confidence);
        const confPct = Number.isFinite(conf)
          ? `${(conf * 100).toFixed(0)}%`
          : "";
        const streak = Number(p.current_streak);
        const streakText =
          Number.isFinite(streak) && streak > 1
            ? ` ${Math.round(streak)}x`
            : "";
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
        labelCount++;
      }
    });
  };

  if (isWeekly) {
    renderYoloGroup(weeklyYolo, true);
  } else {
    renderYoloGroup(dailyYolo, false);
    renderYoloGroup(weeklyYolo, true);
  }

  // HMM regime background shading + probability sub-pane
  const hasHmm = hmmRegime !== null && hmmRegime.regimes.length > 0;

  if (hasHmm) {
    const regimes = hmmRegime.regimes;

    // Background shading: group consecutive same-label days into spans
    let spanStart = 0;
    for (let i = 1; i <= regimes.length; i++) {
      if (i === regimes.length || regimes[i].label !== regimes[spanStart].label) {
        const startDate = regimes[spanStart].date;
        const endDate = regimes[i - 1].date;
        const color = regimes[spanStart].color;
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: startDate,
          x1: endDate,
          y0: 0,
          y1: 1,
          fillcolor: color.replace(")", ",0.08)").replace("rgb", "rgba").startsWith("rgba")
            ? color.replace(")", ",0.08)").replace("rgb", "rgba")
            : `${color}14`,  // 14 hex = ~8% opacity
          line: { width: 0 },
        });
        spanStart = i;
      }
    }

    // Regime probability stacked area traces
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
      line: { color: "#38d39f", width: 0.5 },
      fill: "tozeroy",
      fillcolor: "rgba(56,211,159,0.3)",
      xaxis: "x",
      yaxis: "y3",
      hoverinfo: "text+x",
      hovertext: probLow.map((p) => `Low Vol: ${(p * 100).toFixed(1)}%`),
    });

    // Stack normal on top of low
    const probLowPlusNormal = probLow.map((p, i) => p + probNormal[i]);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: regimeDates,
      y: probLowPlusNormal,
      name: "P(Normal)",
      line: { color: "#f8c24e", width: 0.5 },
      fill: "tonexty",
      fillcolor: "rgba(248,194,78,0.3)",
      xaxis: "x",
      yaxis: "y3",
      hoverinfo: "text+x",
      hovertext: probNormal.map((p) => `Normal: ${(p * 100).toFixed(1)}%`),
    });

    // Stack high on top of normal
    traces.push({
      type: "scatter",
      mode: "lines",
      x: regimeDates,
      y: probLowPlusNormal.map((p, i) => p + probHigh[i]),
      name: "P(High Vol)",
      line: { color: "#ff6b6b", width: 0.5 },
      fill: "tonexty",
      fillcolor: "rgba(255,107,107,0.3)",
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

  // Layout domains shift when HMM pane is present
  const priceDomain: [number, number] = hasHmm ? [0.36, 1] : [0.28, 1];
  const volumeDomain: [number, number] = hasHmm ? [0.18, 0.30] : [0, 0.22];
  const regimeDomain: [number, number] = [0, 0.14];
  const chartHeight = hasHmm ? 660 : 580;

  const layout: Record<string, unknown> = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#8ea0bd", size: 11 },
    margin: { t: 40, r: 200, b: 50, l: 60 },
    dragmode: "zoom" as const,
    legend: { orientation: "h" as const, y: -0.04, x: 0, xanchor: "left" as const },
    xaxis: {
      gridcolor: "rgba(255,255,255,0.04)",
      rangeslider: { visible: false },
      range: defaultThreeMonthRange(x),
      rangebreaks: isWeekly
        ? [{ bounds: ["sat", "mon"] }]
        : [{ bounds: ["sat", "mon"] }],
      rangeselector: {
        bgcolor: "#e9eef8",
        activecolor: "#6aa9ff",
        bordercolor: "#0b0f16",
        borderwidth: 1,
        font: { color: "#0b0f16", size: 12 },
        buttons: [
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
      title: "Price",
    },
    yaxis2: {
      gridcolor: "rgba(255,255,255,0.04)",
      domain: volumeDomain,
      title: "Volume",
    },
    shapes,
    annotations,
    height: chartHeight,
  };

  if (hasHmm) {
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

/* ── Commentary sidebar ── */

function ChartCommentarySidebar({
  commentary,
  hmmRegime,
}: {
  commentary: ChartCommentary | null;
  hmmRegime: HmmRegime | null;
}) {
  if (!commentary) {
    return (
      <GlassCard label="Chart Commentary">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No chart commentary available. Load a ticker to generate commentary.
        </p>
      </GlassCard>
    );
  }

  const debate = commentary.debate_v1;
  const debateState =
    commentary.debate_consensus_state ??
    debate?.consensus?.consensus_state ??
    null;
  const agreementScore =
    commentary.debate_agreement_score ??
    debate?.consensus?.agreement_score ??
    null;

  const regimeLabel = hmmRegime?.current_state ?? null;
  const regimeProbs = hmmRegime?.current_probs ?? null;
  const regimeDays = hmmRegime?.days_in_current ?? null;
  const regimeConf = regimeProbs && regimeLabel ? regimeProbs[regimeLabel] : null;
  const regimeVariant: "green" | "amber" | "red" | "muted" =
    regimeLabel === "low_vol"
      ? "green"
      : regimeLabel === "normal"
        ? "amber"
        : regimeLabel === "high_vol"
          ? "red"
          : "muted";
  const regimeDisplay: Record<string, string> = {
    low_vol: "LOW VOL",
    normal: "NORMAL",
    high_vol: "HIGH VOL",
  };

  return (
    <GlassCard>
      {/* Badges row */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {commentary.setup_tier && (
          <Badge variant={tierVariant(commentary.setup_tier)}>
            Tier {commentary.setup_tier}
          </Badge>
        )}
        {commentary.signal_bias && (
          <Badge variant={biasVariant(commentary.signal_bias)}>
            {commentary.signal_bias.toUpperCase()}
          </Badge>
        )}
        {commentary.actionability && (
          <Badge variant="default">
            {commentary.actionability.toUpperCase()}
          </Badge>
        )}
        {regimeLabel ? (
          <Badge variant={regimeVariant}>
            HMM {regimeDisplay[regimeLabel] ?? regimeLabel.toUpperCase()}
            {regimeConf != null && ` ${(regimeConf * 100).toFixed(0)}%`}
            {regimeDays != null && ` (${regimeDays}d)`}
          </Badge>
        ) : (
          <Badge variant="muted">REGIME N/A</Badge>
        )}
        {debateState && (
          <Badge
            variant={
              debateState === "ready"
                ? "green"
                : debateState === "conditional"
                  ? "amber"
                  : "red"
            }
          >
            DEBATE {debateState.toUpperCase()}
            {agreementScore != null && ` ${agreementScore.toFixed(0)}%`}
          </Badge>
        )}
        {commentary.yolo_direction_conflict && (
          <Badge variant="red">YOLO CONFLICT</Badge>
        )}
      </div>

      <div className="space-y-2 text-xs">
        {commentary.observation ? (
          <p className="text-[var(--text)]">{String(commentary.observation)}</p>
        ) : (
          <p className="text-[var(--muted)]">No observation available.</p>
        )}

        {commentary.action && (
          <p className="text-[var(--muted)]">
            <strong className="text-[var(--text)]">Action:</strong>{" "}
            {String(commentary.action)}
          </p>
        )}

        {commentary.risk_note && (
          <p className="text-[var(--muted)]">
            <strong className="text-[var(--text)]">Risk:</strong>{" "}
            {String(commentary.risk_note)}
          </p>
        )}

        {commentary.technical_read && (
          <p className="text-[var(--muted)]">
            <strong className="text-[var(--text)]">Technical:</strong>{" "}
            {String(commentary.technical_read)}
          </p>
        )}
      </div>

      {/* Debate roles */}
      {debate && debate.roles && debate.roles.length > 0 && (
        <DebateRolesInline debate={debate} />
      )}

      {commentary.asof && (
        <p className="mt-2 text-[10px] text-[var(--muted)]">
          As of {String(commentary.asof)} | Source: {String(commentary.narrative_source ?? "rule")}
        </p>
      )}
    </GlassCard>
  );
}

function DebateRolesInline({
  debate,
}: {
  debate: NonNullable<ChartCommentary["debate_v1"]>;
}) {
  const [expanded, setExpanded] = useState(false);
  const consensus = debate.consensus;
  const roles = debate.roles ?? [];

  return (
    <div className="mt-3 border-t border-[var(--line)] pt-3">
      <button
        onClick={() => setExpanded((p) => !p)}
        className="text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
      >
        {expanded ? "Hide" : "Show"} debate ({roles.length} roles)
      </button>
      {expanded && (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--muted)]">
            <span>
              Consensus:{" "}
              <strong className="text-[var(--text)]">
                {String(consensus.consensus_bias ?? "\u2014")}
              </strong>
            </span>
            <span>
              State:{" "}
              <strong className="text-[var(--text)]">
                {String(consensus.consensus_state ?? "\u2014")}
              </strong>
            </span>
            <span>
              Agreement:{" "}
              <strong className="text-[var(--text)]">
                {typeof consensus.agreement_score === "number" ? consensus.agreement_score.toFixed(0) : "\u2014"}%
              </strong>
            </span>
            <span>
              Disagreements:{" "}
              <strong className="text-[var(--text)]">
                {String(consensus.disagreement_count ?? "\u2014")}
              </strong>
            </span>
          </div>
          {roles.map((role, i) => {
            const isBull =
              role.stance.toLowerCase().includes("bull") ||
              role.stance.toLowerCase() === "long";
            const isBear =
              role.stance.toLowerCase().includes("bear") ||
              role.stance.toLowerCase() === "short";
            const barColor = isBull
              ? "var(--green)"
              : isBear
                ? "var(--red)"
                : "var(--amber)";
            const confPct = Math.min(100, Math.max(0, role.confidence * 100));
            return (
              <div key={i} className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="w-24 text-[10px] font-medium capitalize text-[var(--text)]">
                    {role.role.replace(/_/g, " ")}
                  </span>
                  <Badge
                    variant={isBull ? "green" : isBear ? "red" : "amber"}
                  >
                    {role.stance.toUpperCase()}
                  </Badge>
                  <div className="relative h-1.5 flex-1 rounded-full bg-[var(--line)]">
                    <div
                      className="absolute left-0 top-0 h-full rounded-full"
                      style={{
                        width: `${confPct}%`,
                        backgroundColor: barColor,
                      }}
                    />
                  </div>
                  <span className="w-8 text-right text-[10px] tabular-nums text-[var(--muted)]">
                    {confPct.toFixed(0)}%
                  </span>
                </div>
                {role.evidence.filter(Boolean).length > 0 && (
                  <ul className="ml-28 space-y-0">
                    {role.evidence.filter(Boolean).map((ev, j) => (
                      <li
                        key={j}
                        className="text-[10px] text-[var(--muted)] before:mr-1 before:content-['\u2022']"
                      >
                        {String(ev)}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ── Pattern tables ── */

function PatternTabs({ payload }: { payload: DashboardPayload }) {
  const [activeTab, setActiveTab] = useState<
    "rule" | "hybrid" | "candlestick"
  >("rule");

  const rulePatterns = payload.patterns ?? [];
  const hybridPatterns = payload.hybrid_patterns ?? [];
  const candlestickPatterns = payload.candlestick_patterns ?? [];

  const tabs = [
    { key: "rule" as const, label: `Rule (${rulePatterns.length})` },
    { key: "hybrid" as const, label: `Hybrid (${hybridPatterns.length})` },
    {
      key: "candlestick" as const,
      label: `Candlestick (${candlestickPatterns.length})`,
    },
  ];

  const ruleColumns: Array<{
    key: keyof PatternRow & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "pattern", label: "Pattern" },
    { key: "status", label: "Status" },
    {
      key: "confidence",
      label: "Confidence",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    { key: "start_date", label: "Start" },
    { key: "end_date", label: "End" },
  ];

  const hybridColumns: Array<{
    key: keyof HybridPatternRow & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "pattern", label: "Pattern" },
    { key: "status", label: "Status" },
    {
      key: "hybrid_confidence",
      label: "Hybrid Conf",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    {
      key: "base_confidence",
      label: "Base Conf",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    { key: "candle_bias", label: "Candle Bias" },
    {
      key: "vol_ratio",
      label: "Vol Ratio",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    { key: "start_date", label: "Start" },
    { key: "end_date", label: "End" },
  ];

  const candleColumns: Array<{
    key: keyof CandlestickPatternRow & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "date", label: "Date" },
    { key: "pattern", label: "Pattern" },
    { key: "bias", label: "Bias" },
    {
      key: "confidence",
      label: "Confidence",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    { key: "explanation", label: "Explanation" },
  ];

  return (
    <div>
      <div className="mb-2 flex gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
              activeTab === tab.key
                ? "bg-[var(--accent)] text-white"
                : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "rule" && (
        rulePatterns.length > 0 ? (
          <Table
            columns={ruleColumns}
            data={rulePatterns as unknown as Record<string, unknown>[]}
            sortable
          />
        ) : (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
            No rule patterns detected.
          </div>
        )
      )}
      {activeTab === "hybrid" && (
        hybridPatterns.length > 0 ? (
          <Table
            columns={hybridColumns}
            data={hybridPatterns as unknown as Record<string, unknown>[]}
            sortable
          />
        ) : (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
            No hybrid patterns detected.
          </div>
        )
      )}
      {activeTab === "candlestick" && (
        candlestickPatterns.length > 0 ? (
          <Table
            columns={candleColumns}
            data={candlestickPatterns as unknown as Record<string, unknown>[]}
            sortable
          />
        ) : (
          <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
            No candlestick patterns detected.
          </div>
        )
      )}
    </div>
  );
}

/* ── YOLO Audit Table ── */

function YoloAuditSection({
  yoloAudit,
}: {
  yoloAudit: YoloAuditRow[];
}) {
  if (yoloAudit.length === 0) {
    return (
      <div>
        <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
          YOLO Audit
        </h3>
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
          No YOLO audit data available.
        </div>
      </div>
    );
  }

  const columns: Array<{
    key: keyof YoloAuditRow & string;
    label: string;
    render?: (v: unknown, row: unknown) => React.ReactNode;
  }> = [
    { key: "timeframe", label: "TF" },
    { key: "pattern", label: "Pattern" },
    { key: "signal_role", label: "Role" },
    {
      key: "active_now",
      label: "Active",
      render: (v: unknown) => (v ? "Yes" : "No"),
    },
    { key: "yolo_recency", label: "Recency" },
    { key: "confirmation_trend", label: "Trend" },
    { key: "lifecycle_state", label: "Lifecycle" },
    {
      key: "age_days",
      label: "Age (d)",
      render: (v: unknown) => fmt(v as number | null, 0),
    },
    {
      key: "current_streak",
      label: "Streak",
      render: (v: unknown) => fmt(v as number | null, 0),
    },
    {
      key: "confidence",
      label: "Conf",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    { key: "first_seen_asof", label: "First Seen" },
    { key: "last_seen_asof", label: "Last Seen" },
  ];

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        YOLO Audit ({yoloAudit.length} entries)
      </h3>
      <Table
        columns={columns}
        data={yoloAudit as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}

/* ── Main Page ── */

export default function ChartPage() {
  const { ticker, timeframe, setTicker, setTimeframe } = useChartStore();
  const [searchParams] = useSearchParams();
  const [inputValue, setInputValue] = useState(ticker);
  const [commentaryExpanded, setCommentaryExpanded] = useState(true);
  const [chartOverlays, setChartOverlays] = useState<ChartOverlayState>(
    DEFAULT_CHART_OVERLAYS,
  );
  const { livePrice, streamingActive } = useLiveEquityPrice(ticker);

  // Pick up ticker from URL query param ?t=AAPL
  useEffect(() => {
    const urlTicker = searchParams.get("t");
    if (urlTicker) {
      const clean = urlTicker.trim().toUpperCase();
      if (clean && clean !== ticker) {
        setTicker(clean);
        setInputValue(clean);
      }
    }
  }, [searchParams, setTicker, ticker]);

  const { data, isLoading, error, refetch } = useChart(ticker);

  const handleLoad = useCallback(() => {
    const clean = inputValue.trim().toUpperCase();
    if (clean) {
      setTicker(clean);
    }
  }, [inputValue, setTicker]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") handleLoad();
    },
    [handleLoad],
  );

  const fundamentals = data?.fundamentals ?? {
    price: null,
    pe: null,
    peg: null,
    target_price: null,
    discount_pct: null,
  };
  const currentPrice = livePrice?.price ?? fundamentals.price;
  const options = data?.options_summary ?? { put_call_oi_ratio: null };
  const commentary = data?.chart_commentary ?? null;
  const isWeekly = timeframe === "weekly";
  const livePayload = useMemo(
    () => applyLivePriceToPayload(data, livePrice),
    [data, livePrice],
  );

  // Build effective live candle: start from API live_candle, overlay WS price if newer
  const effectiveLiveCandle = useMemo((): LiveCandle | null => {
    const apiCandle =
      typeof data?.live_candle === "object" && data.live_candle !== null
        ? data.live_candle
        : null;
    if (!apiCandle) return null;

    // If WS price exists and belongs to the same ticker, update close/high/low
    if (
      livePrice &&
      typeof livePrice.price === "number" &&
      Number.isFinite(livePrice.price) &&
      livePrice.symbol === data?.ticker
    ) {
      const wsTime = livePrice.timestamp ? new Date(livePrice.timestamp).getTime() : 0;
      const apiTime = apiCandle.timestamp ? new Date(apiCandle.timestamp).getTime() : 0;
      if (wsTime > apiTime) {
        return {
          ...apiCandle,
          close: livePrice.price,
          high: Math.max(apiCandle.high, livePrice.price),
          low: Math.min(apiCandle.low, livePrice.price),
        };
      }
    }
    return apiCandle;
  }, [data?.live_candle, data?.ticker, livePrice]);

  const chartResult = useMemo(() => {
    if (!livePayload || !livePayload.chart || livePayload.chart.length === 0) {
      return null;
    }
    return buildChartData(livePayload, isWeekly, chartOverlays, effectiveLiveCandle);
  }, [livePayload, isWeekly, chartOverlays, effectiveLiveCandle]);

  const chartBarCount = (
    isWeekly ? resampleToWeekly(livePayload?.chart ?? []) : livePayload?.chart ?? []
  ).length;

  return (
    <div className="space-y-6">
      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-xl font-bold tracking-tight">
          Chart
          {data?.ticker && (
            <span className="ml-2 text-[var(--accent)]">{data.ticker}</span>
          )}
        </h2>
        {livePrice && (
          <span className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)] animate-pulse" />
            <span className="font-semibold text-[var(--text)] tabular-nums">
              ${livePrice.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            {livePrice.prev_price != null && livePrice.prev_price > 0 && (
              <span
                className={`text-[10px] font-semibold tabular-nums ${
                  livePrice.price >= livePrice.prev_price
                    ? "text-[var(--green)]"
                    : "text-[var(--red)]"
                }`}
              >
                {livePrice.price >= livePrice.prev_price ? "+" : ""}
                {(((livePrice.price - livePrice.prev_price) / livePrice.prev_price) * 100).toFixed(2)}%
              </span>
            )}
            <span className="text-[9px] font-semibold uppercase tracking-wider text-[var(--green)]">Live</span>
          </span>
        )}
        {!livePrice && streamingActive && (
          <span className="flex items-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-[10px] text-[var(--amber)]">
            Streaming...
          </span>
        )}
        {!livePrice && !streamingActive && data?.ticker && (
          <span className="flex items-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-[10px] text-[var(--muted)]">
            Delayed
          </span>
        )}
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value.toUpperCase())}
          onKeyDown={handleKeyDown}
          placeholder="Ticker (e.g. SPY)"
          className="w-28 rounded-lg border border-[var(--line)] bg-[var(--bg)] px-3 py-1.5 text-sm font-mono text-[var(--text)] placeholder-[var(--muted)] focus:border-[var(--accent)] focus:outline-none transition-colors"
        />
        <button
          onClick={handleLoad}
          disabled={isLoading}
          className="rounded-lg bg-[var(--accent)] px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-[var(--blue)] disabled:opacity-50"
        >
          Load
        </button>
        <div className="flex gap-1">
          {(["daily", "weekly"] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                timeframe === tf
                  ? "bg-[var(--blue)] text-white"
                  : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {tf.charAt(0).toUpperCase() + tf.slice(1)}
            </button>
          ))}
        </div>
        <button
          onClick={() => {
            refetch();
          }}
          disabled={isLoading}
          className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-xs text-[var(--muted)] transition-colors hover:text-[var(--text)] disabled:opacity-50"
        >
          Refresh
        </button>
        <Link
          to="/report"
          className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-xs text-[var(--muted)] transition-colors hover:text-[var(--text)]"
        >
          &larr; Back to Report
        </Link>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
          Indicators
        </span>
        {CHART_OVERLAY_OPTIONS.map((option) => {
          const unavailable = chartBarCount > 0 && chartBarCount < option.minBars;
          const active = chartOverlays[option.key];
          return (
            <button
              key={option.key}
              type="button"
              disabled={unavailable}
              title={unavailable ? `Needs ${option.minBars} bars on this timeframe` : undefined}
              onClick={() =>
                setChartOverlays((current) => ({
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
        {chartBarCount > 0 && (
          <span className="text-xs text-[var(--muted)]">
            Default view opens on the latest 3 months.
          </span>
        )}
      </div>

      {isLoading && <Spinner className="mt-12" />}
      {error && (
        <div className="mt-4 text-center text-sm text-[var(--red)]">
          Failed to load chart: {String((error as Error)?.message ?? "Unknown error")}
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Fundamental cards */}
          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
            <GlassCard
              label="Price"
              value={currentPrice != null ? `$${currentPrice.toFixed(2)}` : "\u2014"}
            />
            <GlassCard
              label="P/E"
              value={fmt(fundamentals.pe, 1)}
            />
            <GlassCard
              label="PEG"
              value={fmt(fundamentals.peg, 2)}
            />
            <GlassCard
              label="Target"
              value={fundamentals.target_price != null ? `$${fundamentals.target_price.toFixed(2)}` : "\u2014"}
            />
            <GlassCard
              label="Discount %"
              value={
                fundamentals.discount_pct != null
                  ? `${fundamentals.discount_pct > 0 ? "+" : ""}${fundamentals.discount_pct.toFixed(1)}%`
                  : "\u2014"
              }
              className={
                fundamentals.discount_pct != null && fundamentals.discount_pct > 0
                  ? "border-[rgba(56,211,159,0.3)]"
                  : fundamentals.discount_pct != null && fundamentals.discount_pct < -10
                    ? "border-[rgba(255,107,107,0.3)]"
                    : ""
              }
            />
            <GlassCard
              label="Put/Call OI"
              value={
                options.put_call_oi_ratio != null
                  ? options.put_call_oi_ratio.toFixed(3)
                  : "\u2014"
              }
            />
          </div>

          {/* Main content: chart + commentary sidebar */}
          <div className="flex gap-4">
            {/* Chart area */}
            <div className="flex-1 min-w-0">
              {chartResult ? (
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-2">
                  <PlotlyWrapper
                    data={chartResult.traces as unknown as Record<string, unknown>[]}
                    layout={chartResult.layout as unknown as Record<string, unknown>}
                    config={{
                      responsive: true,
                      displayModeBar: true,
                      scrollZoom: true,
                    }}
                    style={{ width: "100%", height: ((chartResult.layout as Record<string, unknown>).height as number) ?? 580 }}
                  />
                </div>
              ) : (
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-12 text-center text-sm text-[var(--muted)]">
                  No chart data available.
                </div>
              )}
            </div>

            {/* Commentary sidebar */}
            <div
              className={`transition-all duration-200 ${
                commentaryExpanded ? "w-80 shrink-0" : "w-8 shrink-0"
              } hidden lg:block`}
            >
              {commentaryExpanded ? (
                <div className="relative">
                  <button
                    onClick={() => setCommentaryExpanded(false)}
                    className="absolute -left-3 top-2 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--panel)] text-xs text-[var(--muted)] hover:text-[var(--text)] transition-colors"
                    title="Collapse commentary"
                  >
                    &rsaquo;
                  </button>
                  <ChartCommentarySidebar commentary={commentary} hmmRegime={data?.hmm_regime ?? null} />
                </div>
              ) : (
                <button
                  onClick={() => setCommentaryExpanded(true)}
                  className="flex h-full w-8 items-start justify-center rounded-lg border border-[var(--line)] bg-[var(--panel)] pt-3 text-xs text-[var(--muted)] hover:text-[var(--text)] transition-colors"
                  title="Expand commentary"
                >
                  &lsaquo;
                </button>
              )}
            </div>
          </div>

          {/* Commentary for mobile/tablet (below chart) */}
          <div className="lg:hidden">
            <ChartCommentarySidebar commentary={commentary} hmmRegime={data?.hmm_regime ?? null} />
          </div>

          {/* Levels + Gaps summary */}
          <div className="grid gap-4 lg:grid-cols-2">
            <LevelsCard levels={data.levels ?? []} />
            <GapsCard gaps={data.gaps ?? []} />
          </div>

          {/* Pattern tables */}
          <PatternTabs payload={data} />

          {/* YOLO Audit */}
          <YoloAuditSection yoloAudit={data.yolo_audit ?? []} />

          {/* Footer */}
          <div className="text-xs text-[var(--muted)]">
            As of {data.asof ?? "\u2014"} &middot; {data.ticker}
          </div>

          {/* NFA disclaimer */}
          <p className="text-[10px] text-[var(--muted)]">
            Technical analysis patterns and AI-generated signals are for research purposes only. They do not constitute buy or sell recommendations.
          </p>
        </>
      )}
    </div>
  );
}

/* ── Levels card ── */

function LevelsCard({ levels }: { levels: LevelRow[] }) {
  if (levels.length === 0) {
    return (
      <Card label="Support / Resistance Levels">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No levels data available.
        </p>
      </Card>
    );
  }

  const columns: Array<{
    key: keyof LevelRow & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    {
      key: "type",
      label: "Type",
      render: (v: unknown) => {
        const t = String(v ?? "");
        return (
          <Badge variant={t === "support" ? "blue" : "red"}>
            {t.toUpperCase()}
          </Badge>
        );
      },
    },
    {
      key: "level",
      label: "Level",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    { key: "tier", label: "Tier" },
    { key: "touches", label: "Touches" },
    { key: "last_touch_date", label: "Last Touch" },
  ];

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        Support / Resistance ({levels.length})
      </h3>
      <Table
        columns={columns}
        data={levels as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}

/* ── Gaps card ── */

function GapsCard({ gaps }: { gaps: GapRow[] }) {
  if (gaps.length === 0) {
    return (
      <Card label="Gaps">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No gap data available.
        </p>
      </Card>
    );
  }

  const columns: Array<{
    key: keyof GapRow & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "date", label: "Date" },
    {
      key: "type",
      label: "Type",
      render: (v: unknown) => {
        const t = String(v ?? "");
        return (
          <Badge variant={t === "bull_gap" ? "green" : "blue"}>
            {t.replace(/_/g, " ").toUpperCase()}
          </Badge>
        );
      },
    },
    {
      key: "gap_low",
      label: "Low",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    {
      key: "gap_high",
      label: "High",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
  ];

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        Gaps ({gaps.length})
      </h3>
      <Table
        columns={columns}
        data={gaps as unknown as Record<string, unknown>[]}
        sortable
      />
    </div>
  );
}

import type {
  CryptoBar,
  CryptoStructurePayload,
  LevelRow,
} from "../../api/types";
import type { FormingCandleData } from "../../hooks/useCryptoSubscription";
import { getPlotlyColors } from "../plotlyTheme";

export interface CryptoOverlayState {
  sma20: boolean;
  sma50: boolean;
  sma200: boolean;
  bollinger: boolean;
}

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

function addLevelOverlays(
  levels: LevelRow[],
  annotations: Record<string, unknown>[],
  shapes: Record<string, unknown>[],
) {
  levels.forEach((level) => {
    if (!Number.isFinite(level.level)) {
      return;
    }

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
      text: `${level.type === "support" ? "SUP" : "RES"} ${formatPrice(level.level)}`,
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

function computeRollingSma(values: number[], period: number): number[] {
  const result: number[] = [];
  for (let i = period - 1; i < values.length; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) {
      sum += values[j];
    }
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

export interface CandlePatternRow {
  date: string;
  pattern: string;
  bias: string;
  confidence: number;
}

export function buildCandlestickChart(
  bars: CryptoBar[],
  symbol: string,
  structure: CryptoStructurePayload | null,
  overlays: CryptoOverlayState,
  formingCandle?: FormingCandleData | null,
  uirevisionKey?: string,
  candlestickPatterns?: CandlePatternRow[],
): { traces: Record<string, unknown>[]; layout: Record<string, unknown> } {
  const timestamps = bars.map((bar) => bar.timestamp);
  const open = bars.map((bar) => bar.open);
  const high = bars.map((bar) => bar.high);
  const low = bars.map((bar) => bar.low);
  const close = bars.map((bar) => bar.close);
  const volume = bars.map((bar) => bar.volume);

  const traces: Record<string, unknown>[] = [
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
      whiskerwidth: 0.8,
      increasing: {
        line: { color: "#38d39f", width: 1.5 },
        fillcolor: "rgba(56,211,159,0.85)",
      },
      decreasing: {
        line: { color: "#ff6b6b", width: 1.5 },
        fillcolor: "rgba(255,107,107,0.85)",
      },
    },
    {
      type: "bar",
      x: timestamps,
      y: volume,
      name: "Volume",
      marker: {
        color: bars.map((bar) =>
          bar.close >= bar.open
            ? "rgba(56,211,159,0.5)"
            : "rgba(255,107,107,0.5)",
        ),
      },
      xaxis: "x",
      yaxis: "y2",
    },
  ];

  if (formingCandle && formingCandle.timestamp) {
    const fillColor = formingCandle.close >= formingCandle.open
      ? "rgba(56,211,159,0.35)"
      : "rgba(255,107,107,0.35)";
    const lineColor = formingCandle.close >= formingCandle.open
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
        line: { color: lineColor, width: 1, dash: "dot" },
        fillcolor: fillColor,
      },
      decreasing: {
        line: { color: lineColor, width: 1, dash: "dot" },
        fillcolor: fillColor,
      },
    });
    traces.push({
      type: "bar",
      x: [formingCandle.timestamp],
      y: [formingCandle.volume],
      name: "Forming Vol",
      marker: { color: fillColor },
      xaxis: "x",
      yaxis: "y2",
      showlegend: false,
    });
  }

  if (overlays.sma20 && bars.length >= 20) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: timestamps.slice(19),
      y: computeRollingSma(close, 20),
      name: "SMA 20",
      line: { color: "#f0c040", width: 1.2 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.sma50 && bars.length >= 50) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: timestamps.slice(49),
      y: computeRollingSma(close, 50),
      name: "SMA 50",
      line: { color: "#6baed6", width: 1.2 },
      xaxis: "x",
      yaxis: "y",
    });
  }
  if (overlays.sma200 && bars.length >= 200) {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: timestamps.slice(199),
      y: computeRollingSma(close, 200),
      name: "SMA 200",
      line: { color: "#d291ff", width: 1.3 },
      xaxis: "x",
      yaxis: "y",
    });
  }

  let latestBollinger: { upper: number; middle: number; lower: number } | null = null;
  if (overlays.bollinger && bars.length >= 20) {
    const bands = computeRollingBollinger(close, 20, 2);
    const bandTimestamps = timestamps.slice(19);
    traces.push(
      {
        type: "scatter",
        mode: "lines",
        x: bandTimestamps,
        y: bands.upper,
        name: "BOLL Upper",
        line: { color: "#ffd21f", width: 1.8 },
        showlegend: false,
        xaxis: "x",
        yaxis: "y",
      },
      {
        type: "scatter",
        mode: "lines",
        x: bandTimestamps,
        y: bands.middle,
        name: "BOLL Mid",
        line: { color: "#f6bed8", width: 1.6 },
        showlegend: false,
        xaxis: "x",
        yaxis: "y",
      },
      {
        type: "scatter",
        mode: "lines",
        x: bandTimestamps,
        y: bands.lower,
        name: "BOLL Lower",
        line: { color: "#16c7ff", width: 1.8 },
        showlegend: false,
        xaxis: "x",
        yaxis: "y",
      },
    );

    latestBollinger = {
      upper: bands.upper[bands.upper.length - 1],
      middle: bands.middle[bands.middle.length - 1],
      lower: bands.lower[bands.lower.length - 1],
    };
  }

  const shapes: Record<string, unknown>[] = [];
  const annotations: Record<string, unknown>[] = [];
  addLevelOverlays(structure?.levels ?? [], annotations, shapes);

  if (
    latestBollinger &&
    Number.isFinite(latestBollinger.upper) &&
    Number.isFinite(latestBollinger.middle) &&
    Number.isFinite(latestBollinger.lower)
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
        text: `MID: ${formatPrice(latestBollinger.middle)}`,
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
        text: `UPPER: ${formatPrice(latestBollinger.upper)}`,
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
        text: `LOWER: ${formatPrice(latestBollinger.lower)}`,
        showarrow: false,
        xanchor: "left",
        yanchor: "top",
        font: { color: "#16c7ff", size: 12 },
      },
    );
  }

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

  // Candlestick pattern markers (bullish/bearish triangles)
  if (candlestickPatterns && candlestickPatterns.length > 0) {
    const bullish = candlestickPatterns.filter((p) => p.bias === "bullish");
    const bearish = candlestickPatterns.filter((p) => p.bias === "bearish");
    if (bullish.length > 0) {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: bullish.map((p) => p.date),
        y: bullish.map((p) => {
          const idx = timestamps.indexOf(p.date);
          return idx >= 0 ? low[idx] * 0.997 : null;
        }),
        name: "Bullish Signal",
        marker: { symbol: "triangle-up", size: 9, color: "#38d39f" },
        hovertext: bullish.map((p) => `${p.pattern.replace(/_/g, " ")} (${(p.confidence * 100).toFixed(0)}%)`),
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
          const idx = timestamps.indexOf(p.date);
          return idx >= 0 ? high[idx] * 1.003 : null;
        }),
        name: "Bearish Signal",
        marker: { symbol: "triangle-down", size: 9, color: "#ff6b6b" },
        hovertext: bearish.map((p) => `${p.pattern.replace(/_/g, " ")} (${(p.confidence * 100).toFixed(0)}%)`),
        hoverinfo: "text+x",
        xaxis: "x",
        yaxis: "y",
      });
    }
  }

  const theme = getPlotlyColors();

  const layout: Record<string, unknown> = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    uirevision: uirevisionKey ?? symbol,
    font: { color: theme.font, size: 11 },
    margin: { t: 30, r: 60, b: 50, l: 60 },
    dragmode: "zoom",
    legend: {
      orientation: "h",
      y: -0.04,
      x: 0,
      xanchor: "left",
    },
    xaxis: {
      gridcolor: theme.grid,
      rangeslider: { visible: false },
    },
    yaxis: {
      gridcolor: theme.grid,
      domain: [0.28, 1],
      title: "Price",
      autorange: true,
      fixedrange: false,
      rangemode: "normal",
    },
    yaxis2: {
      gridcolor: theme.grid,
      domain: [0, 0.22],
      title: "Volume",
    },
    shapes,
    annotations,
    height: 500,
  };

  return { traces, layout };
}

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
  rsiPeriod: number; // 0 = hidden, 7/14/21 = active
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

export interface DirectionalRegimePoint {
  date: string;
  label: string;
  color: string;
}

export function buildCandlestickChart(
  bars: CryptoBar[],
  symbol: string,
  structure: CryptoStructurePayload | null,
  overlays: CryptoOverlayState,
  formingCandle?: FormingCandleData | null,
  uirevisionKey?: string,
  candlestickPatterns?: CandlePatternRow[],
  directionalRegimes?: DirectionalRegimePoint[],
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
    // Build date lookup: pattern dates are "YYYY-MM-DD HH:MM", timestamps are ISO
    const tsDateMap = new Map<string, number>();
    timestamps.forEach((ts, i) => {
      tsDateMap.set(ts.slice(0, 10), i);  // match by date portion
      tsDateMap.set(ts.slice(0, 16).replace("T", " "), i);  // match with time
      tsDateMap.set(ts, i);  // exact match
    });

    const findIdx = (date: string): number => {
      return tsDateMap.get(date) ?? tsDateMap.get(date.slice(0, 10)) ?? -1;
    };

    const bullish = candlestickPatterns.filter((p) => p.bias === "bullish");
    const bearish = candlestickPatterns.filter((p) => p.bias === "bearish");
    if (bullish.length > 0) {
      const validBullish = bullish.filter((p) => findIdx(p.date) >= 0);
      if (validBullish.length > 0) {
        traces.push({
          type: "scatter",
          mode: "markers",
          x: validBullish.map((p) => timestamps[findIdx(p.date)]),
          y: validBullish.map((p) => low[findIdx(p.date)] * 0.995),
          name: "Bullish Signal",
          marker: { symbol: "triangle-up", size: 11, color: "#38d39f" },
          hovertext: validBullish.map((p) => `${p.pattern.replace(/_/g, " ")} (${(p.confidence * 100).toFixed(0)}%)`),
          hoverinfo: "text+x",
          xaxis: "x",
          yaxis: "y",
        });
      }
    }
    if (bearish.length > 0) {
      const validBearish = bearish.filter((p) => findIdx(p.date) >= 0);
      if (validBearish.length > 0) {
        traces.push({
          type: "scatter",
          mode: "markers",
          x: validBearish.map((p) => timestamps[findIdx(p.date)]),
          y: validBearish.map((p) => high[findIdx(p.date)] * 1.005),
          name: "Bearish Signal",
          marker: { symbol: "triangle-down", size: 11, color: "#ff6b6b" },
          hovertext: validBearish.map((p) => `${p.pattern.replace(/_/g, " ")} (${(p.confidence * 100).toFixed(0)}%)`),
          hoverinfo: "text+x",
          xaxis: "x",
          yaxis: "y",
        });
      }
    }
  }

  // Directional HMM background coloring (bullish=green, bearish=red, chop=purple)
  if (directionalRegimes && directionalRegimes.length > 1) {
    const dirColors: Record<string, string> = {
      bullish: "rgba(56,211,159,0.08)",
      bearish: "rgba(255,107,107,0.08)",
      chop: "rgba(168,85,247,0.08)",
    };
    const dirTextColors: Record<string, string> = {
      bullish: "#38d39f",
      bearish: "#ff6b6b",
      chop: "#a855f7",
    };
    const dirLabels: Record<string, string> = {
      bullish: "BULL",
      bearish: "BEAR",
      chop: "CHOP",
    };
    let spanStart = 0;
    for (let i = 1; i <= directionalRegimes.length; i++) {
      if (i === directionalRegimes.length || directionalRegimes[i].label !== directionalRegimes[spanStart].label) {
        const label = directionalRegimes[spanStart].label;
        const fillcolor = dirColors[label] ?? "rgba(128,128,128,0.04)";
        const startDate = directionalRegimes[spanStart].date;
        const endDate = directionalRegimes[Math.min(i, directionalRegimes.length - 1)].date;
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "y",
          x0: startDate,
          x1: endDate,
          y0: Math.min(...low.filter((v) => v > 0)) * 0.99,
          y1: Math.max(...high) * 1.01,
          fillcolor,
          line: { width: 0 },
          layer: "below",
        });
        // Label at top of each regime span
        const spanDays = i - spanStart;
        if (spanDays >= 5) {
          annotations.push({
            xref: "x",
            yref: "paper",
            x: startDate,
            y: 0.98,
            text: dirLabels[label] ?? label,
            showarrow: false,
            font: { color: dirTextColors[label] ?? "#8ea0bd", size: 9 },
            opacity: 0.6,
            xanchor: "left",
          });
        }
        spanStart = i;
      }
    }
  }

  // Compute RSI time series for subplot (always on, period defaults to 14)
  const rsiPeriod = (typeof overlays.rsiPeriod === "number" && overlays.rsiPeriod > 0) ? overlays.rsiPeriod : 14;
  const rsiValues: (number | null)[] = new Array(close.length).fill(null);
  if (rsiPeriod > 0 && close.length >= rsiPeriod + 1) {
    const deltas = close.map((c, i) => i === 0 ? 0 : c - close[i - 1]);
    let avgGain = 0;
    let avgLoss = 0;
    for (let i = 1; i <= rsiPeriod; i++) {
      if (deltas[i] > 0) avgGain += deltas[i];
      else avgLoss += Math.abs(deltas[i]);
    }
    avgGain /= rsiPeriod;
    avgLoss /= rsiPeriod;
    rsiValues[rsiPeriod] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    for (let i = rsiPeriod + 1; i < close.length; i++) {
      const d = deltas[i];
      avgGain = (avgGain * (rsiPeriod - 1) + Math.max(d, 0)) / rsiPeriod;
      avgLoss = (avgLoss * (rsiPeriod - 1) + Math.abs(Math.min(d, 0))) / rsiPeriod;
      rsiValues[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    }
  }

  const hasRsi = rsiPeriod > 0 && rsiValues.some((v) => v !== null);

  if (hasRsi) {
    // RSI line
    traces.push({
      type: "scatter",
      mode: "lines",
      x: timestamps,
      y: rsiValues,
      name: `RSI ${rsiPeriod}`,
      line: { color: "#a855f7", width: 1.5 },
      xaxis: "x",
      yaxis: "y3",
      showlegend: false,
      hovertemplate: "RSI: %{y:.1f}<extra></extra>",
    });

    // Overbought/oversold horizontal lines (as shapes)
    shapes.push(
      { type: "line", xref: "paper", yref: "y3", x0: 0, x1: 1, y0: 70, y1: 70, line: { color: "rgba(255,107,107,0.4)", width: 1, dash: "dot" } },
      { type: "line", xref: "paper", yref: "y3", x0: 0, x1: 1, y0: 30, y1: 30, line: { color: "rgba(56,211,159,0.4)", width: 1, dash: "dot" } },
      { type: "line", xref: "paper", yref: "y3", x0: 0, x1: 1, y0: 50, y1: 50, line: { color: "rgba(142,160,189,0.2)", width: 1, dash: "dot" } },
    );
  }

  const theme = getPlotlyColors();

  // Adjust domains for 3 panels: Price (top), Volume (mid), RSI (bottom)
  const priceDomain: [number, number] = hasRsi ? [0.40, 1] : [0.28, 1];
  const volumeDomain: [number, number] = hasRsi ? [0.22, 0.36] : [0, 0.22];
  const rsiDomain: [number, number] = [0, 0.18];

  const layout: Record<string, unknown> = {
    paper_bgcolor: theme.bg,
    plot_bgcolor: theme.bg,
    uirevision: uirevisionKey ?? symbol,
    font: { color: theme.font, size: 11 },
    margin: { t: 30, r: 90, b: 50, l: 60 },
    dragmode: "zoom",
    legend: {
      orientation: "h",
      y: 1.02,
      x: 1,
      xanchor: "right",
      yanchor: "bottom",
      bgcolor: "rgba(11,15,22,0.6)",
      bordercolor: "rgba(255,255,255,0.08)",
      borderwidth: 1,
      font: { size: 10 },
    },
    xaxis: {
      gridcolor: theme.grid,
      rangeslider: { visible: false },
    },
    yaxis: {
      gridcolor: theme.grid,
      domain: priceDomain,
      title: "Price",
      autorange: true,
      fixedrange: false,
      rangemode: "normal",
    },
    yaxis2: {
      gridcolor: theme.grid,
      domain: volumeDomain,
      title: "Volume",
    },
    shapes,
    annotations,
    height: hasRsi ? 750 : 500,
  };

  if (hasRsi) {
    (layout as Record<string, unknown>).yaxis3 = {
      gridcolor: theme.grid,
      domain: rsiDomain,
      title: "RSI",
      range: [0, 100],
      tickvals: [30, 50, 70],
      ticktext: ["30", "50", "70"],
    };
  }

  return { traces, layout };
}

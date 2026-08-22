import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { useChartQuick, useChartCommentary } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import { useLiveEquityPrice } from "../hooks/useLiveEquityPrice";
import type { DashboardPayload, LiveCandle } from "../api/types";
import Spinner from "../components/ui/Spinner";
import ChartToolbar from "../components/chart/ChartToolbar";
import ChartFundamentals from "../components/chart/ChartFundamentals";
import ChartWorkspace from "../components/chart/ChartWorkspace";
import LevelsCard from "../components/chart/LevelsCard";
import GapsCard from "../components/chart/GapsCard";
import ChartOverlayControls from "../components/chart/ChartOverlayControls";
import ChartPlotPanel from "../components/chart/ChartPlotPanel";
import PatternTabs from "../components/chart/PatternTabs";
import YoloAuditSection from "../components/chart/YoloAuditSection";
import ChartCommentarySidebar from "../components/chart/ChartCommentarySidebar";
import {
  applyLivePriceToPayload,
  buildChartData,
  CHART_OVERLAY_OPTIONS,
  DEFAULT_CHART_OVERLAYS,
  formatChartNumber,
  resampleToMonthly,
  resampleToWeekly,
  type ChartOverlayKey,
  type ChartOverlayState,
} from "../lib/chart/buildEquityChartData";

/* ── Main Page ── */

// Throttle the live price that drives the full Plotly rebuild. Equity ticks can
// arrive several times per second; coalescing them to a fixed cadence keeps the
// LIVE candle/badge visually current while avoiding a full trace rebuild per tick.
// Toolbar/fundamentals keep using the unthrottled livePrice for instant display.
const CHART_LIVE_PRICE_THROTTLE_MS = 1000;

function useThrottledValue<T>(value: T, intervalMs: number): T {
  const [throttled, setThrottled] = useState(value);
  const lastEmit = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const now = Date.now();
    const elapsed = now - lastEmit.current;
    if (elapsed >= intervalMs) {
      timer.current = setTimeout(() => {
        lastEmit.current = Date.now();
        setThrottled(value);
      }, 0);
    } else {
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => {
        lastEmit.current = Date.now();
        setThrottled(value);
      }, intervalMs - elapsed);
    }
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [value, intervalMs]);

  return throttled;
}

export default function ChartPage() {
  const { ticker, timeframe, setTicker, setTimeframe } = useChartStore();
  const [searchParams] = useSearchParams();
  const [inputValue, setInputValue] = useState(ticker);
  const [commentaryExpanded, setCommentaryExpanded] = useState(true);
  const [chartOverlays, setChartOverlays] = useState<ChartOverlayState>(
    DEFAULT_CHART_OVERLAYS,
  );
  const [greenBarrierThreshold] = useState(() => {
    const rawThreshold = searchParams.get("threshold");
    const parsedThreshold = rawThreshold === null ? Number.NaN : Number(rawThreshold);
    return Number.isFinite(parsedThreshold) && parsedThreshold >= -100 && parsedThreshold <= 0
      ? parsedThreshold
      : -98;
  });
  const [reportGreenBarrierSnapshot] = useState(() => {
    const asof = searchParams.get("asof")?.trim() ?? "";
    const rawValue = searchParams.get("value");
    const value = rawValue === null ? Number.NaN : Number(rawValue);
    return /^\d{4}-\d{2}-\d{2}$/.test(asof) && Number.isFinite(value)
      ? { asof, value }
      : null;
  });
  const [compactChart, setCompactChart] = useState(
    () => typeof window !== "undefined" && window.innerWidth < 768,
  );
  const { livePrice, streamingActive } = useLiveEquityPrice(ticker);

  useEffect(() => {
    document.title = "Chart \u2014 Trader Koo";
  }, []);

  // Pick up ticker from URL query param ?t=AAPL (initial mount only)
  const urlConsumed = useRef(false);
  useEffect(() => {
    if (urlConsumed.current) return undefined;
    urlConsumed.current = true;
    const urlTicker = searchParams.get("t") || searchParams.get("ticker");
    const urlTimeframe = searchParams.get("timeframe");
    if (
      urlTimeframe === "daily" ||
      urlTimeframe === "weekly" ||
      urlTimeframe === "monthly"
    ) {
      setTimeframe(urlTimeframe);
    }
    if (urlTicker) {
      const clean = urlTicker.trim().toUpperCase();
      if (clean) {
        const timer = window.setTimeout(() => {
          setTicker(clean);
          setInputValue(clean);
        }, 0);
        return () => window.clearTimeout(timer);
      }
    }
    return undefined;
  }, [searchParams, setTicker, setTimeframe]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handleResize = () => setCompactChart(window.innerWidth < 768);
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Progressive loading: fast path first, commentary in background
  const {
    data: quickData,
    isLoading: quickLoading,
    error: quickError,
    refetch: refetchQuick,
  } = useChartQuick(ticker);

  const {
    data: commentaryData,
    isLoading: commentaryLoading,
    refetch: refetchCommentary,
  } = useChartCommentary(ticker, !!quickData);

  // Merge quick + commentary into a DashboardPayload-compatible shape
  const data = useMemo((): DashboardPayload | undefined => {
    if (!quickData) return undefined;
    return {
      ...quickData,
      report_generated_ts: commentaryData?.report_generated_ts ?? null,
      chart_commentary: commentaryData?.chart_commentary ?? ({} as DashboardPayload["chart_commentary"]),
      hmm_regime: commentaryData?.hmm_regime ?? null,
    };
  }, [quickData, commentaryData]);

  const isLoading = quickLoading;
  const error = quickError;

  const handleRefresh = useCallback(() => {
    void refetchQuick();
    void refetchCommentary();
  }, [refetchQuick, refetchCommentary]);

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
  const commentary = commentaryData?.chart_commentary ?? null;
  const freshness = quickData?.data_freshness ?? undefined;
  const priceContract = quickData?.data_sources;
  // Throttled copy for the chart rebuild only; toolbar/fundamentals stay instant.
  const chartLivePrice = useThrottledValue(livePrice, CHART_LIVE_PRICE_THROTTLE_MS);
  const livePayload = useMemo(
    () => applyLivePriceToPayload(data, chartLivePrice),
    [data, chartLivePrice],
  );

  // Build effective live candle: start from API live_candle, overlay WS price if newer
  const effectiveLiveCandle = useMemo((): LiveCandle | null => {
    const apiCandle =
      typeof data?.live_candle === "object" && data.live_candle !== null
        ? data.live_candle
        : null;
    if (!apiCandle) return null;

    // If WS price exists and belongs to the same ticker, update close/high/low.
    // Use the throttled price so this memo (and the downstream full chart rebuild)
    // coalesces ticks instead of recomputing on every WS message.
    if (
      chartLivePrice &&
      typeof chartLivePrice.price === "number" &&
      Number.isFinite(chartLivePrice.price) &&
      chartLivePrice.symbol === data?.ticker
    ) {
      const wsTime = chartLivePrice.timestamp ? new Date(chartLivePrice.timestamp).getTime() : 0;
      const apiTime = apiCandle.timestamp ? new Date(apiCandle.timestamp).getTime() : 0;
      if (wsTime > apiTime) {
        return {
          ...apiCandle,
          close: chartLivePrice.price,
          high: Math.max(apiCandle.high, chartLivePrice.price),
          low: Math.min(apiCandle.low, chartLivePrice.price),
        };
      }
    }
    return apiCandle;
  }, [data, chartLivePrice]);

  const chartResult = useMemo(() => {
    if (!livePayload || !livePayload.chart || livePayload.chart.length === 0) {
      return null;
    }
    return buildChartData(
      livePayload,
      timeframe,
      chartOverlays,
      effectiveLiveCandle,
      compactChart,
      greenBarrierThreshold,
    );
  }, [
    livePayload,
    timeframe,
    chartOverlays,
    effectiveLiveCandle,
    compactChart,
    greenBarrierThreshold,
  ]);

  const chartBarCount = (
    timeframe === "monthly"
      ? resampleToMonthly(livePayload?.chart ?? [])
      : timeframe === "weekly"
        ? resampleToWeekly(livePayload?.chart ?? [])
        : livePayload?.chart ?? []
  ).length;
  const reportSnapshotMatches = reportGreenBarrierSnapshot && chartResult
    ? chartResult.greenBarrier.asof === reportGreenBarrierSnapshot.asof &&
      chartResult.greenBarrier.value !== null &&
      Math.abs(chartResult.greenBarrier.value - reportGreenBarrierSnapshot.value) <= 0.11
    : null;

  return (
    <div className="space-y-6">
      {/* Controls row */}
      <ChartToolbar
        ticker={data?.ticker}
        livePrice={livePrice}
        streamingActive={streamingActive}
        inputValue={inputValue}
        isLoading={isLoading}
        timeframe={timeframe}
        onInputChange={setInputValue}
        onInputKeyDown={handleKeyDown}
        onLoad={handleLoad}
        onSelectTimeframe={setTimeframe}
        onRefresh={handleRefresh}
      />

      {/* Data freshness indicator */}
      {freshness?.latest_price_date && (
        <div className={`text-[10px] mb-1 ${freshness.is_stale ? "text-[var(--red)]" : "text-[var(--muted)]"}`}>
          Price data as of <strong>{freshness.latest_price_date}</strong>
          {freshness.age_hours != null && ` (${freshness.age_hours < 24 ? `${freshness.age_hours.toFixed(0)}h ago` : `${(freshness.age_hours / 24).toFixed(1)}d ago`})`}
          {freshness.is_stale && " — STALE"}
        </div>
      )}

      {priceContract && (
        <div
          role="status"
          className={`rounded-lg border px-3 py-2 text-xs ${
            priceContract.research_eligible
              ? "border-[var(--green)]/30 bg-[var(--green)]/5 text-[var(--muted)]"
              : "border-[var(--red)]/40 bg-[var(--red)]/5 text-[var(--red)]"
          }`}
        >
          Price basis: <strong>{priceContract.adjustment_basis}</strong> · {priceContract.price} ·{" "}
          {priceContract.adjustment_version} · {priceContract.basis_status}
          {!priceContract.research_eligible && " — excluded from research until resolved"}
        </div>
      )}

      {reportGreenBarrierSnapshot && (
        <div
          role="status"
          className={`rounded-lg border px-3 py-2 text-xs ${
            reportSnapshotMatches
              ? "border-[var(--green)]/30 bg-[var(--green)]/5 text-[var(--green)]"
              : "border-[var(--amber)]/30 bg-[var(--amber)]/5 text-[var(--amber)]"
          }`}
        >
          Report snapshot: Williams %R {reportGreenBarrierSnapshot.value.toFixed(1)} as of{" "}
          {reportGreenBarrierSnapshot.asof}.{" "}
          {reportSnapshotMatches
            ? "The plotted snapshot matches the report."
            : "The loaded chart no longer matches that snapshot; the threshold is context only."}
        </div>
      )}

      <ChartOverlayControls
        overlayOptions={CHART_OVERLAY_OPTIONS}
        barCount={chartBarCount}
        overlays={chartOverlays}
        onToggleOverlay={(overlayKey) => {
          const key = overlayKey as ChartOverlayKey;
          setChartOverlays((current) => ({
            ...current,
            [key]: !current[key],
          }));
        }}
      />

      {isLoading && <Spinner className="mt-12" />}
      {error && (
        <div className="mt-4 text-center text-sm text-[var(--red)]">
          Failed to load chart: {String((error as Error)?.message ?? "Unknown error")}
        </div>
      )}

      {!data && !isLoading && !error && (
        <div className="mt-16 text-center text-sm text-[var(--muted)]">
          Enter a ticker symbol above to view chart analysis
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Fundamental cards */}
          <ChartFundamentals
            currentPrice={currentPrice}
            fundamentals={fundamentals}
            putCallOiRatio={options.put_call_oi_ratio}
            formatNumber={formatChartNumber}
          />

          <ChartWorkspace
            commentaryExpanded={commentaryExpanded}
            onCollapse={() => setCommentaryExpanded(false)}
            onExpand={() => setCommentaryExpanded(true)}
            chartContent={
              <ChartPlotPanel
                chartData={chartResult ? (chartResult.traces as unknown as Record<string, unknown>[]) : null}
                chartLayout={chartResult ? (chartResult.layout as unknown as Record<string, unknown>) : null}
              />
            }
            desktopCommentary={
              <ChartCommentarySidebar
                commentary={commentary}
                hmmRegime={commentaryData?.hmm_regime ?? null}
                isLoading={commentaryLoading}
              />
            }
            mobileCommentary={
              <ChartCommentarySidebar
                commentary={commentary}
                hmmRegime={commentaryData?.hmm_regime ?? null}
                isLoading={commentaryLoading}
              />
            }
          />

          {/* Levels + Gaps summary */}
          <div className="grid gap-4 lg:grid-cols-2">
            <LevelsCard levels={data.levels ?? []} formatNumber={formatChartNumber} />
            <GapsCard gaps={data.gaps ?? []} formatNumber={formatChartNumber} />
          </div>

          {/* Pattern tables */}
          <PatternTabs payload={data} formatNumber={formatChartNumber} />

          {/* YOLO Audit */}
          <YoloAuditSection
            yoloAudit={data.yolo_audit ?? []}
            formatNumber={formatChartNumber}
          />

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

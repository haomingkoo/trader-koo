import { useState, useCallback, useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useChart } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import { useLiveEquityPrice } from "../hooks/useLiveEquityPrice";
import type { LiveCandle } from "../api/types";
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
  resampleToWeekly,
  type ChartOverlayKey,
  type ChartOverlayState,
} from "../lib/chart/buildEquityChartData";

/* ── Main Page ── */

export default function ChartPage() {
  const { ticker, timeframe, setTicker, setTimeframe } = useChartStore();
  const [searchParams] = useSearchParams();
  const [inputValue, setInputValue] = useState(ticker);
  const [commentaryExpanded, setCommentaryExpanded] = useState(true);
  const [chartOverlays, setChartOverlays] = useState<ChartOverlayState>(
    DEFAULT_CHART_OVERLAYS,
  );
  const [compactChart, setCompactChart] = useState(
    () => typeof window !== "undefined" && window.innerWidth < 768,
  );
  const { livePrice, streamingActive } = useLiveEquityPrice(ticker);

  useEffect(() => {
    document.title = "Chart \u2014 Trader Koo";
  }, []);

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

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handleResize = () => setCompactChart(window.innerWidth < 768);
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

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
    return buildChartData(
      livePayload,
      isWeekly,
      chartOverlays,
      effectiveLiveCandle,
      compactChart,
    );
  }, [livePayload, isWeekly, chartOverlays, effectiveLiveCandle, compactChart]);

  const chartBarCount = (
    isWeekly ? resampleToWeekly(livePayload?.chart ?? []) : livePayload?.chart ?? []
  ).length;

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
        onRefresh={() => {
          refetch();
        }}
      />

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
              <ChartCommentarySidebar commentary={commentary} hmmRegime={data?.hmm_regime ?? null} />
            }
            mobileCommentary={
              <ChartCommentarySidebar commentary={commentary} hmmRegime={data?.hmm_regime ?? null} />
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

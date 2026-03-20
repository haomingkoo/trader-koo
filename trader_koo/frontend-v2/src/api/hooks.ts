import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "./client";
import type {
  DailyReportPayload,
  DashboardPayload,
  DashboardQuickPayload,
  DashboardCommentaryPayload,
  PaperTradeSummary,
  PaperTradeList,
  OpportunitiesPayload,
  EarningsPayload,
  MarketSummary,
  PipelineStatus,
  CryptoSummary,
  CryptoHistoryPayload,
  CryptoIndicatorsPayload,
  CryptoStructurePayload,
  CryptoCorrelationPayload,
  CryptoMarketStructurePayload,
  CryptoOpenInterestPayload,
  VixMetricsPayload,
  FearGreedPayload,
  AlertsPayload,
  MacroLivePayload,
} from "./types";

export function useReport() {
  return useQuery({
    queryKey: ["report"],
    queryFn: () => apiFetch<DailyReportPayload>("/api/daily-report?limit=20"),
    staleTime: 2 * 60 * 1000,
  });
}

export function useChart(ticker: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ["chart", ticker],
    queryFn: () => apiFetch<DashboardPayload>(`/api/dashboard/${ticker}?months=0`),
    staleTime: 2 * 60 * 1000,
    enabled: enabled && ticker.length > 0,
  });
}

/** Fast path: price data, levels, patterns, fundamentals (no LLM/HMM). */
export function useChartQuick(ticker: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ["chart-quick", ticker],
    queryFn: () =>
      apiFetch<DashboardQuickPayload>(
        `/api/dashboard/${ticker}/quick?months=0`,
      ),
    staleTime: 2 * 60 * 1000,
    enabled: enabled && ticker.length > 0,
  });
}

/** Slow path: chart commentary, debate engine, HMM regime. */
export function useChartCommentary(ticker: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ["chart-commentary", ticker],
    queryFn: () =>
      apiFetch<DashboardCommentaryPayload>(
        `/api/dashboard/${ticker}/commentary?months=0`,
      ),
    staleTime: 2 * 60 * 1000,
    enabled: enabled && ticker.length > 0,
  });
}

export function usePaperTradeSummary() {
  return useQuery({
    queryKey: ["paper-trades-summary"],
    queryFn: () => apiFetch<PaperTradeSummary>("/api/paper-trades/summary"),
    staleTime: 2 * 60 * 1000,
  });
}

export function usePaperTrades(status: string = "all", direction: string = "all") {
  // Backend rejects direction=all — only send direction if it's long or short
  const dirParam = direction === "long" || direction === "short" ? `&direction=${direction}` : "";
  return useQuery({
    queryKey: ["paper-trades", status, direction],
    queryFn: () =>
      apiFetch<PaperTradeList>(
        `/api/paper-trades?status=${status}${dirParam}&limit=500`,
      ),
    staleTime: 2 * 60 * 1000,
  });
}

export function useOpportunities(params: {
  view?: string;
  limit?: number;
  min_discount?: number;
  max_peg?: number;
  overvalued_threshold?: number;
}) {
  const qs = new URLSearchParams({
    limit: String(params.limit ?? 1000),
    min_discount: String(params.min_discount ?? 10),
    max_peg: String(params.max_peg ?? 2),
    overvalued_threshold: String(params.overvalued_threshold ?? -10),
    view: params.view ?? "all",
  });
  return useQuery({
    queryKey: ["opportunities", params],
    queryFn: () => apiFetch<OpportunitiesPayload>(`/api/opportunities?${qs.toString()}`),
    staleTime: 2 * 60 * 1000,
  });
}

export function useEarnings(days: number = 21, tickers?: string) {
  const qs = new URLSearchParams({ days: String(days), limit: "1000" });
  if (tickers) qs.set("tickers", tickers);
  return useQuery({
    queryKey: ["earnings", days, tickers],
    queryFn: () => apiFetch<EarningsPayload>(`/api/earnings-calendar?${qs.toString()}`),
    staleTime: 2 * 60 * 1000,
  });
}

export function useMarketSummary(days: number = 90) {
  return useQuery({
    queryKey: ["market-summary", days],
    queryFn: () => apiFetch<MarketSummary>(`/api/market-summary?days=${days}`),
    staleTime: 5 * 60 * 1000,
  });
}

export function usePipelineStatus() {
  return useQuery({
    queryKey: ["pipeline-status"],
    queryFn: () => apiFetch<PipelineStatus>("/api/status"),
    refetchInterval: 30 * 1000,
    staleTime: 15 * 1000,
  });
}

export function useCryptoSummary() {
  return useQuery({
    queryKey: ["crypto-summary"],
    queryFn: () => apiFetch<CryptoSummary>("/api/crypto/summary"),
    refetchInterval: 5000,
  });
}

export function useCryptoHistory(symbol: string, interval = "1m", limit = 100) {
  return useQuery({
    queryKey: ["crypto-history", symbol, interval, limit],
    queryFn: () =>
      apiFetch<CryptoHistoryPayload>(
        `/api/crypto/history/${symbol}?interval=${interval}&limit=${limit}`,
      ),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

export function useCryptoIndicators(symbol: string) {
  return useQuery({
    queryKey: ["crypto-indicators", symbol],
    queryFn: () =>
      apiFetch<CryptoIndicatorsPayload>(`/api/crypto/indicators/${symbol}`),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });
}

export function useCryptoStructure(symbol: string, interval = "1m", limit = 240) {
  return useQuery({
    queryKey: ["crypto-structure", symbol, interval, limit],
    queryFn: () =>
      apiFetch<CryptoStructurePayload>(
        `/api/crypto/structure/${symbol}?interval=${interval}&limit=${limit}`,
      ),
    staleTime: 10_000,
  });
}

export function useCryptoCorrelation(symbol = "BTC-USD", benchmark = "SPY", limit = 40) {
  return useQuery({
    queryKey: ["crypto-correlation", symbol, benchmark, limit],
    queryFn: () =>
      apiFetch<CryptoCorrelationPayload>(
        `/api/crypto/correlation/${symbol}?benchmark=${benchmark}&limit=${limit}`,
      ),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
}

export function useCryptoOpenInterest(symbol = "BTC-USD", period = "1h", limit = 100) {
  return useQuery({
    queryKey: ["crypto-open-interest", symbol, period, limit],
    queryFn: () =>
      apiFetch<CryptoOpenInterestPayload>(
        `/api/crypto/open-interest/${symbol}?period=${period}&limit=${limit}`,
      ),
    staleTime: 300_000,
    refetchInterval: 300_000,
  });
}

export function useCryptoMarketStructure(interval = "1h", limit = 168) {
  return useQuery({
    queryKey: ["crypto-market-structure", interval, limit],
    queryFn: () =>
      apiFetch<CryptoMarketStructurePayload>(
        `/api/crypto/market-structure?interval=${interval}&limit=${limit}`,
      ),
    staleTime: 30_000,
    refetchInterval: 30_000,
  });
}

export function useVixMetrics() {
  return useQuery({
    queryKey: ["vix-metrics"],
    queryFn: () => apiFetch<VixMetricsPayload>("/api/vix-metrics"),
    staleTime: 2 * 60 * 1000,
  });
}

export function useFearGreed() {
  return useQuery({
    queryKey: ["market-sentiment"],
    queryFn: () => apiFetch<FearGreedPayload>("/api/market-sentiment"),
    staleTime: 5 * 60 * 1000,
  });
}

export function useUpdateTradeNotes() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ tradeId, notes }: { tradeId: number; notes: string }) => {
      return apiFetch<{ ok: boolean; trade_id: number; notes: string }>(
        `/api/paper-trades/${tradeId}/notes`,
        {
          method: "PATCH",
          body: JSON.stringify({ notes }),
        },
      );
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["paper-trades"] });
      void queryClient.invalidateQueries({ queryKey: ["paper-trades-summary"] });
    },
  });
}

export function useAlerts(limit: number = 50) {
  return useQuery({
    queryKey: ["alerts", limit],
    queryFn: () => apiFetch<AlertsPayload>(`/api/alerts?limit=${limit}`),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useMacroLive() {
  return useQuery({
    queryKey: ["macro-live"],
    queryFn: () => apiFetch<MacroLivePayload>("/api/macro-live"),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useTriggerUpdate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ mode, apiKey }: { mode: "full" | "yolo" | "report"; apiKey: string }) => {
      const result = await apiFetch<{ ok: boolean; detail?: string; message?: string }>(
        `/api/admin/trigger-update?mode=${encodeURIComponent(mode)}`,
        {
          method: "POST",
          headers: { "X-API-Key": apiKey },
        },
      );
      return typeof result.detail === "string"
        ? result.detail
        : typeof result.message === "string"
          ? result.message
          : "Triggered successfully";
    },
    onSuccess: () => {
      // Auto-refresh pipeline status after triggering
      void queryClient.invalidateQueries({ queryKey: ["pipeline-status"] });
    },
  });
}

/* ── Pipeline / Status ── */
export interface PipelineStatus {
  pipeline: {
    stage: string;
  };
  pipeline_stage?: string;
  latest_run: {
    status: string;
    tickers_processed: number | null;
    tickers_total: number | null;
  };
  counts?: {
    tracked_tickers: number | null;
  };
  freshness?: {
    price_age_days: number | null;
  };
}

/* ── Daily Report ── */
export interface ReportCounts {
  tracked_tickers: number | null;
  price_rows: number | null;
}

export interface YoloTimeframe {
  timeframe: string;
  tickers_with_patterns: number | null;
}

export interface YoloSummary {
  rows_total: number | null;
  tickers_with_patterns: number | null;
}

export interface YoloBlock {
  summary: YoloSummary;
  timeframes: YoloTimeframe[];
}

export interface IngestRun {
  status: string | null;
}

export interface RiskCondition {
  severity: string;
  reason: string;
  code?: string;
}

export interface RiskFilters {
  trade_mode: string;
  hard_blocks: number;
  soft_flags: number;
  conditions: RiskCondition[];
}

export interface KeyChange {
  title: string;
  detail: string;
}

export interface SetupEvaluationOverall {
  hit_rate_pct: number | null;
  avg_signed_return_pct: number | null;
  expectancy_pct: number | null;
}

export interface SetupEvalFamily {
  setup_family: string;
  call_direction: string;
  calls: number;
  hit_rate_pct: number | null;
  avg_signed_return_pct: number | null;
  expectancy_pct: number | null;
}

export interface SetupEvalValidity {
  validity_days: number;
  calls: number;
  hit_rate_pct: number | null;
  avg_signed_return_pct: number | null;
  expectancy_pct: number | null;
}

export interface SetupEvalAction {
  priority: string;
  scope: string;
  reason: string;
  recommendation: string;
}

export interface SetupEvaluation {
  enabled: boolean;
  error?: string;
  reason?: string;
  overall: SetupEvaluationOverall;
  by_family: SetupEvalFamily[];
  by_validity_days: SetupEvalValidity[];
  improvement_actions: SetupEvalAction[];
  window_days: number;
  min_sample: number;
  scored_calls: number;
  open_calls: number;
  hit_threshold_pct: number;
  latest_scored_asof: string | null;
}

export interface SetupRow {
  ticker: string;
  setup_score: number | null;
  setup_tier: string | null;
  setup_label: string | null;
  bias_label: string | null;
  yolo_context: string | null;
  level_event: string | null;
  observation_short: string | null;
  next_step_short: string | null;
  audit_notes: string | null;
  pct_change?: number | null;
}

export interface RegimeContext {
  vix: VixData;
  health: {
    state: string;
    score: number | null;
    drivers: string[];
    warnings: string[];
  };
  overall: {
    participation_bias: string;
  };
  participation: Array<Record<string, unknown>>;
  timeframes: Array<Record<string, unknown>>;
  levels: Array<Record<string, unknown>>;
  ma_matrix: Array<{
    metric: string;
    value_pct: number | null;
    state: string;
    risk_read: string;
  }>;
  comparison: {
    series: Array<Record<string, unknown>>;
    symbols: Record<string, unknown>;
  };
  summary: string;
  llm_commentary: {
    source: string;
    [key: string]: unknown;
  };
  asof_date: string;
  source: string;
}

export interface VixData {
  close: number | null;
  risk_state: string;
  level_source: string;
  percentile_1y?: number | null;
  pct_vs_ma20?: number | null;
  pct_vs_ma50?: number | null;
  pct_vs_ma100?: number | null;
  ma20?: number | null;
  ma50?: number | null;
  ma100?: number | null;
  ma_cross_state?: string;
  ma_state?: string;
}

export interface ReportSignals {
  tonight_key_changes: KeyChange[];
  regime_context: RegimeContext | null;
  setup_quality_top: SetupRow[];
  setup_evaluation: SetupEvaluation | Record<string, never>;
  volatility_context?: Record<string, unknown>;
  market_breadth?: Record<string, unknown>;
}

export interface ReportLatest {
  generated_ts: string | null;
  counts: ReportCounts;
  yolo: YoloBlock;
  latest_ingest_run: IngestRun;
  signals: ReportSignals;
  latest_data?: { price_date: string | null };
  risk_filters: RiskFilters;
}

export interface DailyReportPayload {
  ok: boolean;
  latest: ReportLatest;
}

/* ── Dashboard / Chart ── */
export interface OhlcvRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface LevelRow {
  type: string;
  level: number;
  zone_low: number;
  zone_high: number;
  tier: string;
  touches: number;
  last_touch_date: string;
}

export interface GapRow {
  date: string;
  type: string;
  gap_low: number;
  gap_high: number;
}

export interface TrendlineRow {
  type: string;
  touch_count: number;
  score: number;
  last_touch_date: string;
}

export interface PatternRow {
  pattern: string;
  status: string;
  confidence: number;
  start_date: string;
  end_date: string;
}

export interface HybridPatternRow {
  pattern: string;
  status: string;
  hybrid_confidence: number;
  base_confidence: number;
  candle_score: number;
  volume_score: number;
  breakout_score: number;
  candle_bias: string;
  vol_ratio: number;
  start_date: string;
  end_date: string;
}

export interface PatternOverlayRow {
  source: string;
  class_name: string;
  status: string;
  confidence: number;
  start_date: string;
  end_date: string;
}

export interface CandlestickPatternRow {
  date: string;
  pattern: string;
  bias: string;
  confidence: number;
  explanation: string;
}

export interface YoloAuditRow {
  timeframe: string;
  pattern: string;
  signal_role: string;
  active_now: boolean;
  yolo_recency: string | null;
  confirmation_trend: string | null;
  lifecycle_state: string | null;
  latest_close_in_pattern: boolean | null;
  age_days: number | null;
  current_streak: number | null;
  snapshots_seen: number | null;
  first_seen_asof: string | null;
  last_seen_asof: string | null;
  confidence_delta: number | null;
  confidence: number | null;
}

export interface ChartCommentary {
  setup_tier: string | null;
  signal_bias: string | null;
  actionability: string | null;
  primary_yolo_recency: string | null;
  primary_yolo_role: string | null;
  observation: string | null;
  action: string | null;
  risk_note: string | null;
  technical_read: string | null;
  asof: string | null;
  narrative_source: string | null;
  yolo_direction_conflict: boolean;
  debate_consensus_state: string | null;
  debate_agreement_score: number | null;
  debate_disagreement_count: number | null;
  debate_v1?: {
    version: number;
    consensus: {
      consensus_state: string;
      consensus_bias: string;
      agreement_score: number;
      disagreement_count: number;
    };
    roles: Array<{
      role: string;
      stance: string;
      confidence: number;
      evidence: string[];
    }>;
  };
  narrative_trace?: Record<string, unknown>;
  llm_ready_prompt?: string;
}

export interface Fundamentals {
  price: number | null;
  pe: number | null;
  peg: number | null;
  target_price: number | null;
  discount_pct: number | null;
}

export interface OptionsSummary {
  put_call_oi_ratio: number | null;
}

export interface EarningsMarker {
  date: string;
  session: string;
}

export interface DashboardPayload {
  ticker: string;
  asof: string;
  report_generated_ts: string | null;
  chart: OhlcvRow[];
  levels: LevelRow[];
  gaps: GapRow[];
  trendlines: TrendlineRow[];
  patterns: PatternRow[];
  hybrid_patterns: HybridPatternRow[];
  pattern_overlays: PatternOverlayRow[];
  candlestick_patterns: CandlestickPatternRow[];
  yolo_audit: YoloAuditRow[];
  yolo_patterns: Array<Record<string, unknown>>;
  chart_commentary: ChartCommentary;
  fundamentals: Fundamentals;
  options_summary: OptionsSummary;
  earnings_markers: EarningsMarker[];
}

/* ── Opportunities ── */
export interface OpportunityRow {
  ticker: string;
  price: number | null;
  pe: number | null;
  peg: number | null;
  target_price: number | null;
  discount_pct: number | null;
  valuation_label: string | null;
  eps_growth_5y: number | null;
}

export interface OpportunitiesPayload {
  rows: OpportunityRow[];
  filter_help: string | null;
  universe_count: number | null;
  eligible_count: number | null;
}

/* ── Earnings ── */
export interface EarningsRow {
  ticker: string;
  earnings_date: string;
  earnings_session: string;
  schedule_quality: string | null;
  days_until: number | null;
  recommendation_state: string | null;
  score: number | null;
  signal_bias: string | null;
  earnings_risk: string | null;
  yolo_pattern: string | null;
  sector: string | null;
  price: number | null;
  discount_pct: number | null;
  peg: number | null;
  observation: string | null;
  action: string | null;
}

export interface EarningsGroup {
  date: string;
  rows: EarningsRow[];
}

export interface EarningsSummary {
  setup_ready: number;
  watch: number;
  calendar_only: number;
  unverified: number;
}

export interface EarningsPayload {
  rows: EarningsRow[];
  groups: EarningsGroup[];
  summary: EarningsSummary;
  provider_status: { detail: string };
  count: number;
  market_date: string;
  detail: string | null;
  provider: string;
}

/* ── Paper Trades ── */
export interface PaperTrade {
  ticker: string;
  direction: string;
  entry_price: number | null;
  current_price: number | null;
  stop_loss: number | null;
  target_price: number | null;
  pnl_pct: number | null;
  unrealized_pnl_pct: number | null;
  r_multiple: number | null;
  status: string;
  setup_family: string | null;
  setup_tier: string | null;
  entry_date: string | null;
}

export interface PaperTradeList {
  trades: PaperTrade[];
}

export interface PaperTradeSummaryOverall {
  total_trades: number | null;
  win_rate_pct: number | null;
  avg_pnl_pct: number | null;
  total_pnl_pct: number | null;
  avg_r_multiple: number | null;
}

export interface PaperTradeDirectionStats {
  total: number;
  win_rate_pct: number;
  avg_pnl_pct: number;
}

export interface EquityCurvePoint {
  date: string;
  equity_index: number;
}

export interface PaperTradeSummary {
  overall: PaperTradeSummaryOverall;
  by_direction: Record<string, PaperTradeDirectionStats>;
  by_exit_reason: Record<string, number>;
  equity_curve: EquityCurvePoint[];
}

/* ── Market Summary ── */
export interface MarketTickerSummary {
  price: number;
  change_pct_1d: number;
  change_pct_period: number;
  history: Array<{ date: string; close: number }>;
}

export interface MarketSummary {
  as_of: string | null;
  tickers: Record<string, MarketTickerSummary | null>;
}

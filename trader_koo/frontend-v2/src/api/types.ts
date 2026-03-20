/* ── Pipeline / Status ── */
export interface PipelineStatus {
  ok: boolean;
  service: string;
  now_utc: string;
  db_exists: boolean;
  warnings: string[];
  warning_count: number;
  pipeline_active: boolean;
  pipeline_stage: string;
  pipeline: {
    active: boolean;
    stage: string;
    stage_line: string | null;
    stage_line_ts: string | null;
    stage_age_sec: number | null;
    stale_timeout_sec: number | null;
    stale_inference: string | null;
    running_age_sec: number | null;
    running_stale_min: number | null;
    running_stale: boolean | null;
    last_completed_stage: string | null;
    last_completed_status: string | null;
    last_completed_line: string | null;
    last_completed_ts: string | null;
    post_ingest_resume: unknown | null;
    run_log_path: string;
  };
  latest_run: {
    run_id: string | null;
    started_ts: string | null;
    finished_ts: string | null;
    status: string;
    tickers_total: number | null;
    tickers_ok: number | null;
    tickers_failed: number | null;
    tickers_processed: number | null;
    error_message: string | null;
  } | null;
  counts: {
    tracked_tickers: number;
    price_rows: number;
    fundamentals_rows: number;
    options_rows: number;
  };
  freshness: {
    price_age_days: number | null;
    fund_age_hours: number | null;
    opt_age_hours: number | null;
  };
  errors: {
    failed_runs_7d: number;
    latest_error_message: string | null;
    latest_error_ts: string | null;
    latest_failed_run: Record<string, unknown> | null;
  };
  activity: {
    tracked_tickers: number;
    tickers_processed: number;
    tickers_total: number;
    tickers_ok: number;
    tickers_failed: number;
    price_rows: number;
    fundamentals_rows: number;
    options_rows: number;
  };
  latest_data: {
    price_date: string | null;
    fund_snapshot: string | null;
    options_snapshot: string | null;
  };
  llm: Record<string, unknown>;
  service_meta: {
    service: string;
    contract: string;
    contract_version: string;
    version: string;
    auth_header: string;
    admin_auth_configured: boolean;
    runtime_started_ts: string;
    base_url?: string;
    app_url?: string;
    repo_url?: string;
    git_sha?: string;
    deployed_ts?: string;
    deployment_id?: string;
    actions: Array<{
      id: string;
      label: string;
      description: string;
      method: string;
      path: string;
      confirm_text?: string;
    }>;
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
  score: number | null;
  setup_tier: string | null;
  setup_family: string | null;
  signal_bias: string | null;
  pct_change: number | null;
  observation: string | null;
  action: string | null;
  risk_note: string | null;
  technical_read: string | null;
  actionability: string | null;
  yolo_pattern: string | null;
  yolo_recency: string | null;
  yolo_bias: string | null;
  yolo_direction_conflict: boolean;
  level_context: string | null;
  debate_consensus_state: string | null;
  debate_agreement_score: number | null;
  debate_consensus_bias: string | null;
  debate_disagreement_count: number | null;
  narrative_source: string | null;
  discount_pct: number | null;
  peg: number | null;
  sector: string | null;
  [key: string]: unknown;
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
    observation: string;
    action: string;
    risk_note: string;
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
  latest_data?: {
    price_date: string | null;
    fund_snapshot?: string | null;
    options_snapshot?: string | null;
    yolo_detected_ts?: string | null;
  };
  risk_filters: RiskFilters;
  warnings?: string[];
  freshness?: {
    price_age_days: number | null;
    fund_age_hours?: number | null;
    opt_age_hours?: number | null;
    yolo_age_hours?: number | null;
  };
  market_session?: {
    market_date: string;
    market_day: string;
    market_open: boolean;
  };
}

export interface DailyReportPayload {
  ok: boolean;
  latest: ReportLatest;
  history?: Array<{
    file: string;
    size_bytes: number;
    modified_ts: string;
  }>;
  detail: string | null;
  pipeline?: {
    active: boolean | null;
    stage: string | null;
  };
  latest_markdown?: string;
  vix_metrics?: VixMetrics | null;
}

/* ── HMM Regime ── */
export interface HmmRegimeDay {
  date: string;
  state: number;
  label: string;
  color: string;
  prob_low: number;
  prob_normal: number;
  prob_high: number;
}

export interface HmmRegime {
  regimes: HmmRegimeDay[];
  current_state: string;
  current_probs: Record<string, number>;
  transition_risk_pct: number | null;
  transition_matrix: number[][];
  days_in_current: number;
}

/* ── Dashboard / Chart ── */
export interface OhlcvRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  ma20?: number | null;
  ma50?: number | null;
  ma100?: number | null;
  ma200?: number | null;
  atr?: number | null;
  atr_pct?: number | null;
}

export interface LevelRow {
  type: string;
  level: number;
  zone_low: number;
  zone_high: number;
  tier: string;
  touches: number;
  last_touch_date: string;
  source?: string;
  distance_pct?: number;
}

export interface GapRow {
  date: string;
  type: string;
  gap_low: number;
  gap_high: number;
}

export interface TrendlineRow {
  type: string;
  x0_date?: string;
  x1_date?: string;
  y0?: number;
  y1?: number;
  slope?: number;
  intercept?: number;
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
  x0_date?: string;
  x1_date?: string;
  y0?: number;
  y1?: number;
  y0b?: number;
  y1b?: number;
  notes?: string;
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
  ticker: string;
  asof: string | null;
  score: number | null;
  setup_tier: string | null;
  setup_family: string | null;
  signal_bias: string | null;
  actionability: string | null;
  observation: string | null;
  action: string | null;
  risk_note: string | null;
  technical_read: string | null;
  narrative_source: string | null;
  primary_yolo_role: string | null;
  primary_yolo_recency: string | null;
  primary_yolo_confirmation_trend: string | null;
  primary_yolo_lifecycle_state: string | null;
  primary_yolo_latest_close_in_pattern: boolean | null;
  yolo_bias: string | null;
  yolo_direction_conflict: boolean;
  yolo_conflict_strength: number | null;
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
  debate_consensus_state: string | null;
  debate_consensus_bias: string | null;
  debate_agreement_score: number | null;
  debate_disagreement_count: number | null;
  debate_safety_adjustment: number | null;
  commentary_context?: Record<string, unknown>;
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

export interface YoloPatternRow {
  timeframe: string;
  pattern: string;
  confidence: number;
  age_days: number | null;
  signal_role: string | null;
  first_seen_asof: string | null;
  last_seen_asof: string | null;
  snapshots_seen: number | null;
  current_streak: number | null;
  x0_date?: string;
  x1_date?: string;
  y0?: number;
  y1?: number;
}

export interface LiveCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  tick_count: number;
  forming: boolean;
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
  yolo_patterns: YoloPatternRow[];
  chart_commentary: ChartCommentary;
  fundamentals: Fundamentals;
  options_summary: OptionsSummary;
  earnings_markers: EarningsMarker[];
  hmm_regime: HmmRegime | null;
  cv_proxy_patterns?: PatternOverlayRow[];
  hybrid_cv_compare?: Record<string, unknown>[];
  data_sources?: Record<string, unknown>;
  live_candle?: LiveCandle;
}

/* ── Opportunities ── */
export interface OpportunityRow {
  ticker: string;
  price: number | null;
  pe: number | null;
  peg: number | null;
  eps_ttm: number | null;
  eps_growth_5y: number | null;
  target_price: number | null;
  discount_pct: number | null;
  target_reason: string | null;
  target_source: string | null;
  valuation_label: string | null;
}

export interface OpportunitiesPayload {
  rows: OpportunityRow[];
  count: number;
  universe_count: number | null;
  eligible_count: number | null;
  snapshot_ts: string | null;
  filter_help: Record<string, string> | null;
  source_counts?: Record<string, number>;
  filters?: {
    view: string;
    min_discount: number;
    max_peg: number;
    overvalued_threshold: number;
    limit: number;
  };
}

/* ── Earnings ── */
export interface EarningsRow {
  ticker: string;
  company_name: string | null;
  earnings_date: string;
  display_date: string;
  earnings_session: string;
  earnings_session_label: string;
  schedule_quality: string | null;
  days_until: number | null;
  recommendation_state: string | null;
  recommendation_note: string | null;
  recommendation_ready: boolean;
  score: number | null;
  signal_bias: string | null;
  actionability: string | null;
  earnings_risk: string | null;
  earnings_risk_note: string | null;
  yolo_pattern: string | null;
  sector: string | null;
  industry: string | null;
  price: number | null;
  discount_pct: number | null;
  peg: number | null;
  observation: string | null;
  action: string | null;
  technical_read: string | null;
  risk_note: string | null;
  source: string | null;
  provider: string | null;
  tracked: boolean;
  setup_tier: string | null;
  setup_family: string | null;
}

export interface EarningsGroupSession {
  session: string;
  label: string;
  rows: EarningsRow[];
}

export interface EarningsGroup {
  date: string;
  display_date: string;
  count: number;
  sessions: EarningsGroupSession[];
}

export interface EarningsSummary {
  window_days: number;
  total_events: number;
  high_risk: number;
  elevated_risk: number;
  setup_ready: number;
  watch: number;
  calendar_only: number;
  unverified: number;
  by_session: Record<string, number>;
  scored_rows: number;
}

export interface EarningsPayload {
  ok: boolean;
  rows: EarningsRow[];
  groups: EarningsGroup[];
  summary: EarningsSummary;
  provider_status: Record<string, unknown>;
  count: number;
  market_date: string;
  detail: string | null;
  provider: string;
  snapshot_ts: string | null;
  universe_count: number | null;
}

/* ── Paper Trades ── */
export interface PaperTrade {
  id: number;
  report_date: string | null;
  ticker: string;
  direction: string;
  entry_price: number | null;
  entry_date: string | null;
  target_price: number | null;
  stop_loss: number | null;
  exit_price: number | null;
  exit_date: string | null;
  exit_reason: string | null;
  status: string;
  current_price: number | null;
  unrealized_pnl_pct: number | null;
  pnl_pct: number | null;
  r_multiple: number | null;
  setup_family: string | null;
  setup_tier: string | null;
  score: number | null;
  signal_bias: string | null;
  actionability: string | null;
  observation: string | null;
  action_text: string | null;
  risk_note: string | null;
  debate_agreement_score: number | null;
  high_water_mark: number | null;
  low_water_mark: number | null;
  decision_version?: string | null;
  decision_state?: string | null;
  analyst_stage?: string | null;
  debate_stage?: string | null;
  risk_stage?: string | null;
  portfolio_decision?: string | null;
  decision_summary?: string | null;
  decision_reasons?: string[];
  risk_flags?: string[];
  position_size_pct?: number | null;
  risk_budget_pct?: number | null;
  stop_distance_pct?: number | null;
  expected_reward_pct?: number | null;
  expected_r_multiple?: number | null;
  entry_plan?: string | null;
  exit_plan?: string | null;
  sizing_summary?: string | null;
  review_status?: string | null;
  review_summary?: string | null;
  ml_predicted_win_prob?: number | null;
  notes?: string | null;
}

export interface PaperTradeList {
  ok: boolean;
  count: number;
  trades: PaperTrade[];
}

export interface PaperTradeSummaryOverall {
  total_trades: number;
  open_count: number;
  wins?: number;
  losses?: number;
  win_rate_pct: number | null;
  avg_pnl_pct: number | null;
  expectancy_pct?: number | null;
  total_pnl_pct: number | null;
  avg_r_multiple: number | null;
  best_trade_pct?: number | null;
  worst_trade_pct?: number | null;
  avg_win_pct?: number | null;
  avg_loss_pct?: number | null;
  payoff_ratio?: number | null;
  profit_factor?: number | null;
  target_hit_rate_pct?: number | null;
  stopped_out_rate_pct?: number | null;
  starting_capital?: number;
  portfolio_value?: number;
  realized_pnl?: number;
  unrealized_pnl?: number;
  total_return_pct?: number;
}

export interface PaperTradeDirectionStats {
  total: number;
  wins: number;
  win_rate_pct: number;
  avg_pnl_pct: number;
}

export interface PaperTradePolicy {
  bot_version: string;
  decision_version: string;
  min_tier: string;
  min_score: number;
  max_open: number;
  expiry_days: number;
  min_reward_r_multiple: number;
  high_vol_atr_pct: number;
  qualifying_tiers: string[];
  qualifying_actionability: string[];
  position_size_pct: Record<string, number>;
  caution_position_scale: number;
  high_vol_position_scale: number;
  earnings_position_scale: number;
}

export interface PaperTradeFeedbackItem {
  kind: string;
  severity: string;
  title: string;
  detail: string;
  action: string;
}

export interface PaperTradeEdgeRow {
  trade_count: number;
  wins: number;
  win_rate_pct: number;
  avg_pnl_pct: number;
  avg_r_multiple: number | null;
  window_days: number;
}

export interface PaperTradeFamilyEdgeRow extends PaperTradeEdgeRow {
  setup_family: string;
  direction: string;
  losses: number;
  total_pnl_pct: number;
  best_r: number | null;
  worst_r: number | null;
  target_hit_rate_pct: number;
  stopped_out_rate_pct: number;
  edge_label: string;
  bot_version?: string | null;
}

export interface PaperTradeRegimeEdgeRow extends PaperTradeEdgeRow {
  regime: string;
}

export interface PaperTradeVixBucketEdgeRow extends PaperTradeEdgeRow {
  vix_bucket: string;
}

export interface EquityCurvePoint {
  date: string;
  equity_index: number;
}

export interface PaperTradeSummary {
  ok: boolean;
  overall: PaperTradeSummaryOverall;
  by_direction: Record<string, PaperTradeDirectionStats>;
  by_family: Record<string, PaperTradeDirectionStats>;
  by_tier: Record<string, PaperTradeDirectionStats>;
  by_exit_reason: Record<string, number>;
  equity_curve: EquityCurvePoint[];
  recent_trades: PaperTrade[];
  policy?: PaperTradePolicy | null;
  feedback?: PaperTradeFeedbackItem[];
  family_edges?: PaperTradeFamilyEdgeRow[];
  regime_edges?: PaperTradeRegimeEdgeRow[];
  vix_bucket_edges?: PaperTradeVixBucketEdgeRow[];
}

/* ── Crypto ── */
export interface CryptoPrice {
  symbol: string;
  price: number;
  volume_24h: number;
  change_pct_24h: number;
  timestamp: string;
}

export interface CryptoBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CryptoSummary {
  prices: Record<string, CryptoPrice>;
  connected: boolean;
}

export interface CryptoHistoryPayload {
  ok: boolean;
  symbol: string;
  interval: string;
  count: number;
  bars: CryptoBar[];
  forming_candle?: CryptoBar | null;
}

export interface CryptoIndicators {
  sma_20: number | null;
  sma_50: number | null;
  rsi_14: number | null;
  macd: {
    macd: number | null;
    signal: number | null;
    histogram: number | null;
  };
  bollinger: {
    upper: number | null;
    middle: number | null;
    lower: number | null;
    width: number | null;
  };
  vwap: number | null;
}

export interface CryptoIndicatorsPayload {
  ok: boolean;
  symbol: string;
  indicators: CryptoIndicators;
  bar_count: number;
}

export interface CryptoStructureContext {
  latest_close: number | null;
  support_level: number | null;
  support_zone_low: number | null;
  support_zone_high: number | null;
  resistance_level: number | null;
  resistance_zone_low: number | null;
  resistance_zone_high: number | null;
  pct_to_support: number | null;
  pct_to_resistance: number | null;
  range_position: number | null;
  level_context: string;
  ma_trend: string;
  ma20: number | null;
  ma50: number | null;
  atr_pct: number | null;
  momentum_20: number | null;
  realized_vol_20: number | null;
}

export interface CryptoStructurePayload {
  ok: boolean;
  symbol: string;
  interval: string;
  bar_count: number;
  levels: LevelRow[];
  trendlines: TrendlineRow[];
  hmm_regime: HmmRegime | null;
  context: CryptoStructureContext;
}

export interface CryptoOpenInterestBar {
  timestamp: string;
  open_interest: number;
  open_interest_value: number;
}

export interface CryptoOpenInterestPayload {
  ok: boolean;
  symbol: string;
  period: string;
  count: number;
  oi_bars: CryptoOpenInterestBar[];
  current_oi: {
    symbol: string;
    open_interest: number;
    timestamp: string;
  } | null;
  oi_change_24h_pct: number | null;
}

export interface CryptoCorrelationWindow {
  window_days: number;
  sample_size: number;
  correlation: number | null;
  beta: number | null;
  asset_return_pct: number | null;
  benchmark_return_pct: number | null;
  relative_performance_pct: number | null;
  ratio_zscore: number | null;
}

export interface CryptoCorrelationHistoryPoint {
  date: string;
  asset_close: number;
  benchmark_close: number;
  asset_rebased: number;
  benchmark_rebased: number;
}

export interface CryptoCorrelationPayload {
  ok: boolean;
  symbol: string;
  benchmark: string;
  as_of: string | null;
  sample_size: number;
  relationship_label: string;
  note: string;
  latest: {
    asset_close: number | null;
    benchmark_close: number | null;
    asset_change_1d_pct: number | null;
    benchmark_change_1d_pct: number | null;
  };
  windows: Record<string, CryptoCorrelationWindow>;
  aligned_history: CryptoCorrelationHistoryPoint[];
}

export interface CryptoMarketStructureSymbol {
  symbol: string;
  price: number | null;
  change_pct_24h: number | null;
  ma_trend: string;
  level_context: string;
  support_level: number | null;
  resistance_level: number | null;
  pct_to_support: number | null;
  pct_to_resistance: number | null;
  atr_pct: number | null;
  realized_vol_20: number | null;
  momentum_20: number | null;
  range_position: number | null;
  hmm_state: string | null;
  hmm_confidence: number | null;
}

export interface CryptoMarketStructurePayload {
  ok: boolean;
  interval: string;
  as_of: string | null;
  overview: {
    tracked_symbols: number;
    bullish_trend_count: number;
    bearish_or_mixed_count: number;
    at_support_count: number;
    at_resistance_count: number;
    avg_change_pct_24h: number | null;
    avg_momentum_20: number | null;
    avg_atr_pct: number | null;
    avg_realized_vol_20: number | null;
    volatility_regime: string;
    market_posture: string;
  };
  leaders: CryptoMarketStructureSymbol[];
  laggards: CryptoMarketStructureSymbol[];
  symbols: CryptoMarketStructureSymbol[];
}

/* ── Equity Streaming ── */
export interface EquityTick {
  symbol: string;
  price: number;
  volume: number;
  timestamp: string;
  prev_price?: number | null;
}

export interface StreamingStatus {
  connected: boolean;
  subscribed_count: number;
  max_symbols: number;
  symbols: string[];
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

/* ── VIX Metrics ── */
export interface VixMetrics {
  vix_vix3m_ratio: number | null;
  term_structure_signal: string;
  realized_vol_20d: number | null;
  vol_risk_premium: number | null;
  vol_premium_signal: string;
  vix_percentile_252d: number | null;
  percentile_zone: string;
  above_80th_pctile: boolean;
  vix_daily_change_pct: number | null;
  is_spike: boolean;
  spike_magnitude: string | null;
  recommended_position_pct: number;
  sizing_reason: string;
  gauge_zone: string;
  gauge_color: string;
}

export interface VixMetricsPayload {
  ok: boolean;
  vix_vix3m_ratio: number | null;
  term_structure_signal: string;
  realized_vol_20d: number | null;
  vol_risk_premium: number | null;
  vol_premium_signal: string;
  vix_percentile_252d: number | null;
  percentile_zone: string;
  above_80th_pctile: boolean;
  vix_daily_change_pct: number | null;
  is_spike: boolean;
  spike_magnitude: string | null;
  recommended_position_pct: number;
  sizing_reason: string;
  gauge_zone: string;
  gauge_color: string;
  error?: string;
}

/* ── Market Sentiment ── */
export interface FearGreedComponent {
  name: string;
  score: number | null;
  signal: string;
  detail: string;
}

export interface ExternalSentimentHeadline {
  title: string;
  source: string | null;
  url: string | null;
  time_published: string | null;
  score: number | null;
  label: string | null;
}

export interface ExternalNewsSentiment {
  provider: string;
  source_type: string;
  available: boolean;
  score: number | null;
  raw_score: number | null;
  label: string | null;
  article_count: number;
  updated_at: string | null;
  lookback_hours: number;
  tickers: string[];
  topics: string[];
  note: string;
  headlines: ExternalSentimentHeadline[];
}

export interface SocialSentimentPost {
  title: string;
  subreddit: string;
  url: string | null;
  upvotes: number;
  num_comments: number;
  created_at: string | null;
  excerpt?: string | null;
  raw_score: number;
  sentiment_score: number | null;
  label: string | null;
  bullish_terms: number;
  bearish_terms: number;
  engagement: number;
}

export interface SocialSentimentBreakdown {
  subreddit: string;
  post_count: number;
  avg_sentiment_score: number | null;
  note: string | null;
}

export interface SocialSentiment {
  provider: string;
  source_type: string;
  available: boolean;
  score: number | null;
  raw_score: number | null;
  label: string | null;
  post_count: number;
  subreddit_count: number;
  updated_at: string | null;
  lookback_hours: number;
  subreddits: string[];
  note: string;
  bullish_terms_total: number;
  bearish_terms_total: number;
  posts: SocialSentimentPost[];
  source_breakdown: SocialSentimentBreakdown[];
}

export interface MethodologyMeta {
  version: string;
  internal_model: string;
  configured_weights: Record<string, number>;
  active_weights: Record<string, number>;
  blend_formula: string | null;
  optional_sources: {
    external_news: string;
    social_sentiment: string;
  };
}

export interface FearGreedPayload {
  ok: boolean;
  score: number | null;
  label: string;
  color: string;
  previous_close: number | null;
  one_week_ago: number | null;
  one_month_ago: number | null;
  methodology: string;
  summary: string;
  basis: string[];
  uses_social_sentiment: boolean;
  external_news: ExternalNewsSentiment;
  social_sentiment: SocialSentiment;
  blended_score: number | null;
  blended_label: string | null;
  blended_color: string | null;
  blended_summary: string | null;
  methodology_meta: MethodologyMeta;
  components: FearGreedComponent[];
  error?: string;
}

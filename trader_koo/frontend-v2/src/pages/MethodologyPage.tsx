import { useEffect, useState, useRef, useCallback } from "react";
import {
  Database,
  Eye,
  Brain,
  MessageSquare,
  ShieldCheck,
  PlayCircle,
  Clock,
  TrendingUp,
  BarChart3,
  Activity,
  Layers,
  Zap,
  Target,
  AlertTriangle,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface MethodologyStats {
  ok: boolean;
  tickers_tracked: number;
  patterns_detected_today: number;
  ml_features: number;
  ml_auc: number;
  paper_trades_total: number;
  paper_trades_open: number;
  win_rate: number | null;
  data_sources: number;
}

interface SectionProps {
  children: React.ReactNode;
  className?: string;
}

interface DataSourceCardProps {
  emoji: string;
  name: string;
  provides: string;
  frequency: string;
}

interface AnalystCardProps {
  name: string;
  focus: string;
  outputs: string;
  color: string;
}

interface TimelineStepProps {
  time: string;
  label: string;
  detail: string;
  isLast?: boolean;
}

/* ------------------------------------------------------------------ */
/* Animated section wrapper (fade-in on scroll)                        */
/* ------------------------------------------------------------------ */

function AnimatedSection({ children, className = "" }: SectionProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.unobserve(el);
        }
      },
      { threshold: 0.08 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ${visible ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"} ${className}`}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Section heading                                                     */
/* ------------------------------------------------------------------ */

function SectionHeading({
  step,
  icon: Icon,
  title,
  subtitle,
}: {
  step: number;
  icon: React.ElementType;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="mb-6 flex items-start gap-4">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--accent)]/15 text-[var(--accent)]">
        <Icon size={20} strokeWidth={1.75} />
      </div>
      <div>
        <div className="mb-0.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--accent)]">
          Step {step}
        </div>
        <h2 className="text-lg font-bold text-[var(--text)]">{title}</h2>
        <p className="mt-1 text-sm leading-relaxed text-[var(--muted)]">
          {subtitle}
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Reusable cards                                                      */
/* ------------------------------------------------------------------ */

function GlassCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-[var(--line)] bg-[var(--panel)]/80 p-4 backdrop-blur-sm ${className}`}
    >
      {children}
    </div>
  );
}

function DataSourceCard({ emoji, name, provides, frequency }: DataSourceCardProps) {
  return (
    <GlassCard className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-lg" aria-hidden="true">
          {emoji}
        </span>
        <span className="text-sm font-semibold text-[var(--text)]">{name}</span>
      </div>
      <p className="text-xs leading-relaxed text-[var(--muted)]">{provides}</p>
      <span className="mt-auto text-[11px] font-medium text-[var(--accent)]">
        {frequency}
      </span>
    </GlassCard>
  );
}

function AnalystCard({ name, focus, outputs, color }: AnalystCardProps) {
  return (
    <GlassCard className="flex flex-col gap-2">
      <div
        className="text-xs font-bold uppercase tracking-[0.12em]"
        style={{ color }}
      >
        {name}
      </div>
      <p className="text-xs leading-relaxed text-[var(--muted)]">{focus}</p>
      <p className="mt-auto text-[11px] text-[var(--text)]">{outputs}</p>
    </GlassCard>
  );
}

function TimelineStep({ time, label, detail, isLast = false }: TimelineStepProps) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className="h-3 w-3 rounded-full border-2 border-[var(--accent)] bg-[var(--bg)]" />
        {!isLast && <div className="w-px flex-1 bg-[var(--line)]" />}
      </div>
      <div className={`pb-6 ${isLast ? "pb-0" : ""}`}>
        <span className="text-xs font-semibold tabular-nums text-[var(--accent)]">
          {time}
        </span>
        <div className="text-sm font-medium text-[var(--text)]">{label}</div>
        <p className="text-xs leading-relaxed text-[var(--muted)]">{detail}</p>
      </div>
    </div>
  );
}

function StatBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/50 px-3 py-2 text-center">
      <div className="text-lg font-bold tabular-nums text-[var(--accent)]">
        {value}
      </div>
      <div className="text-[11px] text-[var(--muted)]">{label}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Layer card for pattern detection                                    */
/* ------------------------------------------------------------------ */

function LayerCard({
  layer,
  title,
  description,
}: {
  layer: number;
  title: string;
  description: string;
}) {
  return (
    <div className="flex gap-3 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[var(--accent)]/10 text-xs font-bold text-[var(--accent)]">
        {layer}
      </div>
      <div>
        <div className="text-sm font-semibold text-[var(--text)]">{title}</div>
        <p className="mt-0.5 text-xs leading-relaxed text-[var(--muted)]">
          {description}
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main page component                                                 */
/* ------------------------------------------------------------------ */

export default function MethodologyPage() {
  const [stats, setStats] = useState<MethodologyStats | null>(null);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/methodology-stats");
      if (res.ok) {
        const data: MethodologyStats = await res.json();
        if (data.ok) setStats(data);
      }
    } catch {
      /* non-critical — page works without live stats */
    }
  }, []);

  useEffect(() => {
    document.title = "Methodology \u2014 Trader Koo";
    fetchStats();
  }, [fetchStats]);

  return (
    <div className="mx-auto max-w-4xl space-y-12 pb-16">
      {/* ============================================================ */}
      {/* Hero                                                          */}
      {/* ============================================================ */}
      <AnimatedSection>
        <div className="space-y-4 pt-2">
          <h1 className="text-3xl font-bold tracking-tight text-[var(--text)]">
            How Trader Koo Works
          </h1>
          <p className="max-w-2xl text-sm leading-relaxed text-[var(--muted)]">
            A multi-layered decision pipeline for swing trade analysis &mdash;
            from data ingestion to paper trade execution.
          </p>
          <div className="rounded-xl border-2 border-[var(--amber)]/40 bg-[rgba(248,194,78,0.08)] px-4 py-3">
            <div className="flex items-start gap-2.5">
              <AlertTriangle
                size={16}
                className="mt-0.5 shrink-0 text-[var(--amber)]"
              />
              <p className="text-xs leading-relaxed text-[var(--muted)]">
                For informational and educational purposes only. Not financial
                advice. Past performance does not guarantee future results.
              </p>
            </div>
          </div>

          {/* Live stats ribbon */}
          {stats && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatBadge
                label="Tickers Tracked"
                value={stats.tickers_tracked.toLocaleString()}
              />
              <StatBadge
                label="Patterns Today"
                value={stats.patterns_detected_today.toLocaleString()}
              />
              <StatBadge
                label="ML Features"
                value={String(stats.ml_features)}
              />
              <StatBadge
                label="Data Sources"
                value={String(stats.data_sources)}
              />
            </div>
          )}
        </div>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Section 1: Data Foundation                                    */}
      {/* ============================================================ */}
      <AnimatedSection>
        <GlassCard>
          <SectionHeading
            step={1}
            icon={Database}
            title="Data Foundation"
            subtitle="Seven integrated data sources refresh on schedule to build a comprehensive market picture."
          />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <DataSourceCard
              emoji="📈"
              name="yfinance"
              provides="OHLCV price history for 500+ S&P tickers, splits, dividends"
              frequency="Daily at 22:00 UTC"
            />
            <DataSourceCard
              emoji="📊"
              name="Finviz"
              provides="Fundamentals: P/E, PEG, market cap, sector, analyst targets"
              frequency="Daily at 22:00 UTC"
            />
            <DataSourceCard
              emoji="🔔"
              name="Finnhub"
              provides="Real-time WebSocket quotes, company news feed"
              frequency="Real-time (market hours)"
            />
            <DataSourceCard
              emoji="₿"
              name="Binance"
              provides="BTC/ETH live prices, 1-min candles, 24h volume"
              frequency="Real-time WebSocket"
            />
            <DataSourceCard
              emoji="🏛️"
              name="FRED"
              provides="Yield curve, high-yield OAS, fed funds rate, M2"
              frequency="Daily (macro snapshot)"
            />
            <DataSourceCard
              emoji="🎯"
              name="Polymarket"
              provides="Prediction market odds: Fed cuts, recession, geopolitical"
              frequency="On-demand (page load)"
            />
            <DataSourceCard
              emoji="📰"
              name="Alpha Vantage"
              provides="News sentiment scores, topic-level market sentiment"
              frequency="Daily (optional provider)"
            />
          </div>
        </GlassCard>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Section 2: Pattern Detection                                  */}
      {/* ============================================================ */}
      <AnimatedSection>
        <GlassCard>
          <SectionHeading
            step={2}
            icon={Eye}
            title="Pattern Detection (5 Layers)"
            subtitle="Patterns are scored through multiple independent detection systems that vote together."
          />
          <div className="space-y-3">
            <LayerCard
              layer={1}
              title="Rule-Based Geometric"
              description="Detects flags, wedges, channels, and triangles using slope regression on swing highs/lows with configurable tolerances."
            />
            <LayerCard
              layer={2}
              title="Candlestick Patterns"
              description="Identifies reversal and continuation signals: hammer, morning star, engulfing, doji clusters, and more across 1-3 bar windows."
            />
            <LayerCard
              layer={3}
              title="Hybrid Scoring"
              description="Combines all signals into a weighted composite: 50% rule-based geometry + 20% candlestick + 15% volume confirmation + 15% breakout proximity."
            />
            <LayerCard
              layer={4}
              title="CV Proxy (Consensus)"
              description="An independent geometry scorer re-evaluates the same price data with different parameters, providing a second opinion for consensus validation."
            />
            <LayerCard
              layer={5}
              title="YOLOv8 AI Detection"
              description="Pre-computed dual-timeframe scan (daily 180d + weekly 730d). Price charts are rendered as images, run through a fine-tuned YOLOv8 model, and bounding-box coordinates are mapped back to price/date levels."
            />
          </div>
        </GlassCard>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Section 3: ML Pipeline                                        */}
      {/* ============================================================ */}
      <AnimatedSection>
        <GlassCard>
          <SectionHeading
            step={3}
            icon={Brain}
            title="ML Pipeline"
            subtitle="A LightGBM classifier trained with walk-forward validation filters false positives from pattern detection."
          />
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
                <div className="mb-1 text-xs font-semibold text-[var(--accent)]">
                  Feature Engineering
                </div>
                <p className="text-xs leading-relaxed text-[var(--muted)]">
                  {stats ? stats.ml_features : 51} features across momentum
                  (multi-horizon returns), volatility (ATR, Bollinger width),
                  volume, trend position, VIX regime, cross-sectional ranks,
                  macro indicators, sector rotation, and news sentiment.
                </p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
                <div className="mb-1 text-xs font-semibold text-[var(--accent)]">
                  Labeling Strategy
                </div>
                <p className="text-xs leading-relaxed text-[var(--muted)]">
                  Triple-barrier method: 2x ATR profit target, 2x ATR stop loss,
                  10-day time barrier. A trade is labeled as a win only if the
                  profit target is hit before the stop or time expiry.
                </p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
                <div className="mb-1 text-xs font-semibold text-[var(--accent)]">
                  Training Process
                </div>
                <p className="text-xs leading-relaxed text-[var(--muted)]">
                  LightGBM walk-forward with purged validation windows (no
                  data leakage). The model is retrained periodically on
                  expanding windows. SHAP values provide feature importance for
                  interpretability.
                </p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
                <div className="mb-1 text-xs font-semibold text-[var(--accent)]">
                  Meta-Labeling Filter
                </div>
                <p className="text-xs leading-relaxed text-[var(--muted)]">
                  A secondary model acts as a filter on pattern detections,
                  adjusting position size or vetoing setups that the primary
                  model flags as low-probability.
                </p>
              </div>
            </div>
            <div className="rounded-lg border border-[var(--amber)]/30 bg-[rgba(248,194,78,0.05)] p-3">
              <p className="text-xs leading-relaxed text-[var(--muted)]">
                <span className="font-semibold text-[var(--amber)]">
                  Current AUC: {stats ? stats.ml_auc : 0.5235}
                </span>{" "}
                &mdash; The model is deliberately used as a filter (reject
                low-confidence setups) rather than a standalone signal generator.
                A slight edge in filtering compounds over many trades.
              </p>
            </div>
          </div>
        </GlassCard>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Section 4: Multi-Angle Debate Engine                          */}
      {/* ============================================================ */}
      <AnimatedSection>
        <GlassCard>
          <SectionHeading
            step={4}
            icon={MessageSquare}
            title="Multi-Angle Debate Engine"
            subtitle="Five specialist analysts evaluate each setup from different perspectives, then an arbiter synthesizes a consensus."
          />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <AnalystCard
              name="Trend / Structure"
              focus="Evaluates price trend, support/resistance levels, market structure (higher highs/lows), and relative position to key moving averages."
              outputs="Stance, confidence, S/R levels"
              color="var(--accent)"
            />
            <AnalystCard
              name="Momentum / Participation"
              focus="Checks RSI, MACD, volume trends, breadth confirmation, and whether momentum supports the setup direction."
              outputs="Momentum score, divergence flags"
              color="var(--green)"
            />
            <AnalystCard
              name="Valuation"
              focus="Assesses P/E, PEG, discount-to-target, sector comparisons, and whether price has room to move."
              outputs="Fair value estimate, upside %"
              color="var(--blue, var(--accent))"
            />
            <AnalystCard
              name="Pattern / YOLO"
              focus="Integrates the 5-layer pattern detection results, YOLO AI confidence, and historical pattern reliability statistics."
              outputs="Pattern quality, timeframe alignment"
              color="var(--amber)"
            />
            <AnalystCard
              name="Risk Manager"
              focus="Evaluates VIX regime, earnings proximity, sector risk, correlation clustering, and portfolio-level exposure."
              outputs="Risk flags, position size cap"
              color="var(--red)"
            />
          </div>
          <div className="mt-4 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
            <div className="mb-1 text-xs font-semibold text-[var(--accent)]">
              Arbiter: Deterministic Consensus
            </div>
            <p className="text-xs leading-relaxed text-[var(--muted)]">
              The arbiter computes an agreement score across all five analysts.
              Primary disagreements are surfaced explicitly. When disagreement
              is high, conviction is automatically downgraded regardless of
              individual analyst confidence.
            </p>
          </div>
        </GlassCard>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Section 5: Risk Gating                                        */}
      {/* ============================================================ */}
      <AnimatedSection>
        <GlassCard>
          <SectionHeading
            step={5}
            icon={ShieldCheck}
            title="Risk Gating"
            subtitle="Multiple risk filters must pass before a setup can open a paper trade."
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="flex gap-3 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
              <Activity
                size={18}
                className="mt-0.5 shrink-0 text-[var(--accent)]"
              />
              <div>
                <div className="text-sm font-semibold text-[var(--text)]">
                  VIX Regime (3-State HMM)
                </div>
                <p className="mt-0.5 text-xs leading-relaxed text-[var(--muted)]">
                  Hidden Markov Model classifies the market into low-vol, normal,
                  and high-vol regimes. High-vol regimes reduce position sizes
                  and raise conviction thresholds.
                </p>
              </div>
            </div>
            <div className="flex gap-3 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
              <BarChart3
                size={18}
                className="mt-0.5 shrink-0 text-[var(--accent)]"
              />
              <div>
                <div className="text-sm font-semibold text-[var(--text)]">
                  Fear/Greed Composite
                </div>
                <p className="mt-0.5 text-xs leading-relaxed text-[var(--muted)]">
                  Weighted blend: VIX 30% + breadth 25% + momentum 25% + market
                  strength 15% + put/call 5%. Extreme fear or greed adjusts
                  risk tolerance.
                </p>
              </div>
            </div>
            <div className="flex gap-3 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
              <Layers
                size={18}
                className="mt-0.5 shrink-0 text-[var(--accent)]"
              />
              <div>
                <div className="text-sm font-semibold text-[var(--text)]">
                  ATR-Based Position Sizing
                </div>
                <p className="mt-0.5 text-xs leading-relaxed text-[var(--muted)]">
                  Position size is calibrated to each stock&apos;s volatility using
                  14-day ATR. No single trade risks more than a fixed percentage
                  of the portfolio.
                </p>
              </div>
            </div>
            <div className="flex gap-3 rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
              <Zap
                size={18}
                className="mt-0.5 shrink-0 text-[var(--accent)]"
              />
              <div>
                <div className="text-sm font-semibold text-[var(--text)]">
                  Drawdown Breaker
                </div>
                <p className="mt-0.5 text-xs leading-relaxed text-[var(--muted)]">
                  Stops opening new trades if the portfolio drawdown exceeds a
                  threshold. Existing positions are managed to expiry but no new
                  risk is added.
                </p>
              </div>
            </div>
          </div>
        </GlassCard>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Section 6: Paper Trade Execution                              */}
      {/* ============================================================ */}
      <AnimatedSection>
        <GlassCard>
          <SectionHeading
            step={6}
            icon={PlayCircle}
            title="Paper Trade Execution"
            subtitle="Setups that pass all gates open simulated trades with full lifecycle tracking."
          />
          <div className="space-y-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
                <div className="mb-1 flex items-center gap-2">
                  <Target
                    size={14}
                    className="text-[var(--green)]"
                  />
                  <span className="text-sm font-semibold text-[var(--text)]">
                    Entry
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-[var(--muted)]">
                  Setup passes all gates (pattern + ML filter + debate consensus
                  + risk checks) and a paper trade is opened at the next
                  available price.
                </p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
                <div className="mb-1 flex items-center gap-2">
                  <TrendingUp
                    size={14}
                    className="text-[var(--red)]"
                  />
                  <span className="text-sm font-semibold text-[var(--text)]">
                    Exit (Triple-Barrier)
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-[var(--muted)]">
                  Profit target hit, stop loss hit, or time expiry at 10 days
                  &mdash; whichever comes first. All exit reasons are tracked.
                </p>
              </div>
            </div>
            <div className="rounded-lg border border-[var(--line)] bg-[var(--bg)]/40 p-3">
              <div className="mb-1 text-xs font-semibold text-[var(--accent)]">
                Tracking &amp; Audit
              </div>
              <p className="text-xs leading-relaxed text-[var(--muted)]">
                Daily mark-to-market updates equity curve, unrealized P&amp;L,
                R-multiple, and Sharpe ratio. Every trade records the full
                decision audit trail: which analysts voted, debate consensus
                score, risk gate results, and portfolio context at entry.
              </p>
            </div>
            {/* Live paper trade stats */}
            {stats && stats.paper_trades_total > 0 && (
              <div className="grid grid-cols-3 gap-3">
                <StatBadge
                  label="Total Trades"
                  value={String(stats.paper_trades_total)}
                />
                <StatBadge
                  label="Open Now"
                  value={String(stats.paper_trades_open)}
                />
                <StatBadge
                  label="Win Rate"
                  value={
                    stats.win_rate !== null
                      ? `${(stats.win_rate * 100).toFixed(1)}%`
                      : "\u2014"
                  }
                />
              </div>
            )}
          </div>
        </GlassCard>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Section 7: Daily Pipeline                                     */}
      {/* ============================================================ */}
      <AnimatedSection>
        <GlassCard>
          <SectionHeading
            step={7}
            icon={Clock}
            title="Daily Pipeline"
            subtitle="Automated nightly orchestration runs Monday through Friday."
          />
          <div className="ml-1">
            <TimelineStep
              time="22:00 UTC"
              label="Market Data Ingestion"
              detail="yfinance OHLCV + Finviz fundamentals for all tracked tickers. Earnings calendar refresh, options IV snapshot."
            />
            <TimelineStep
              time="22:05 UTC"
              label="YOLO Pattern Detection"
              detail="Dual-timeframe batch scan: daily (180-day window) and weekly (730-day window) chart images through YOLOv8."
            />
            <TimelineStep
              time="22:10 UTC"
              label="Report Generation"
              detail="Setup rankings, risk filters, macro context, multi-angle debate for top setups. ML scoring and meta-label filtering."
            />
            <TimelineStep
              time="22:15 UTC"
              label="Email Delivery"
              detail="HTML report emailed with top setups, risk summary, VIX regime, and paper trade performance."
            />
            <TimelineStep
              time="Market Hours"
              label="Real-Time Monitoring"
              detail="Finnhub WebSocket for live quotes. Telegram alerts when tracked setups approach entry zones."
              isLast
            />
          </div>
        </GlassCard>
      </AnimatedSection>

      {/* ============================================================ */}
      {/* Footer                                                        */}
      {/* ============================================================ */}
      <AnimatedSection>
        <div className="border-t border-[var(--line)] pt-6 text-center">
          <p className="text-xs text-[var(--muted)]">
            Built by Koo &middot; Not Financial Advice &middot;{" "}
            <a
              href="https://trader.kooexperience.com"
              className="text-[var(--accent)] transition-colors hover:text-[var(--text)]"
            >
              trader.kooexperience.com
            </a>
          </p>
        </div>
      </AnimatedSection>
    </div>
  );
}

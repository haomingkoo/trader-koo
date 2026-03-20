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
  ChevronDown,
  Play,
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

interface PipelineStep {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  description: string;
}

/* ------------------------------------------------------------------ */
/* Reveal wrapper (IntersectionObserver scroll animation)              */
/* Matches WMSS pattern: opacity .6s ease, transform .6s ease         */
/* with staggered delays via transitionDelay                           */
/* ------------------------------------------------------------------ */

function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          obs.unobserve(el);
        }
      },
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`transition-all duration-[600ms] ease-out ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-7"
      } ${className}`}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Stat card (WMSS pattern: monospace big number + muted label)        */
/* ------------------------------------------------------------------ */

function StatCard({ value, label }: { value: string; label: string }) {
  return (
    <div className="flex-1 min-w-[120px] rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3.5 text-center">
      <div className="text-2xl font-bold font-mono text-[var(--accent)]">
        {value}
      </div>
      <div className="mt-0.5 text-[11px] text-[var(--muted)] uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Viz panel (WMSS pattern: surface bg + border + radial gradient      */
/* top accent line)                                                    */
/* ------------------------------------------------------------------ */

function VizPanel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`relative rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-7 overflow-hidden ${className}`}
    >
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{
          background:
            "radial-gradient(50% 100% at 50% 0%, var(--accent) 0%, transparent 100%)",
          opacity: 0.5,
        }}
      />
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Section heading (WMSS pattern: step label + clamp title + desc)     */
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
    <div className="mb-8">
      <Reveal>
        <div className="flex items-center gap-2.5 mb-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent)]/10 text-[var(--accent)]">
            <Icon size={16} strokeWidth={2} />
          </div>
          <span className="text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
            Step {step}
          </span>
        </div>
      </Reveal>
      <Reveal delay={100}>
        <h2
          className="font-bold tracking-[-0.5px] leading-tight mb-4 text-[var(--text)]"
          style={{ fontSize: "clamp(24px, 4vw, 36px)" }}
        >
          {title}
        </h2>
      </Reveal>
      <Reveal delay={200}>
        <p className="text-[15px] leading-[1.7] text-[var(--muted)] max-w-[640px]">
          {subtitle}
        </p>
      </Reveal>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Pipeline node (WMSS pattern: idle -> active -> done transitions)    */
/* ------------------------------------------------------------------ */

type PipelineNodeState = "idle" | "active" | "done";

function PipelineNode({
  icon: Icon,
  label,
  state,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  state: PipelineNodeState;
}) {
  return (
    <div
      className={`flex flex-col items-center gap-2 transition-all duration-[400ms] ${
        state === "idle" ? "opacity-40 scale-100" : ""
      } ${state === "active" ? "opacity-100 scale-110" : ""} ${
        state === "done" ? "opacity-60 scale-100" : ""
      }`}
    >
      <div
        className={`w-14 h-14 rounded-2xl flex items-center justify-center border-2 transition-all duration-[400ms] ${
          state === "active"
            ? "border-[var(--accent)] shadow-[0_0_20px_rgba(74,158,255,0.25)]"
            : state === "done"
              ? "border-[var(--green)] opacity-70"
              : "border-[var(--line)] bg-[var(--panel)]"
        }`}
        style={
          state === "active"
            ? { background: "rgba(74,158,255,0.1)" }
            : undefined
        }
      >
        <Icon
          className={`h-6 w-6 ${
            state === "active"
              ? "text-[var(--accent)]"
              : state === "done"
                ? "text-[var(--green)]"
                : "text-[var(--muted)]"
          }`}
        />
      </div>
      <span className="text-[11px] font-medium text-center max-w-[80px] leading-tight">
        {label}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Pipeline arrow (WMSS pattern: line + triangle, active state)        */
/* ------------------------------------------------------------------ */

function PipelineArrow({ active }: { active: boolean }) {
  return (
    <div className="items-center flex-shrink-0 hidden sm:flex">
      <div
        className={`w-10 h-0.5 transition-colors duration-[400ms] ${
          active ? "bg-[var(--accent)]" : "bg-[var(--line)]"
        }`}
      />
      <div
        className={`w-0 h-0 border-y-[5px] border-y-transparent border-l-[6px] transition-colors duration-[400ms] ${
          active
            ? "border-l-[var(--accent)]"
            : "border-l-[var(--line)]"
        }`}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Data source card                                                    */
/* ------------------------------------------------------------------ */

function DataSourceCard({
  emoji,
  name,
  provides,
  frequency,
  delay,
}: {
  emoji: string;
  name: string;
  provides: string;
  frequency: string;
  delay: number;
}) {
  return (
    <Reveal delay={delay}>
      <VizPanel className="flex flex-col gap-2 h-full">
        <div className="flex items-center gap-2">
          <span className="text-lg" aria-hidden="true">
            {emoji}
          </span>
          <span className="text-sm font-semibold text-[var(--text)]">
            {name}
          </span>
        </div>
        <p className="text-xs leading-relaxed text-[var(--muted)] flex-1">
          {provides}
        </p>
        <span className="mt-auto text-[11px] font-medium text-[var(--accent)]">
          {frequency}
        </span>
      </VizPanel>
    </Reveal>
  );
}

/* ------------------------------------------------------------------ */
/* Analyst card                                                        */
/* ------------------------------------------------------------------ */

function AnalystCard({
  name,
  focus,
  outputs,
  color,
  delay,
}: {
  name: string;
  focus: string;
  outputs: string;
  color: string;
  delay: number;
}) {
  return (
    <Reveal delay={delay}>
      <VizPanel className="flex flex-col gap-2 h-full">
        <div
          className="text-xs font-bold uppercase tracking-[0.12em]"
          style={{ color }}
        >
          {name}
        </div>
        <p className="text-xs leading-relaxed text-[var(--muted)] flex-1">
          {focus}
        </p>
        <p className="mt-auto text-[11px] text-[var(--text)]">{outputs}</p>
      </VizPanel>
    </Reveal>
  );
}

/* ------------------------------------------------------------------ */
/* Layer card for pattern detection                                    */
/* ------------------------------------------------------------------ */

function LayerCard({
  layer,
  title,
  description,
  delay,
}: {
  layer: number;
  title: string;
  description: string;
  delay: number;
}) {
  return (
    <Reveal delay={delay}>
      <div className="flex gap-3 rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent)]/10 text-sm font-bold text-[var(--accent)]">
          {layer}
        </div>
        <div>
          <div className="text-sm font-semibold text-[var(--text)]">
            {title}
          </div>
          <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
            {description}
          </p>
        </div>
      </div>
    </Reveal>
  );
}

/* ------------------------------------------------------------------ */
/* Timeline step                                                       */
/* ------------------------------------------------------------------ */

function TimelineStep({
  time,
  label,
  detail,
  isLast = false,
  delay,
}: {
  time: string;
  label: string;
  detail: string;
  isLast?: boolean;
  delay: number;
}) {
  return (
    <Reveal delay={delay}>
      <div className="flex gap-4">
        <div className="flex flex-col items-center">
          <div className="h-3 w-3 rounded-full border-2 border-[var(--accent)] bg-[var(--bg)]" />
          {!isLast && <div className="w-px flex-1 bg-[var(--line)]" />}
        </div>
        <div className={isLast ? "pb-0" : "pb-6"}>
          <span className="text-xs font-semibold tabular-nums text-[var(--accent)] font-mono">
            {time}
          </span>
          <div className="text-sm font-medium text-[var(--text)]">{label}</div>
          <p className="text-xs leading-relaxed text-[var(--muted)]">
            {detail}
          </p>
        </div>
      </div>
    </Reveal>
  );
}

/* ------------------------------------------------------------------ */
/* Pipeline steps data                                                 */
/* ------------------------------------------------------------------ */

const PIPELINE_STEPS: PipelineStep[] = [
  {
    icon: Database,
    label: "Data Ingest",
    description:
      "yfinance OHLCV, Finviz fundamentals, FRED macro, Finnhub news, Binance crypto, and Polymarket odds are collected and normalized.",
  },
  {
    icon: Eye,
    label: "Pattern Scan",
    description:
      "5-layer detection: rule-based geometry, candlestick patterns, hybrid scoring, CV proxy consensus, and YOLOv8 AI on chart images.",
  },
  {
    icon: Brain,
    label: "ML Filter",
    description:
      "LightGBM meta-labeling classifier with early stopping, 3 target modes (return sign, barrier, cross-sectional rank), and slim feature sets (7\u201315 features). Used as observation filter, not signal generator (AUC 0.52\u20130.53).",
  },
  {
    icon: MessageSquare,
    label: "Debate Engine",
    description:
      "5 specialist analysts (trend, momentum, valuation, pattern, risk) evaluate each setup. Arbiter synthesizes consensus.",
  },
  {
    icon: ShieldCheck,
    label: "Risk Gate",
    description:
      "VIX regime, Fear/Greed composite, ATR position sizing, and drawdown breaker must all pass before trade entry.",
  },
  {
    icon: Target,
    label: "Paper Trade",
    description:
      "Setup opens a simulated trade with triple-barrier exit (2x ATR profit, 2x ATR stop, 10-day time). Full audit trail recorded.",
  },
];

/* ------------------------------------------------------------------ */
/* Main page component                                                 */
/* ------------------------------------------------------------------ */

export default function MethodologyPage() {
  const [stats, setStats] = useState<MethodologyStats | null>(null);
  const [pipelineStates, setPipelineStates] = useState<PipelineNodeState[]>(
    PIPELINE_STEPS.map(() => "idle"),
  );
  const [pipelineDesc, setPipelineDesc] = useState(
    'Click "Replay Pipeline" to watch the trading decision flow.',
  );
  const [pipelinePlaying, setPipelinePlaying] = useState(false);
  const pipelineRef = useRef<HTMLDivElement>(null);
  const pipelineTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const autoPlayedRef = useRef(false);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/methodology-stats");
      if (res.ok) {
        const data: MethodologyStats = await res.json();
        if (data.ok) setStats(data);
      }
    } catch {
      /* non-critical -- page works without live stats */
    }
  }, []);

  useEffect(() => {
    document.title = "Methodology \u2014 Trader Koo";
    fetchStats();
  }, [fetchStats]);

  /* ---- Pipeline animation (WMSS pattern: sequential with 600ms delays) ---- */
  const playPipeline = useCallback(() => {
    if (pipelinePlaying) return;
    setPipelinePlaying(true);

    setPipelineStates(PIPELINE_STEPS.map(() => "idle"));
    setPipelineDesc(PIPELINE_STEPS[0].description);

    let step = 0;
    pipelineTimerRef.current = setInterval(() => {
      if (step >= PIPELINE_STEPS.length) {
        if (pipelineTimerRef.current) clearInterval(pipelineTimerRef.current);
        setPipelinePlaying(false);
        return;
      }

      setPipelineStates((prev) => {
        const next = [...prev];
        if (step > 0) next[step - 1] = "done";
        next[step] = "active";
        return next;
      });
      setPipelineDesc(PIPELINE_STEPS[step].description);
      step++;
    }, 1800);
  }, [pipelinePlaying]);

  /* Auto-play when pipeline scrolls into view */
  useEffect(() => {
    const el = pipelineRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !autoPlayedRef.current) {
          autoPlayedRef.current = true;
          setTimeout(() => playPipeline(), 400);
        }
      },
      { threshold: 0.3 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [playPipeline]);

  useEffect(() => {
    return () => {
      if (pipelineTimerRef.current) clearInterval(pipelineTimerRef.current);
    };
  }, []);

  return (
    <div className="mx-auto max-w-[900px] pb-16">
      {/* ============================================================ */}
      {/* Hero (near full-viewport)                                     */}
      {/* ============================================================ */}
      <section className="flex flex-col justify-center min-h-[calc(100dvh-120px)] py-16">
        <Reveal>
          <div className="mb-4 text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
            Trader Koo
          </div>
        </Reveal>

        <Reveal delay={100}>
          <h1
            className="font-extrabold tracking-tight leading-[1.1] mb-6"
            style={{
              fontSize: "clamp(32px, 6vw, 56px)",
              background:
                "linear-gradient(135deg, var(--text) 30%, var(--accent) 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            How the Trading
            <br />
            Pipeline Works
          </h1>
        </Reveal>

        <Reveal delay={200}>
          <p className="text-[15px] leading-[1.7] text-[var(--muted)] max-w-[640px] mb-8">
            A multi-layered decision pipeline for swing trade analysis &mdash;
            from data ingestion through pattern detection, ML filtering,
            multi-angle debate, and risk-gated paper trade execution.
          </p>
        </Reveal>

        <Reveal delay={300}>
          <div className="rounded-xl border-2 border-[var(--amber)]/40 bg-[rgba(248,194,78,0.06)] px-5 py-3.5 max-w-[640px]">
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
        </Reveal>

        {/* Live stats ribbon */}
        {stats && (
          <Reveal delay={400}>
            <div className="flex flex-wrap gap-4 mt-10">
              <StatCard
                label="Tickers Tracked"
                value={stats.tickers_tracked.toLocaleString()}
              />
              <StatCard
                label="Patterns Today"
                value={stats.patterns_detected_today.toLocaleString()}
              />
              <StatCard
                label="ML Features"
                value={String(stats.ml_features)}
              />
              <StatCard
                label="Data Sources"
                value={String(stats.data_sources)}
              />
            </div>
          </Reveal>
        )}

        {/* Bobbing scroll hint (WMSS pattern) */}
        <Reveal delay={600}>
          <div
            className="mt-12 flex flex-col items-center"
            style={{ animation: "bobDown 2s ease-in-out infinite" }}
          >
            <span className="text-xs text-[var(--muted)]">
              Scroll to explore
            </span>
            <ChevronDown size={20} className="mt-1 text-[var(--muted)]" />
          </div>
        </Reveal>
      </section>

      {/* ============================================================ */}
      {/* Pipeline Overview (animated flow, WMSS pipeline pattern)      */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={0}
          icon={Layers}
          title="Pipeline Overview"
          subtitle="Six stages transform raw market data into risk-managed paper trades. Watch the flow animate below."
        />

        <Reveal delay={300}>
          <VizPanel className="mt-8">
            <div ref={pipelineRef}>
              {/* Pipeline nodes row */}
              <div className="flex items-center justify-center gap-2 sm:gap-0 flex-wrap py-4">
                {PIPELINE_STEPS.map((step, i) => (
                  <div key={i} className="flex items-center">
                    <PipelineNode
                      icon={step.icon}
                      label={step.label}
                      state={pipelineStates[i]}
                    />
                    {i < PIPELINE_STEPS.length - 1 && (
                      <PipelineArrow
                        active={
                          pipelineStates[i] === "done" ||
                          pipelineStates[i + 1] === "active" ||
                          pipelineStates[i + 1] === "done"
                        }
                      />
                    )}
                  </div>
                ))}
              </div>

              {/* Description area */}
              <div className="mt-6 px-2 min-h-[60px]">
                <p className="text-sm leading-relaxed text-[var(--muted)] text-center max-w-[560px] mx-auto">
                  {pipelineDesc}
                </p>
              </div>

              {/* Play button (WMSS pattern: surface bg with hover accent) */}
              <div className="mt-4 flex justify-center">
                <button
                  onClick={() => {
                    autoPlayedRef.current = true;
                    setPipelineStates(PIPELINE_STEPS.map(() => "idle"));
                    if (pipelineTimerRef.current) {
                      clearInterval(pipelineTimerRef.current);
                      setPipelinePlaying(false);
                    }
                    setTimeout(() => playPipeline(), 100);
                  }}
                  disabled={pipelinePlaying}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all duration-200 border ${
                    pipelinePlaying
                      ? "border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] cursor-not-allowed"
                      : "border-[var(--line)] bg-[var(--panel)] text-[var(--text)] hover:bg-[var(--accent)]/10 hover:border-[var(--accent)]/40"
                  }`}
                >
                  <Play size={14} />
                  {pipelinePlaying ? "Playing..." : "Replay Pipeline"}
                </button>
              </div>
            </div>
          </VizPanel>
        </Reveal>
      </section>

      {/* ============================================================ */}
      {/* Section 1: Data Foundation                                    */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={1}
          icon={Database}
          title="Data Foundation"
          subtitle="Seven integrated data sources refresh on schedule to build a comprehensive market picture."
        />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 mt-8">
          <DataSourceCard
            emoji={"\uD83D\uDCC8"}
            name="yfinance"
            provides="OHLCV price history for 500+ S&P tickers, splits, dividends"
            frequency="Daily at 22:00 UTC"
            delay={0}
          />
          <DataSourceCard
            emoji={"\uD83D\uDCCA"}
            name="Finviz"
            provides="Fundamentals: P/E, PEG, market cap, sector, analyst targets"
            frequency="Daily at 22:00 UTC"
            delay={100}
          />
          <DataSourceCard
            emoji={"\uD83D\uDD14"}
            name="Finnhub"
            provides="Real-time WebSocket quotes, company news feed"
            frequency="Real-time (market hours)"
            delay={200}
          />
          <DataSourceCard
            emoji={"\u20BF"}
            name="Binance"
            provides="BTC/ETH live prices, 1-min candles, 24h volume"
            frequency="Real-time WebSocket"
            delay={300}
          />
          <DataSourceCard
            emoji={"\uD83C\uDFDB\uFE0F"}
            name="FRED"
            provides="Yield curve, high-yield OAS, fed funds rate, M2"
            frequency="Daily (macro snapshot)"
            delay={400}
          />
          <DataSourceCard
            emoji={"\uD83C\uDFAF"}
            name="Polymarket"
            provides="Prediction market odds: Fed cuts, recession, geopolitical"
            frequency="On-demand (page load)"
            delay={500}
          />
          <DataSourceCard
            emoji={"\uD83D\uDCF0"}
            name="Alpha Vantage"
            provides="News sentiment scores, topic-level market sentiment"
            frequency="Daily (optional provider)"
            delay={600}
          />
        </div>
      </section>

      {/* ============================================================ */}
      {/* Section 2: Pattern Detection                                  */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={2}
          icon={Eye}
          title="Pattern Detection (5 Layers)"
          subtitle="Patterns are scored through multiple independent detection systems that vote together."
        />
        <div className="space-y-4 mt-8">
          <LayerCard
            layer={1}
            title="Rule-Based Geometric"
            description="Detects flags, wedges, channels, and triangles using slope regression on swing highs/lows with configurable tolerances."
            delay={0}
          />
          <LayerCard
            layer={2}
            title="Candlestick Patterns"
            description="Identifies reversal and continuation signals: hammer, morning star, engulfing, doji clusters, and more across 1-3 bar windows."
            delay={100}
          />
          <LayerCard
            layer={3}
            title="Hybrid Scoring"
            description="Combines all signals into a weighted composite: 50% rule-based geometry + 20% candlestick + 15% volume confirmation + 15% breakout proximity."
            delay={200}
          />
          <LayerCard
            layer={4}
            title="CV Proxy (Consensus)"
            description="An independent geometry scorer re-evaluates the same price data with different parameters, providing a second opinion for consensus validation."
            delay={300}
          />
          <LayerCard
            layer={5}
            title="YOLOv8 AI Detection"
            description="Pre-computed dual-timeframe scan (daily 180d + weekly 730d). Price charts are rendered as images, run through a fine-tuned YOLOv8 model, and bounding-box coordinates are mapped back to price/date levels."
            delay={400}
          />
        </div>
      </section>

      {/* ============================================================ */}
      {/* Section 3: ML Pipeline                                        */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={3}
          icon={Brain}
          title="ML Pipeline"
          subtitle="A LightGBM classifier trained with walk-forward validation filters false positives from pattern detection."
        />
        <div className="grid gap-4 sm:grid-cols-2 mt-8">
          <Reveal delay={0}>
            <VizPanel className="h-full">
              <div className="mb-2 text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
                Feature Engineering
              </div>
              <p className="text-[13px] leading-[1.7] text-[var(--muted)]">
                Slim default set of 7&ndash;15 cross-sectionally ranked features
                (full 51-feature set available). Includes volume-confirmed
                momentum, ATR expansion, gap percentage, multi-horizon returns,
                volatility, VIX regime, macro indicators, and sector rotation.
                All per-ticker features use cross-sectional rank normalization.
              </p>
            </VizPanel>
          </Reveal>
          <Reveal delay={100}>
            <VizPanel className="h-full">
              <div className="mb-2 text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
                Labeling Strategy
              </div>
              <p className="text-[13px] leading-[1.7] text-[var(--muted)]">
                Triple-barrier method: 2x ATR profit target, 2x ATR stop loss,
                10-day time barrier. Labels check intraday high/low for barrier
                touches (not just close). Three target modes: return sign,
                barrier hit, and cross-sectional rank.
              </p>
            </VizPanel>
          </Reveal>
          <Reveal delay={200}>
            <VizPanel className="h-full">
              <div className="mb-2 text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
                Training Process
              </div>
              <p className="text-[13px] leading-[1.7] text-[var(--muted)]">
                LightGBM walk-forward with purged validation windows (no data
                leakage) and early stopping (50 rounds, lr=0.01, 15 leaves,
                depth 3). The model is retrained periodically on expanding
                windows. SHAP values provide feature importance for
                interpretability.
              </p>
            </VizPanel>
          </Reveal>
          <Reveal delay={300}>
            <VizPanel className="h-full">
              <div className="mb-2 text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
                Meta-Labeling Filter
              </div>
              <p className="text-[13px] leading-[1.7] text-[var(--muted)]">
                A secondary model acts as a filter on pattern detections,
                adjusting position size or vetoing setups that the primary model
                flags as low-probability.
              </p>
            </VizPanel>
          </Reveal>
        </div>

        <Reveal delay={400}>
          <div className="mt-6 rounded-xl border border-[var(--amber)]/30 bg-[rgba(248,194,78,0.04)] p-5">
            <p className="text-[13px] leading-[1.7] text-[var(--muted)]">
              <span className="font-bold font-mono text-[var(--amber)]">
                AUC: {stats ? stats.ml_auc : "0.52\u20130.53"}
              </span>{" "}
              &mdash; The model is deliberately used as an observation filter
              (reject low-confidence setups) rather than a standalone signal
              generator. Honest assessment: the edge is marginal, but even
              slight filtering accuracy compounds over many trades.
            </p>
          </div>
        </Reveal>
      </section>

      {/* ============================================================ */}
      {/* Section 4: Multi-Angle Debate Engine                          */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={4}
          icon={MessageSquare}
          title="Multi-Angle Debate Engine"
          subtitle="Five specialist analysts evaluate each setup from different perspectives, then an arbiter synthesizes a consensus."
        />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 mt-8">
          <AnalystCard
            name="Trend / Structure"
            focus="Evaluates price trend, support/resistance levels, market structure (higher highs/lows), and relative position to key moving averages."
            outputs="Stance, confidence, S/R levels"
            color="var(--accent)"
            delay={0}
          />
          <AnalystCard
            name="Momentum / Participation"
            focus="Checks RSI, MACD, volume trends, breadth confirmation, and whether momentum supports the setup direction."
            outputs="Momentum score, divergence flags"
            color="var(--green)"
            delay={100}
          />
          <AnalystCard
            name="Valuation"
            focus="Assesses P/E, PEG, discount-to-target, sector comparisons, and whether price has room to move."
            outputs="Fair value estimate, upside %"
            color="var(--blue, var(--accent))"
            delay={200}
          />
          <AnalystCard
            name="Pattern / YOLO"
            focus="Integrates the 5-layer pattern detection results, YOLO AI confidence, and historical pattern reliability statistics."
            outputs="Pattern quality, timeframe alignment"
            color="var(--amber)"
            delay={300}
          />
          <AnalystCard
            name="Risk Manager"
            focus="Evaluates VIX regime, earnings proximity, sector risk, correlation clustering, and portfolio-level exposure."
            outputs="Risk flags, position size cap"
            color="var(--red)"
            delay={400}
          />
        </div>

        <Reveal delay={500}>
          <VizPanel className="mt-6">
            <div className="mb-2 text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
              Arbiter: Deterministic Consensus
            </div>
            <p className="text-[13px] leading-[1.7] text-[var(--muted)]">
              The arbiter computes an agreement score across all five analysts.
              Primary disagreements are surfaced explicitly. When disagreement is
              high, conviction is automatically downgraded regardless of
              individual analyst confidence.
            </p>
          </VizPanel>
        </Reveal>
      </section>

      {/* ============================================================ */}
      {/* Section 5: Risk Gating                                        */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={5}
          icon={ShieldCheck}
          title="Risk Gating"
          subtitle="Multiple risk filters must pass before a setup can open a paper trade."
        />
        <div className="grid gap-4 sm:grid-cols-2 mt-8">
          {[
            {
              Icon: Activity,
              title: "VIX Regime (3-State HMM)",
              desc: "Hidden Markov Model classifies the market into low-vol, normal, and high-vol regimes. High-vol regimes reduce position sizes and raise conviction thresholds.",
              delay: 0,
            },
            {
              Icon: BarChart3,
              title: "Fear/Greed Composite",
              desc: "Weighted blend: VIX 30% + breadth 25% + momentum 25% + market strength 15% + put/call 5%. Extreme fear or greed adjusts risk tolerance.",
              delay: 100,
            },
            {
              Icon: Layers,
              title: "ATR-Based Position Sizing",
              desc: "Position size is calibrated to each stock's volatility using 14-day ATR. No single trade risks more than a fixed percentage of the portfolio.",
              delay: 200,
            },
            {
              Icon: Zap,
              title: "Drawdown Breaker",
              desc: "Stops opening new trades if the portfolio drawdown exceeds a threshold. Existing positions are managed to expiry but no new risk is added.",
              delay: 300,
            },
          ].map(({ Icon, title, desc, delay }) => (
            <Reveal key={title} delay={delay}>
              <VizPanel className="h-full">
                <div className="flex gap-3">
                  <Icon
                    size={18}
                    className="mt-0.5 shrink-0 text-[var(--accent)]"
                  />
                  <div>
                    <div className="text-sm font-semibold text-[var(--text)]">
                      {title}
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
                      {desc}
                    </p>
                  </div>
                </div>
              </VizPanel>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ============================================================ */}
      {/* Section 6: Paper Trade Execution                              */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={6}
          icon={PlayCircle}
          title="Paper Trade Execution"
          subtitle="Setups that pass all gates open simulated trades with full lifecycle tracking."
        />
        <div className="grid gap-4 sm:grid-cols-2 mt-8">
          <Reveal delay={0}>
            <VizPanel className="h-full">
              <div className="mb-2 flex items-center gap-2">
                <Target size={14} className="text-[var(--green)]" />
                <span className="text-sm font-semibold text-[var(--text)]">
                  Entry
                </span>
              </div>
              <p className="text-xs leading-relaxed text-[var(--muted)]">
                Setup passes all gates (pattern + ML filter + debate consensus +
                risk checks) and a paper trade is opened at the next available
                price.
              </p>
            </VizPanel>
          </Reveal>
          <Reveal delay={100}>
            <VizPanel className="h-full">
              <div className="mb-2 flex items-center gap-2">
                <TrendingUp size={14} className="text-[var(--red)]" />
                <span className="text-sm font-semibold text-[var(--text)]">
                  Exit (Triple-Barrier)
                </span>
              </div>
              <p className="text-xs leading-relaxed text-[var(--muted)]">
                Profit target hit, stop loss hit, or time expiry at 10 days
                &mdash; whichever comes first. All exit reasons are tracked.
              </p>
            </VizPanel>
          </Reveal>
        </div>

        <Reveal delay={200}>
          <VizPanel className="mt-4">
            <div className="mb-2 text-xs font-bold uppercase tracking-[1.5px] text-[var(--accent)]">
              Tracking &amp; Audit
            </div>
            <p className="text-[13px] leading-[1.7] text-[var(--muted)]">
              Daily mark-to-market updates equity curve, unrealized P&amp;L,
              R-multiple, and Sharpe ratio. Every trade records the full decision
              audit trail: which analysts voted, debate consensus score, risk
              gate results, and portfolio context at entry.
            </p>
          </VizPanel>
        </Reveal>

        {/* Live paper trade stats */}
        {stats && stats.paper_trades_total > 0 && (
          <Reveal delay={300}>
            <div className="flex flex-wrap gap-4 mt-6">
              <StatCard
                label="Total Trades"
                value={String(stats.paper_trades_total)}
              />
              <StatCard
                label="Open Now"
                value={String(stats.paper_trades_open)}
              />
              <StatCard
                label="Win Rate"
                value={
                  stats.win_rate !== null
                    ? `${(stats.win_rate * 100).toFixed(1)}%`
                    : "\u2014"
                }
              />
            </div>
          </Reveal>
        )}
      </section>

      {/* ============================================================ */}
      {/* Section 7: Daily Pipeline                                     */}
      {/* ============================================================ */}
      <section className="py-20 border-t border-[var(--line)]">
        <SectionHeading
          step={7}
          icon={Clock}
          title="Daily Pipeline"
          subtitle="Automated nightly orchestration runs Monday through Friday."
        />
        <VizPanel className="mt-8">
          <div className="ml-1">
            <TimelineStep
              time="22:00 UTC"
              label="Market Data Ingestion"
              detail="yfinance OHLCV + Finviz fundamentals for all tracked tickers. Earnings calendar refresh, options IV snapshot."
              delay={0}
            />
            <TimelineStep
              time="22:05 UTC"
              label="YOLO Pattern Detection"
              detail="Dual-timeframe batch scan: daily (180-day window) and weekly (730-day window) chart images through YOLOv8."
              delay={100}
            />
            <TimelineStep
              time="22:10 UTC"
              label="Report Generation"
              detail="Setup rankings, risk filters, macro context, multi-angle debate for top setups. ML scoring and meta-label filtering."
              delay={200}
            />
            <TimelineStep
              time="22:15 UTC"
              label="Email Delivery"
              detail="HTML report emailed with top setups, risk summary, VIX regime, and paper trade performance."
              delay={300}
            />
            <TimelineStep
              time="Market Hours"
              label="Real-Time Monitoring"
              detail="Finnhub WebSocket for live quotes. Telegram alerts when tracked setups approach entry zones."
              isLast
              delay={400}
            />
          </div>
        </VizPanel>
      </section>

      {/* ============================================================ */}
      {/* Footer                                                        */}
      {/* ============================================================ */}
      <Reveal>
        <div className="border-t border-[var(--line)] pt-10 pb-4 text-center">
          <p className="text-xs text-[var(--muted)]">
            Built by Koo &middot; Not Financial Advice &middot;{" "}
            <a
              href="https://trader.kooexperience.com"
              className="text-[var(--accent)] transition-colors duration-200 hover:text-[var(--text)]"
            >
              trader.kooexperience.com
            </a>
          </p>
        </div>
      </Reveal>
    </div>
  );
}

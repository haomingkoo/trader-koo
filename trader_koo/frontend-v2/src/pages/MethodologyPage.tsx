import { useEffect, useRef, useState, useCallback } from "react";

/* ══════════════════════════════════════════════════════════════════════
   HOOKS
   ══════════════════════════════════════════════════════════════════════ */

function useInView(threshold = 0.15) {
  const ref = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) setInView(true);
      },
      { threshold },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);

  return { ref, inView };
}

/* ══════════════════════════════════════════════════════════════════════
   ANIMATED COUNTER
   ══════════════════════════════════════════════════════════════════════ */

function AnimatedCounter({
  value,
  suffix = "",
  decimals = 0,
}: {
  value: number;
  suffix?: string;
  decimals?: number;
}) {
  const [display, setDisplay] = useState(0);
  const { ref, inView } = useInView(0.3);

  useEffect(() => {
    if (!inView) return;
    let frame: number;
    const duration = 1400;
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(
        decimals > 0
          ? parseFloat((eased * value).toFixed(decimals))
          : Math.round(eased * value),
      );
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [inView, value, decimals]);

  return (
    <span ref={ref} className="tabular-nums">
      {decimals > 0 ? display.toFixed(decimals) : display.toLocaleString()}
      {suffix}
    </span>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   FULL-VIEWPORT SECTION WITH FADE-IN
   ══════════════════════════════════════════════════════════════════════ */

function Section({
  children,
  className = "",
  id,
  full = true,
}: {
  children: React.ReactNode;
  className?: string;
  id?: string;
  full?: boolean;
}) {
  const { ref, inView } = useInView(0.08);
  return (
    <section
      ref={ref}
      id={id}
      className={`relative flex items-center justify-center px-6 py-24 transition-all duration-[1200ms] ease-out ${
        full ? "min-h-screen" : ""
      } ${
        inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-16"
      } ${className}`}
    >
      {children}
    </section>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   STAGGER-ANIMATED ELEMENT
   ══════════════════════════════════════════════════════════════════════ */

function Stagger({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const { ref, inView } = useInView(0.15);
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${
        inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      } ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   STICKY SECTION LABEL
   ══════════════════════════════════════════════════════════════════════ */

function StickyLabel({ text }: { text: string }) {
  return (
    <div className="sticky top-0 z-10 mb-16 backdrop-blur-sm bg-transparent pt-6 pb-2">
      <span className="text-[10px] font-bold uppercase tracking-[0.3em] text-[var(--accent)]/80">
        {text}
      </span>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   FLOW STEP + ARROW (horizontal pipeline)
   ══════════════════════════════════════════════════════════════════════ */

function FlowStep({
  label,
  sublabel,
  delay = 0,
}: {
  label: string;
  sublabel: string;
  delay?: number;
}) {
  const { ref, inView } = useInView(0.2);
  return (
    <div
      ref={ref}
      className={`flex flex-col items-center rounded-2xl border border-[var(--line)] bg-[var(--panel)] px-6 py-5 text-center transition-all duration-700 ease-out min-w-[130px] ${
        inView ? "opacity-100 scale-100" : "opacity-0 scale-90"
      }`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <span className="text-sm font-bold text-[var(--text)]">{label}</span>
      <span className="mt-1.5 text-[11px] text-[var(--muted)]">{sublabel}</span>
    </div>
  );
}

function FlowArrow() {
  return (
    <div className="hidden items-center text-[var(--accent)]/30 lg:flex">
      <div className="h-px w-10 bg-gradient-to-r from-[var(--accent)]/20 to-[var(--accent)]/40" />
      <svg width="10" height="10" viewBox="0 0 10 10" className="text-[var(--accent)]/50">
        <path d="M1 1L8 5L1 9" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   STAT CARD (big number)
   ══════════════════════════════════════════════════════════════════════ */

function StatCard({
  value,
  label,
  delay = 0,
  suffix = "",
  decimals = 0,
}: {
  value: number;
  label: string;
  delay?: number;
  suffix?: string;
  decimals?: number;
}) {
  return (
    <Stagger
      delay={delay}
      className="flex flex-col items-center rounded-2xl border border-[var(--line)] bg-[var(--panel)] px-8 py-10"
    >
      <div className="text-5xl font-bold tracking-tight text-[var(--accent)]">
        <AnimatedCounter value={value} suffix={suffix} decimals={decimals} />
      </div>
      <div className="mt-3 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">
        {label}
      </div>
    </Stagger>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   PIPELINE NODE (vertical stepper)
   ══════════════════════════════════════════════════════════════════════ */

function PipelineNode({
  step,
  title,
  description,
  highlight,
  isLast = false,
  delay = 0,
}: {
  step: number;
  title: string;
  description: string;
  highlight?: string;
  isLast?: boolean;
  delay?: number;
}) {
  const { ref, inView } = useInView(0.15);
  return (
    <div
      ref={ref}
      className={`relative flex gap-8 transition-all duration-800 ease-out ${
        inView ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-10"
      }`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <div className="flex flex-col items-center">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full border-2 border-[var(--accent)] bg-[var(--accent)]/10 text-base font-bold text-[var(--accent)]">
          {step}
        </div>
        {!isLast && (
          <div className="w-px flex-1 bg-gradient-to-b from-[var(--accent)]/40 to-[var(--accent)]/10" />
        )}
      </div>
      <div className="pb-16">
        <h4 className="text-xl font-semibold text-[var(--text)]">{title}</h4>
        <p className="mt-2 max-w-lg text-sm leading-relaxed text-[var(--muted)]">
          {description}
        </p>
        {highlight && (
          <p className="mt-4 max-w-lg rounded-xl border border-[var(--accent)]/20 bg-[var(--accent)]/5 px-5 py-3 text-xs leading-relaxed text-[var(--muted)] italic">
            {highlight}
          </p>
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   GATE STEP (risk gating)
   ══════════════════════════════════════════════════════════════════════ */

function GateStep({
  gate,
  description,
  isLast = false,
  delay = 0,
}: {
  gate: string;
  description: string;
  isLast?: boolean;
  delay?: number;
}) {
  const { ref, inView } = useInView(0.15);
  return (
    <div
      ref={ref}
      className={`flex items-start gap-6 transition-all duration-700 ease-out ${
        inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
      }`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <div className="flex flex-col items-center">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-[var(--green)]/40 bg-[var(--green)]/10">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M3 8.5L6.5 12L13 4"
              stroke="var(--green)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        {!isLast && <div className="h-10 w-px bg-[var(--green)]/15" />}
      </div>
      <div className={isLast ? "pb-0" : "pb-4"}>
        <h4 className="text-base font-semibold text-[var(--text)]">{gate}</h4>
        <p className="mt-1 text-sm leading-relaxed text-[var(--muted)]">
          {description}
        </p>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════
   LIVE STATS HOOK
   ══════════════════════════════════════════════════════════════════════ */

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

function useMethodologyStats() {
  const [stats, setStats] = useState<MethodologyStats | null>(null);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/methodology-stats");
      if (res.ok) {
        const data: MethodologyStats = await res.json();
        if (data.ok) setStats(data);
      }
    } catch {
      /* non-critical */
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return stats;
}

/* ══════════════════════════════════════════════════════════════════════
   DATA
   ══════════════════════════════════════════════════════════════════════ */

const DATA_SOURCES = [
  { name: "yfinance", provides: "OHLCV price history, splits, dividends for 500+ S&P tickers", frequency: "Daily" },
  { name: "Finviz", provides: "Fundamentals, P/E, PEG, market cap, analyst targets, sector data", frequency: "Daily" },
  { name: "Finnhub", provides: "Real-time WebSocket quotes, company news, earnings calendar", frequency: "Real-time" },
  { name: "FRED", provides: "Treasury yields, unemployment, fed funds rate, macro indicators", frequency: "Daily" },
  { name: "Binance", provides: "BTC/ETH WebSocket feed, 1-minute candles, 24h volume", frequency: "Real-time" },
  { name: "Polymarket", provides: "Prediction market odds for Fed cuts, recession, geopolitical events", frequency: "On-demand" },
  { name: "Alpha Vantage", provides: "News sentiment scoring and topic-level market sentiment", frequency: "On-demand" },
];

const PATTERN_LAYERS = [
  {
    title: "Rule-Based Screening",
    description: "Support/resistance levels, gap zones, trendline channels, and moving average crossovers computed from raw price data. Detects flags, wedges, channels, and triangles using slope regression on swing highs and lows.",
  },
  {
    title: "Candlestick Pattern Recognition",
    description: "Identifies reversal and continuation signals across 1-3 bar windows: hammer, morning star, engulfing, doji clusters, and more. Each pattern is weighted by historical reliability.",
  },
  {
    title: "Hybrid Scoring",
    description: "Combines all rule-based signals into a weighted composite: 50% geometry + 20% candlestick + 15% volume confirmation + 15% breakout proximity. An independent scorer re-evaluates with different parameters for consensus.",
  },
  {
    title: "LLM Debate Analysis",
    description: "Multi-angle commentary from five synthetic analysts. Gemini-powered narrative generation with structured schema validation. Sanitized output ensures reliable downstream consumption.",
  },
  {
    title: "YOLOv8 Visual Pattern Detection",
    description: "Render candlestick chart to image. Run object detection model to find head-and-shoulders, double tops, wedges, flags, and more. Map bounding boxes back to price coordinates for actionable levels.",
    highlight: "Uses foduucom/stockmarket-pattern-detection-yolov8 fine-tuned on chart pattern datasets. Dual-timeframe scan: daily (180d) and weekly (730d). Runs nightly on the full universe.",
  },
];

const ML_FEATURES = [
  { category: "Price", items: ["Multi-horizon returns", "Volatility ratios", "Distance from SMA", "Bollinger %B"] },
  { category: "Volume", items: ["Volume z-score", "OBV slope", "Volume-price divergence"] },
  { category: "Technical", items: ["RSI", "MACD signal", "ATR ratio", "Trend position"] },
  { category: "Macro", items: ["VIX level", "VIX term structure", "Yield curve slope"] },
  { category: "Pattern", items: ["YOLO detection count", "Pattern confidence", "Cross-sectional rank"] },
];

const ANALYSTS = [
  { name: "Trend / Structure", role: "Price trend, S/R levels, market structure, MA positioning", color: "var(--accent)" },
  { name: "Momentum", role: "RSI, MACD, volume trends, breadth confirmation, divergences", color: "var(--green)" },
  { name: "Valuation", role: "P/E, PEG, discount-to-target, sector comparisons, fair value", color: "var(--blue)" },
  { name: "Pattern / YOLO", role: "5-layer detection results, AI confidence, pattern reliability", color: "var(--amber)" },
  { name: "Risk Manager", role: "VIX regime, earnings proximity, sector risk, correlation", color: "var(--red)" },
];

const RISK_GATES = [
  { gate: "VIX Regime Check", description: "3-state HMM classifies low-vol, normal, and high-vol. High-vol regimes raise conviction thresholds and reduce position sizes." },
  { gate: "Fear & Greed Filter", description: "Weighted composite: VIX 30% + breadth 25% + momentum 25% + market strength 15% + put/call 5%. Extreme readings adjust risk tolerance." },
  { gate: "Correlation Guard", description: "Limits exposure to correlated positions. No portfolio concentration in a single sector or factor." },
  { gate: "ATR Position Sizing", description: "Each position is calibrated to the stock's 14-day ATR. No single trade risks more than a fixed percentage of the portfolio." },
  { gate: "Drawdown Breaker", description: "Halts new entries when portfolio drawdown exceeds threshold. Existing positions managed to expiry but no new risk is added." },
];

const PIPELINE_STEPS = [
  { time: "22:00", label: "Market Data Ingest", detail: "yfinance OHLCV + Finviz fundamentals for 500+ tickers" },
  { time: "22:03", label: "YOLO Pattern Detection", detail: "Render charts, run YOLOv8, map detections to price levels" },
  { time: "22:10", label: "Report Generation", detail: "LLM debate, ML scoring, setup quality ranking, risk filters" },
  { time: "22:14", label: "Paper Trade MTM", detail: "Mark-to-market all open positions, check exit rules" },
  { time: "22:15", label: "Telegram Alerts", detail: "Push key changes, new setups, and exit notifications" },
];

/* ══════════════════════════════════════════════════════════════════════
   MAIN PAGE
   ══════════════════════════════════════════════════════════════════════ */

export default function MethodologyPage() {
  const stats = useMethodologyStats();

  useEffect(() => {
    document.title = "Methodology \u2014 Trader Koo";
  }, []);

  return (
    <div className="scrollbar-none -mx-4 -mt-3 -mb-4 overflow-y-auto overflow-x-hidden">

      {/* ────────────────────────────────────────────────────────────────
          HERO — full viewport, centered, cinematic
          ──────────────────────────────────────────────────────────────── */}
      <section className="relative flex min-h-screen flex-col items-center justify-center px-6 py-24 bg-gradient-to-b from-[#060a12] via-[#0b0f16] to-[#0d1220]">
        <div className="text-center max-w-4xl">
          <h1 className="text-5xl font-bold tracking-tight text-[var(--text)] sm:text-6xl lg:text-7xl leading-[1.08]">
            How Trader Koo
            <br />
            <span className="bg-gradient-to-r from-[var(--accent)] to-[var(--blue)] bg-clip-text text-transparent">
              Works
            </span>
          </h1>
          <p className="mx-auto mt-8 max-w-xl text-lg leading-relaxed text-[var(--muted)] sm:text-xl">
            A multi-layered decision pipeline
            <br className="hidden sm:block" />
            for swing trade analysis.
          </p>

          {/* Horizontal pipeline summary */}
          <div className="mx-auto mt-20 flex flex-wrap items-center justify-center gap-3 lg:gap-0">
            <FlowStep label="7 Sources" sublabel="Market data" delay={300} />
            <FlowArrow />
            <FlowStep label="5 Layers" sublabel="Pattern detection" delay={500} />
            <FlowArrow />
            <FlowStep label="51 Features" sublabel="ML pipeline" delay={700} />
            <FlowArrow />
            <FlowStep label="5 Analysts" sublabel="Debate engine" delay={900} />
            <FlowArrow />
            <FlowStep label="Paper Trades" sublabel="Risk-gated" delay={1100} />
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 animate-bounce text-[var(--muted)]/60">
          <div className="flex flex-col items-center gap-2">
            <span className="text-[10px] uppercase tracking-[0.25em]">Scroll to explore</span>
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M9 3V15M9 15L4 10M9 15L14 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        </div>
      </section>

      {/* ────────────────────────────────────────────────────────────────
          DATA FOUNDATION
          ──────────────────────────────────────────────────────────────── */}
      <Section className="bg-gradient-to-b from-[#0d1220] to-[var(--bg)]" id="data">
        <div className="w-full max-w-5xl">
          <StickyLabel text="01 / Data Foundation" />

          <div className="text-center mb-20">
            <h2 className="text-3xl font-bold tracking-tight text-[var(--text)] sm:text-4xl lg:text-5xl">
              Seven data sources.
              <br />
              <span className="text-[var(--muted)]">One unified view.</span>
            </h2>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)]">
              Price history, fundamentals, real-time quotes, macro indicators,
              crypto feeds, prediction markets, and news sentiment converge
              into a single analytical surface.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {DATA_SOURCES.map((src, i) => (
              <Stagger
                key={src.name}
                delay={i * 80}
                className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-6 transition-colors hover:border-[var(--accent)]/30"
              >
                <div className="flex items-start justify-between gap-3">
                  <h4 className="text-sm font-bold text-[var(--text)]">{src.name}</h4>
                  <span
                    className={`shrink-0 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                      src.frequency === "Real-time"
                        ? "bg-[var(--green)]/12 text-[var(--green)]"
                        : src.frequency === "Daily"
                          ? "bg-[var(--blue)]/12 text-[var(--blue)]"
                          : "bg-[var(--amber)]/12 text-[var(--amber)]"
                    }`}
                  >
                    {src.frequency}
                  </span>
                </div>
                <p className="mt-3 text-xs leading-relaxed text-[var(--muted)]">
                  {src.provides}
                </p>
              </Stagger>
            ))}
          </div>
        </div>
      </Section>

      {/* ────────────────────────────────────────────────────────────────
          PATTERN DETECTION
          ──────────────────────────────────────────────────────────────── */}
      <Section className="bg-gradient-to-b from-[var(--bg)] to-[#0a0e18]" id="patterns">
        <div className="w-full max-w-4xl">
          <StickyLabel text="02 / Pattern Detection" />

          <div className="text-center mb-20">
            <h2 className="text-3xl font-bold tracking-tight text-[var(--text)] sm:text-4xl lg:text-5xl">
              Five layers of detection.
            </h2>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)]">
              From simple moving-average crossovers to computer-vision pattern
              recognition. Each layer adds signal. No single layer decides.
            </p>
          </div>

          <div className="mt-8">
            {PATTERN_LAYERS.map((layer, i) => (
              <PipelineNode
                key={layer.title}
                step={i + 1}
                title={layer.title}
                description={layer.description}
                highlight={layer.highlight}
                isLast={i === PATTERN_LAYERS.length - 1}
                delay={i * 150}
              />
            ))}
          </div>
        </div>
      </Section>

      {/* ────────────────────────────────────────────────────────────────
          ML PIPELINE
          ──────────────────────────────────────────────────────────────── */}
      <Section className="bg-gradient-to-b from-[#0a0e18] to-[#0d1220]" id="ml">
        <div className="w-full max-w-5xl">
          <StickyLabel text="03 / ML Pipeline" />

          <div className="text-center mb-20">
            <h2 className="text-3xl font-bold tracking-tight text-[var(--text)] sm:text-4xl lg:text-5xl">
              Machine learning
              <br />
              <span className="text-[var(--muted)]">as a filter, not a signal.</span>
            </h2>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)]">
              LightGBM walk-forward classifier trained on real market data.
              Honest about its edge: useful for filtering weak setups, never
              for generating trades.
            </p>
          </div>

          {/* Big stat callouts */}
          <div className="mx-auto grid max-w-4xl gap-4 sm:grid-cols-3">
            <StatCard value={stats?.ml_features ?? 51} label="Engineered Features" delay={0} />
            <StatCard value={93} label="Walk-Forward Folds" delay={200} />
            <StatCard value={22697} label="Training Samples" delay={400} />
          </div>

          {/* Horizontal flow */}
          <div className="mx-auto mt-16 flex flex-wrap items-center justify-center gap-3 lg:gap-0">
            <FlowStep label="Features" sublabel="51 engineered" delay={100} />
            <FlowArrow />
            <FlowStep label="Labels" sublabel="Triple-barrier" delay={300} />
            <FlowArrow />
            <FlowStep label="Training" sublabel="Walk-forward CV" delay={500} />
            <FlowArrow />
            <FlowStep label="Scoring" sublabel="Probability filter" delay={700} />
          </div>

          {/* Feature categories */}
          <Stagger delay={200} className="mx-auto mt-16 max-w-4xl">
            <div className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-8">
              <h4 className="mb-6 text-[11px] font-bold uppercase tracking-[0.2em] text-[var(--muted)]">
                Feature Categories
              </h4>
              <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {ML_FEATURES.map((cat) => (
                  <div key={cat.category}>
                    <span className="inline-block rounded-full bg-[var(--accent)]/10 px-3.5 py-1.5 text-xs font-bold text-[var(--accent)]">
                      {cat.category}
                    </span>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {cat.items.map((f) => (
                        <span
                          key={f}
                          className="rounded-md border border-[var(--line)] bg-[var(--bg)] px-2.5 py-1 text-[10px] text-[var(--muted)]"
                        >
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Stagger>

          {/* Honest AUC callout */}
          <Stagger delay={400} className="mx-auto mt-10 max-w-xl text-center">
            <div className="rounded-2xl border border-[var(--amber)]/25 bg-[var(--amber)]/5 px-8 py-8">
              <div className="text-4xl font-bold tracking-tight text-[var(--amber)]">
                <AnimatedCounter value={stats?.ml_auc ?? 0.5235} decimals={4} />
              </div>
              <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--amber)]/70">
                AUC Score
              </div>
              <p className="mt-4 text-sm leading-relaxed text-[var(--muted)]">
                Honest evaluation. Slight edge over random. Used as a probability
                filter to downweight weak setups &mdash; never as the primary signal.
              </p>
            </div>
          </Stagger>
        </div>
      </Section>

      {/* ────────────────────────────────────────────────────────────────
          DEBATE ENGINE
          ──────────────────────────────────────────────────────────────── */}
      <Section className="bg-gradient-to-b from-[#0d1220] to-[var(--bg)]" id="debate">
        <div className="w-full max-w-5xl">
          <StickyLabel text="04 / Debate Engine" />

          <div className="text-center mb-20">
            <h2 className="text-3xl font-bold tracking-tight text-[var(--text)] sm:text-4xl lg:text-5xl">
              Five analysts.
              <br />
              <span className="text-[var(--muted)]">One verdict.</span>
            </h2>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)]">
              Every ticker is evaluated from five perspectives. An arbiter
              synthesizes the debate into a conviction-weighted recommendation.
              High disagreement automatically downgrades conviction.
            </p>
          </div>

          {/* Analyst cards */}
          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-5">
            {ANALYSTS.map((analyst, i) => (
              <Stagger
                key={analyst.name}
                delay={i * 120}
                className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center transition-colors hover:border-[var(--accent)]/20"
              >
                <div
                  className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full text-xl font-bold"
                  style={{
                    backgroundColor: `color-mix(in srgb, ${analyst.color} 12%, transparent)`,
                    color: analyst.color,
                  }}
                >
                  {analyst.name.charAt(0)}
                </div>
                <h4 className="text-sm font-bold text-[var(--text)]">
                  {analyst.name}
                </h4>
                <p className="mt-3 text-[11px] leading-relaxed text-[var(--muted)]">
                  {analyst.role}
                </p>
              </Stagger>
            ))}
          </div>

          {/* Arbiter */}
          <Stagger delay={700} className="mx-auto mt-10 max-w-lg">
            <div className="rounded-2xl border-2 border-[var(--accent)]/25 bg-[var(--accent)]/5 p-8 text-center">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-[var(--accent)]/12 text-2xl font-bold text-[var(--accent)]">
                A
              </div>
              <h4 className="text-lg font-bold text-[var(--text)]">Arbiter</h4>
              <p className="mt-3 text-sm leading-relaxed text-[var(--muted)]">
                Synthesizes all five perspectives. Weights by relevance. Assigns
                final conviction score and directional bias. Disagreement between
                analysts automatically caps the conviction ceiling.
              </p>
            </div>
          </Stagger>
        </div>
      </Section>

      {/* ────────────────────────────────────────────────────────────────
          RISK GATING
          ──────────────────────────────────────────────────────────────── */}
      <Section className="bg-gradient-to-b from-[var(--bg)] to-[#0a0e18]" id="risk">
        <div className="w-full max-w-3xl">
          <StickyLabel text="05 / Risk Gating" />

          <div className="text-center mb-20">
            <h2 className="text-3xl font-bold tracking-tight text-[var(--text)] sm:text-4xl lg:text-5xl">
              Five gates.
              <br />
              <span className="text-[var(--muted)]">Every trade must pass.</span>
            </h2>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)]">
              No setup reaches execution without clearing each risk gate in
              sequence. One failure is enough to reject.
            </p>
          </div>

          <div className="space-y-0">
            {RISK_GATES.map((gate, i) => (
              <GateStep
                key={gate.gate}
                gate={gate.gate}
                description={gate.description}
                isLast={i === RISK_GATES.length - 1}
                delay={i * 150}
              />
            ))}
          </div>
        </div>
      </Section>

      {/* ────────────────────────────────────────────────────────────────
          EXECUTION & TRACKING
          ──────────────────────────────────────────────────────────────── */}
      <Section className="bg-gradient-to-b from-[#0a0e18] to-[#0d1220]" id="execution">
        <div className="w-full max-w-5xl">
          <StickyLabel text="06 / Execution & Tracking" />

          <div className="text-center mb-20">
            <h2 className="text-3xl font-bold tracking-tight text-[var(--text)] sm:text-4xl lg:text-5xl">
              Paper trade lifecycle.
            </h2>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)]">
              Every position is tracked from entry signal through daily
              mark-to-market to final exit. Full P&L attribution, R-multiples,
              and equity curve.
            </p>
          </div>

          {/* Lifecycle flow */}
          <div className="mx-auto flex flex-wrap items-center justify-center gap-3 lg:gap-0">
            <FlowStep label="Entry Signal" sublabel="All gates pass" delay={0} />
            <FlowArrow />
            <FlowStep label="Position Open" sublabel="Paper entry logged" delay={200} />
            <FlowArrow />
            <FlowStep label="Daily MTM" sublabel="Mark-to-market" delay={400} />
            <FlowArrow />
            <FlowStep label="Exit" sublabel="Profit / Stop / Time" delay={600} />
          </div>

          {/* Tracking metrics */}
          <div className="mx-auto mt-16 grid max-w-4xl gap-4 sm:grid-cols-4">
            {[
              { label: "Win Rate", detail: "Per-trade outcome tracking" },
              { label: "R-Multiple", detail: "Risk-adjusted return per trade" },
              { label: "Sharpe Ratio", detail: "Portfolio risk-adjusted edge" },
              { label: "Equity Curve", detail: "Cumulative NAV over time" },
            ].map((metric, i) => (
              <Stagger
                key={metric.label}
                delay={i * 120}
                className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center"
              >
                <div className="text-base font-bold text-[var(--accent)]">
                  {metric.label}
                </div>
                <div className="mt-2 text-[11px] leading-relaxed text-[var(--muted)]">
                  {metric.detail}
                </div>
              </Stagger>
            ))}
          </div>

          {/* Live paper trade stats */}
          {stats && stats.paper_trades_total > 0 && (
            <div className="mx-auto mt-12 grid max-w-3xl gap-4 sm:grid-cols-3">
              <StatCard
                value={stats.paper_trades_total}
                label="Total Trades"
                delay={0}
              />
              <StatCard
                value={stats.paper_trades_open}
                label="Open Now"
                delay={150}
              />
              {stats.win_rate !== null && (
                <StatCard
                  value={Math.round(stats.win_rate * 100)}
                  label="Win Rate"
                  delay={300}
                  suffix="%"
                />
              )}
            </div>
          )}
        </div>
      </Section>

      {/* ────────────────────────────────────────────────────────────────
          DAILY PIPELINE
          ──────────────────────────────────────────────────────────────── */}
      <Section className="bg-gradient-to-b from-[#0d1220] to-[var(--bg)]" id="pipeline">
        <div className="w-full max-w-4xl">
          <StickyLabel text="07 / Daily Pipeline" />

          <div className="text-center mb-20">
            <h2 className="text-3xl font-bold tracking-tight text-[var(--text)] sm:text-4xl lg:text-5xl">
              Automated.
              <br />
              <span className="text-[var(--muted)]">Every night at 22:00 UTC.</span>
            </h2>
            <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-[var(--muted)]">
              The full pipeline executes every trading day. No manual
              intervention. Weekends run YOLO seed passes on weekly timeframes.
            </p>
          </div>

          {/* Timeline */}
          <div className="space-y-0">
            {PIPELINE_STEPS.map((item, i) => (
              <Stagger key={item.label} delay={i * 150}>
                <div className="flex items-start gap-6">
                  <div className="flex flex-col items-center">
                    <div className="flex h-11 w-24 shrink-0 items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--panel)] text-[11px] font-mono font-bold text-[var(--accent)]">
                      {item.time}
                    </div>
                    {i < PIPELINE_STEPS.length - 1 && (
                      <div className="h-10 w-px bg-[var(--line)]/50" />
                    )}
                  </div>
                  <div className={i < PIPELINE_STEPS.length - 1 ? "pb-5" : ""}>
                    <h4 className="text-base font-semibold text-[var(--text)]">
                      {item.label}
                    </h4>
                    <p className="mt-1 text-sm leading-relaxed text-[var(--muted)]">
                      {item.detail}
                    </p>
                  </div>
                </div>
              </Stagger>
            ))}
          </div>

          {/* Weekend note */}
          <Stagger delay={600} className="mt-12">
            <div className="rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-6">
              <h4 className="text-[11px] font-bold uppercase tracking-[0.2em] text-[var(--muted)]">
                Weekend Schedule
              </h4>
              <p className="mt-3 text-sm leading-relaxed text-[var(--muted)]">
                Saturday 00:30 UTC: full YOLO seed run on both daily and weekly
                timeframes across the entire universe. No market data ingest or
                report generation on weekends.
              </p>
            </div>
          </Stagger>
        </div>
      </Section>

      {/* ────────────────────────────────────────────────────────────────
          FOOTER
          ──────────────────────────────────────────────────────────────── */}
      <section className="flex flex-col items-center justify-center px-6 py-32 bg-gradient-to-b from-[var(--bg)] to-[#060a12]">
        <div className="text-center">
          <p className="text-lg font-medium text-[var(--muted)]">
            Built with conviction, not certainty.
          </p>
          <p className="mt-2 text-sm text-[var(--muted)]/50">
            Not financial advice. Research tool only.
          </p>
          <div className="mt-10 flex items-center justify-center gap-8">
            <a
              href="https://trader.kooexperience.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-[var(--accent)] transition-colors hover:text-[var(--text)]"
            >
              trader.kooexperience.com
            </a>
            <span className="text-[var(--line)]">|</span>
            <a
              href="https://github.com/haomingkoo/trader-koo"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-[var(--accent)] transition-colors hover:text-[var(--text)]"
            >
              GitHub
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}

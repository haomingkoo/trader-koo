import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import {
  ArrowRight,
  BarChart3,
  Bell,
  Bitcoin,
  BookOpen,
  FileText,
  Search,
  ShieldCheck,
  TrendingUp,
  Wallet,
} from "lucide-react";
import Card from "../components/ui/Card";
import PipelineOpsPanel from "../components/PipelineOpsPanel";

interface FeatureCard {
  to: string;
  title: string;
  description: string;
  group: string;
  Icon: LucideIcon;
}

const features: FeatureCard[] = [
  {
    to: "/report",
    title: "Daily Report",
    group: "Research",
    Icon: FileText,
    description:
      "A ranked market brief: setup quality, bias, risk filters, level events, and what changed since the last run.",
  },
  {
    to: "/chart",
    title: "Chart Analysis",
    group: "Research",
    Icon: TrendingUp,
    description:
      "Ticker-level workspace with candles, levels, gaps, trendlines, pattern overlays, commentary, and fundamentals.",
  },
  {
    to: "/vix",
    title: "VIX / Regime",
    group: "Market context",
    Icon: ShieldCheck,
    description:
      "Volatility regime, market health, participation bias, drivers, warnings, and commentary before taking risk.",
  },
  {
    to: "/earnings",
    title: "Earnings Calendar",
    group: "Catalysts",
    Icon: Bell,
    description:
      "Upcoming events with schedule quality, recommendation state, setup scores, bias, and risk notes.",
  },
  {
    to: "/opportunities",
    title: "Opportunities",
    group: "Research",
    Icon: Search,
    description:
      "Valuation screen with P/E, PEG, discount-to-target, and setup quality filters across the universe.",
  },
  {
    to: "/paper-trades",
    title: "Paper Trades",
    group: "Risk loop",
    Icon: Wallet,
    description:
      "The feedback loop: simulated entries, equity curve, P&L, win rate, R-multiples, critic notes, and calibration.",
  },
  {
    to: "/crypto",
    title: "Crypto",
    group: "Live markets",
    Icon: Bitcoin,
    description:
      "Live crypto view with Binance streaming prices, 1-minute candles, 24h volume, and structure signals.",
  },
  {
    to: "/markets",
    title: "Prediction Markets",
    group: "Market context",
    Icon: BarChart3,
    description:
      "Finance-relevant Polymarket events sorted by volume for macro, rates, recession, and geopolitical context.",
  },
];

const workflow = [
  {
    to: "/report",
    title: "1. Read the map",
    detail: "Start with the daily report to see where the system thinks risk/reward is changing.",
    action: "Open report",
  },
  {
    to: "/chart",
    title: "2. Validate one ticker",
    detail: "Use the chart workspace to inspect levels, patterns, fundamentals, and commentary evidence.",
    action: "Open chart",
  },
  {
    to: "/vix",
    title: "3. Check regime risk",
    detail: "Confirm whether volatility, market breadth, and macro context support taking risk today.",
    action: "Check regime",
  },
  {
    to: "/paper-trades",
    title: "4. Review the feedback loop",
    detail: "See whether paper trades are actually improving through P&L, R-multiples, and critic notes.",
    action: "Review trades",
  },
];

export default function GuidePage() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<"general" | "operations">(
    "general",
  );

  useEffect(() => {
    document.title = "Guide \u2014 Trader Koo";
  }, []);

  const handleEscapeKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") setSettingsOpen(false);
    },
    [],
  );

  useEffect(() => {
    if (settingsOpen) {
      document.addEventListener("keydown", handleEscapeKey);
      return () => document.removeEventListener("keydown", handleEscapeKey);
    }
  }, [settingsOpen, handleEscapeKey]);

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-5">
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <div className="space-y-3">
            <div className="label-sm">Research cockpit</div>
            <h1 className="text-3xl font-bold tracking-tight text-[var(--text)]">
              Trader Koo
            </h1>
            <p className="max-w-2xl text-sm leading-relaxed text-[var(--muted)]">
              A swing-trade research workflow: find setups, check regime risk, validate evidence, and learn from paper trades.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Link
              to="/report"
              className="inline-flex items-center gap-2 rounded-lg bg-[var(--accent)] px-4 py-2 text-xs font-semibold text-white transition-opacity hover:opacity-90"
            >
              Today's report <ArrowRight size={14} />
            </Link>
            <Link
              to="/methodology"
              className="inline-flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--bg)] px-4 py-2 text-xs font-semibold text-[var(--text)] transition-colors hover:border-[var(--accent)]"
            >
              How it works <BookOpen size={14} />
            </Link>
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        {workflow.map((step) => (
          <Link
            key={step.to}
            to={step.to}
            className="group rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 transition-colors hover:border-[var(--accent)] hover:bg-[var(--panel-hover)]"
          >
            <div className="text-sm font-semibold text-[var(--text)]">
              {step.title}
            </div>
            <p className="mt-2 min-h-[4.5rem] text-xs leading-relaxed text-[var(--muted)]">
              {step.detail}
            </p>
            <div className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-[var(--accent)]">
              {step.action}
              <ArrowRight size={13} className="transition-transform group-hover:translate-x-0.5" />
            </div>
          </Link>
        ))}
      </div>

      <div className="rounded-xl border border-[var(--amber)]/40 bg-[rgba(248,194,78,0.08)] p-4">
        <div className="flex items-start gap-3">
          <ShieldCheck size={18} className="mt-0.5 text-[var(--amber)]" aria-hidden="true" />
          <div>
            <div className="text-sm font-bold text-[var(--amber)]">Research only</div>
            <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
              Trader Koo is an educational research and paper-trading tool. It
              does not provide financial advice, cannot guarantee data accuracy,
              and should not be the only basis for any investment decision.
            </p>
          </div>
        </div>
      </div>

      <div>
        <div className="mb-3 flex items-end justify-between gap-3">
          <div>
            <div className="label-sm">Capability map</div>
            <h2 className="mt-1 text-lg font-semibold text-[var(--text)]">
              Pick the view that matches the decision
            </h2>
          </div>
          <Link
            to="/methodology"
            className="hidden text-xs font-semibold text-[var(--accent)] hover:text-[var(--blue)] sm:inline-flex"
          >
            Methodology
          </Link>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <Link
              key={feature.to}
              to={feature.to}
              className="group block rounded-xl border border-[var(--line)] bg-[var(--panel)] p-5 transition-all hover:border-[var(--accent)] hover:bg-[var(--panel-hover)]"
            >
              <div className="flex items-start gap-3">
                <feature.Icon
                  size={18}
                  className="mt-0.5 text-[var(--accent)]"
                  aria-hidden="true"
                />
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--muted)]">
                    {feature.group}
                  </div>
                  <div className="mt-1 text-sm font-semibold text-[var(--text)] transition-colors group-hover:text-[var(--accent)]">
                    {feature.title}
                  </div>
                </div>
              </div>
              <p className="mt-3 text-xs leading-relaxed text-[var(--muted)]">
                {feature.description}
              </p>
            </Link>
          ))}
        </div>
      </div>

      {/* Data freshness + quiet settings entry */}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-start">
        <Card label="Data Freshness">
          <div className="mt-1 space-y-1.5 text-xs text-[var(--muted)]">
            <p>
              All data is processed nightly via an automated pipeline:
              <span className="ml-1 font-medium text-[var(--text)]">
                Ingest &rarr; YOLO Detection &rarr; Report Generation
              </span>
            </p>
            <p>
              Market data refreshes Monday through Friday at 22:00 UTC.
              Weekend snapshots include weekly timeframe YOLO seed runs.
              Data shown is always from the most recent pipeline run.
            </p>
          </div>
        </Card>
        <button
          type="button"
          onClick={() => setSettingsOpen(true)}
          className="inline-flex items-center gap-2 self-start rounded-lg border border-[var(--line)] bg-[var(--panel)] px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text)]"
        >
          <span aria-hidden="true">&#9881;</span>
          Settings &amp; Admin
        </button>
      </div>

      {/* Quick links footer */}
      <div className="flex flex-wrap gap-3 text-xs">
        {features.map((feature) => (
          <Link
            key={feature.to}
            to={feature.to}
            className="rounded-md border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[var(--muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text)]"
          >
            {feature.title}
          </Link>
        ))}
      </div>

      {settingsOpen && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 px-4 py-8 backdrop-blur-sm">
          <div role="dialog" aria-modal="true" aria-label="Settings and Admin" className="max-h-[85vh] w-full max-w-4xl overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--panel)] shadow-2xl">
            <div className="flex items-center justify-between border-b border-[var(--line)] px-5 py-4">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[var(--text)]">
                  Settings &amp; Admin
                </h2>
                <p className="mt-1 text-xs text-[var(--muted)]">
                  Pipeline diagnostics and admin controls.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className="rounded-md border border-[var(--line)] px-3 py-1.5 text-xs font-semibold text-[var(--muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--text)]"
              >
                Close
              </button>
            </div>

            <div className="flex gap-2 border-b border-[var(--line)] px-5 py-3">
              {[
                { key: "general", label: "General" },
                { key: "operations", label: "Operations" },
              ].map((tab) => (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() =>
                    setSettingsTab(tab.key as "general" | "operations")
                  }
                  className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] transition-colors ${
                    settingsTab === tab.key
                      ? "bg-[var(--accent)]/15 text-[var(--text)] ring-1 ring-inset ring-[var(--accent)]/35"
                      : "border border-[var(--line)] bg-[var(--bg)] text-[var(--muted)] hover:text-[var(--text)]"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="max-h-[calc(85vh-8rem)] overflow-y-auto px-5 py-5">
              {settingsTab === "general" ? (
                <div className="space-y-4">
                  <Card label="Usage Notes">
                    <div className="space-y-2 text-xs leading-relaxed text-[var(--muted)]">
                      <p>
                        Research outputs can be stale, partial, or unavailable
                        when upstream data providers fail.
                      </p>
                    </div>
                  </Card>
                  <Card label="Data Freshness">
                    <div className="space-y-2 text-xs leading-relaxed text-[var(--muted)]">
                      <p>
                        Nightly pipeline:
                        <span className="ml-1 font-medium text-[var(--text)]">
                          Ingest &rarr; YOLO Detection &rarr; Report Generation
                        </span>
                      </p>
                      <p>
                        Market data refreshes Monday through Friday at 22:00 UTC.
                        Weekend snapshots include weekly timeframe YOLO seed
                        runs.
                      </p>
                    </div>
                  </Card>
                </div>
              ) : (
                <PipelineOpsPanel />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

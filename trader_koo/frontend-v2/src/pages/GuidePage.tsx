import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import Card from "../components/ui/Card";
import PipelineOpsPanel from "../components/PipelineOpsPanel";

interface FeatureCard {
  to: string;
  title: string;
  description: string;
}

const features: FeatureCard[] = [
  {
    to: "/report",
    title: "Daily Report",
    description:
      "Setup quality rankings with tier scoring, bias labels, YOLO context, and level events across the tracked universe. Includes risk filters and key changes.",
  },
  {
    to: "/chart",
    title: "Chart Analysis",
    description:
      "Interactive candlestick chart with support/resistance levels, gap zones, trendlines, pattern overlays, multi-angle debate commentary, and fundamentals.",
  },
  {
    to: "/vix",
    title: "VIX / Regime",
    description:
      "Regime context dashboard with VIX levels, MA matrix, market health scoring, participation bias, drivers, warnings, and LLM commentary.",
  },
  {
    to: "/earnings",
    title: "Earnings Calendar",
    description:
      "Upcoming earnings events with schedule quality, recommendation state, setup scores, bias, risk analysis, and calendar/table views per ticker.",
  },
  {
    to: "/opportunities",
    title: "Opportunities",
    description:
      "Valuation screener with P/E, PEG, and discount-to-target filtering. View undervalued, overvalued, deep value, or all tickers.",
  },
  {
    to: "/paper-trades",
    title: "Paper Trades",
    description:
      "Simulated trade log with equity curve tracking, P&L analysis, win rates, R-multiples, direction breakdowns, and exit reason statistics.",
  },
  {
    to: "/crypto",
    title: "Crypto",
    description:
      "Live BTC and ETH prices via Binance WebSocket feed with 1-minute candlestick charts, 24h volume, and real-time price change tracking.",
  },
  {
    to: "/markets",
    title: "Prediction Markets",
    description:
      "Finance-relevant Polymarket events sorted by volume. Fed rate cuts, recession odds, geopolitical events — macro regime signals from prediction markets.",
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
      {/* Header */}
      <div className="space-y-3">
        <h1 className="text-3xl font-bold tracking-tight text-[var(--text)]">
          trader_koo
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-[var(--muted)]">
          S&amp;P 500 market analysis dashboard with YOLOv8 pattern detection,
          VIX regime context, multi-angle debate commentary, and automated
          nightly pipeline processing. Built for structured research and trade
          idea evaluation.
        </p>
      </div>

      {/* NFA disclaimer — prominent placement */}
      <div className="rounded-xl border-2 border-[var(--amber)]/40 bg-[rgba(248,194,78,0.08)] p-5">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 text-lg text-[var(--amber)]" aria-hidden="true">
            &#9888;
          </span>
          <div className="space-y-2">
            <div className="text-sm font-bold text-[var(--amber)]">
              Not Financial Advice
            </div>
            <p className="text-xs leading-relaxed text-[var(--muted)]">
              This is a personal analysis tool for educational and research
              purposes only. No content on this dashboard constitutes a
              recommendation to buy, sell, or hold any security.
            </p>
            <p className="text-xs leading-relaxed text-[var(--muted)]">
              Past performance does not guarantee future results. All data may
              be delayed, inaccurate, or incomplete. AI-generated signals,
              pattern detections, and commentary can be wrong.
            </p>
            <p className="text-xs leading-relaxed text-[var(--muted)]">
              Do not make investment decisions based solely on this dashboard.
              All investment decisions carry risk and should be made with
              independent analysis and qualified professional guidance.
            </p>
          </div>
        </div>
      </div>

      {/* Feature cards grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {features.map((feature) => (
          <Link
            key={feature.to}
            to={feature.to}
            className="group block rounded-xl border border-[var(--line)] bg-[var(--panel)] p-5 transition-all hover:border-[var(--accent)] hover:bg-[var(--panel-hover)]"
          >
            <div className="mb-2 text-sm font-semibold text-[var(--accent)] transition-colors group-hover:text-[var(--blue)]">
              {feature.title}
            </div>
            <p className="text-xs leading-relaxed text-[var(--muted)]">
              {feature.description}
            </p>
          </Link>
        ))}
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
              Data shown is always as of the most recent pipeline completion.
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
                  Advanced controls and pipeline diagnostics stay tucked away
                  from the main guide flow.
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
                        This is the current production interface for Trader
                        Koo.
                      </p>
                      <p>
                        Pipeline internals and rerun controls are intentionally
                        hidden from regular users unless someone opens the admin
                        settings panel.
                      </p>
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

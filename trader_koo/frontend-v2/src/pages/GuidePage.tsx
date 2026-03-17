import { Link } from "react-router-dom";
import Card from "../components/ui/Card";

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
];

export default function GuidePage() {
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

      {/* Data freshness disclaimer */}
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

      {/* Disclaimer banner */}
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
    </div>
  );
}

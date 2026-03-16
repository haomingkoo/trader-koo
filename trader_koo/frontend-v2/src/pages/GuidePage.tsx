import Card from "../components/ui/Card";

export default function GuidePage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h2 className="mb-2 text-2xl font-bold tracking-tight text-[var(--text)]">
          trader_koo Dashboard
        </h2>
        <p className="text-sm text-[var(--muted)]">
          S&P 500 market analysis dashboard with YOLO pattern detection, regime
          context, and multi-angle debate commentary.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Card label="Report">
          <p className="mt-1 text-xs text-[var(--muted)]">
            Daily setup quality rankings with tier scoring, bias labels, YOLO
            context, and level events across the tracked universe.
          </p>
        </Card>
        <Card label="VIX Analysis">
          <p className="mt-1 text-xs text-[var(--muted)]">
            Regime context dashboard with VIX levels, MA matrix, market health
            scoring, participation breadth, and comparison series.
          </p>
        </Card>
        <Card label="Earnings">
          <p className="mt-1 text-xs text-[var(--muted)]">
            Upcoming earnings calendar with schedule quality, recommendation
            state, setup scores, and bias analysis per ticker.
          </p>
        </Card>
        <Card label="Chart">
          <p className="mt-1 text-xs text-[var(--muted)]">
            Interactive candlestick chart with support/resistance levels, gap
            zones, trendlines, pattern overlays, and AI commentary.
          </p>
        </Card>
        <Card label="Opportunities">
          <p className="mt-1 text-xs text-[var(--muted)]">
            Valuation screener with P/E, PEG, discount-to-target filtering
            across undervalued, overvalued, and deep value views.
          </p>
        </Card>
        <Card label="Paper Trades">
          <p className="mt-1 text-xs text-[var(--muted)]">
            Simulated trade log with equity curve tracking, P&L analysis,
            direction breakdowns, and exit reason statistics.
          </p>
        </Card>
      </div>

      <Card className="border-[var(--accent)] border-opacity-30">
        <p className="text-xs text-[var(--muted)]">
          Not financial advice. This is a personal analysis tool for educational
          purposes only. All data is processed nightly via an automated pipeline
          (ingest &rarr; YOLO detection &rarr; report generation).
        </p>
      </Card>
    </div>
  );
}

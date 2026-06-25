import type { RegimeContext, VixData, VixMetricsPayload } from "../../api/types";
import Badge from "../ui/Badge";
import Card from "../ui/Card";
import GaugeSvg from "../ui/GaugeSvg";
import { formatVixState } from "./vixUtils";

const riskReadBadgeVariant = (
  read: string | null | undefined,
): "green" | "red" | "amber" | "muted" => {
  const lower = (read ?? "").toLowerCase().replace(/_/g, " ");
  if (lower.includes("risk off") || lower.includes("elevated") || lower.includes("above")) {
    return "red";
  }
  if (lower.includes("risk on") || lower.includes("relief") || lower.includes("below")) {
    return "green";
  }
  return "muted";
};

const GAUGE_ZONES: Array<[number, number, string, string]> = [
  [0, 12, "Complacency", "#00c853"],
  [12, 16, "Calm", "#4caf50"],
  [16, 20, "Normal", "#c0ca33"],
  [20, 25, "Caution", "#fdd835"],
  [25, 30, "Stress", "#ff9800"],
  [30, 40, "Fear", "#f44336"],
  [40, 60, "Extreme Fear", "#b71c1c"],
  [60, 80, "Panic", "#5b1521"],
];

const GAUGE_MAX = 80;

function VixGauge({ vixClose }: { vixClose: number }) {
  return (
    <div className="flex flex-col items-center">
      <GaugeSvg
        value={vixClose}
        zones={GAUGE_ZONES}
        max={GAUGE_MAX}
        valueLabel={vixClose.toFixed(2)}
        valueColor="var(--text)"
      />
    </div>
  );
}

export function SpikeAlertBanner({ metrics }: { metrics: VixMetricsPayload }) {
  if (!metrics.is_spike || metrics.vix_daily_change_pct == null) return null;

  const magnitude = metrics.spike_magnitude ?? "moderate";
  const bannerColor =
    magnitude === "extreme"
      ? "border-[var(--red)] bg-[rgba(255,107,107,0.12)]"
      : magnitude === "large"
        ? "border-[var(--red)] bg-[rgba(255,107,107,0.08)]"
        : "border-[var(--amber)] bg-[rgba(248,194,78,0.12)]";
  const textColor =
    magnitude === "extreme" || magnitude === "large"
      ? "text-[var(--red)]"
      : "text-[var(--amber)]";
  const direction = metrics.vix_daily_change_pct > 0 ? "spiked" : "dropped";
  const absChange = Math.abs(metrics.vix_daily_change_pct).toFixed(1);

  return (
    <div className={`rounded-lg border p-3 ${bannerColor}`} role="alert">
      <div className={`text-sm font-semibold ${textColor}`}>
        VIX {direction} {absChange}% today ({magnitude})
      </div>
      <p className="mt-1 text-xs text-[var(--muted)]">
        {metrics.vix_daily_change_pct > 0
          ? "Historically, VIX spikes of this magnitude tend to mean-revert within 5-10 trading days. Reduce position sizes and wait for stabilization."
          : "A sharp VIX drop may signal complacency. Monitor for potential snap-back."}
      </p>
    </div>
  );
}

export function VixPrimaryPanels({
  vix,
  metrics,
}: {
  vix: VixData;
  metrics: VixMetricsPayload | null;
}) {
  const pct = metrics?.recommended_position_pct ?? 0;
  const barColor =
    pct >= 100
      ? "bg-[var(--green)]"
      : pct >= 75
        ? "bg-[var(--blue)]"
        : pct >= 50
          ? "bg-[var(--amber)]"
          : "bg-[var(--red)]";

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card label="VIX Gauge">
        {vix.close != null ? (
          <VixGauge vixClose={vix.close} />
        ) : (
          <p className="py-8 text-center text-sm text-[var(--muted)]">
            VIX data unavailable
          </p>
        )}
      </Card>

      {metrics ? (
        <Card label="Position Sizing">
          <div className="mt-2 space-y-3">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-[var(--text)]">{pct}%</span>
              <span className="text-xs text-[var(--muted)]">recommended</span>
            </div>
            <div className="h-2 w-full rounded-full bg-[var(--panel-hover)]">
              <div
                className={`h-2 rounded-full transition-all ${barColor}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="text-xs leading-relaxed text-[var(--muted)]">
              {String(metrics.sizing_reason ?? "")}
            </p>
          </div>
        </Card>
      ) : (
        <Card label="Position Sizing">
          <p className="mt-2 text-xs text-[var(--muted)]">VIX metrics unavailable</p>
        </Card>
      )}
    </div>
  );
}

export function VixMetricCardsGrid({ metrics }: { metrics: VixMetricsPayload }) {
  const termVariant: "green" | "red" | "amber" | "muted" =
    metrics.term_structure_signal === "fear"
      ? "red"
      : metrics.term_structure_signal === "elevated"
        ? "amber"
        : metrics.term_structure_signal === "complacent"
          ? "green"
          : "muted";

  const percentileVariant: "green" | "red" | "amber" | "muted" =
    metrics.percentile_zone === "extreme_high"
      ? "red"
      : metrics.percentile_zone === "elevated" ||
          metrics.percentile_zone === "extreme_low" ||
          metrics.percentile_zone === "low"
        ? "amber"
        : "muted";

  const premiumVariant: "green" | "red" | "amber" | "muted" =
    metrics.vol_premium_signal === "cheap_vol"
      ? "green"
      : metrics.vol_premium_signal === "expensive_vol"
        ? "red"
        : "muted";

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      <Card
        glass
        label="VIX/VIX3M Ratio"
        value={metrics.vix_vix3m_ratio != null ? metrics.vix_vix3m_ratio.toFixed(3) : "\u2014"}
      >
        <Badge variant={termVariant} className="mt-1">
          {formatVixState(metrics.term_structure_signal)}
        </Badge>
      </Card>

      <Card
        glass
        label="VIX Percentile (252d)"
        value={
          metrics.vix_percentile_252d != null
            ? `${metrics.vix_percentile_252d.toFixed(1)}%`
            : "\u2014"
        }
      >
        <Badge variant={percentileVariant} className="mt-1">
          {formatVixState(metrics.percentile_zone)}
        </Badge>
        {metrics.above_80th_pctile && (
          <div className="mt-1 text-[10px] font-semibold text-[var(--blue)]">
            Contrarian note: elevated VIX can improve forward entry conditions after price stabilizes.
          </div>
        )}
      </Card>

      <Card label="Vol Risk Premium">
        <div className="mt-2 space-y-2">
          <div className="flex items-center gap-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">
                Implied (VIX)
              </div>
              <div className="text-sm font-semibold text-[var(--text)]">
                {metrics.realized_vol_20d != null && metrics.vol_risk_premium != null
                  ? (metrics.realized_vol_20d + metrics.vol_risk_premium).toFixed(1)
                  : "\u2014"}
              </div>
            </div>
            <div className="text-lg text-[var(--muted)]">vs</div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">
                Realized (20d)
              </div>
              <div className="text-sm font-semibold text-[var(--text)]">
                {metrics.realized_vol_20d != null
                  ? `${metrics.realized_vol_20d.toFixed(1)}%`
                  : "\u2014"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-[var(--muted)]">Spread:</span>
            <span className="text-sm font-semibold text-[var(--text)]">
              {metrics.vol_risk_premium != null
                ? `${metrics.vol_risk_premium > 0 ? "+" : ""}${metrics.vol_risk_premium.toFixed(1)}`
                : "\u2014"}
            </span>
            <Badge variant={premiumVariant}>{formatVixState(metrics.vol_premium_signal)}</Badge>
          </div>
        </div>
      </Card>
    </div>
  );
}

export function MarketHealthCard({ health }: { health: RegimeContext["health"] }) {
  return (
    <Card label="Market Health">
      <div className="mt-2 space-y-3">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span className="text-[var(--muted)]">State:</span>
          <Badge
            variant={
              health.state === "healthy"
                ? "green"
                : health.state === "caution"
                  ? "amber"
                  : health.state === "stress"
                    ? "red"
                    : "muted"
            }
          >
            {formatVixState(health.state)}
          </Badge>
          <span className="text-[var(--muted)]">Score:</span>
          <span className="font-semibold text-[var(--text)]">
            {health.score != null ? `${health.score} / 100` : "\u2014"}
          </span>
        </div>

        {health.drivers.length > 0 && (
          <div>
            <div className="mb-1 text-xs font-medium text-[var(--muted)]">Drivers</div>
            <ul className="space-y-0.5 text-xs text-[var(--text)]">
              {health.drivers.map((driver, index) => (
                <li key={index} className="flex items-start gap-1.5">
                  <span className="mt-0.5 text-[var(--green)]" aria-hidden="true">&#8226;</span>
                  {String(driver)}
                </li>
              ))}
            </ul>
          </div>
        )}

        {health.warnings.length > 0 && (
          <div>
            <div className="mb-1 text-xs font-medium text-[var(--muted)]">Warnings</div>
            <ul className="space-y-0.5 text-xs text-[var(--red)]">
              {health.warnings.map((warning, index) => (
                <li key={index} className="flex items-start gap-1.5">
                  <span className="mt-0.5" aria-hidden="true">&#9888;</span>
                  <span>Warning: {String(warning)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {health.drivers.length === 0 && health.warnings.length === 0 && (
          <p className="text-xs text-[var(--muted)]">
            No health drivers or warnings for this snapshot.
          </p>
        )}
      </div>
    </Card>
  );
}

export function ParticipationBiasCard({ participationBias }: { participationBias: string }) {
  return (
    <Card label="Participation Bias">
      <div className="mt-1 flex items-center gap-2">
        <Badge
          variant={
            participationBias.toLowerCase().includes("bull")
              ? "green"
              : participationBias.toLowerCase().includes("bear")
                ? "red"
                : "muted"
          }
        >
          {formatVixState(participationBias)}
        </Badge>
      </div>
    </Card>
  );
}

export function MAMatrixCard({ rows }: { rows: RegimeContext["ma_matrix"] }) {
  return (
    <Card label="MA Matrix">
      {rows.length > 0 ? (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                <th className="px-3 py-2 font-semibold uppercase tracking-wider">Metric</th>
                <th className="px-3 py-2 font-semibold uppercase tracking-wider">Value %</th>
                <th className="px-3 py-2 font-semibold uppercase tracking-wider">State</th>
                <th className="px-3 py-2 font-semibold uppercase tracking-wider">Risk Read</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index} className="border-b border-[var(--line)] last:border-b-0">
                  <td className="px-3 py-2 font-medium text-[var(--text)]">{String(row.metric)}</td>
                  <td className="px-3 py-2 tabular-nums text-[var(--text)]">
                    {typeof row.value_pct === "number"
                      ? `${row.value_pct > 0 ? "+" : ""}${row.value_pct.toFixed(2)}%`
                      : "\u2014"}
                  </td>
                  <td className="px-3 py-2">
                    <Badge
                      variant={
                        row.state.toLowerCase().includes("above")
                          ? "red"
                          : row.state.toLowerCase().includes("below")
                            ? "green"
                            : "muted"
                      }
                    >
                      {formatVixState(row.state)}
                    </Badge>
                  </td>
                  <td className="px-3 py-2">
                    <Badge variant={riskReadBadgeVariant(row.risk_read)}>
                      {String(row.risk_read).replace(/_/g, " ")}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="mt-2 text-xs text-[var(--muted)]">
          No MA matrix data available for this snapshot.
        </p>
      )}
    </Card>
  );
}

export function RegimeSummaryCard({ summary }: { summary: string }) {
  return (
    <Card label="Regime Summary">
      <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
        {summary || "No summary available."}
      </p>
    </Card>
  );
}

export function CommentaryCard({
  commentary,
}: {
  commentary: RegimeContext["llm_commentary"];
}) {
  const commentarySource = (commentary.source ?? "rule").trim().toLowerCase();

  return (
    <Card label="Commentary">
      <div className="mt-2 space-y-2">
        <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--muted)]">
          Source:{" "}
          {commentarySource === "llm"
            ? "LLM (generated from current regime snapshot)"
            : "Rule (rule-based fallback)"}
        </div>
        {commentary.observation ? (
          <div className="space-y-2 text-xs text-[var(--muted)]">
            <p>{String(commentary.observation)}</p>
            {commentary.action && (
              <p>
                <strong className="text-[var(--text)]">Action:</strong>{" "}
                {String(commentary.action)}
              </p>
            )}
            {commentary.risk_note && (
              <p>
                <strong className="text-[var(--text)]">Risk:</strong>{" "}
                {String(commentary.risk_note)}
              </p>
            )}
          </div>
        ) : (
          <p className="text-xs text-[var(--muted)]">
            No LLM commentary available for this snapshot.
          </p>
        )}
      </div>
    </Card>
  );
}

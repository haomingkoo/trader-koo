import { useReport, useVixMetrics } from "../api/hooks";
import type { VixMetricsPayload } from "../api/types";
import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";
import Spinner from "../components/ui/Spinner";

const riskReadBadgeVariant = (
  read: string,
): "green" | "red" | "amber" | "muted" => {
  const lower = read.toLowerCase().replace(/_/g, " ");
  if (lower.includes("risk off") || lower.includes("elevated") || lower.includes("above"))
    return "red";
  if (lower.includes("risk on") || lower.includes("relief") || lower.includes("below"))
    return "green";
  return "muted";
};

const formatState = (s: string): string =>
  s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

/* ── Gauge zones: [lo, hi, label, color] ── */
const GAUGE_ZONES: Array<[number, number, string, string]> = [
  [0, 12, "Complacency", "#00c853"],
  [12, 16, "Calm", "#4caf50"],
  [16, 20, "Normal", "#c0ca33"],
  [20, 25, "Caution", "#fdd835"],
  [25, 30, "Stress", "#ff9800"],
  [30, 40, "Fear", "#f44336"],
  [40, 60, "Extreme Fear", "#b71c1c"],
  [60, 80, "Panic", "#212121"],
];

const GAUGE_MAX = 80;

/* ── SVG VIX Gauge (semicircle speedometer) ── */
function VixGauge({ vixClose }: { vixClose: number }) {
  const cx = 150;
  const cy = 140;
  const r = 110;
  const startAngle = Math.PI; // left (180deg)
  const totalArc = Math.PI;

  // Build zone arcs
  const zoneArcs = GAUGE_ZONES.map(([lo, hi, label, color]) => {
    const a1 = startAngle - (lo / GAUGE_MAX) * totalArc;
    const a2 = startAngle - (Math.min(hi, GAUGE_MAX) / GAUGE_MAX) * totalArc;
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy - r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2);
    const y2 = cy - r * Math.sin(a2);
    const largeArc = Math.abs(a1 - a2) > Math.PI ? 1 : 0;
    const d = `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 0 ${x2} ${y2}`;
    return { d, color, label, lo, hi };
  });

  // Needle position
  const clampedVix = Math.max(0, Math.min(vixClose, GAUGE_MAX));
  const needleAngle = startAngle - (clampedVix / GAUGE_MAX) * totalArc;
  const needleLen = r - 15;
  const nx = cx + needleLen * Math.cos(needleAngle);
  const ny = cy - needleLen * Math.sin(needleAngle);

  // Find current zone label
  const currentZone = GAUGE_ZONES.find(
    ([lo, hi]) => vixClose >= lo && vixClose < hi,
  );
  const zoneLabel = currentZone ? currentZone[2] : "Panic";
  const zoneColor = currentZone ? currentZone[3] : "#212121";

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 300 170" className="w-full max-w-[340px]">
        {/* Zone arcs */}
        {zoneArcs.map((arc) => (
          <path
            key={arc.lo}
            d={arc.d}
            fill="none"
            stroke={arc.color}
            strokeWidth={18}
            strokeLinecap="butt"
          />
        ))}
        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={nx}
          y2={ny}
          stroke="var(--text)"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        {/* Center dot */}
        <circle cx={cx} cy={cy} r={5} fill="var(--text)" />
        {/* VIX value */}
        <text
          x={cx}
          y={cy + 28}
          textAnchor="middle"
          fontSize={22}
          fontWeight={700}
          fill="var(--text)"
        >
          {vixClose.toFixed(2)}
        </text>
      </svg>
      <div className="mt-1 flex items-center gap-2">
        <span
          className="inline-block h-3 w-3 rounded-full"
          style={{ backgroundColor: zoneColor }}
        />
        <span className="text-sm font-semibold" style={{ color: zoneColor }}>
          {zoneLabel}
        </span>
      </div>
    </div>
  );
}

/* ── Spike Alert Banner ── */
function SpikeAlertBanner({ metrics }: { metrics: VixMetricsPayload }) {
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
    <div
      className={`rounded-lg border p-3 ${bannerColor}`}
      role="alert"
    >
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

/* ── Position Sizing Card ── */
function PositionSizingCard({ metrics }: { metrics: VixMetricsPayload }) {
  const pct = metrics.recommended_position_pct;
  const barColor =
    pct >= 100
      ? "bg-[var(--green)]"
      : pct >= 75
        ? "bg-[var(--blue)]"
        : pct >= 50
          ? "bg-[var(--amber)]"
          : "bg-[var(--red)]";

  return (
    <Card label="Position Sizing">
      <div className="mt-2 space-y-3">
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-[var(--text)]">{pct}%</span>
          <span className="text-xs text-[var(--muted)]">recommended</span>
        </div>
        {/* Bar */}
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
  );
}

/* ── Vol Risk Premium Card ── */
function VolRiskPremiumCard({ metrics }: { metrics: VixMetricsPayload }) {
  const signal = metrics.vol_premium_signal;
  const variant: "green" | "red" | "amber" | "muted" =
    signal === "cheap_vol"
      ? "green"
      : signal === "expensive_vol"
        ? "red"
        : signal === "normal"
          ? "muted"
          : "muted";

  return (
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
          <Badge variant={variant}>
            {formatState(signal)}
          </Badge>
        </div>
      </div>
    </Card>
  );
}

/* ── Term Structure Card ── */
function TermStructureCard({ metrics }: { metrics: VixMetricsPayload }) {
  const signal = metrics.term_structure_signal;
  const variant: "green" | "red" | "amber" | "muted" =
    signal === "fear"
      ? "red"
      : signal === "elevated"
        ? "amber"
        : signal === "complacent"
          ? "green"
          : "muted";

  return (
    <Card glass label="VIX/VIX3M Ratio" value={
      metrics.vix_vix3m_ratio != null
        ? metrics.vix_vix3m_ratio.toFixed(3)
        : "\u2014"
    }>
      <Badge variant={variant} className="mt-1">
        {formatState(signal)}
      </Badge>
    </Card>
  );
}

/* ── Percentile Card ── */
function PercentileCard({ metrics }: { metrics: VixMetricsPayload }) {
  const zone = metrics.percentile_zone;
  const variant: "green" | "red" | "amber" | "muted" =
    zone === "extreme_high" || zone === "elevated"
      ? "red"
      : zone === "extreme_low" || zone === "low"
        ? "green"
        : "muted";

  return (
    <Card glass label="VIX Percentile (252d)" value={
      metrics.vix_percentile_252d != null
        ? `${metrics.vix_percentile_252d.toFixed(1)}%`
        : "\u2014"
    }>
      <Badge variant={variant} className="mt-1">
        {formatState(zone)}
      </Badge>
      {metrics.above_80th_pctile && (
        <div className="mt-1 text-[10px] font-semibold text-[var(--amber)]">
          Above 80th pctile -- historically optimal buy signal
        </div>
      )}
    </Card>
  );
}


export default function VixPage() {
  const { data, isLoading, error } = useReport();
  const {
    data: metricsData,
    isLoading: metricsLoading,
    error: metricsError,
  } = useVixMetrics();

  if (isLoading || metricsLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load VIX data: {(error as Error).message}
      </div>
    );
  }

  const regime = data?.latest?.signals?.regime_context;

  if (!regime || !Object.keys(regime).length) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--muted)]">
        No regime context available yet. Data populates after the nightly
        pipeline run.
      </div>
    );
  }

  const vix = regime.vix;
  const health = regime.health ?? {
    state: "unknown",
    score: null,
    drivers: [],
    warnings: [],
  };
  const overall = regime.overall ?? { participation_bias: "unknown" };
  const maMatrix = regime.ma_matrix ?? [];
  const commentary = regime.llm_commentary;
  const commentarySource = (commentary.source ?? "rule").trim().toLowerCase();

  // VIX metrics from the standalone endpoint
  const metrics = metricsData?.ok ? metricsData : null;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold tracking-tight">
        VIX / Regime Analysis
      </h2>

      {/* Spike Alert Banner */}
      {metrics && <SpikeAlertBanner metrics={metrics} />}

      {/* VIX Gauge + Top Metrics Row */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Gauge */}
        <Card label="VIX Gauge">
          {vix.close != null ? (
            <VixGauge vixClose={vix.close} />
          ) : (
            <p className="py-8 text-center text-sm text-[var(--muted)]">
              VIX data unavailable
            </p>
          )}
        </Card>

        {/* Position Sizing */}
        {metrics ? (
          <PositionSizingCard metrics={metrics} />
        ) : (
          <Card label="Position Sizing">
            <p className="mt-2 text-xs text-[var(--muted)]">
              VIX metrics unavailable
            </p>
          </Card>
        )}
      </div>

      {/* Metrics cards row */}
      {metrics && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <TermStructureCard metrics={metrics} />
          <PercentileCard metrics={metrics} />
          <VolRiskPremiumCard metrics={metrics} />
        </div>
      )}
      {metricsError && (
        <div className="text-xs text-[var(--red)]">
          Failed to load VIX metrics: {(metricsError as Error).message}
        </div>
      )}

      {/* VIX KPI cards */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Card
          glass
          label="VIX Close"
          value={vix.close != null ? vix.close.toFixed(2) : "\u2014"}
        />
        <Card
          glass
          label="Risk State"
          value={formatState(vix.risk_state ?? "unknown")}
        />
        <Card
          glass
          label="Percentile 1Y"
          value={
            vix.percentile_1y != null
              ? `${Number(vix.percentile_1y).toFixed(1)}%`
              : "\u2014"
          }
        />
        <Card
          glass
          label="MA Cross State"
          value={formatState(vix.ma_cross_state ?? vix.ma_state ?? "unknown")}
        />
      </div>

      {/* Health panel */}
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
              {formatState(health.state)}
            </Badge>
            <span className="text-[var(--muted)]">Score:</span>
            <span className="font-semibold text-[var(--text)]">
              {health.score != null ? `${health.score} / 100` : "\u2014"}
            </span>
          </div>

          {health.drivers.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-medium text-[var(--muted)]">
                Drivers
              </div>
              <ul className="space-y-0.5 text-xs text-[var(--text)]">
                {health.drivers.map((d: string, i: number) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="mt-0.5 text-[var(--green)]" aria-hidden="true">&#8226;</span>
                    {String(d)}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {health.warnings.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-medium text-[var(--muted)]">
                Warnings
              </div>
              <ul className="space-y-0.5 text-xs text-[var(--red)]">
                {health.warnings.map((w: string, i: number) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="mt-0.5" aria-hidden="true">&#9888;</span>
                    <span>Warning: {String(w)}</span>
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

      {/* Participation Bias */}
      <Card label="Participation Bias">
        <div className="mt-1 flex items-center gap-2">
          <Badge
            variant={
              overall.participation_bias?.toLowerCase().includes("bull")
                ? "green"
                : overall.participation_bias?.toLowerCase().includes("bear")
                  ? "red"
                  : "muted"
            }
          >
            {formatState(overall.participation_bias ?? "unknown")}
          </Badge>
        </div>
      </Card>

      {/* MA Matrix table */}
      <Card label="MA Matrix">
        {maMatrix.length > 0 ? (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[var(--line)] text-[var(--muted)]">
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    Value %
                  </th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    State
                  </th>
                  <th className="px-3 py-2 font-semibold uppercase tracking-wider">
                    Risk Read
                  </th>
                </tr>
              </thead>
              <tbody>
                {maMatrix.map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-[var(--line)] last:border-b-0"
                  >
                    <td className="px-3 py-2 font-medium text-[var(--text)]">
                      {String(row.metric)}
                    </td>
                    <td className="px-3 py-2 tabular-nums text-[var(--text)]">
                      {row.value_pct != null
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
                        {formatState(row.state)}
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

      {/* Regime Summary */}
      <Card label="Regime Summary">
        <p className="mt-1 text-xs leading-relaxed text-[var(--muted)]">
          {typeof regime.summary === "string" ? regime.summary : "No summary available."}
        </p>
      </Card>

      {/* LLM Commentary */}
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

      {/* Footer */}
      <div className="text-xs text-[var(--muted)]">
        As of {String(regime.asof_date ?? "\u2014")} &middot; Source:{" "}
        {String(regime.source ?? "unknown")}
      </div>
    </div>
  );
}

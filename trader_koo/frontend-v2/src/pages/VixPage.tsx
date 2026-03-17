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
  [0, 12, "Complacency", "#0d9f6e"],
  [12, 18, "Calm", "#4caf50"],
  [18, 24, "Caution", "#f8c24e"],
  [24, 32, "Stress", "#ff9800"],
  [32, 50, "Fear", "#ff6b6b"],
  [50, 80, "Panic", "#d32f2f"],
];

const GAUGE_MAX = 80;

/* ── SVG arc helpers ── */
function vixPolarToCartesian(
  cx: number,
  cy: number,
  r: number,
  angleDeg: number,
) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function vixDescribeArc(
  cx: number,
  cy: number,
  r: number,
  startAngle: number,
  endAngle: number,
) {
  const start = vixPolarToCartesian(cx, cy, r, endAngle);
  const end = vixPolarToCartesian(cx, cy, r, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
}

/* ── SVG VIX Gauge ── */
function VixGauge({ vixClose }: { vixClose: number }) {
  const cx = 150;
  const cy = 155;
  const r = 120;
  const arcWidth = 14;

  // Semicircle: 180 deg (left) to 360 deg (right)
  // Map VIX 0 -> 180 deg, VIX 80 -> 360 deg
  const vixToAngle = (v: number) => 180 + (v / GAUGE_MAX) * 180;

  // Build zone arc paths
  const zoneArcs = GAUGE_ZONES.map(([lo, hi, , color]) => {
    const a1 = vixToAngle(lo);
    const a2 = vixToAngle(hi);
    return { d: vixDescribeArc(cx, cy, r, a1, a2), color };
  });

  // SVG rotation is anchored on the vertical needle where 0deg points up.
  // Convert the left->right semicircle into that coordinate system.
  const clampedVix = Math.max(0, Math.min(vixClose, GAUGE_MAX));
  const needleRotation = vixToAngle(clampedVix) - 270;

  // Current zone
  const currentZone = GAUGE_ZONES.find(
    ([lo, hi]) => vixClose >= lo && vixClose < hi,
  );
  const zoneLabel = currentZone ? currentZone[2] : "Panic";
  const zoneColor = currentZone ? currentZone[3] : "#d32f2f";

  const needleLen = r - arcWidth / 2 - 6;

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 300 180" className="w-full max-w-[340px]">
        {/* Dim background track */}
        <path
          d={vixDescribeArc(cx, cy, r, 180, 360)}
          fill="none"
          stroke="var(--panel-hover)"
          strokeWidth={arcWidth + 4}
          strokeLinecap="round"
          opacity={0.3}
        />

        {/* Colored zone arcs */}
        {zoneArcs.map((arc, i) => (
          <path
            key={i}
            d={arc.d}
            fill="none"
            stroke={arc.color}
            strokeWidth={arcWidth}
            strokeLinecap="butt"
          />
        ))}

        {/* Round caps at the ends of the full arc */}
        {(() => {
          const leftCap = vixPolarToCartesian(cx, cy, r, 180);
          const rightCap = vixPolarToCartesian(cx, cy, r, 360);
          return (
            <>
              <circle
                cx={leftCap.x}
                cy={leftCap.y}
                r={arcWidth / 2}
                fill={GAUGE_ZONES[0][3]}
              />
              <circle
                cx={rightCap.x}
                cy={rightCap.y}
                r={arcWidth / 2}
                fill={GAUGE_ZONES[GAUGE_ZONES.length - 1][3]}
              />
            </>
          );
        })()}

        {/* Scale labels: 0 and MAX */}
        <text
          x={cx - r}
          y={cy + 18}
          textAnchor="middle"
          fontSize={9}
          fill="var(--muted)"
          opacity={0.6}
        >
          0
        </text>
        <text
          x={cx + r}
          y={cy + 18}
          textAnchor="middle"
          fontSize={9}
          fill="var(--muted)"
          opacity={0.6}
        >
          {GAUGE_MAX}
        </text>

        {/* Needle -- rotated from center */}
        <line
          x1={cx}
          y1={cy}
          x2={cx}
          y2={cy - needleLen}
          stroke="var(--text)"
          strokeWidth={2}
          strokeLinecap="round"
          transform={`rotate(${needleRotation}, ${cx}, ${cy})`}
        />

        {/* Center dot */}
        <circle cx={cx} cy={cy} r={6} fill="var(--panel-hover)" />
        <circle cx={cx} cy={cy} r={3} fill="var(--text)" />

        {/* VIX value -- large centered */}
        <text
          x={cx}
          y={cy - 24}
          textAnchor="middle"
          fontSize={34}
          fontWeight={800}
          fill="var(--text)"
          style={{ fontFamily: "'Inter', system-ui, sans-serif" }}
        >
          {vixClose.toFixed(2)}
        </text>

        {/* Zone label below score */}
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          fontSize={11}
          fontWeight={700}
          fill={zoneColor}
          style={{
            textTransform: "uppercase" as const,
            letterSpacing: "0.12em",
          }}
        >
          {zoneLabel}
        </text>
      </svg>
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
        Failed to load VIX data: {String((error as Error)?.message ?? "Unknown error")}
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

      {/* Position sizing NFA disclaimer */}
      <p className="text-[10px] text-[var(--muted)]">
        Position sizing recommendations are for educational purposes only and should not be construed as financial advice.
      </p>

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
          Failed to load VIX metrics: {String((metricsError as Error)?.message ?? "Unknown error")}
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

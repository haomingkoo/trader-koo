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
  [0, 12, "Complacency", "#38d39f"],
  [12, 18, "Calm", "#6dd5a0"],
  [18, 24, "Caution", "#f8c24e"],
  [24, 32, "Stress", "#ff9800"],
  [32, 50, "Fear", "#ff6b6b"],
  [50, 80, "Panic", "#e53935"],
];

const GAUGE_MAX = 80;

/** Gradient stop colors for the smooth arc */
const GRADIENT_STOPS: Array<{ offset: string; color: string }> = [
  { offset: "0%", color: "#38d39f" },
  { offset: "20%", color: "#6dd5a0" },
  { offset: "35%", color: "#f8c24e" },
  { offset: "50%", color: "#ff9800" },
  { offset: "70%", color: "#ff6b6b" },
  { offset: "100%", color: "#e53935" },
];

/* ── SVG VIX Gauge (CNN Fear & Greed inspired) ── */
function VixGauge({ vixClose }: { vixClose: number }) {
  const cx = 150;
  const cy = 150;
  const r = 120;
  const arcWidth = 14;
  const startAngle = Math.PI;
  const totalArc = Math.PI;

  // Smooth background arc path (semicircle)
  const arcStartX = cx + r * Math.cos(startAngle);
  const arcStartY = cy - r * Math.sin(startAngle);
  const arcEndX = cx + r * Math.cos(0);
  const arcEndY = cy - r * Math.sin(0);
  const arcPath = `M ${arcStartX} ${arcStartY} A ${r} ${r} 0 0 0 ${arcEndX} ${arcEndY}`;

  // Track arc uses the same path as the colored arc
  const trackPath = arcPath;

  // Zone labels positioned along the outside of the arc
  const labelR = r + 22;
  const zoneLabels = GAUGE_ZONES.map(([lo, hi, label]) => {
    const midVal = (lo + hi) / 2;
    const angle = startAngle - (midVal / GAUGE_MAX) * totalArc;
    const lx = cx + labelR * Math.cos(angle);
    const ly = cy - labelR * Math.sin(angle);
    return { label, x: lx, y: ly, angle };
  });

  // Tick marks at zone boundaries
  const tickInner = r - arcWidth / 2 - 2;
  const tickOuter = r + arcWidth / 2 + 2;
  const tickMarks = GAUGE_ZONES.slice(1).map(([lo]) => {
    const angle = startAngle - (lo / GAUGE_MAX) * totalArc;
    return {
      x1: cx + tickInner * Math.cos(angle),
      y1: cy - tickInner * Math.sin(angle),
      x2: cx + tickOuter * Math.cos(angle),
      y2: cy - tickOuter * Math.sin(angle),
    };
  });

  // Needle
  const clampedVix = Math.max(0, Math.min(vixClose, GAUGE_MAX));
  const needleAngle = startAngle - (clampedVix / GAUGE_MAX) * totalArc;
  const needleLen = r - arcWidth / 2 - 8;
  const nx = cx + needleLen * Math.cos(needleAngle);
  const ny = cy - needleLen * Math.sin(needleAngle);

  // Needle tip glow position
  const glowR = needleLen + 4;
  const glowX = cx + glowR * Math.cos(needleAngle);
  const glowY = cy - glowR * Math.sin(needleAngle);

  // Current zone
  const currentZone = GAUGE_ZONES.find(
    ([lo, hi]) => vixClose >= lo && vixClose < hi,
  );
  const zoneLabel = currentZone ? currentZone[2] : "Panic";
  const zoneColor = currentZone ? currentZone[3] : "#e53935";

  const glowColor = zoneColor;

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 300 190" className="w-full max-w-[340px]">
        <defs>
          {/* Gradient along the arc */}
          <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
            {GRADIENT_STOPS.map((s) => (
              <stop key={s.offset} offset={s.offset} stopColor={s.color} />
            ))}
          </linearGradient>
          {/* Needle tip glow */}
          <filter id="needleGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Subtle shadow beneath arc */}
          <filter id="arcShadow" x="-10%" y="-10%" width="120%" height="130%">
            <feGaussianBlur stdDeviation="2" result="shadow" />
            <feMerge>
              <feMergeNode in="shadow" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Dim track behind the colored arc */}
        <path
          d={trackPath}
          fill="none"
          stroke="var(--panel-hover)"
          strokeWidth={arcWidth + 4}
          strokeLinecap="round"
          opacity={0.5}
        />

        {/* Colored gradient arc */}
        <path
          d={arcPath}
          fill="none"
          stroke="url(#gaugeGrad)"
          strokeWidth={arcWidth}
          strokeLinecap="round"
          filter="url(#arcShadow)"
        />

        {/* Zone boundary tick marks */}
        {tickMarks.map((t, i) => (
          <line
            key={i}
            x1={t.x1}
            y1={t.y1}
            x2={t.x2}
            y2={t.y2}
            stroke="var(--bg)"
            strokeWidth={1.5}
            opacity={0.6}
          />
        ))}

        {/* Zone labels along the outside */}
        {zoneLabels.map((z) => {
          const angleDeg = (z.angle * 180) / Math.PI;
          // Rotate text to follow the arc, but keep readable
          const textRotation = angleDeg > 90 ? angleDeg - 180 : angleDeg;
          return (
            <text
              key={z.label}
              x={z.x}
              y={z.y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={7}
              fontWeight={600}
              fill="var(--muted)"
              opacity={0.7}
              transform={`rotate(${-textRotation}, ${z.x}, ${z.y})`}
            >
              {z.label}
            </text>
          );
        })}

        {/* Scale labels: 0 and GAUGE_MAX */}
        <text
          x={cx - r - 4}
          y={cy + 14}
          textAnchor="middle"
          fontSize={8}
          fill="var(--muted)"
          opacity={0.5}
        >
          0
        </text>
        <text
          x={cx + r + 4}
          y={cy + 14}
          textAnchor="middle"
          fontSize={8}
          fill="var(--muted)"
          opacity={0.5}
        >
          {GAUGE_MAX}
        </text>

        {/* Needle glow dot */}
        <circle
          cx={glowX}
          cy={glowY}
          r={4}
          fill={glowColor}
          opacity={0.6}
          filter="url(#needleGlow)"
        />

        {/* Needle line */}
        <line
          x1={cx}
          y1={cy}
          x2={nx}
          y2={ny}
          stroke="var(--text)"
          strokeWidth={1.5}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 2px ${glowColor})` }}
        />

        {/* Center dot */}
        <circle cx={cx} cy={cy} r={4} fill="var(--text)" />
        <circle cx={cx} cy={cy} r={2} fill="var(--bg)" />

        {/* VIX value (large, bold, centered below arc) */}
        <text
          x={cx}
          y={cy - 18}
          textAnchor="middle"
          fontSize={32}
          fontWeight={800}
          fill="var(--text)"
          style={{ fontFamily: "'Inter', system-ui, sans-serif" }}
        >
          {vixClose.toFixed(2)}
        </text>

        {/* Zone label below score */}
        <text
          x={cx}
          y={cy + 2}
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

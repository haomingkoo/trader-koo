import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useReport } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import type {
  SetupRow,
  RiskCondition,
  KeyChange,
  SetupEvaluation,
  SetupEvalFamily,
  SetupEvalValidity,
  SetupEvalAction,
  RegimeContext,
  YoloBlock,
} from "../api/types";
import Spinner from "../components/ui/Spinner";
import Badge, { tierVariant } from "../components/ui/Badge";
import Table from "../components/ui/Table";
import FearGreedGauge from "../components/FearGreedGauge";
import { PipelineStatusInline } from "../components/PipelineOpsPanel";

/* ── Helpers ── */

function formatTs(ts: string | null): { local: string; ny: string } {
  if (!ts) return { local: "\u2014", ny: "\u2014" };
  try {
    const d = new Date(ts);
    const local = d.toLocaleString();
    const ny = d.toLocaleString("en-US", { timeZone: "America/New_York" });
    return { local, ny };
  } catch {
    return { local: ts, ny: ts };
  }
}

function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null || !Number.isFinite(n)) return "\u2014";
  return n.toFixed(decimals);
}

function severityVariant(
  severity: string,
): "red" | "amber" | "green" | "muted" {
  const s = severity.toLowerCase();
  if (s === "hard" || s === "block" || s === "critical") return "red";
  if (s === "soft" || s === "warning" || s === "elevated") return "amber";
  if (s === "ok" || s === "normal" || s === "low") return "green";
  return "muted";
}

function biasVariant(
  bias: string | null,
): "green" | "red" | "amber" | "muted" {
  if (!bias) return "muted";
  const b = bias.toLowerCase();
  if (b.includes("bull") || b === "long") return "green";
  if (b.includes("bear") || b === "short") return "red";
  if (b.includes("neutral") || b === "flat") return "amber";
  return "muted";
}

/* ── Glassmorphism card wrapper ── */

function GlassCard({
  label,
  value,
  children,
  className = "",
}: {
  label?: string;
  value?: string | number;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl backdrop-blur-sm bg-[var(--panel)]/80 border border-[var(--line)] p-4 ${className}`}
    >
      {label && (
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          {label}
        </div>
      )}
      {value !== undefined && (
        <div className="text-lg font-bold tabular-nums text-[var(--text)]">
          {typeof value === "string" || typeof value === "number" ? (value ?? "\u2014") : String(value ?? "\u2014")}
        </div>
      )}
      {children}
    </div>
  );
}

/* ── Sub-components ── */

function SummaryKpiRow({
  generatedTs,
  trackedTickers,
  priceRows,
  priceDate,
  ingestStatus,
}: {
  generatedTs: string | null;
  trackedTickers: number | null;
  priceRows: number | null;
  priceDate: string | null;
  ingestStatus: string | null;
}) {
  const ts = formatTs(generatedTs);
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
      <GlassCard label="Generated (Local)" value={ts.local} />
      <GlassCard label="Generated (NY)" value={ts.ny} />
      <GlassCard
        label="Tracked Tickers"
        value={trackedTickers ?? "\u2014"}
      />
      <GlassCard
        label="Price Rows"
        value={priceRows != null ? priceRows.toLocaleString() : "\u2014"}
      />
      <GlassCard label="Latest Price Date" value={priceDate ?? "\u2014"} />
      <GlassCard label="Last Ingest" value={ingestStatus ?? "\u2014"} />
    </div>
  );
}

function RiskFiltersPanel({
  tradeMode,
  hardBlocks,
  softFlags,
  conditions,
}: {
  tradeMode: string;
  hardBlocks: number;
  softFlags: number;
  conditions: RiskCondition[];
}) {
  const [expanded, setExpanded] = useState(false);
  const modeUpper = tradeMode.toUpperCase();
  const modeVariant =
    modeUpper === "NORMAL"
      ? "green"
      : modeUpper === "CAUTION"
        ? "amber"
        : "red";

  return (
    <GlassCard>
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant={modeVariant} className="text-xs">
          {modeUpper} MODE
        </Badge>
        <span className="text-xs text-[var(--muted)]">
          Hard blocks:{" "}
          <strong className="text-[var(--text)]">{hardBlocks}</strong>
        </span>
        <span className="text-xs text-[var(--muted)]">
          Soft flags:{" "}
          <strong className="text-[var(--text)]">{softFlags}</strong>
        </span>
        {conditions.length > 0 && (
          <button
            onClick={() => setExpanded((p) => !p)}
            className="ml-auto text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
          >
            {expanded ? "Hide" : "Show"} conditions ({conditions.length})
          </button>
        )}
      </div>
      {expanded && conditions.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {conditions.map((c, i) => (
            <li key={i} className="flex items-start gap-2 text-xs">
              <Badge variant={severityVariant(c.severity)} className="shrink-0">
                {c.severity.toUpperCase()}
              </Badge>
              <span className="text-[var(--muted)]">
                {c.code && (
                  <span className="font-mono text-[var(--text)]">
                    {String(c.code)}
                  </span>
                )}{" "}
                {String(c.reason)}
              </span>
            </li>
          ))}
        </ul>
      )}
      {conditions.length === 0 && (
        <p className="mt-2 text-xs text-[var(--muted)]">
          No active risk conditions.
        </p>
      )}
    </GlassCard>
  );
}

function KeyChangesSection({ changes }: { changes: KeyChange[] }) {
  if (changes.length === 0) {
    return (
      <GlassCard label="Key Changes">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No key changes tonight.
        </p>
      </GlassCard>
    );
  }
  return (
    <GlassCard label="Key Changes">
      <ul className="mt-2 space-y-1.5">
        {changes.map((kc, i) => (
          <li key={i} className="text-xs">
            <strong className="text-[var(--text)]">{String(kc.title)}:</strong>{" "}
            <span className="text-[var(--muted)]">{String(kc.detail)}</span>
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}

function VixRegimeWidget({ regime }: { regime: RegimeContext | null }) {
  if (!regime) {
    return (
      <GlassCard label="VIX Regime Context">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No VIX regime data available.
        </p>
      </GlassCard>
    );
  }
  const vix = regime.vix;
  const health = regime.health;
  const overall = regime.overall;
  const riskState = (vix.risk_state ?? "unknown").replace(/_/g, " ");
  const healthState = (health.state ?? "unknown").replace(/_/g, " ");
  const participationBias = (overall.participation_bias ?? "unknown").replace(
    /_/g,
    " ",
  );

  const riskVariant =
    riskState.toLowerCase().includes("low") ||
    riskState.toLowerCase().includes("normal")
      ? "green"
      : riskState.toLowerCase().includes("elevated") ||
          riskState.toLowerCase().includes("caution")
        ? "amber"
        : "red";

  return (
    <GlassCard>
      <div className="flex flex-wrap items-center gap-4">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
          VIX Regime
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold tabular-nums text-[var(--text)]">
            {fmt(vix.close, 2)}
          </span>
          <Badge variant={riskVariant}>{riskState.toUpperCase()}</Badge>
        </div>
        <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
          <span>
            Health:{" "}
            <strong className="text-[var(--text)]">
              {healthState} ({fmt(health.score, 1)}/100)
            </strong>
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
          <span>
            Bias:{" "}
            <strong className="text-[var(--text)]">
              {participationBias}
            </strong>
          </span>
        </div>
        <Link
          to="/vix"
          className="ml-auto text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          Full VIX Analysis &rarr;
        </Link>
      </div>
    </GlassCard>
  );
}

function YoloDetectionCards({ yolo }: { yolo: YoloBlock }) {
  const summary = yolo.summary;
  const timeframes = yolo.timeframes ?? [];
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--muted)]">
        YOLO Pattern Detection
      </h3>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard
          label="Total Rows"
          value={summary.rows_total ?? "\u2014"}
        />
        <GlassCard
          label="Tickers w/ Patterns"
          value={summary.tickers_with_patterns ?? "\u2014"}
        />
        {timeframes.map((tf) => (
          <GlassCard
            key={tf.timeframe}
            label={`${tf.timeframe} Tickers`}
            value={tf.tickers_with_patterns ?? "\u2014"}
          />
        ))}
      </div>
    </div>
  );
}

/* ── Setup Quality Table ── */

function SetupQualitySection({
  rows,
  debateMap,
  activeTicker,
}: {
  rows: SetupRow[];
  debateMap: Map<string, NonNullable<import("../api/types").ChartCommentary["debate_v1"]>>;
  activeTicker: string;
}) {
  const [sortCol, setSortCol] = useState<string>("score");
  const [sortAsc, setSortAsc] = useState(false);
  const [filterTier, setFilterTier] = useState<string>("all");
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

  const filtered = useMemo(() => {
    let result = [...rows];
    if (filterTier !== "all") {
      result = result.filter(
        (r) => (r.setup_tier ?? "").toUpperCase() === filterTier,
      );
    }
    const dir = sortAsc ? 1 : -1;
    result.sort((a, b) => {
      const av = a[sortCol as keyof SetupRow];
      const bv = b[sortCol as keyof SetupRow];
      const an = Number(av);
      const bn = Number(bv);
      if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * dir;
      return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
    });
    return result;
  }, [rows, filterTier, sortCol, sortAsc]);

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc((p) => !p);
    } else {
      setSortCol(col);
      setSortAsc(col === "ticker");
    }
  };

  const columns: Array<{
    key: string;
    label: string;
    render?: (row: SetupRow) => React.ReactNode;
  }> = [
    {
      key: "ticker",
      label: "Ticker",
      render: (row) => (
        <Link
          to={`/chart?t=${row.ticker}`}
          className="font-mono font-bold text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
        >
          {row.ticker}
        </Link>
      ),
    },
    {
      key: "score",
      label: "Score",
      render: (row) => (
        <span className="tabular-nums">{fmt(row.score, 1)}</span>
      ),
    },
    {
      key: "pct_change",
      label: "% Chg",
      render: (row) => {
        const val = row.pct_change;
        if (val == null) return "\u2014";
        const color =
          val > 0
            ? "text-[var(--green)]"
            : val < 0
              ? "text-[var(--red)]"
              : "text-[var(--muted)]";
        return (
          <span className={`tabular-nums ${color}`}>
            {val > 0 ? "+" : ""}
            {val.toFixed(2)}%
          </span>
        );
      },
    },
    {
      key: "setup_tier",
      label: "Tier",
      render: (row) =>
        row.setup_tier ? (
          <Badge variant={tierVariant(row.setup_tier)}>
            {row.setup_tier}
          </Badge>
        ) : (
          "\u2014"
        ),
    },
    {
      key: "signal_bias",
      label: "Bias",
      render: (row) =>
        row.signal_bias ? (
          <Badge variant={biasVariant(row.signal_bias)}>
            {row.signal_bias}
          </Badge>
        ) : (
          "\u2014"
        ),
    },
    {
      key: "yolo_pattern",
      label: "YOLO",
      render: (row) => (
        <span className="text-xs text-[var(--muted)]">
          {row.yolo_pattern ?? "\u2014"}
        </span>
      ),
    },
    {
      key: "observation",
      label: "Observation",
      render: (row) => (
        <span className="text-xs text-[var(--muted)] max-w-[200px] truncate block" title={row.observation ?? ""}>
          {row.observation ?? "\u2014"}
        </span>
      ),
    },
    {
      key: "action",
      label: "Action",
      render: (row) => (
        <span className="text-xs text-[var(--muted)] max-w-[200px] truncate block" title={row.action ?? ""}>
          {row.action ?? "\u2014"}
        </span>
      ),
    },
  ];

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h3 className="text-sm font-semibold text-[var(--muted)]">
          Setup Quality ({filtered.length} setups)
        </h3>
        <div className="flex gap-1">
          {["all", "A", "B", "C"].map((tier) => (
            <button
              key={tier}
              onClick={() => setFilterTier(tier)}
              className={`rounded-md px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider transition-colors ${
                filterTier === tier
                  ? "bg-[var(--accent)] text-white"
                  : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {tier === "all" ? "All" : `Tier ${tier}`}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
          No setups match the current filter.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--panel)]">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-[var(--line)]">
                {columns.map((col) => (
                  <th
                    key={col.key}
                    className="cursor-pointer select-none px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)] hover:text-[var(--text)]"
                    onClick={() => handleSort(col.key)}
                  >
                    {col.label}
                    {sortCol === col.key && (
                      <span className="ml-1">
                        {sortAsc ? "\u25B2" : "\u25BC"}
                      </span>
                    )}
                  </th>
                ))}
                <th className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                  Debate
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => {
                const debate = debateMap.get(row.ticker);
                const isExpanded = expandedTicker === row.ticker;
                return (
                  <SetupTableRow
                    key={row.ticker}
                    row={row}
                    columns={columns}
                    debate={debate}
                    isExpanded={isExpanded}
                    isHighlighted={row.ticker === activeTicker}
                    onToggle={() =>
                      setExpandedTicker(isExpanded ? null : row.ticker)
                    }
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function SetupTableRow({
  row,
  columns,
  debate,
  isExpanded,
  isHighlighted,
  onToggle,
}: {
  row: SetupRow;
  columns: Array<{
    key: string;
    label: string;
    render?: (row: SetupRow) => React.ReactNode;
  }>;
  debate?: NonNullable<import("../api/types").ChartCommentary["debate_v1"]>;
  isExpanded: boolean;
  isHighlighted: boolean;
  onToggle: () => void;
}) {
  const agreementScore = debate?.consensus?.agreement_score;

  return (
    <>
      <tr className={`border-b border-[var(--line)] last:border-b-0 hover:bg-[var(--panel-hover)] transition-colors ${isHighlighted ? "bg-[var(--accent)]/10 ring-1 ring-inset ring-[var(--accent)]/30" : ""}`}>
        {columns.map((col) => (
          <td key={col.key} className="px-3 py-2 text-[var(--text)]">
            {col.render ? col.render(row) : String(row[col.key as keyof SetupRow] ?? "\u2014")}
          </td>
        ))}
        <td className="px-3 py-2">
          {debate ? (
            <button
              onClick={onToggle}
              className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
            >
              {agreementScore != null && (
                <span className="tabular-nums">
                  {agreementScore.toFixed(0)}%
                </span>
              )}
              <span>{isExpanded ? "\u25B2" : "\u25BC"}</span>
            </button>
          ) : (
            <span className="text-xs text-[var(--muted)]">{"\u2014"}</span>
          )}
        </td>
      </tr>
      {isExpanded && debate && (
        <tr className="border-b border-[var(--line)]">
          <td
            colSpan={columns.length + 1}
            className="bg-[var(--bg)] px-4 py-3"
          >
            <DebateVisualization debate={debate} />
          </td>
        </tr>
      )}
    </>
  );
}

function DebateVisualization({
  debate,
}: {
  debate: NonNullable<import("../api/types").ChartCommentary["debate_v1"]>;
}) {
  const consensus = debate.consensus;
  const roles = debate.roles ?? [];

  const agreementPct = consensus.agreement_score ?? 0;
  const meterColor =
    agreementPct >= 70
      ? "var(--green)"
      : agreementPct >= 40
        ? "var(--amber)"
        : "var(--red)";

  return (
    <div className="space-y-3">
      {/* Consensus header */}
      <div className="flex flex-wrap items-center gap-3">
        <Badge
          variant={
            consensus.consensus_state === "ready"
              ? "green"
              : consensus.consensus_state === "conditional"
                ? "amber"
                : "red"
          }
        >
          {(consensus.consensus_state ?? "unknown").toUpperCase()}
        </Badge>
        <Badge variant={biasVariant(consensus.consensus_bias)}>
          {(consensus.consensus_bias ?? "neutral").toUpperCase()}
        </Badge>
        <span className="text-xs text-[var(--muted)]">
          Disagreements:{" "}
          <strong className="text-[var(--text)]">
            {String(consensus.disagreement_count ?? "\u2014")}
          </strong>
        </span>
      </div>

      {/* Agreement meter */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-[var(--muted)]">
          Agreement
        </span>
        <div className="relative h-2 flex-1 rounded-full bg-[var(--line)]">
          <div
            className="absolute left-0 top-0 h-full rounded-full transition-all"
            style={{
              width: `${Math.min(100, Math.max(0, agreementPct))}%`,
              backgroundColor: meterColor,
            }}
          />
        </div>
        <span
          className="text-xs font-bold tabular-nums"
          style={{ color: meterColor }}
        >
          {agreementPct.toFixed(0)}%
        </span>
      </div>

      {/* Role breakdown */}
      {roles.length > 0 ? (
        <div className="space-y-2">
          {roles.map((role, i) => {
            const isBull =
              role.stance.toLowerCase().includes("bull") ||
              role.stance.toLowerCase() === "long";
            const isBear =
              role.stance.toLowerCase().includes("bear") ||
              role.stance.toLowerCase() === "short";
            const barColor = isBull
              ? "var(--green)"
              : isBear
                ? "var(--red)"
                : "var(--amber)";
            const confPct = Math.min(100, Math.max(0, role.confidence * 100));
            return (
              <div key={i} className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="w-28 text-xs font-medium capitalize text-[var(--text)]">
                    {role.role.replace(/_/g, " ")}
                  </span>
                  <Badge
                    variant={isBull ? "green" : isBear ? "red" : "amber"}
                    className="w-16 justify-center"
                  >
                    {role.stance.toUpperCase()}
                  </Badge>
                  <div className="relative h-1.5 flex-1 rounded-full bg-[var(--line)]">
                    <div
                      className="absolute left-0 top-0 h-full rounded-full transition-all"
                      style={{
                        width: `${confPct}%`,
                        backgroundColor: barColor,
                      }}
                    />
                  </div>
                  <span className="w-10 text-right text-[10px] tabular-nums text-[var(--muted)]">
                    {(role.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                {role.evidence.length > 0 && (
                  <ul className="ml-32 space-y-0.5">
                    {role.evidence.filter(Boolean).map((ev, j) => (
                      <li
                        key={j}
                        className="text-[10px] text-[var(--muted)] before:mr-1 before:content-['\u2022']"
                      >
                        {String(ev)}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-xs text-[var(--muted)]">No debate roles available.</p>
      )}
    </div>
  );
}

/* ── Setup Evaluation Panel ── */

function SetupEvaluationPanel({
  evaluation,
}: {
  evaluation: SetupEvaluation | Record<string, never>;
}) {
  const eval_ = evaluation as SetupEvaluation;

  if (!eval_ || !Object.keys(eval_).length) {
    return (
      <GlassCard label="Setup Evaluation">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No setup evaluation data available.
        </p>
      </GlassCard>
    );
  }

  if (!eval_.enabled || eval_.error) {
    return (
      <GlassCard label="Setup Evaluation">
        <p className="mt-1 text-xs text-[var(--muted)]">
          Setup evaluation unavailable ({eval_.error ?? eval_.reason ?? "disabled"}).
        </p>
      </GlassCard>
    );
  }

  const overall = eval_.overall ?? {};
  const families = eval_.by_family ?? [];
  const validities = eval_.by_validity_days ?? [];
  const actions = eval_.improvement_actions ?? [];

  const longFamilies = families.filter(
    (f) => f.call_direction.toLowerCase() === "long",
  );
  const shortFamilies = families.filter(
    (f) => f.call_direction.toLowerCase() === "short",
  );

  const bestValidity = validities
    .filter((v) => (v.calls ?? 0) > 0)
    .sort(
      (a, b) => (b.expectancy_pct ?? -999) - (a.expectancy_pct ?? -999),
    )[0];

  const familyColumns: Array<{
    key: keyof SetupEvalFamily & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "setup_family", label: "Setup Family" },
    { key: "calls", label: "Calls" },
    {
      key: "hit_rate_pct",
      label: "Hit Rate %",
      render: (v: unknown) => fmt(v as number | null, 1),
    },
    {
      key: "avg_signed_return_pct",
      label: "Avg Return %",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    {
      key: "expectancy_pct",
      label: "Expectancy %",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
  ];

  const validityColumns: Array<{
    key: keyof SetupEvalValidity & string;
    label: string;
    render?: (v: unknown) => React.ReactNode;
  }> = [
    { key: "validity_days", label: "Validity (days)" },
    { key: "calls", label: "Calls" },
    {
      key: "hit_rate_pct",
      label: "Hit Rate %",
      render: (v: unknown) => fmt(v as number | null, 1),
    },
    {
      key: "avg_signed_return_pct",
      label: "Avg Return %",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
    {
      key: "expectancy_pct",
      label: "Expectancy %",
      render: (v: unknown) => fmt(v as number | null, 2),
    },
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-[var(--muted)]">
        Setup Evaluation
      </h3>

      <p className="text-xs text-[var(--muted)]">
        Historical calibration built from archived setup calls and the later price outcomes that followed their
        validity windows. These figures are computed from stored report snapshots and subsequent closes, not manually keyed in.
      </p>

      {/* Overall KPIs */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <GlassCard
          label="Hit Rate"
          value={
            overall.hit_rate_pct != null
              ? `${overall.hit_rate_pct.toFixed(1)}%`
              : "\u2014"
          }
        />
        <GlassCard
          label="Avg Signed Return"
          value={
            overall.avg_signed_return_pct != null
              ? `${overall.avg_signed_return_pct.toFixed(2)}%`
              : "\u2014"
          }
        />
        <GlassCard
          label="Expectancy"
          value={
            overall.expectancy_pct != null
              ? `${overall.expectancy_pct.toFixed(2)}%`
              : "\u2014"
          }
        />
        <GlassCard label="Scored Calls" value={eval_.scored_calls ?? "\u2014"} />
      </div>

      {/* Summary line */}
      <p className="text-xs text-[var(--muted)]">
        Window: {eval_.window_days ?? "\u2014"}d | Min sample:{" "}
        {eval_.min_sample ?? "\u2014"} | Open calls:{" "}
        {eval_.open_calls ?? "\u2014"} | Hit threshold:{" "}
        {eval_.hit_threshold_pct ?? "\u2014"}%
        {eval_.latest_scored_asof &&
          ` | Latest scored: ${eval_.latest_scored_asof}`}
        {bestValidity &&
          ` | Best horizon: ${bestValidity.validity_days}d (${fmt(bestValidity.expectancy_pct, 2)}%)`}
      </p>

      {/* By-family tables */}
      {longFamilies.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-semibold text-[var(--green)]">
            Long Families
          </h4>
          <Table
            columns={familyColumns}
            data={longFamilies}
          />
        </div>
      )}
      {shortFamilies.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-semibold text-[var(--red)]">
            Short Families
          </h4>
          <Table
            columns={familyColumns}
            data={shortFamilies}
          />
        </div>
      )}

      {/* Validity breakdown */}
      {validities.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-semibold text-[var(--muted)]">
            By Validity Window
          </h4>
          <Table
            columns={validityColumns}
            data={validities}
          />
        </div>
      )}

      {/* Improvement actions */}
      <div>
        <h4 className="mb-1 text-xs font-semibold text-[var(--muted)]">
          Improvement Actions
        </h4>
        {actions.length > 0 ? (
          <ul className="space-y-2">
            {actions.map((action, i) => (
              <ImprovementAction key={i} action={action} />
            ))}
          </ul>
        ) : (
          <p className="text-xs text-[var(--muted)]">
            No tuning actions yet. Collect more scored calls.
          </p>
        )}
      </div>
    </div>
  );
}

function ImprovementAction({ action }: { action: SetupEvalAction }) {
  const priorityVariant =
    action.priority.toLowerCase() === "high"
      ? "red"
      : action.priority.toLowerCase() === "medium"
        ? "amber"
        : "muted";
  return (
    <li className="flex items-start gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3 text-xs">
      <Badge variant={priorityVariant} className="shrink-0">
        {action.priority.toUpperCase()}
      </Badge>
      <div>
        <span className="font-medium text-[var(--text)]">
          {action.scope.replace(/_/g, " ")}:
        </span>{" "}
        <span className="text-[var(--muted)]">{String(action.reason)}</span>
        {action.recommendation && (
          <p className="mt-1 text-[var(--amber)]">
            Tune: {String(action.recommendation)}
          </p>
        )}
      </div>
    </li>
  );
}

/* ── Main Page ── */

export default function ReportPage() {
  const { data, isLoading, error } = useReport();
  const activeTicker = useChartStore((s) => s.ticker);

  if (isLoading) return <Spinner className="mt-12" />;
  if (error) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--red)]">
        Failed to load report: {String((error as Error)?.message ?? "Unknown error")}
      </div>
    );
  }

  const latest = data?.latest;
  if (!data?.ok || !latest || !Object.keys(latest).length) {
    return (
      <div className="mt-12 text-center text-sm text-[var(--muted)]">
        No report data available yet.
      </div>
    );
  }

  const counts = latest.counts ?? { tracked_tickers: null, price_rows: null };
  const yoloBlock: YoloBlock = latest.yolo ?? {
    summary: { rows_total: null, tickers_with_patterns: null },
    timeframes: [],
  };
  const ingestRun = latest.latest_ingest_run ?? { status: null };
  const signals = latest.signals ?? {
    tonight_key_changes: [],
    regime_context: null,
    setup_quality_top: [],
    setup_evaluation: {},
  };
  const reportDetail = typeof data.detail === "string" && data.detail.trim().length > 0 ? data.detail.trim() : null;
  const risk = latest.risk_filters ?? {
    trade_mode: "normal",
    hard_blocks: 0,
    soft_flags: 0,
    conditions: [],
  };
  const setupRows = signals.setup_quality_top ?? [];

  // Build a debate map from setup rows (debate_v1 data is on chart_commentary per-ticker,
  // but for the report view we don't have individual chart data loaded. We pass an empty map
  // and the debate column will show "\u2014" for tickers without preloaded debate data.)
  const debateMap = new Map<
    string,
    NonNullable<import("../api/types").ChartCommentary["debate_v1"]>
  >();

  return (
    <div className="space-y-6">
      {/* NFA disclaimer banner */}
      <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/5 px-4 py-2 text-xs text-[var(--amber)]">
        <strong>Research only. Not financial advice.</strong> This dashboard can be wrong, stale, or incomplete.
        Treat it as a decision-support tool, not an instruction to buy or sell. Past performance does not guarantee future results.
      </div>

      {/* Pipeline status inline */}
      <PipelineStatusInline />

      {reportDetail && (
        <div className="rounded-lg border border-[var(--amber)]/30 bg-[var(--amber)]/8 px-4 py-3 text-xs text-[var(--amber)]">
          <strong>Report status:</strong> {reportDetail}
        </div>
      )}

      <h2 className="text-xl font-bold tracking-tight">Daily Report</h2>

      {/* 1. Summary KPI cards */}
      <SummaryKpiRow
        generatedTs={latest.generated_ts}
        trackedTickers={counts.tracked_tickers}
        priceRows={counts.price_rows}
        priceDate={latest.latest_data?.price_date ?? null}
        ingestStatus={ingestRun.status}
      />

      {/* 2. Risk Filters */}
      <RiskFiltersPanel
        tradeMode={String(risk.trade_mode ?? "normal")}
        hardBlocks={risk.hard_blocks ?? 0}
        softFlags={risk.soft_flags ?? 0}
        conditions={risk.conditions ?? []}
      />

      {/* 3. Key Changes */}
      <KeyChangesSection changes={signals.tonight_key_changes ?? []} />

      {/* 4. Macro context */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <FearGreedGauge />
        <VixRegimeWidget regime={signals.regime_context} />
      </div>

      {/* 5. YOLO Detection */}
      <YoloDetectionCards yolo={yoloBlock} />

      {/* 6 + 7. Setup Quality Table with Debate */}
      <SetupQualitySection rows={setupRows} debateMap={debateMap} activeTicker={activeTicker} />

      {/* 8. Setup Evaluation */}
      <SetupEvaluationPanel
        evaluation={signals.setup_evaluation ?? {}}
      />
    </div>
  );
}

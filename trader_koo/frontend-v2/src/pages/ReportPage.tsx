import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useReport } from "../api/hooks";
import { useChartStore } from "../stores/chartStore";
import type {
  SetupRow,
  YoloBlock,
} from "../api/types";
import Spinner from "../components/ui/Spinner";
import Badge, { tierVariant } from "../components/ui/Badge";
import FearGreedGauge from "../components/FearGreedGauge";
import { PipelineStatusInline } from "../components/PipelineOpsPanel";
import {
  KeyChangesSection,
  RiskFiltersPanel,
  SummaryKpiRow,
  VixRegimeWidget,
  YoloDetectionCards,
} from "../components/report/ReportOverviewSections";
import {
  biasVariant,
  formatReportNumber as fmt,
} from "../components/report/reportShared";
import SetupEvaluationPanel from "../components/report/SetupEvaluationPanel";

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

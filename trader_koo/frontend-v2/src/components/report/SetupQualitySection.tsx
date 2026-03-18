import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { ChartCommentary, SetupRow } from "../../api/types";
import Badge, { tierVariant } from "../ui/Badge";
import { biasVariant, formatReportNumber as fmt } from "./reportShared";

type DebateData = NonNullable<ChartCommentary["debate_v1"]>;

function DebateVisualization({ debate }: { debate: DebateData }) {
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
        <span className="text-xs font-bold tabular-nums" style={{ color: meterColor }}>
          {agreementPct.toFixed(0)}%
        </span>
      </div>

      {roles.length > 0 ? (
        <div className="space-y-2">
          {roles.map((role, index) => {
            const stance = role.stance.toLowerCase();
            const isBull = stance.includes("bull") || stance === "long";
            const isBear = stance.includes("bear") || stance === "short";
            const barColor = isBull
              ? "var(--green)"
              : isBear
                ? "var(--red)"
                : "var(--amber)";
            const confPct = Math.min(100, Math.max(0, role.confidence * 100));

            return (
              <div key={index} className="space-y-1">
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
                      style={{ width: `${confPct}%`, backgroundColor: barColor }}
                    />
                  </div>
                  <span className="w-10 text-right text-[10px] tabular-nums text-[var(--muted)]">
                    {(role.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                {role.evidence.length > 0 && (
                  <ul className="ml-32 space-y-0.5">
                    {role.evidence.filter(Boolean).map((evidence, evidenceIndex) => (
                      <li
                        key={evidenceIndex}
                        className="text-[10px] text-[var(--muted)] before:mr-1 before:content-['\u2022']"
                      >
                        {String(evidence)}
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
  debate?: DebateData;
  isExpanded: boolean;
  isHighlighted: boolean;
  onToggle: () => void;
}) {
  const agreementScore = debate?.consensus?.agreement_score;

  return (
    <>
      <tr
        className={`border-b border-[var(--line)] last:border-b-0 hover:bg-[var(--panel-hover)] transition-colors ${
          isHighlighted
            ? "bg-[var(--accent)]/10 ring-1 ring-inset ring-[var(--accent)]/30"
            : ""
        }`}
      >
        {columns.map((column) => (
          <td key={column.key} className="px-3 py-2 text-[var(--text)]">
            {column.render
              ? column.render(row)
              : String(row[column.key as keyof SetupRow] ?? "\u2014")}
          </td>
        ))}
        <td className="px-3 py-2">
          {debate ? (
            <button
              onClick={onToggle}
              className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
            >
              {agreementScore != null && (
                <span className="tabular-nums">{agreementScore.toFixed(0)}%</span>
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
          <td colSpan={columns.length + 1} className="bg-[var(--bg)] px-4 py-3">
            <DebateVisualization debate={debate} />
          </td>
        </tr>
      )}
    </>
  );
}

export default function SetupQualitySection({
  rows,
  debateMap,
  activeTicker,
}: {
  rows: SetupRow[];
  debateMap: Map<string, DebateData>;
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
        (row) => (row.setup_tier ?? "").toUpperCase() === filterTier,
      );
    }
    const direction = sortAsc ? 1 : -1;
    result.sort((a, b) => {
      const av = a[sortCol as keyof SetupRow];
      const bv = b[sortCol as keyof SetupRow];
      const an = Number(av);
      const bn = Number(bv);
      if (Number.isFinite(an) && Number.isFinite(bn)) return (an - bn) * direction;
      return String(av ?? "").localeCompare(String(bv ?? "")) * direction;
    });
    return result;
  }, [rows, filterTier, sortCol, sortAsc]);

  const handleSort = (column: string) => {
    if (sortCol === column) {
      setSortAsc((prev) => !prev);
    } else {
      setSortCol(column);
      setSortAsc(column === "ticker");
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
          className="font-mono font-bold text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
        >
          {row.ticker}
        </Link>
      ),
    },
    {
      key: "score",
      label: "Score",
      render: (row) => <span className="tabular-nums">{fmt(row.score, 1)}</span>,
    },
    {
      key: "pct_change",
      label: "% Chg",
      render: (row) => {
        const value = row.pct_change;
        if (value == null) return "\u2014";
        const color =
          value > 0
            ? "text-[var(--green)]"
            : value < 0
              ? "text-[var(--red)]"
              : "text-[var(--muted)]";
        return (
          <span className={`tabular-nums ${color}`}>
            {value > 0 ? "+" : ""}
            {value.toFixed(2)}%
          </span>
        );
      },
    },
    {
      key: "setup_tier",
      label: "Tier",
      render: (row) =>
        row.setup_tier ? (
          <Badge variant={tierVariant(row.setup_tier)}>{row.setup_tier}</Badge>
        ) : (
          "\u2014"
        ),
    },
    {
      key: "signal_bias",
      label: "Bias",
      render: (row) =>
        row.signal_bias ? (
          <Badge variant={biasVariant(row.signal_bias)}>{row.signal_bias}</Badge>
        ) : (
          "\u2014"
        ),
    },
    {
      key: "yolo_pattern",
      label: "YOLO",
      render: (row) => (
        <span className="text-xs text-[var(--muted)]">{row.yolo_pattern ?? "\u2014"}</span>
      ),
    },
    {
      key: "observation",
      label: "Observation",
      render: (row) => (
        <span
          className="block max-w-[200px] truncate text-xs text-[var(--muted)]"
          title={row.observation ?? ""}
        >
          {row.observation ?? "\u2014"}
        </span>
      ),
    },
    {
      key: "action",
      label: "Action",
      render: (row) => (
        <span
          className="block max-w-[200px] truncate text-xs text-[var(--muted)]"
          title={row.action ?? ""}
        >
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
                {columns.map((column) => (
                  <th
                    key={column.key}
                    className="cursor-pointer select-none px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)] hover:text-[var(--text)]"
                    onClick={() => handleSort(column.key)}
                  >
                    {column.label}
                    {sortCol === column.key && (
                      <span className="ml-1">{sortAsc ? "\u25B2" : "\u25BC"}</span>
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
                    onToggle={() => setExpandedTicker(isExpanded ? null : row.ticker)}
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

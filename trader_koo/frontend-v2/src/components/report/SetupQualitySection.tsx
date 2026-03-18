import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { ChartCommentary, SetupRow } from "../../api/types";
import Badge, { tierVariant } from "../ui/Badge";
import { biasVariant, formatReportNumber as fmt } from "./reportShared";

type DebateData = NonNullable<ChartCommentary["debate_v1"]>;

function normalizeDebate(
  raw: SetupRow["debate_v1"] | DebateData | null | undefined,
): DebateData | undefined {
  const debate = raw as DebateData | null | undefined;
  if (!debate || typeof debate !== "object" || !debate.consensus) return undefined;
  return {
    version: Number(debate.version ?? 1),
    consensus: {
      consensus_state: String(debate.consensus.consensus_state ?? "unknown"),
      consensus_bias: String(debate.consensus.consensus_bias ?? "neutral"),
      agreement_score: Number(debate.consensus.agreement_score ?? 0),
      disagreement_count: Number(debate.consensus.disagreement_count ?? 0),
    },
    roles: Array.isArray(debate.roles)
      ? debate.roles.map((role: DebateData["roles"][number]) => ({
          role: String(role.role ?? "unknown"),
          stance: String(role.stance ?? "neutral"),
          confidence: Number(role.confidence ?? 0),
          evidence: Array.isArray(role.evidence)
            ? role.evidence.map((item: string) => String(item))
            : [],
        }))
      : [],
  };
}

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
          <button
            onClick={onToggle}
            className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
          >
            {agreementScore != null ? (
              <span className="tabular-nums">{agreementScore.toFixed(0)}%</span>
            ) : (
              <span>View</span>
            )}
            <span>{isExpanded ? "\u25B2" : "\u25BC"}</span>
          </button>
        </td>
      </tr>
      {isExpanded && (
        <tr className="border-b border-[var(--line)]">
          <td colSpan={columns.length + 1} className="bg-[var(--bg)] px-4 py-3">
            <div className="space-y-3">
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3">
                  <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
                    Setup Read
                  </div>
                  <p className="mt-2 text-sm text-[var(--text)]">
                    {row.observation ?? "No observation available."}
                  </p>
                </div>
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3">
                  <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
                    Plan
                  </div>
                  <p className="mt-2 text-sm text-[var(--text)]">
                    {row.action ?? "No action plan available."}
                  </p>
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3">
                  <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
                    Risk
                  </div>
                  <p className="mt-2 text-sm text-[var(--muted)]">
                    {row.risk_note ?? "No risk note available."}
                  </p>
                </div>
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3">
                  <div className="text-[10px] font-semibold uppercase tracking-widest text-[var(--muted)]">
                    Technical
                  </div>
                  <p className="mt-2 text-sm text-[var(--muted)]">
                    {row.technical_read ?? "No technical summary available."}
                  </p>
                </div>
              </div>

              {row.yolo_pattern && (
                <div className="flex flex-wrap gap-2">
                  <Badge variant="muted">YOLO {String(row.yolo_pattern)}</Badge>
                  {Boolean(row.primary_yolo_recency) && (
                    <Badge variant="muted">{String(row.primary_yolo_recency)}</Badge>
                  )}
                  {Boolean(row.yolo_bias) && (
                    <Badge variant={biasVariant(String(row.yolo_bias))}>{String(row.yolo_bias)}</Badge>
                  )}
                </div>
              )}

              {debate ? (
                <DebateVisualization debate={debate} />
              ) : (
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3 text-xs text-[var(--muted)]">
                  Debate detail is not available for this setup snapshot.
                </div>
              )}
            </div>
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
      key: "setup",
      label: "Setup",
      render: (row) => (
        <span
          className="block max-w-[260px] truncate text-xs text-[var(--muted)]"
          title={row.action ?? row.observation ?? ""}
        >
          {row.action ?? row.observation ?? "\u2014"}
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
        <div className="space-y-3">
          <div className="space-y-3 md:hidden">
            {filtered.map((row) => {
              const debate = debateMap.get(row.ticker) ?? normalizeDebate(row.debate_v1);
              const isExpanded = expandedTicker === row.ticker;
              return (
                <div
                  key={row.ticker}
                  className={`rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 ${
                    row.ticker === activeTicker ? "ring-1 ring-[var(--accent)]/40" : ""
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <Link
                        to={`/chart?t=${row.ticker}`}
                        className="font-mono text-lg font-bold text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
                      >
                        {row.ticker}
                      </Link>
                      <div className="mt-1 text-xs text-[var(--muted)]">
                        Score {fmt(row.score, 1)}
                      </div>
                    </div>
                    <div className="flex flex-wrap justify-end gap-2">
                      {row.setup_tier ? (
                        <Badge variant={tierVariant(row.setup_tier)}>{row.setup_tier}</Badge>
                      ) : null}
                      {row.signal_bias ? (
                        <Badge variant={biasVariant(row.signal_bias)}>{row.signal_bias}</Badge>
                      ) : null}
                      {debate?.consensus?.agreement_score != null ? (
                        <Badge variant="muted">
                          Debate {debate.consensus.agreement_score.toFixed(0)}%
                        </Badge>
                      ) : null}
                    </div>
                  </div>

                  <p className="mt-3 text-sm text-[var(--text)]">
                    {row.action ?? row.observation ?? "No setup note available."}
                  </p>

                  <button
                    type="button"
                    onClick={() => setExpandedTicker(isExpanded ? null : row.ticker)}
                    className="mt-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
                  >
                    {isExpanded ? "Hide details" : "Show details"}
                  </button>

                  {isExpanded && (
                    <div className="mt-3 space-y-3 border-t border-[var(--line)] pt-3">
                      <div className="text-xs text-[var(--muted)]">
                        <strong className="text-[var(--text)]">Observation:</strong>{" "}
                        {row.observation ?? "\u2014"}
                      </div>
                      <div className="text-xs text-[var(--muted)]">
                        <strong className="text-[var(--text)]">Risk:</strong>{" "}
                        {row.risk_note ?? "\u2014"}
                      </div>
                      <div className="text-xs text-[var(--muted)]">
                        <strong className="text-[var(--text)]">Technical:</strong>{" "}
                        {row.technical_read ?? "\u2014"}
                      </div>
                      {debate ? (
                        <DebateVisualization debate={debate} />
                      ) : (
                        <div className="rounded-xl border border-[var(--line)] bg-[var(--bg)] p-3 text-xs text-[var(--muted)]">
                          Debate detail is not available for this setup snapshot.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="hidden overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--panel)] md:block">
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
                    Details
                  </th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((row) => {
                  const debate = debateMap.get(row.ticker) ?? normalizeDebate(row.debate_v1);
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
        </div>
      )}
    </div>
  );
}

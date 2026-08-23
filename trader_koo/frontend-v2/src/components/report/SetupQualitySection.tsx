import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { SetupRow } from "../../api/types";
import Badge from "../ui/Badge";
import { tierVariant } from "../ui/badgeUtils";
import { biasVariant } from "./reportShared";
import {
  buildSetupQualityView,
  type DebateData,
  type SetupQualityItem,
} from "./setupQualityView";

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
                        className="text-[10px] text-[var(--muted)] before:mr-1 before:content-['·']"
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
        <p className="text-xs text-[var(--muted)]">No rule perspectives available.</p>
      )}
    </div>
  );
}

function SetupTableRow({
  item,
  columns,
  isExpanded,
  isHighlighted,
  onToggle,
}: {
  item: SetupQualityItem;
  columns: Array<{
    key: string;
    label: string;
    className?: string;
    render: (item: SetupQualityItem) => React.ReactNode;
  }>;
  isExpanded: boolean;
  isHighlighted: boolean;
  onToggle: () => void;
}) {
  const { row, debate } = item;

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
          <td
            key={column.key}
            className={`px-3 py-3 align-top text-[var(--text)] ${column.className ?? ""}`}
          >
            {column.render(item)}
          </td>
        ))}
        <td className="px-3 py-3 align-top">
          <button
            type="button"
            onClick={onToggle}
            aria-expanded={isExpanded}
            className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
          >
            <span className="whitespace-nowrap tabular-nums">{item.ruleReviewLabel}</span>
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

              <div className="flex flex-wrap gap-2">
                {row.earnings_within_5d && (
                  <Badge variant="amber">
                    Earnings {row.earnings_date ?? ""}{" "}
                    ({row.days_to_earnings != null ? `${row.days_to_earnings}d` : "TBD"})
                  </Badge>
                )}
                {row.yolo_pattern && (
                  <>
                    <Badge variant="muted">YOLO {String(row.yolo_pattern)}</Badge>
                    {Boolean(row.primary_yolo_recency) && (
                      <Badge variant="muted">{String(row.primary_yolo_recency)}</Badge>
                    )}
                    {Boolean(row.yolo_bias) && (
                      <Badge variant={biasVariant(String(row.yolo_bias))}>{String(row.yolo_bias)}</Badge>
                    )}
                    {row.yolo_score_eligible === false && (
                      <Badge variant="amber">Low-conf pattern</Badge>
                    )}
                  </>
                )}
              </div>

              {debate ? (
                <DebateVisualization debate={debate} />
              ) : (
                <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3 text-xs text-[var(--muted)]">
                  Deterministic rule-review detail is not available for this setup snapshot.
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
  const [showAll, setShowAll] = useState(false);
  const DEFAULT_VISIBLE = 10;

  const items = useMemo(
    () => buildSetupQualityView(rows, debateMap, {
      tier: filterTier,
      sortKey: sortCol,
      ascending: sortAsc,
    }),
    [rows, debateMap, filterTier, sortCol, sortAsc],
  );

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
    className?: string;
    render: (item: SetupQualityItem) => React.ReactNode;
  }> = [
    {
      key: "ticker",
      label: "Ticker",
      className: "w-[92px]",
      render: ({ row, ticker }) => (
        <span className="flex items-center gap-1.5">
          <Link
            to={`/chart?t=${ticker}`}
            className="font-mono font-bold text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
          >
            {ticker}
          </Link>
          {row.earnings_within_5d && (
            <Badge variant="amber" className="text-[9px] px-1.5 py-0">
              E {row.days_to_earnings != null ? `${row.days_to_earnings}d` : ""}
            </Badge>
          )}
        </span>
      ),
    },
    {
      key: "score",
      label: "Score",
      className: "w-[72px]",
      render: (item) => <span className="tabular-nums">{item.scoreText}</span>,
    },
    {
      key: "calibrated_hit_prob",
      label: "Prob",
      className: "w-[88px]",
      render: (item) => {
        if (!item.probability) return "\u2014";
        return (
          <div className="flex flex-col gap-1">
            <Badge variant={item.probability.tone}>{item.probability.pctText}</Badge>
            {item.probability.sampleText && (
              <span className="text-[10px] text-[var(--muted)]">
                {item.probability.sampleText}
              </span>
            )}
          </div>
        );
      },
    },
    {
      key: "setup_tier",
      label: "Tier",
      className: "w-[64px]",
      render: ({ row }) =>
        row.setup_tier ? (
          <Badge variant={tierVariant(row.setup_tier)}>{row.setup_tier}</Badge>
        ) : (
          "\u2014"
        ),
    },
    {
      key: "signal_bias",
      label: "Bias",
      className: "w-[96px]",
      render: ({ row }) =>
        row.signal_bias ? (
          <Badge variant={biasVariant(row.signal_bias)}>{row.signal_bias}</Badge>
        ) : (
          "\u2014"
        ),
    },
    {
      key: "options_underpriced_score",
      label: "Options",
      className: "w-[178px]",
      render: (item) => {
        if (!item.options) return "\u2014";
        return (
          <div className="flex flex-col gap-1">
            <Badge
              variant={item.options.tone}
              className="h-auto w-fit max-w-[160px] whitespace-normal py-1 text-left leading-4"
            >
              {item.options.label}
            </Badge>
            {item.options.ranksText && (
              <span className="text-[10px] tabular-nums text-[var(--muted)]">
                {item.options.ranksText}
              </span>
            )}
          </div>
        );
      },
    },
    {
      key: "news_sentiment_score",
      label: "News",
      className: "w-[92px]",
      render: (item) => {
        if (!item.news) return "\u2014";
        return (
          <div className="flex flex-col gap-1">
            <Badge variant={item.news.tone}>{item.news.scoreText}</Badge>
            {item.news.macroText ? (
              <span className="text-[10px] tabular-nums text-[var(--muted)]">
                {item.news.macroText}
              </span>
            ) : null}
          </div>
        );
      },
    },
    {
      key: "setup",
      label: "Decision note",
      className: "w-[42%] min-w-[360px]",
      render: (item) => (
        <span className="block max-w-[68ch] whitespace-normal text-xs leading-5 text-[var(--text)]">
          {item.decisionNote}
        </span>
      ),
    },
  ];

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h3 className="text-sm font-semibold text-[var(--muted)]">
          Setup Quality ({items.length} setups)
        </h3>
        <div className="flex gap-1">
          {["all", "A", "B", "C"].map((tier) => (
            <button
              type="button"
              key={tier}
              onClick={() => setFilterTier(tier)}
              aria-pressed={filterTier === tier}
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

      {items.length === 0 ? (
        <div className="rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 text-center text-sm text-[var(--muted)]">
          No setups match the current filter.
        </div>
      ) : (
        <div className="space-y-3">
          <div className="space-y-3 2xl:hidden">
            {(showAll ? items : items.slice(0, DEFAULT_VISIBLE)).map((item) => {
              const { row, ticker, debate } = item;
              const isExpanded = expandedTicker === ticker;
              return (
                <div
                  key={ticker}
                  className={`rounded-xl border border-[var(--line)] bg-[var(--panel)] p-4 ${
                    ticker === activeTicker ? "ring-1 ring-[var(--accent)]/40" : ""
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <span className="flex items-center gap-2">
                        <Link
                          to={`/chart?t=${ticker}`}
                          className="font-mono text-lg font-bold text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
                        >
                          {ticker}
                        </Link>
                        {row.earnings_within_5d && (
                          <Badge variant="amber" className="text-[9px] px-1.5 py-0">
                            E {row.days_to_earnings != null ? `${row.days_to_earnings}d` : ""}
                          </Badge>
                        )}
                      </span>
                      <div className="mt-1 text-xs text-[var(--muted)]">
                        Score {item.scoreText}
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
                          {item.ruleReviewLabel}
                        </Badge>
                      ) : null}
                      {item.probability ? (
                        <Badge variant={item.probability.tone}>
                          P {item.probability.pctText}
                        </Badge>
                      ) : null}
                    </div>
                  </div>

                  <p className="mt-3 text-sm text-[var(--text)]">
                    {item.decisionNote}
                  </p>

                  <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-[var(--line)] pt-3 sm:grid-cols-4">
                    <div>
                      <dt className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                        Probability
                      </dt>
                      <dd className="mt-1 text-xs tabular-nums text-[var(--text)]">
                        {item.probability?.compactText ?? "Unavailable"}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                        Options
                      </dt>
                      <dd className="mt-1 text-xs text-[var(--text)]">
                        {item.options?.label ?? "Unavailable"}
                      </dd>
                      {item.options?.ranksText && (
                        <dd className="mt-0.5 text-[10px] tabular-nums text-[var(--muted)]">
                          {item.options.ranksText}
                        </dd>
                      )}
                    </div>
                    <div>
                      <dt className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                        News
                      </dt>
                      <dd className="mt-1 text-xs tabular-nums text-[var(--text)]">
                        {item.news?.scoreText ?? "Unavailable"}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                        Macro
                      </dt>
                      <dd className="mt-1 text-xs tabular-nums text-[var(--text)]">
                        {item.news?.macroText?.replace(/^Macro /, "") ?? "Unavailable"}
                      </dd>
                    </div>
                  </dl>

                  <button
                    type="button"
                    onClick={() => setExpandedTicker(isExpanded ? null : ticker)}
                    aria-expanded={isExpanded}
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
                          Deterministic rule-review detail is not available for this setup snapshot.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="hidden overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--panel)] 2xl:block">
            <table className="w-full min-w-[1220px] table-fixed border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-[var(--line)]">
                  {columns.map((column) => (
                    <th
                      key={column.key}
                      className={`px-3 py-2 ${column.className ?? ""}`}
                    >
                      <button
                        type="button"
                        onClick={() => handleSort(column.key)}
                        aria-label={`Sort by ${column.label}`}
                        className="select-none text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)] hover:text-[var(--text)]"
                      >
                        {column.label}
                        {sortCol === column.key && (
                          <span className="ml-1">{sortAsc ? "\u25B2" : "\u25BC"}</span>
                        )}
                      </button>
                    </th>
                  ))}
                  <th className="w-[110px] px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
                    Rule review
                  </th>
                </tr>
              </thead>
              <tbody>
                {(showAll ? items : items.slice(0, DEFAULT_VISIBLE)).map((item) => {
                  const isExpanded = expandedTicker === item.ticker;
                  return (
                    <SetupTableRow
                      key={item.ticker}
                      item={item}
                      columns={columns}
                      isExpanded={isExpanded}
                      isHighlighted={item.ticker === activeTicker}
                      onToggle={() => setExpandedTicker(isExpanded ? null : item.ticker)}
                    />
                  );
                })}
              </tbody>
            </table>
          </div>

          {items.length > DEFAULT_VISIBLE && (
            <div className="flex justify-center pt-1">
              <button
                type="button"
                onClick={() => setShowAll((prev) => !prev)}
                className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-4 py-2 text-xs font-semibold uppercase tracking-wider text-[var(--accent)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--blue)]"
              >
                {showAll
                  ? `Show Top ${DEFAULT_VISIBLE}`
                  : `Show All ${items.length} Setups`}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

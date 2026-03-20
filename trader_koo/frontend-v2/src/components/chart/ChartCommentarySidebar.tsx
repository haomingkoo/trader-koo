import { useState } from "react";
import type { ChartCommentary, HmmRegime } from "../../api/types";
import Badge, { tierVariant } from "../ui/Badge";
import GlassCard from "../ui/GlassCard";

function biasVariant(
  bias: string | null,
): "green" | "red" | "amber" | "muted" {
  if (!bias) return "muted";
  const value = bias.toLowerCase();
  if (value.includes("bull") || value === "long") return "green";
  if (value.includes("bear") || value === "short") return "red";
  return "amber";
}

function DebateRolesInline({
  debate,
}: {
  debate: NonNullable<ChartCommentary["debate_v1"]>;
}) {
  const [expanded, setExpanded] = useState(false);
  const consensus = debate.consensus;
  const roles = debate.roles ?? [];

  return (
    <div className="mt-3 border-t border-[var(--line)] pt-3">
      <button
        onClick={() => setExpanded((current) => !current)}
        className="text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] hover:text-[var(--blue)] transition-colors"
      >
        {expanded ? "Hide" : "Show"} debate ({roles.length} roles)
      </button>
      {expanded && (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--muted)]">
            <span>
              Consensus:{" "}
              <strong className="text-[var(--text)]">
                {String(consensus.consensus_bias ?? "\u2014")}
              </strong>
            </span>
            <span>
              State:{" "}
              <strong className="text-[var(--text)]">
                {String(consensus.consensus_state ?? "\u2014")}
              </strong>
            </span>
            <span>
              Agreement:{" "}
              <strong className="text-[var(--text)]">
                {typeof consensus.agreement_score === "number" ? consensus.agreement_score.toFixed(0) : "\u2014"}%
              </strong>
            </span>
            <span>
              Disagreements:{" "}
              <strong className="text-[var(--text)]">
                {String(consensus.disagreement_count ?? "\u2014")}
              </strong>
            </span>
          </div>
          {roles.map((role, index) => {
            const normalizedStance = role.stance.toLowerCase();
            const isBull = normalizedStance.includes("bull") || normalizedStance === "long";
            const isBear = normalizedStance.includes("bear") || normalizedStance === "short";
            const barColor = isBull
              ? "var(--green)"
              : isBear
                ? "var(--red)"
                : "var(--amber)";
            const confidencePct = Math.min(100, Math.max(0, role.confidence * 100));

            return (
              <div key={index} className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="w-24 text-[10px] font-medium capitalize text-[var(--text)]">
                    {role.role.replace(/_/g, " ")}
                  </span>
                  <Badge variant={isBull ? "green" : isBear ? "red" : "amber"}>
                    {role.stance.toUpperCase()}
                  </Badge>
                  <div className="relative h-1.5 flex-1 rounded-full bg-[var(--line)]">
                    <div
                      className="absolute left-0 top-0 h-full rounded-full"
                      style={{
                        width: `${confidencePct}%`,
                        backgroundColor: barColor,
                      }}
                    />
                  </div>
                  <span className="w-8 text-right text-[10px] tabular-nums text-[var(--muted)]">
                    {confidencePct.toFixed(0)}%
                  </span>
                </div>
                {role.evidence.filter(Boolean).length > 0 && (
                  <ul className="ml-28 space-y-0">
                    {role.evidence.filter(Boolean).map((evidence, evidenceIndex) => (
                      <li
                        key={evidenceIndex}
                        className="text-[10px] text-[var(--muted)]"
                      >
                        <span className="mr-1 text-[var(--line)]">-</span>{String(evidence)}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

interface ChartCommentarySidebarProps {
  commentary: ChartCommentary | null;
  hmmRegime: HmmRegime | null;
}

export default function ChartCommentarySidebar({
  commentary,
  hmmRegime,
}: ChartCommentarySidebarProps) {
  if (!commentary) {
    return (
      <GlassCard label="Chart Commentary">
        <p className="mt-1 text-xs text-[var(--muted)]">
          No chart commentary available. Load a ticker to generate commentary.
        </p>
      </GlassCard>
    );
  }

  const debate = commentary.debate_v1;
  const debateState =
    commentary.debate_consensus_state ??
    debate?.consensus?.consensus_state ??
    null;
  const agreementScore =
    commentary.debate_agreement_score ??
    debate?.consensus?.agreement_score ??
    null;

  const regimeLabel = hmmRegime?.current_state ?? null;
  const regimeDays = hmmRegime?.days_in_current ?? null;
  const transitionRisk = hmmRegime?.transition_risk_pct ?? null;
  const regimeVariant: "green" | "amber" | "red" | "muted" =
    regimeLabel === "low_vol"
      ? "green"
      : regimeLabel === "normal"
        ? "amber"
        : regimeLabel === "high_vol"
          ? "red"
          : "muted";
  const regimeDisplay: Record<string, string> = {
    low_vol: "LOW VOL",
    normal: "NORMAL",
    high_vol: "HIGH VOL",
  };

  return (
    <GlassCard>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {commentary.setup_tier && (
          <Badge variant={tierVariant(commentary.setup_tier)}>
            Tier {commentary.setup_tier}
          </Badge>
        )}
        {commentary.signal_bias && (
          <Badge variant={biasVariant(commentary.signal_bias)}>
            {commentary.signal_bias.toUpperCase()}
          </Badge>
        )}
        {commentary.actionability && (
          <Badge variant="default">
            {commentary.actionability.toUpperCase()}
          </Badge>
        )}
        {regimeLabel ? (
          <Badge variant={regimeVariant}>
            HMM {regimeDisplay[regimeLabel] ?? regimeLabel.toUpperCase()}
            {regimeDays != null && ` (${regimeDays}d)`}
            {transitionRisk != null && ` · ${transitionRisk.toFixed(1)}% shift risk`}
          </Badge>
        ) : (
          <Badge variant="muted">REGIME N/A</Badge>
        )}
        {debateState && (
          <Badge
            variant={
              debateState === "ready"
                ? "green"
                : debateState === "conditional"
                  ? "amber"
                  : "red"
            }
          >
            DEBATE {debateState.toUpperCase()}
            {agreementScore != null && ` ${agreementScore.toFixed(0)}%`}
          </Badge>
        )}
        {commentary.yolo_direction_conflict && (
          <Badge variant="red">YOLO CONFLICT</Badge>
        )}
      </div>

      <div className="space-y-2 text-xs">
        {commentary.observation ? (
          <p className="text-[var(--text)]">{String(commentary.observation)}</p>
        ) : (
          <p className="text-[var(--muted)]">No observation available.</p>
        )}

        {commentary.action && (
          <p className="text-[var(--muted)]">
            <strong className="text-[var(--text)]">Action:</strong>{" "}
            {String(commentary.action)}
          </p>
        )}

        {commentary.risk_note && (
          <p className="text-[var(--muted)]">
            <strong className="text-[var(--text)]">Risk:</strong>{" "}
            {String(commentary.risk_note)}
          </p>
        )}

        {commentary.technical_read && (
          <p className="text-[var(--muted)]">
            <strong className="text-[var(--text)]">Technical:</strong>{" "}
            {String(commentary.technical_read)}
          </p>
        )}
      </div>

      {debate && debate.roles && debate.roles.length > 0 && (
        <DebateRolesInline debate={debate} />
      )}

      {commentary.asof && (
        <p className="mt-2 text-[10px] text-[var(--muted)]">
          As of {String(commentary.asof)} | Source: {String(commentary.narrative_source ?? "rule")}
        </p>
      )}
    </GlassCard>
  );
}

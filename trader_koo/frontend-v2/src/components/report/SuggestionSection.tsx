import { Link } from "react-router-dom";
import type { ReportSuggestions } from "../../api/types";
import Badge from "../ui/Badge";
import { formatReportNumber } from "./reportShared";

function actionVariant(action: string) {
  const value = action.toLowerCase();
  if (value.includes("long")) return "green";
  if (value.includes("short")) return "red";
  return "amber";
}

function convictionVariant(conviction: string) {
  if (conviction === "Higher") return "green";
  if (conviction === "Medium") return "blue";
  return "muted";
}

export default function SuggestionSection({
  suggestions,
}: {
  suggestions?: ReportSuggestions;
}) {
  const items = suggestions?.items ?? [];
  if (!items.length) {
    return (
      <section className="rounded-lg border border-[var(--line)] bg-[var(--panel)] p-4">
        <div className="text-sm font-semibold text-[var(--text)]">Suggestions</div>
        <p className="mt-1 text-xs text-[var(--muted)]">
          No clean suggestions right now. The better choice is to wait.
        </p>
        <p className="mt-2 text-[11px] text-[var(--muted)]">
          Research tool only. Not financial advice.
        </p>
      </section>
    );
  }

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-[var(--text)]">Top Suggestions</h3>
          <p className="text-xs text-[var(--muted)]">
            Research only, not financial advice. Compressed from setup quality, calibration, news,
            options, and debate evidence.
          </p>
        </div>
        <Badge variant="muted">{items.length} shown</Badge>
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        {items.map((item) => (
          <article
            key={`${item.ticker}-${item.direction}`}
            className="rounded-lg border border-[var(--line)] bg-[var(--panel)] p-4"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <Link
                  to={`/chart?t=${item.ticker}`}
                  className="font-mono text-lg font-bold text-[var(--accent)] transition-colors hover:text-[var(--blue)]"
                >
                  {item.ticker}
                </Link>
                <div className="mt-1 text-xs text-[var(--muted)]">{item.persona}</div>
              </div>
              <div className="flex flex-col items-end gap-1">
                <Badge variant={actionVariant(item.action)}>{item.action}</Badge>
                <Badge variant={convictionVariant(item.conviction)}>{item.conviction}</Badge>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div>
                <div className="text-[var(--muted)]">Probability</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  {item.probability_pct == null
                    ? "\u2014"
                    : `${formatReportNumber(item.probability_pct, 1)}%`}
                </div>
              </div>
              <div>
                <div className="text-[var(--muted)]">Sample</div>
                <div className="font-semibold tabular-nums text-[var(--text)]">
                  n={item.sample_size ?? 0}
                </div>
              </div>
            </div>

            <ul className="mt-3 space-y-1.5">
              {item.why.slice(0, 3).map((reason) => (
                <li key={reason} className="text-xs text-[var(--text)]">
                  {reason}
                </li>
              ))}
            </ul>

            <div className="mt-3 border-t border-[var(--line)] pt-3 text-xs">
              <div className="text-[var(--muted)]">Invalidation</div>
              <div className="mt-1 text-[var(--text)]">{item.invalidation}</div>
            </div>

            <div className="mt-2 text-xs text-[var(--muted)]">
              Risk: {item.risk}
            </div>

            {item.data_gaps.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {item.data_gaps.slice(0, 2).map((gap) => (
                  <Badge key={gap} variant="muted" className="text-[9px]">
                    {gap}
                  </Badge>
                ))}
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

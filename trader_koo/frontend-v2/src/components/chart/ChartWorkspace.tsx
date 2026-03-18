import type { ReactNode } from "react";

interface ChartWorkspaceProps {
  chartContent: ReactNode;
  desktopCommentary: ReactNode;
  mobileCommentary: ReactNode;
  commentaryExpanded: boolean;
  onCollapse: () => void;
  onExpand: () => void;
}

export default function ChartWorkspace({
  chartContent,
  desktopCommentary,
  mobileCommentary,
  commentaryExpanded,
  onCollapse,
  onExpand,
}: ChartWorkspaceProps) {
  return (
    <>
      <div className="flex gap-4">
        <div className="flex-1 min-w-0">{chartContent}</div>

        <div
          className={`transition-all duration-200 ${
            commentaryExpanded ? "w-80 shrink-0" : "w-8 shrink-0"
          } hidden lg:block`}
        >
          {commentaryExpanded ? (
            <div className="relative">
              <button
                onClick={onCollapse}
                className="absolute -left-3 top-2 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--panel)] text-xs text-[var(--muted)] hover:text-[var(--text)] transition-colors"
                title="Collapse commentary"
              >
                &rsaquo;
              </button>
              {desktopCommentary}
            </div>
          ) : (
            <button
              onClick={onExpand}
              className="flex h-full w-8 items-start justify-center rounded-lg border border-[var(--line)] bg-[var(--panel)] pt-3 text-xs text-[var(--muted)] hover:text-[var(--text)] transition-colors"
              title="Expand commentary"
            >
              &lsaquo;
            </button>
          )}
        </div>
      </div>

      <div className="lg:hidden">{mobileCommentary}</div>
    </>
  );
}

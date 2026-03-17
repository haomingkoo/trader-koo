import { useEffect, useRef } from "react";

interface KeyboardShortcutsModalProps {
  open: boolean;
  onClose: () => void;
}

interface ShortcutEntry {
  keys: string[];
  description: string;
}

const SHORTCUTS: ShortcutEntry[] = [
  { keys: ["1"], description: "Go to Report" },
  { keys: ["2"], description: "Go to VIX Analysis" },
  { keys: ["3"], description: "Go to Earnings" },
  { keys: ["4"], description: "Go to Chart" },
  { keys: ["5"], description: "Go to Opportunities" },
  { keys: ["6"], description: "Go to Paper Trades" },
  { keys: ["7"], description: "Go to Crypto" },
  { keys: ["8"], description: "Go to Guide" },
  { keys: ["/", "\u2318K"], description: "Focus ticker search" },
  { keys: ["R"], description: "Refresh current page data" },
  { keys: ["Esc"], description: "Close modal / sidebar" },
  { keys: ["?"], description: "Show this help" },
];

export default function KeyboardShortcutsModal({
  open,
  onClose,
}: KeyboardShortcutsModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", handler, { capture: true });
    return () => window.removeEventListener("keydown", handler, { capture: true });
  }, [open, onClose]);

  // Focus trap
  useEffect(() => {
    if (open) dialogRef.current?.focus();
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
      aria-label="Keyboard shortcuts dialog overlay"
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        tabIndex={-1}
        className="mx-4 w-full max-w-md rounded-xl border border-[var(--line)] bg-[var(--panel)] p-6 shadow-2xl outline-none"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-bold text-[var(--text)]">
            Keyboard Shortcuts
          </h2>
          <button
            onClick={onClose}
            className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--text)]"
            aria-label="Close shortcuts dialog"
          >
            &#10005;
          </button>
        </div>
        <ul className="space-y-2">
          {SHORTCUTS.map((shortcut, i) => (
            <li
              key={i}
              className="flex items-center justify-between text-sm"
            >
              <span className="text-[var(--muted)]">
                {shortcut.description}
              </span>
              <span className="flex gap-1">
                {shortcut.keys.map((key) => (
                  <kbd
                    key={key}
                    className="inline-block min-w-[1.5rem] rounded border border-[var(--line)] bg-[var(--bg)] px-1.5 py-0.5 text-center text-xs font-mono font-semibold text-[var(--text)]"
                  >
                    {key}
                  </kbd>
                ))}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-[10px] text-[var(--muted)]">
          Shortcuts are disabled when typing in an input field.
        </p>
      </div>
    </div>
  );
}

import { useEffect, useCallback } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";

/** Page routes mapped to number keys 1-8 */
const PAGE_ROUTES: Record<string, string> = {
  "1": "/report",
  "2": "/vix",
  "3": "/earnings",
  "4": "/chart",
  "5": "/opportunities",
  "6": "/paper-trades",
  "7": "/crypto",
  "8": "",
} as const;

/** Query keys per route for targeted refetch on "R" */
const ROUTE_QUERY_KEYS: Record<string, string[]> = {
  "/report": ["report"],
  "/vix": ["market-summary"],
  "/earnings": ["earnings"],
  "/chart": ["chart"],
  "/opportunities": ["opportunities"],
  "/paper-trades": ["paper-trades-summary", "paper-trades"],
  "/crypto": ["crypto-summary", "crypto-history", "crypto-indicators"],
  "": ["report"],
};

interface UseKeyboardShortcutsOptions {
  onToggleHelp: () => void;
  onCloseSidebar: () => void;
}

function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return true;
  if ((el as HTMLElement).isContentEditable) return true;
  return false;
}

export function useKeyboardShortcuts({
  onToggleHelp,
  onCloseSidebar,
}: UseKeyboardShortcutsOptions): void {
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Always let Esc close modals/sidebar
      if (e.key === "Escape") {
        onCloseSidebar();
        return;
      }

      // Cmd+K / Ctrl+K — focus ticker search (navigate to chart)
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        navigate("/chart");
        // Focus the ticker input after navigation
        requestAnimationFrame(() => {
          const input = document.querySelector<HTMLInputElement>(
            'input[placeholder*="Ticker"]',
          );
          input?.focus();
          input?.select();
        });
        return;
      }

      // Skip remaining shortcuts when typing in an input
      if (isInputFocused()) return;

      // "/" — focus ticker search
      if (e.key === "/") {
        e.preventDefault();
        navigate("/chart");
        requestAnimationFrame(() => {
          const input = document.querySelector<HTMLInputElement>(
            'input[placeholder*="Ticker"]',
          );
          input?.focus();
          input?.select();
        });
        return;
      }

      // "?" — toggle help modal
      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        onToggleHelp();
        return;
      }

      // "R" — refresh current page data
      if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        const path = location.pathname.replace(/\/$/, "") || "";
        const normalised = path.startsWith("/") ? path : `/${path}`;
        const keys = ROUTE_QUERY_KEYS[normalised] ?? ROUTE_QUERY_KEYS[path];
        if (keys) {
          for (const key of keys) {
            queryClient.invalidateQueries({ queryKey: [key] });
          }
        }
        return;
      }

      // Number keys 1-8 for page navigation
      if (PAGE_ROUTES[e.key] !== undefined) {
        e.preventDefault();
        navigate(PAGE_ROUTES[e.key]);
        return;
      }
    },
    [navigate, location.pathname, queryClient, onToggleHelp, onCloseSidebar],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);
}

import { useState, useCallback } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "../components/layout/Sidebar";
import Header from "../components/layout/Header";
import ErrorBoundary from "../components/ui/ErrorBoundary";
import KeyboardShortcutsModal from "../components/KeyboardShortcutsModal";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";

export default function DashboardLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const location = useLocation();

  const handleMobileClose = useCallback(() => setMobileOpen(false), []);

  useKeyboardShortcuts({
    onToggleHelp: useCallback(() => setShortcutsOpen((p) => !p), []),
    onCloseSidebar: handleMobileClose,
  });

  return (
    <div className="flex min-h-[100dvh] overflow-hidden bg-[var(--bg)]">
      <Sidebar mobileOpen={mobileOpen} onMobileClose={handleMobileClose} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header onMenuToggle={() => setMobileOpen((p) => !p)} />
        <main className="flex-1 overflow-auto px-4 pb-4 pt-3 [padding-bottom:max(1rem,env(safe-area-inset-bottom))]">
          <ErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
        <footer className="border-t border-[var(--line)] bg-[var(--bg)] px-4 py-2 [padding-bottom:max(0.5rem,env(safe-area-inset-bottom))]">
          <p className="text-center text-[10px] leading-relaxed text-[var(--muted)] sm:text-[10px]">
            Research tool only. Not financial advice. Data may be delayed, inaccurate, or incomplete.
          </p>
        </footer>
      </div>
      <KeyboardShortcutsModal
        open={shortcutsOpen}
        onClose={() => setShortcutsOpen(false)}
      />
    </div>
  );
}

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
    <div className="flex h-screen overflow-hidden">
      <Sidebar mobileOpen={mobileOpen} onMobileClose={handleMobileClose} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header onMenuToggle={() => setMobileOpen((p) => !p)} />
        <main className="flex-1 overflow-auto p-4">
          <ErrorBoundary resetKey={location.pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
        <footer className="border-t border-[var(--line)] bg-[var(--bg)] px-4 py-2">
          <p className="text-center text-[10px] text-[var(--muted)]">
            Research tool only. Not financial advice. Past performance does not guarantee future results.
            All data may be delayed, inaccurate, or incomplete. Do not make investment decisions based solely on this dashboard.
            Always consult a qualified financial advisor.
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

import { useState, useCallback } from "react";
import { Outlet } from "react-router-dom";
import Sidebar from "../components/layout/Sidebar";
import Header from "../components/layout/Header";
import ErrorBoundary from "../components/ui/ErrorBoundary";

export default function DashboardLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleMobileClose = useCallback(() => setMobileOpen(false), []);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar mobileOpen={mobileOpen} onMobileClose={handleMobileClose} />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header onMenuToggle={() => setMobileOpen((p) => !p)} />
        <main className="flex-1 overflow-auto p-4">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
}

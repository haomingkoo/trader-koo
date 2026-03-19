import { NavLink } from "react-router-dom";
import { useState, useEffect } from "react";
import {
  BookOpen,
  FileText,
  Thermometer,
  Calendar,
  TrendingUp,
  Bitcoin,
  Search,
  Wallet,
  ChevronLeft,
  ChevronRight,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface NavItem {
  to: string;
  label: string;
  Icon: LucideIcon;
}

const navItems: NavItem[] = [
  { to: "/", label: "Guide", Icon: BookOpen },
  { to: "/report", label: "Report", Icon: FileText },
  { to: "/vix", label: "VIX Analysis", Icon: Thermometer },
  { to: "/earnings", label: "Earnings", Icon: Calendar },
  { to: "/chart", label: "Chart", Icon: TrendingUp },
  { to: "/crypto", label: "Crypto", Icon: Bitcoin },
  { to: "/opportunities", label: "Opportunities", Icon: Search },
  { to: "/paper-trades", label: "Paper Trades", Icon: Wallet },
];

export default function Sidebar({
  mobileOpen,
  onMobileClose,
}: {
  mobileOpen: boolean;
  onMobileClose: () => void;
}) {
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (mobileOpen) {
      const handleResize = () => {
        if (window.innerWidth >= 768) onMobileClose();
      };
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }
  }, [mobileOpen, onMobileClose]);

  const sidebarContent = (
    <>
      <div className="flex items-center justify-between border-b border-[var(--line)] px-3 py-3">
        {!collapsed && (
          <a
            href="https://kooexperience.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs font-semibold tracking-wider text-[var(--muted)] transition-colors hover:text-[var(--accent)]"
          >
            kooexperience.com
          </a>
        )}
        <button
          onClick={() => setCollapsed((prev) => !prev)}
          className="hidden rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--text)] md:block"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
        <button
          onClick={onMobileClose}
          className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--text)] md:hidden"
          aria-label="Close navigation menu"
        >
          <X size={16} />
        </button>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 p-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            onClick={onMobileClose}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-[var(--panel-hover)] text-[var(--accent)]"
                  : "text-[var(--muted)] hover:bg-[var(--panel-hover)] hover:text-[var(--text)]"
              }`
            }
          >
            <item.Icon size={18} strokeWidth={1.75} aria-hidden="true" />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-[var(--line)] px-3 py-2">
        {!collapsed && (
          <div className="text-[10px] text-[var(--muted)]">
            v2 beta
          </div>
        )}
      </div>
    </>
  );

  return (
    <>
      <aside
        className={`hidden md:flex flex-col border-r border-[var(--line)] bg-[var(--panel)] transition-all duration-200 ${collapsed ? "w-14" : "w-52"}`}
      >
        {sidebarContent}
      </aside>

      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onMobileClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-52 flex-col border-r border-[var(--line)] bg-[var(--panel)] transition-transform duration-200 md:hidden ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {sidebarContent}
      </aside>
    </>
  );
}

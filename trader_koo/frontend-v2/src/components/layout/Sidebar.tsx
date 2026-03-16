import { NavLink } from "react-router-dom";
import { useState } from "react";

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

const navItems: NavItem[] = [
  { to: "/v2", label: "Guide", icon: "\uD83D\uDCD6" },
  { to: "/v2/report", label: "Report", icon: "\uD83D\uDCCB" },
  { to: "/v2/vix", label: "VIX Analysis", icon: "\uD83C\uDF21\uFE0F" },
  { to: "/v2/earnings", label: "Earnings", icon: "\uD83D\uDCC5" },
  { to: "/v2/chart", label: "Chart", icon: "\uD83D\uDCC8" },
  { to: "/v2/opportunities", label: "Opportunities", icon: "\uD83D\uDD0D" },
  { to: "/v2/paper-trades", label: "Paper Trades", icon: "\uD83D\uDCB0" },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`flex flex-col border-r border-[var(--line)] bg-[var(--panel)] transition-all duration-200 ${collapsed ? "w-14" : "w-52"}`}
    >
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
          className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--text)]"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? "\u276F" : "\u276E"}
        </button>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 p-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/v2"}
            className={({ isActive }) =>
              `flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "bg-[var(--panel-hover)] text-[var(--accent)]"
                  : "text-[var(--muted)] hover:bg-[var(--panel-hover)] hover:text-[var(--text)]"
              }`
            }
          >
            <span className="text-base">{item.icon}</span>
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
    </aside>
  );
}

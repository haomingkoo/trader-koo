import { useState, useEffect } from "react";
import { ArrowUpRight, Sun, Moon } from "lucide-react";
import ClockStrip from "./ClockStrip";
import PipelineStatusBadge from "./PipelineStatusBadge";
import {
  CryptoPriceStrip,
  EquityPriceStrip,
} from "./HeaderMarketStrips";
import { usePipelineStatus } from "../../api/hooks";

function useTheme() {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    if (typeof window !== "undefined") {
      return (localStorage.getItem("theme") as "dark" | "light") || "dark";
    }
    return "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  return { theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) };
}

export default function Header({ onMenuToggle }: { onMenuToggle: () => void }) {
  const { data } = usePipelineStatus();
  const { theme, toggle: toggleTheme } = useTheme();

  return (
    <header className="border-b border-[var(--line)] px-4 pb-2 pt-2 [padding-top:max(0.5rem,env(safe-area-inset-top))]">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <button
            onClick={onMenuToggle}
            className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--text)] md:hidden"
            aria-label="Toggle navigation menu"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <a
            href="https://kooexperience.com/"
            className="text-lg font-bold tracking-wide text-[var(--text)] transition-colors hover:text-[var(--accent)] md:text-base"
          >
            HK
          </a>
          <span className="text-[10px] font-medium tracking-wider text-[var(--muted)]">
            Trader Koo
          </span>
          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={toggleTheme}
              className="rounded p-1 text-[var(--muted)] transition-colors hover:bg-[var(--panel-hover)] hover:text-[var(--text)]"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <a
              href="https://kooexperience.com/"
              className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)] transition-colors hover:text-[var(--accent)]"
            >
              Portfolio <ArrowUpRight size={12} />
            </a>
          </div>
        </div>

        <div className="grid gap-2 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] xl:items-start">
          <ClockStrip />
          <div className="flex flex-col gap-2">
            <EquityPriceStrip />
            <CryptoPriceStrip />
          </div>
        </div>

        <PipelineStatusBadge data={data} />
      </div>
    </header>
  );
}

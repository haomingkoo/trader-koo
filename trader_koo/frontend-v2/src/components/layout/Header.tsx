import ClockStrip from "./ClockStrip";
import PipelineStatusBadge from "./PipelineStatusBadge";
import {
  CryptoPriceStrip,
  EquityPriceStrip,
} from "./HeaderMarketStrips";
import { usePipelineStatus } from "../../api/hooks";

export default function Header({ onMenuToggle }: { onMenuToggle: () => void }) {
  const { data } = usePipelineStatus();

  return (
    <header className="border-b border-[var(--line)] px-4 pb-3 pt-3 [padding-top:max(0.75rem,env(safe-area-inset-top))]">
      <div className="flex flex-col gap-3">
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
          <h1 className="text-lg font-bold tracking-wide text-[var(--text)] md:text-base">
            trader_koo
          </h1>
        </div>

        <div className="grid gap-3 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] xl:items-start">
          <ClockStrip />
          <div className="flex flex-col gap-3">
            <EquityPriceStrip />
            <CryptoPriceStrip />
          </div>
        </div>

        <div className="flex items-start justify-start">
          <PipelineStatusBadge data={data} />
        </div>
      </div>
    </header>
  );
}

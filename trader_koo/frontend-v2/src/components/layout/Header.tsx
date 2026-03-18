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
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-3">
      <div className="flex items-center gap-4">
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
        <h1 className="text-base font-bold tracking-wide text-[var(--text)]">
          trader_koo
        </h1>
        <ClockStrip />
        <EquityPriceStrip />
        <CryptoPriceStrip />
      </div>
      <PipelineStatusBadge data={data} />
    </header>
  );
}

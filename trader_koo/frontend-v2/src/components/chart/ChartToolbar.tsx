import { Link } from "react-router-dom";
import type { EquityTick } from "../../api/types";

type TimeframeValue = "daily" | "weekly";

interface ChartToolbarProps {
  ticker: string | null | undefined;
  livePrice: EquityTick | null;
  streamingActive: boolean;
  inputValue: string;
  isLoading: boolean;
  timeframe: TimeframeValue;
  onInputChange: (value: string) => void;
  onInputKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  onLoad: () => void;
  onSelectTimeframe: (timeframe: TimeframeValue) => void;
  onRefresh: () => void;
}

export default function ChartToolbar({
  ticker,
  livePrice,
  streamingActive,
  inputValue,
  isLoading,
  timeframe,
  onInputChange,
  onInputKeyDown,
  onLoad,
  onSelectTimeframe,
  onRefresh,
}: ChartToolbarProps) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <h2 className="text-xl font-bold tracking-tight">
        Chart
        {ticker && (
          <span className="ml-2 text-[var(--accent)]">{ticker}</span>
        )}
      </h2>

      {/* Group 1: Ticker input + Load */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={inputValue}
          onChange={(event) => onInputChange(event.target.value.toUpperCase())}
          onKeyDown={onInputKeyDown}
          placeholder="Ticker (e.g. SPY)"
          className="w-28 rounded-lg border border-[var(--line)] bg-[var(--bg)] px-3 py-1.5 text-sm font-mono text-[var(--text)] placeholder-[var(--muted)] focus:border-[var(--accent)] focus:outline-none transition-colors"
        />
        <button
          onClick={onLoad}
          disabled={isLoading}
          className="rounded-lg bg-[var(--accent)] px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-[var(--blue)] disabled:opacity-50"
        >
          Load
        </button>
      </div>

      {/* Group 2: Daily/Weekly toggle + Refresh */}
      <div className="flex items-center gap-2">
        <div className="flex gap-1">
          {(["daily", "weekly"] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => onSelectTimeframe(tf)}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors ${
                timeframe === tf
                  ? "bg-[var(--blue)] text-white"
                  : "border border-[var(--line)] bg-[var(--panel)] text-[var(--muted)] hover:text-[var(--text)]"
              }`}
            >
              {tf.charAt(0).toUpperCase() + tf.slice(1)}
            </button>
          ))}
        </div>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-xs text-[var(--muted)] transition-colors hover:text-[var(--text)] disabled:opacity-50"
        >
          Refresh
        </button>
      </div>

      {/* Group 3: Live price badge + streaming status */}
      <div className="flex items-center gap-2">
        {livePrice && (
          <span className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-xs">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)] animate-pulse" />
            <span className="font-semibold text-[var(--text)] tabular-nums">
              ${livePrice.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            {livePrice.prev_price != null && livePrice.prev_price > 0 && (
              <span
                className={`text-[10px] font-semibold tabular-nums ${
                  livePrice.price >= livePrice.prev_price
                    ? "text-[var(--green)]"
                    : "text-[var(--red)]"
                }`}
              >
                {livePrice.price >= livePrice.prev_price ? "+" : ""}
                {(((livePrice.price - livePrice.prev_price) / livePrice.prev_price) * 100).toFixed(2)}%
              </span>
            )}
            <span className="text-[9px] font-semibold uppercase tracking-wider text-[var(--green)]">Live</span>
          </span>
        )}
        {!livePrice && streamingActive && (
          <span className="flex items-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-[10px] text-[var(--amber)]">
            Streaming...
          </span>
        )}
        {!livePrice && !streamingActive && ticker && (
          <span className="flex items-center gap-1 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-2.5 py-1 text-[10px] text-[var(--muted)]">
            Delayed
          </span>
        )}
      </div>

      <Link
        to="/report"
        className="rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-xs text-[var(--muted)] transition-colors hover:text-[var(--text)]"
      >
        &larr; Back to Report
      </Link>
    </div>
  );
}

import { useState, useEffect, useRef, useCallback } from "react";
import ClockStrip from "./ClockStrip";
import { usePipelineStatus } from "../../api/hooks";
import type { CryptoPrice } from "../../api/types";

function PipelineDot({ state }: { state: "idle" | "active" | "done" | "error" }) {
  const colors: Record<string, string> = {
    idle: "bg-[var(--line)]",
    active: "bg-[var(--amber)] animate-pulse",
    done: "bg-[var(--green)]",
    error: "bg-[var(--red)]",
  };
  const labels: Record<string, string> = {
    idle: "idle",
    active: "active",
    done: "done",
    error: "error",
  };
  return (
    <div
      className={`h-2.5 w-2.5 rounded-full ${colors[state]}`}
      title={labels[state]}
      aria-label={`Pipeline stage: ${labels[state]}`}
    />
  );
}

function derivePipelineStates(data: { pipeline: { stage: string }; latest_run: { status: string } | null }) {
  const stage = (data.pipeline?.stage ?? "idle").toLowerCase();
  const runStatus = (data.latest_run?.status ?? "").toLowerCase();

  const ingestStages = ["price_daily", "price_seed", "fundamentals", "ingest"];
  const yoloStages = ["yolo", "yolo_batch", "patterns"];
  const reportStages = ["report", "narrative", "scoring", "daily_report"];

  type StageState = "idle" | "active" | "done" | "error";
  let ingest: StageState = "idle";
  let yolo: StageState = "idle";
  let report: StageState = "idle";

  if (runStatus === "running" || runStatus === "in_progress") {
    if (ingestStages.some((s) => stage.includes(s))) {
      ingest = "active";
    } else if (yoloStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "active";
    } else if (reportStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "done";
      report = "active";
    } else {
      ingest = "active";
    }
  } else if (runStatus === "completed" || runStatus === "done") {
    ingest = "done";
    yolo = "done";
    report = "done";
  } else if (runStatus === "failed") {
    if (reportStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "done";
      report = "error";
    } else if (yoloStages.some((s) => stage.includes(s))) {
      ingest = "done";
      yolo = "error";
    } else {
      ingest = "error";
    }
  }

  return { ingest, yolo, report };
}

function formatPrice(price: number): string {
  if (price >= 1000) return price.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 });
  return price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function CryptoPriceChip({ label, tick }: { label: string; tick: CryptoPrice | undefined | null }) {
  if (!tick || typeof tick.price !== "number" || typeof tick.change_pct_24h !== "number") return null;
  const isPositive = tick.change_pct_24h >= 0;
  const changeColor = isPositive ? "text-[var(--green)]" : "text-[var(--red)]";
  const sign = isPositive ? "+" : "";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
        {label}
      </span>
      <span className="tabular-nums text-[var(--text)]">
        ${formatPrice(tick.price)}
      </span>
      <span className={`tabular-nums text-[10px] font-semibold ${changeColor}`}>
        {sign}{tick.change_pct_24h.toFixed(2)}%
      </span>
    </div>
  );
}

function useCryptoWebSocket(): {
  prices: Record<string, CryptoPrice>;
  connected: boolean;
} {
  const [prices, setPrices] = useState<Record<string, CryptoPrice>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/crypto`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setConnected(true);
      backoff.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const tick = JSON.parse(event.data) as CryptoPrice;
        setPrices((prev) => ({ ...prev, [tick.symbol]: tick }));
      } catch { /* malformed message */ }
    };

    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      reconnectTimer.current = setTimeout(() => {
        backoff.current = Math.min(backoff.current * 2, 30000);
        connect();
      }, backoff.current);
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { prices, connected };
}

const CRYPTO_SYMBOLS = [
  { key: "BTC-USD", label: "BTC" },
  { key: "ETH-USD", label: "ETH" },
  { key: "SOL-USD", label: "SOL" },
  { key: "XRP-USD", label: "XRP" },
  { key: "DOGE-USD", label: "DOGE" },
] as const;

function CryptoPriceStrip() {
  const { prices, connected } = useCryptoWebSocket();

  if (!connected && Object.keys(prices).length === 0) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Crypto
        </span>
        <span className="text-[10px] text-[var(--red)]">
          Connecting...
        </span>
      </div>
    );
  }

  const availableChips = CRYPTO_SYMBOLS.filter((s) => prices[s.key]);

  return (
    <div className="flex items-center gap-3 overflow-x-auto rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px] scrollbar-none">
      {availableChips.map((sym, i) => (
        <span key={sym.key} className="flex items-center gap-3 whitespace-nowrap">
          {i > 0 && <span className="text-[var(--line)]">|</span>}
          <CryptoPriceChip label={sym.label} tick={prices[sym.key]} />
        </span>
      ))}
      {availableChips.length === 0 && (
        <span className="text-[10px] text-[var(--muted)]">No data</span>
      )}
      {!connected && (
        <span className="ml-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--red)]" title="Reconnecting..." />
      )}
    </div>
  );
}

export default function Header({ onMenuToggle }: { onMenuToggle: () => void }) {
  const { data } = usePipelineStatus();
  const states = data
    ? derivePipelineStates(data)
    : { ingest: "idle" as const, yolo: "idle" as const, report: "idle" as const };

  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-3">
      <div className="flex items-center gap-4">
        {/* Mobile hamburger */}
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
        <CryptoPriceStrip />
      </div>
      <div className="flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Pipeline
        </span>
        <div className="flex items-center gap-1" aria-label="Pipeline status: Ingest, YOLO, Report">
          <PipelineDot state={states.ingest} />
          <span className="text-[var(--line)]" aria-hidden="true">&rarr;</span>
          <PipelineDot state={states.yolo} />
          <span className="text-[var(--line)]" aria-hidden="true">&rarr;</span>
          <PipelineDot state={states.report} />
        </div>
      </div>
    </header>
  );
}

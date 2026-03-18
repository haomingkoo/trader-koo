import { useState, useEffect, useRef, useCallback } from "react";
import type { CryptoPrice, EquityTick } from "../../api/types";

function formatPrice(price: number): string {
  if (price >= 1000) {
    return price.toLocaleString("en-US", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    });
  }
  return price.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function CryptoPriceChip({ label, tick }: { label: string; tick: CryptoPrice | undefined | null }) {
  if (!tick || typeof tick.price !== "number" || typeof tick.change_pct_24h !== "number") {
    return null;
  }
  const isPositive = tick.change_pct_24h >= 0;
  const changeColor = isPositive ? "text-[var(--green)]" : "text-[var(--red)]";
  const sign = isPositive ? "+" : "";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
        {label}
      </span>
      <span className="tabular-nums text-[var(--text)]">${formatPrice(tick.price)}</span>
      <span className={`tabular-nums text-[10px] font-semibold ${changeColor}`}>
        {sign}{tick.change_pct_24h.toFixed(2)}%
      </span>
    </div>
  );
}

function EquityPriceChip({ label, tick }: { label: string; tick: EquityTick | undefined | null }) {
  if (!tick || typeof tick.price !== "number") return null;
  const prev = tick.prev_price ?? tick.price;
  const isPositive = tick.price >= prev;
  const changeColor = tick.price === prev
    ? "text-[var(--muted)]"
    : isPositive ? "text-[var(--green)]" : "text-[var(--red)]";
  const changePct = prev > 0 ? ((tick.price - prev) / prev) * 100 : 0;
  const sign = changePct >= 0 ? "+" : "";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
        {label}
      </span>
      <span className="tabular-nums text-[var(--text)]">${formatPrice(tick.price)}</span>
      {tick.prev_price != null && (
        <span className={`tabular-nums text-[10px] font-semibold ${changeColor}`}>
          {sign}{changePct.toFixed(2)}%
        </span>
      )}
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
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/crypto`);

    ws.onopen = () => {
      setConnected(true);
      backoff.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const tick = JSON.parse(event.data) as CryptoPrice;
        setPrices((prev) => ({ ...prev, [tick.symbol]: tick }));
      } catch {
        // Ignore malformed messages.
      }
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

function useEquityWebSocket(): {
  prices: Record<string, EquityTick>;
  connected: boolean;
} {
  const [prices, setPrices] = useState<Record<string, EquityTick>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);

  const connect = useCallback(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/equities`);

    ws.onopen = () => {
      setConnected(true);
      backoff.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const tick = JSON.parse(event.data) as EquityTick;
        setPrices((prev) => ({ ...prev, [tick.symbol]: tick }));
      } catch {
        // Ignore malformed messages.
      }
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

const EQUITY_SYMBOLS = [
  { key: "SPY", label: "SPY" },
  { key: "QQQ", label: "QQQ" },
] as const;

export function EquityPriceStrip() {
  const { prices, connected } = useEquityWebSocket();
  const hasData = Object.keys(prices).length > 0;

  if (!connected && !hasData) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Equities
        </span>
        <span className="text-[10px] text-[var(--amber)]">Market closed</span>
      </div>
    );
  }

  const availableChips = EQUITY_SYMBOLS.filter((symbol) => prices[symbol.key]);

  if (availableChips.length === 0 && connected) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Equities
        </span>
        <span className="text-[10px] text-[var(--amber)]">Market closed</span>
      </div>
    );
  }

  return (
    <div className="flex w-full items-center gap-3 overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[11.5px] scrollbar-none">
      {availableChips.map((symbol, index) => (
        <span key={symbol.key} className="flex items-center gap-3 whitespace-nowrap">
          {index > 0 && <span className="text-[var(--line)]">|</span>}
          <EquityPriceChip label={symbol.label} tick={prices[symbol.key]} />
        </span>
      ))}
      {!connected && (
        <span className="ml-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--red)]" title="Reconnecting..." />
      )}
    </div>
  );
}

export function CryptoPriceStrip() {
  const { prices, connected } = useCryptoWebSocket();

  if (!connected && Object.keys(prices).length === 0) {
    return (
      <div className="flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-[var(--panel)] px-3 py-1.5 text-[11.5px]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted)]">
          Crypto
        </span>
        <span className="text-[10px] text-[var(--red)]">Connecting...</span>
      </div>
    );
  }

  const availableChips = CRYPTO_SYMBOLS.filter((symbol) => prices[symbol.key]);

  return (
    <div className="flex w-full items-center gap-3 overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[11.5px] scrollbar-none">
      {availableChips.map((symbol, index) => (
        <span key={symbol.key} className="flex items-center gap-3 whitespace-nowrap">
          {index > 0 && <span className="text-[var(--line)]">|</span>}
          <CryptoPriceChip label={symbol.label} tick={prices[symbol.key]} />
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

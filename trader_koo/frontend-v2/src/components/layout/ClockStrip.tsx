import { useState, useEffect } from "react";

function formatTime(date: Date, timeZone: string): string {
  return date.toLocaleTimeString("en-US", {
    timeZone,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function getMarketStatus(nyDate: Date): { label: string; variant: "open" | "closed" | "pre" } {
  const day = nyDate.getDay();
  const hours = nyDate.getHours();
  const minutes = nyDate.getMinutes();
  const totalMinutes = hours * 60 + minutes;

  // Weekend
  if (day === 0 || day === 6) {
    return { label: "CLOSED", variant: "closed" };
  }

  // Pre-market: 4:00 - 9:30 ET
  if (totalMinutes >= 240 && totalMinutes < 570) {
    return { label: "PRE-MARKET", variant: "pre" };
  }

  // Market hours: 9:30 - 16:00 ET
  if (totalMinutes >= 570 && totalMinutes < 960) {
    return { label: "OPEN", variant: "open" };
  }

  // After hours: 16:00 - 20:00 ET
  if (totalMinutes >= 960 && totalMinutes < 1200) {
    return { label: "AFTER-HOURS", variant: "pre" };
  }

  return { label: "CLOSED", variant: "closed" };
}

const badgeStyles: Record<string, string> = {
  open: "bg-[rgba(56,211,159,0.15)] text-[var(--green)]",
  closed: "bg-[rgba(255,107,107,0.12)] text-[var(--red)]",
  pre: "bg-[rgba(248,194,78,0.15)] text-[var(--amber)]",
};

export default function ClockStrip() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const localTz = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const localLabel = localTz.split("/").pop()?.replace(/_/g, " ") ?? "Local";
  const localTime = formatTime(now, localTz);
  const nyTime = formatTime(now, "America/New_York");

  const nyDate = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" }),
  );
  const market = getMarketStatus(nyDate);

  return (
    <div className="flex w-full flex-wrap items-center gap-2 rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2 text-[11.5px] text-[var(--muted)] md:w-auto md:gap-4 md:px-4">
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
          {localLabel}
        </span>
        <span className="tabular-nums text-[var(--text)]">{localTime}</span>
      </div>
      <span className="hidden text-[var(--line)] md:inline">|</span>
      <div className="flex min-w-0 items-center gap-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text)] opacity-70">
          NY
        </span>
        <span className="tabular-nums text-[var(--text)]">{nyTime}</span>
      </div>
      <span
        className={`ml-auto rounded-md px-2 py-0.5 text-[10px] font-bold tracking-wider ${badgeStyles[market.variant]}`}
      >
        {market.label}
      </span>
    </div>
  );
}

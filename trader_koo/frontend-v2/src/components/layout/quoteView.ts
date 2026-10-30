import type { EquityTick, MacroInstrument, MarketTickerSummary } from "../../api/types";

export interface QuoteView {
  price: number;
  changePct: number | null;
  mode: "live" | "close";
  /** Close date for "close" quotes. */
  asOf: string | null;
}

const NY_DATE = new Intl.DateTimeFormat("en-CA", {
  timeZone: "America/New_York",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

function sessionDate(timestamp: string): string | null {
  const ms = Date.parse(timestamp);
  return Number.isFinite(ms) ? NY_DATE.format(ms) : null;
}

/**
 * A stream tick's `prev_price` is the previous trade, so the day change must
 * use the last stored close from a session before the tick's own session.
 * A tick from a session the stored close already covers is not shown as live.
 */
export function resolveEquityQuote(
  tick: EquityTick | undefined | null,
  snapshot: MarketTickerSummary | undefined | null,
  snapshotAsOf: string | null | undefined,
): QuoteView | null {
  const session = tick ? sessionDate(tick.timestamp) : null;
  if (tick && session && !(snapshotAsOf && session <= snapshotAsOf)) {
    const prior = snapshot?.history.filter((point) => point.date < session).at(-1);
    const changePct = prior && prior.close > 0 ? (tick.price / prior.close - 1) * 100 : null;
    return { price: tick.price, changePct, mode: "live", asOf: null };
  }
  if (snapshot && snapshotAsOf) {
    return { price: snapshot.price, changePct: snapshot.change_pct_1d, mode: "close", asOf: snapshotAsOf };
  }
  return null;
}

export function macroQuoteView(instrument: MacroInstrument): QuoteView | null {
  if (instrument.current != null) {
    return { price: instrument.current, changePct: instrument.change_pct, mode: "live", asOf: null };
  }
  if (instrument.prev_close != null && instrument.prev_close_date) {
    return { price: instrument.prev_close, changePct: null, mode: "close", asOf: instrument.prev_close_date };
  }
  return null;
}

/** "2026-10-29" -> "Oct 29" */
export function shortCloseDate(date: string): string {
  const [, month, day] = date.split("-").map(Number);
  const names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return month && day ? `${names[month - 1]} ${day}` : date;
}

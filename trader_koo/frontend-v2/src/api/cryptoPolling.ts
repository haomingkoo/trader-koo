const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;

const BAR_MS: Record<string, number> = {
  "1m": MINUTE_MS,
  "5m": 5 * MINUTE_MS,
  "15m": 15 * MINUTE_MS,
  "30m": 30 * MINUTE_MS,
  "1h": HOUR_MS,
  "4h": 4 * HOUR_MS,
  "12h": 12 * HOUR_MS,
  "1d": 24 * HOUR_MS,
  "1w": 7 * 24 * HOUR_MS,
};

// Correlation takes seconds to compute and moves on daily equity closes.
export const CRYPTO_CORRELATION_STALE_MS = 30 * MINUTE_MS;

// The socket only keeps the latest closed bar, so history must refetch at least
// once per bar or a second close would leave a gap in the chart.
export function cryptoHistoryRefetchMs(interval: string): number {
  const barMs = BAR_MS[interval] ?? MINUTE_MS;
  return Math.min(Math.max(barMs, MINUTE_MS), HOUR_MS);
}

export function cryptoIndicatorRefetchMs(interval: string): number {
  const barMs = BAR_MS[interval] ?? MINUTE_MS;
  return barMs >= HOUR_MS ? 5 * MINUTE_MS : 30_000;
}

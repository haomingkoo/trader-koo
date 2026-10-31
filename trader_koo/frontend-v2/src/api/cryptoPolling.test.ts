import test from "node:test";
import assert from "node:assert/strict";

import {
  cryptoHistoryRefetchMs,
  cryptoIndicatorRefetchMs,
} from "./cryptoPolling.ts";

test("history refetches once per bar, never faster than a minute", () => {
  assert.equal(cryptoHistoryRefetchMs("1m"), 60_000);
  assert.equal(cryptoHistoryRefetchMs("15m"), 15 * 60_000);
  assert.equal(cryptoHistoryRefetchMs("1h"), 60 * 60_000);
});

test("history refetch is capped at an hour for long bars", () => {
  assert.equal(cryptoHistoryRefetchMs("4h"), 60 * 60_000);
  assert.equal(cryptoHistoryRefetchMs("1w"), 60 * 60_000);
});

test("indicators poll every 30 s below an hour and every 5 min from an hour up", () => {
  assert.equal(cryptoIndicatorRefetchMs("1m"), 30_000);
  assert.equal(cryptoIndicatorRefetchMs("30m"), 30_000);
  assert.equal(cryptoIndicatorRefetchMs("1h"), 300_000);
  assert.equal(cryptoIndicatorRefetchMs("1d"), 300_000);
});

test("unknown intervals fall back to the fast cadence", () => {
  assert.equal(cryptoHistoryRefetchMs("bogus"), 60_000);
  assert.equal(cryptoIndicatorRefetchMs("bogus"), 30_000);
});

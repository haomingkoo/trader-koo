import test from "node:test";
import assert from "node:assert/strict";

import { formatPrice } from "./formatPrice.ts";

test("keeps sub-dollar precision for coins like DOGE", () => {
  assert.equal(formatPrice(0.07506), "0.07506");
});

test("uses two decimals between $1 and $1,000 and none above", () => {
  assert.equal(formatPrice(2.5), "2.50");
  assert.equal(formatPrice(67_123.4), "67,123");
});

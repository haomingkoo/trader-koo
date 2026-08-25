import test from "node:test";
import assert from "node:assert/strict";

import { requireSuccessfulTrigger } from "./triggerUpdate.ts";

test("rejects a successful HTTP response whose payload says the trigger failed", () => {
  assert.throws(
    () => requireSuccessfulTrigger({ ok: false, message: "already running" }),
    /already running/,
  );
});

test("returns the backend message for an accepted trigger", () => {
  assert.equal(
    requireSuccessfulTrigger({ ok: true, message: "report queued" }),
    "report queued",
  );
});

import assert from "node:assert/strict";
import test from "node:test";

import { commentarySourceLabel } from "./commentarySource.ts";

test("commentary provenance is explicit and legacy values fail closed", () => {
  assert.equal(
    commentarySourceLabel("llm"),
    "LLM (generated from current regime snapshot)",
  );
  assert.equal(commentarySourceLabel("rule"), "Deterministic regime rules");
  assert.equal(commentarySourceLabel(undefined), "Provenance not recorded");
  assert.equal(
    commentarySourceLabel("regime_context_2026-08-26"),
    "Provenance not recorded",
  );
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const visibleSurfaces = [
  new URL("../chart/ChartCommentarySidebar.tsx", import.meta.url),
  new URL("../../pages/MethodologyPage.tsx", import.meta.url),
  new URL("../../pages/PaperTradePage.tsx", import.meta.url),
];

test("deterministic rule review is not presented as an agent debate", () => {
  const visibleCopy = visibleSurfaces
    .map((path) => readFileSync(path, "utf8"))
    .join("\n");

  for (const staleLabel of [
    /show[^\n]*debate/i,
    /multi-angle debate/i,
    /debate consensus/i,
    /which analysts voted/i,
    /debate strength/i,
  ]) {
    assert.doesNotMatch(visibleCopy, staleLabel);
  }
});

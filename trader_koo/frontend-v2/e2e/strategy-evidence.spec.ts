import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const snapshotPath = fileURLToPath(
  new URL("../../research/strategy_evidence_20260822.json", import.meta.url),
);
const productionEvidence = JSON.parse(readFileSync(snapshotPath, "utf8"));
productionEvidence.provenance.href =
  `/api/research/strategy-evidence/${productionEvidence.provenance.artifact_sha256}` +
  `/inputs/${productionEvidence.provenance.input_hash_sha256}`;

const maliciousFeedback = [
  {
    kind: "family_strength",
    severity: "green",
    title: "bullish reversal shows edge",
    detail: "Five trades have a positive average.",
    action: "Size up in this regime. Consider priority allocation.",
  },
];

const summaryPayload = {
  ok: true,
  overall: {
    total_trades: 5,
    open_count: 0,
    win_rate_pct: 60,
    avg_pnl_pct: 1.2,
    total_pnl_pct: 6,
    avg_r_multiple: 0.3,
    total_return_pct: 0.5,
  },
  by_direction: {},
  by_family: {},
  by_tier: {},
  by_exit_reason: {},
  equity_curve: [],
  recent_trades: [],
  feedback: maliciousFeedback,
  benchmarks: {},
  strategy_evidence: productionEvidence,
};

async function mockApi(page: Page) {
  await page.routeWebSocket(/\/ws\//, (socket) => socket.close());
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    await route.fulfill({ json: { ok: false } });
  });
  await page.route("**/api/paper-trades?*", async (route) => {
    await route.fulfill({ json: { ok: true, count: 0, trades: [] } });
  });
  await page.route("**/api/paper-trades/summary", async (route) => {
    await route.fulfill({ json: summaryPayload });
  });
  await page.route("**/api/daily-report?*", async (route) => {
    await route.fulfill({
      json: {
        ok: true,
        detail: "Snapshot fixture",
        detail_level: "info",
        detail_blocks_main_report: true,
        latest: {
          generated_ts: "2026-08-22T00:00:00Z",
          latest_data: { price_date: "2026-08-21" },
          freshness: {},
          warnings: [],
          signals: {},
          risk_filters: {},
        },
      },
    });
  });
  await page.route(`**${productionEvidence.provenance.href}`, async (route) => {
    await route.fulfill({
      json: { ok: true, strategy_evidence: productionEvidence },
    });
  });
}

async function expectInadequateEvidence(page: Page) {
  const panel = page.getByTestId("strategy-evidence-state");
  await expect(panel).toBeVisible();
  await expect(page.getByTestId("strategy-evidence-status")).toHaveText(
    "Research only / insufficient history",
  );
  await expect(page.getByTestId("strategy-artifact-hash")).toContainText(
    productionEvidence.provenance.artifact_sha256,
  );
  await expect(page.getByTestId("strategy-input-hash")).toContainText(
    productionEvidence.provenance.input_hash_sha256,
  );
  await expect(panel).toContainText("Results are descriptive");
  await expect(panel).toContainText("Invalid / unresolved");
}

test("portfolio cannot render an actionable recommendation from inadequate evidence", async ({ page }) => {
  await mockApi(page);
  await page.goto("/paper-trades");

  await expectInadequateEvidence(page);
  await expect(page.getByText("Descriptive Review")).toBeVisible();
  await expect(page.getByText("Promotion Review")).toHaveCount(0);
  await expect(page.getByText(/size up/i)).toHaveCount(0);
  await expect(page.getByText(/priority allocation/i)).toHaveCount(0);
  await expect(page.getByText(/shows edge/i)).toHaveCount(0);
  await expect(page.getByText(/do not change allocation or admission/i)).toBeVisible();
});

test("research journey shows the same fail-closed production snapshot", async ({ page }) => {
  await mockApi(page);
  await page.goto("/report");

  await expectInadequateEvidence(page);
  const href = await page.getByTestId("strategy-artifact-hash").getAttribute("href");
  expect(href).toBe(productionEvidence.provenance.href);
  const payload = await page.evaluate(async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`provenance request failed: ${response.status}`);
    return response.json();
  }, href!);
  expect(payload.strategy_evidence.provenance).toMatchObject({
    artifact_sha256: productionEvidence.provenance.artifact_sha256,
    input_hash_sha256: productionEvidence.provenance.input_hash_sha256,
  });
});

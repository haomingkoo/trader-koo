import { expect, test, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const artifactPath = fileURLToPath(
  new URL("../../research/strategy_evidence_artifact_20260822.json", import.meta.url),
);
const inputsPath = fileURLToPath(
  new URL("../../research/strategy_evidence_inputs_20260822.json", import.meta.url),
);
const artifactBytes = readFileSync(artifactPath);
const inputBytes = readFileSync(inputsPath);
const nextOpenArtifactPath = fileURLToPath(
  new URL("../../../tests/fixtures/next_open_baseline_schema_v2.json", import.meta.url),
);
const nextOpenBaseline = {
  ...JSON.parse(readFileSync(nextOpenArtifactPath, "utf8")),
  available: true,
  artifact_path: "next_open_baseline_artifact_20260823.json",
};
const productionEvidence = {
  ...JSON.parse(artifactBytes.toString("utf8")),
  provenance: {
    artifact_name: "strategy_evidence_artifact_20260822.json",
    artifact_sha256: createHash("sha256").update(artifactBytes).digest("hex"),
    input_hash_sha256: createHash("sha256").update(inputBytes).digest("hex"),
    artifact_spec_hash_sha256: null,
    verified: true,
    href: "",
  },
};
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

const outperformingBenchmarks = {
  spy_buy_hold: { return_pct: -1, period_days: 20, start_price: 100, end_price: 99 },
  unfiltered_setups: {
    trades: 100,
    win_rate: 45,
    return_pct: -1,
    total_return_pct: -10,
    sharpe: null,
    hold_days: 10,
  },
};

const summaryPayload = (evidence: Record<string, unknown>) => ({
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
  benchmarks: outperformingBenchmarks,
  strategy_evidence: evidence,
});

async function mockApi(page: Page, evidence: Record<string, unknown> = productionEvidence) {
  await page.routeWebSocket(/\/ws\//, (socket) => socket.close());
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    await route.fulfill({ json: { ok: false } });
  });
  await page.route("**/api/paper-trades?*", async (route) => {
    await route.fulfill({ json: { ok: true, count: 0, trades: [] } });
  });
  await page.route("**/api/paper-trades/summary*", async (route) => {
    await route.fulfill({ json: summaryPayload(evidence) });
  });
  await page.route("**/api/research/next-open-baseline", async (route) => {
    await route.fulfill({ json: { ok: true, baseline: nextOpenBaseline } });
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
  await expect(page.getByText(/portfolio outperforms/i)).toHaveCount(0);
  await expect(page.getByText(/taken trades beat/i)).toHaveCount(0);
  await expect(page.getByText(/observed portfolio return is above/i)).toBeVisible();
  await expect(page.getByText(/observed taken-trade average is above/i)).toBeVisible();
  await expect(page.getByText(/do not change allocation or admission/i)).toBeVisible();
  const baseline = page.getByTestId("next-open-baseline");
  await expect(baseline).toBeVisible();
  await expect(baseline).toContainText("Descriptive only / not promotion eligible");
  await expect(baseline).toContainText("0.45%");
  await expect(baseline).toContainText("Return basis:");
  await expect(baseline).toContainText("Benchmark basis:");
  await expect(baseline).toContainText("Full-investment SPY");
  await expect(baseline).toContainText("0.45%");
  await expect(baseline).toContainText("-8.62%");
  await expect(baseline).toContainText("Matched SPY target / filled:");
  await expect(baseline).toContainText("$49,927.88 / $49,927.88");
  await expect(baseline).toContainText("Excluded calls0");
  await expect(page.getByTestId("next-open-artifact-hash")).toContainText(
    nextOpenBaseline.provenance.artifact_sha256,
  );
});

test("forged eligibility assertions cannot bypass missing evidence gates", async ({ page }) => {
  const forged = structuredClone(productionEvidence);
  Object.assign(forged, {
    lifecycle_stage: "promotion_review",
    readiness_status: "eligible_for_human_promotion_review",
    readiness_reasons: [],
    observation_count: 0,
    traded_signal_date_count: 0,
    effective_non_overlapping_block_count: 0,
    decision_eligible: true,
    causal_validity: { valid: true, reasons: [] },
    consumed_window: { consumed: true, reusable_for_policy_selection: true, status: "fresh" },
    return_basis: "split_adjusted_total_return_net_of_costs",
  });
  await mockApi(page, forged);
  await page.goto("/paper-trades");

  await expect(page.getByText("Promotion Review")).toHaveCount(0);
  await expect(page.getByText(/size up/i)).toHaveCount(0);
  await expect(page.getByText(/portfolio outperforms/i)).toHaveCount(0);
});

test("partial evidence package fails closed instead of crashing", async ({ page }) => {
  await mockApi(page, {
    readiness_status: "eligible_for_human_promotion_review",
    decision_eligible: true,
  });
  await page.goto("/paper-trades");

  await expect(page.getByTestId("strategy-evidence-state")).toBeVisible();
  await expect(page.getByTestId("strategy-evidence-status")).toHaveText(
    "Research only / evidence unavailable",
  );
  await expect(page.getByText("Promotion Review")).toHaveCount(0);
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

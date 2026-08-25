import { expect, test } from "@playwright/test";

const expectedSha = process.env.PLAYWRIGHT_EXPECTED_SHA;
const apiKey = process.env.PLAYWRIGHT_ADMIN_API_KEY;

test("exact inactive release works through production pages", async ({ page, request }) => {
  expect(expectedSha).toBeTruthy();
  expect(apiKey).toBeTruthy();
  const release = await request.get("/api/release");
  expect(release.ok()).toBeTruthy();
  expect(await release.json()).toMatchObject({ ok: true, git_sha: expectedSha });

  await page.addInitScript((key) => {
    localStorage.setItem("trader_koo_admin_key", key);
  }, apiKey as string);

  await page.goto("/report");
  await expect(page.getByRole("heading", { name: "Daily Report" })).toBeVisible();
  const unavailableReport = page.getByTestId("report-unavailable-state");
  if (await unavailableReport.count()) {
    await expect(unavailableReport).toContainText("Report unavailable:");
    await expect(unavailableReport).toContainText("stay hidden until a complete report passes publication checks");
  }
  await page.goto("/chart");
  await expect(page.getByRole("heading", { name: /^Chart/ })).toBeVisible();
  await page.goto("/paper-trades");
  await expect(page.getByTestId("paper-campaign-selector")).toBeVisible();
  await expect(page.getByTestId("paper-campaign-health")).toContainText(/draft|frozen|active/);
  await expect(page.getByTestId("paper-campaign-health")).not.toContainText("Campaign is ready");
  await page.goto("/experiments");
  await expect(page.getByTestId("experiment-results-page")).toBeVisible();
  await expect(page.getByRole("button", { name: /activate/i })).toHaveCount(0);
  await page.goto("/vix");
  await expect(page.getByRole("heading", { name: "VIX / Regime Analysis" })).toBeVisible();
  await expect(page.getByText("No regime context available yet")).toHaveCount(0);
  await page.goto("/methodology");
  await expect(page.getByText(/Current campaign:/)).toBeVisible();
  await expect(page.getByText(/Frozen v1 figures below are historical and unreconciled/)).toBeVisible();
  await expect(page.getByRole("link", { name: "Agent Traces" })).toHaveCount(0);
  await expect(page.getByText("Market closed")).toHaveCount(0);
  await page.goto("/agent-observability");
  await expect(page.getByTestId("agent-observability-page")).toBeVisible();
  await page.getByRole("button", { name: "Load" }).click();
  await expect(page.getByTestId("agent-observability-page")).toContainText("Recorded traces");
});

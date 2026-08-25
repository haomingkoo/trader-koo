import { expect, test } from "@playwright/test";

test("the public product is operational, not merely reachable", async ({ page, request }) => {
  test.skip(
    process.env.PLAYWRIGHT_REQUIRE_LIVE_PRODUCT !== "1",
    "Run after report publication and paper-campaign activation.",
  );

  const [statusResponse, reportResponse, paperResponse, vixResponse, cryptoResponse, marketResponse] =
    await Promise.all([
      request.get("/api/status"),
      request.get("/api/daily-report"),
      request.get("/api/paper-trades/summary"),
      request.get("/api/vix-metrics"),
      request.get("/api/crypto/prices"),
      request.get("/api/market-summary?days=7"),
    ]);
  for (const response of [
    statusResponse,
    reportResponse,
    paperResponse,
    vixResponse,
    cryptoResponse,
    marketResponse,
  ]) {
    expect(response.ok()).toBeTruthy();
  }

  const status = await statusResponse.json();
  const report = await reportResponse.json();
  const paper = await paperResponse.json();
  const vix = await vixResponse.json();
  const crypto = await cryptoResponse.json();
  const market = await marketResponse.json();
  expect(status.freshness.report_fresh).toBe(true);
  expect(report.ok).toBe(true);
  expect(report.latest?.report_run?.publication_verified).toBe(true);
  expect(paper.campaign_health).toMatchObject({ status: "active", write_state: "enabled" });
  expect(typeof vix.vix_vix3m_ratio).toBe("number");
  expect(typeof crypto.prices?.["BTC-USD"]?.price).toBe("number");
  expect(typeof market.tickers?.SPY?.price).toBe("number");

  await page.goto("/");
  await expect(page.getByText("SPY", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("BTC", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Ingest", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Report", { exact: true }).first()).toBeVisible();

  await page.goto("/report");
  await expect(page.getByTestId("report-unavailable-state")).toHaveCount(0);
  await page.goto("/vix");
  await expect(page.getByText("VIX/VIX3M Ratio")).toBeVisible();
  await page.goto("/paper-trades");
  await expect(page.getByTestId("paper-campaign-health")).toContainText("active");
  await expect(page.getByRole("link", { name: "Experiments" })).toHaveCount(0);
});

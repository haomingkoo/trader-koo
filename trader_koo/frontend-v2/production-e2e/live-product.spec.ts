import { expect, test } from "@playwright/test";

test("the public product is operational, not merely reachable", async ({ page, request }) => {
  test.setTimeout(120_000);
  test.skip(
    process.env.PLAYWRIGHT_REQUIRE_LIVE_PRODUCT !== "1",
    "Run after report publication and paper-campaign activation.",
  );

  const [
    statusResponse,
    reportResponse,
    paperResponse,
    vixResponse,
    cryptoResponse,
    marketResponse,
    polymarketResponse,
    hyperliquidResponse,
  ] =
    await Promise.all([
      request.get("/api/status"),
      request.get("/api/daily-report"),
      request.get("/api/paper-trades/summary"),
      request.get("/api/vix-metrics"),
      request.get("/api/crypto/prices"),
      request.get("/api/market-summary?days=7"),
      request.get("/api/polymarket?limit=15"),
      request.get("/api/hyperliquid/history/machibro?days=7"),
    ]);
  for (const response of [
    statusResponse,
    reportResponse,
    paperResponse,
    vixResponse,
    cryptoResponse,
    marketResponse,
    polymarketResponse,
    hyperliquidResponse,
  ]) {
    expect(response.ok()).toBeTruthy();
  }

  const status = await statusResponse.json();
  const report = await reportResponse.json();
  const paper = await paperResponse.json();
  const vix = await vixResponse.json();
  const crypto = await cryptoResponse.json();
  const market = await marketResponse.json();
  const polymarket = await polymarketResponse.json();
  const hyperliquid = await hyperliquidResponse.json();
  expect(status.freshness.report_fresh).toBe(true);
  expect(report.ok).toBe(true);
  expect(report.latest?.report_run?.publication_verified).toBe(true);
  expect(paper.campaign_health).toMatchObject({ status: "active", write_state: "enabled" });
  expect(typeof vix.vix_vix3m_ratio).toBe("number");
  expect(typeof crypto.prices?.["BTC-USD"]?.price).toBe("number");
  expect(typeof market.tickers?.SPY?.price).toBe("number");
  expect(polymarket.ok).toBe(true);
  expect(polymarket.source_fetched_at).toBeTruthy();
  expect(polymarket.events.length).toBeGreaterThan(0);
  for (const event of polymarket.events) {
    expect(event.active_volume_24h).toBeGreaterThanOrEqual(0);
    for (const marketItem of event.markets) {
      expect(marketItem.active).toBe(true);
      expect(marketItem.resolved).toBe(false);
    }
  }
  expect(hyperliquid.portfolio).toMatchObject({ available: true });
  expect(hyperliquid.portfolio.daily.length).toBeGreaterThan(1);
  if (hyperliquid.fill_count === 2000) {
    expect(hyperliquid.execution_coverage).toMatchObject({
      complete: false,
      reason: "hyperliquid_2000_execution_page_cap",
    });
    expect(hyperliquid.stats.win_rate_pct).toBeNull();
  }

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
  await expect(page.getByText("Historical strategy audit")).toBeVisible();
  await expect(page.getByRole("link", { name: "Experiments" })).toHaveCount(0);

  await page.goto("/markets");
  await expect(page.getByText("Source fetched", { exact: false })).toBeVisible();
  await expect(page.getByText("Active 24h", { exact: false }).first()).toBeVisible();

  await page.goto("/hyperliquid");
  await expect(page.getByText("Provider Period P&L")).toBeVisible();
  await expect(page.getByText("Execution statistics are partial", { exact: false })).toBeVisible();
  await expect(page.getByTestId("wallet-win-rate")).toHaveText("—");
});

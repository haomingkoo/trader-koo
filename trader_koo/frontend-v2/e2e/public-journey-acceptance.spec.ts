import { expect, test } from "@playwright/test";

test("public research pages remain useful and lifecycle-honest with partial data", async ({ page }) => {
  await page.routeWebSocket(/\/ws\//, (socket) => socket.close());

  await page.goto("/vix");
  const liveVix = page.getByTestId("vix-live-metrics-only");
  await expect(liveVix).toBeVisible();
  await expect(liveVix).toContainText("sealed daily regime report is unavailable");
  await expect(liveVix).toContainText("13.3%");
  await expect(page.getByText("No regime context available yet")).toHaveCount(0);

  await page.goto("/paper-trades");
  const campaign = page.getByTestId("paper-campaign-health");
  await expect(campaign).toContainText("draft");
  await expect(campaign).not.toContainText("Campaign is ready");

  await page.goto("/methodology");
  await expect(page.getByText(/Current campaign:/)).toContainText("paper-v2");
  await expect(page.getByText(/Frozen v1 figures below/)).toContainText("historical and unreconciled");
  await expect(page.getByText("Frozen v1 Trades")).toBeVisible();
  await expect(page.getByRole("link", { name: "Agent Traces" })).toHaveCount(0);
  await expect(page.getByText("Market closed")).toHaveCount(0);
});

test("experiments prefer usable evidence while retaining failed runs", async ({ page }) => {
  await page.routeWebSocket(/\/ws\//, (socket) => socket.close());
  await page.goto("/experiments");

  const results = page.getByTestId("experiment-results-page");
  await expect(results).toBeVisible();
  await expect(page.getByRole("button", { name: "Next-open setup baseline" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Non-TA challenger tournament" })).toHaveAttribute("aria-pressed", "true");
});

test("missing report fails closed without turning the page into a blank placeholder", async ({ page }) => {
  await page.route("**/api/daily-report**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        ok: false,
        detail: "No complete report passed publication checks.",
        latest: null,
      }),
    });
  });

  await page.goto("/report");
  const unavailable = page.getByTestId("report-unavailable-state");
  await expect(unavailable).toBeVisible();
  await expect(unavailable.getByRole("heading", { name: "Daily Report" })).toBeVisible();
  await expect(unavailable).toContainText("No complete report passed publication checks");
  await expect(unavailable).toContainText("campaign decisions stay hidden");
});

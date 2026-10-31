import { expect, test } from "@playwright/test";

const longDecision =
  "Watch-only. The setup is mixed due to a neutral trend state and lack of candle confirmation. " +
  "Wait for clearer signals from trend, levels, and participation.";

test("setup decisions and metric labels remain readable on desktop", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("/report");

  // Every fixture ticker is unclassified, so the heatmap says so instead of one bucket.
  await expect(page.getByTestId("sector-heatmap-unavailable")).toBeVisible();
  await expect(page.getByText("Unclassified sector")).toHaveCount(0);
  await expect(page.getByText("Not financial advice", { exact: false })).toHaveCount(1);
  await expect(page.getByRole("table").getByText("IP has mixed evidence with a neutral trend and no candle confirmation.")).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Decision note" })).toBeVisible();
  const decision = page.getByRole("table").getByText(longDecision);
  await expect(decision).toBeVisible();
  await expect(page.getByRole("table").getByText("elevated iv event risk")).toBeVisible();
  await expect(page.getByRole("table").getByText("Rules 0%")).toBeVisible();

  await expect(decision).toHaveCSS("white-space", "normal");
  await expect(decision).not.toHaveCSS("text-overflow", "ellipsis");
});

test("setup cards retain labels below the wide-table breakpoint", async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 1200 });
  await page.goto("/report");

  await expect(page.getByRole("paragraph").filter({ hasText: longDecision })).toBeVisible();
  await expect(page.getByText("Probability", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Options", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("News", { exact: true }).first()).toBeVisible();
  // Both setups share macro news score 25, so it is shown once for the section.
  await expect(page.locator("dt", { hasText: /^Macro$/ })).toHaveCount(0);
  await expect(page.getByText("for every setup")).toBeVisible();
  await expect(page.getByText("55%, n=2116")).toBeVisible();
  await expect(page.getByText("Rules 0%").first()).toBeVisible();
});

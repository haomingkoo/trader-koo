import { expect, test } from "@playwright/test";

const longDecision =
  "Watch-only. The setup is mixed due to a neutral trend state and lack of candle confirmation. " +
  "Wait for clearer signals from trend, levels, and participation.";

test("setup decisions and metric labels remain readable on desktop", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("/report");

  await expect(page.getByText("Unclassified sector")).toBeVisible();
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
  await expect(page.getByText("Macro", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("55%, n=2116")).toBeVisible();
  await expect(page.getByText("Rules 0%").first()).toBeVisible();
});

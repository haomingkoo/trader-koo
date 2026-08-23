import { expect, test } from "@playwright/test";

test("authenticated agent view shows only real redacted model spans", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem("trader_koo_admin_key", "e2e-agent-key");
  });
  await page.goto("/agent-observability");
  await page.getByRole("button", { name: "Load" }).click();

  const view = page.getByTestId("agent-observability-page");
  await expect(view).toContainText("Recorded traces");
  await expect(view).toContainText("narrative_rewriter");
  await expect(view).toContainText("azure_openai");
  await expect(view).toContainText("gpt-fixture");
  await expect(view).toContainText("passed");
  await expect(view).toContainText("rephrased");
  await expect(view).toContainText("setup-grounding-v1");
  await expect(view).toContainText("narrative only: changed");
  await expect(view).toContainText("credentials stored: no");
  await expect(view).not.toContainText("api-key");
  await expect(view).not.toContainText("must-not-persist");
});

import { expect, test } from "@playwright/test";

test("paper campaign renders a real sealed decision through API and UI", async ({ page }) => {
  await page.routeWebSocket(/\/ws\//, (socket) => socket.close());

  const api = await page.request.get("http://127.0.0.1:8000/api/paper-trades/decisions");
  expect(api.ok()).toBeTruthy();
  const contract = await api.json();
  expect(contract.decisions[0]).toMatchObject({
    report_run_id: "browser-real-api-report",
    candidate_rank: 1,
    ticker: "REJECT",
    disposition: "rejected",
    final_gate: "eligibility.tier",
    reason_code: "tier_below_minimum",
    tradeability: "not_actionable",
  });

  await page.goto("/paper-trades");

  const panel = page.getByTestId("paper-campaign-health");
  await expect(panel).toContainText("paper-v2");
  await expect(panel).toContainText("draft");
  await expect(panel).toContainText("paper-campaign-v2.0");
  await expect(panel).toContainText("eligibility.tier/tier_below_minimum (1)");
  await expect(panel.getByText("unhealthy")).toBeVisible();

  const decisions = page.getByTestId("paper-campaign-decisions");
  await expect(decisions).toContainText("1");
  await expect(decisions).toContainText("REJECT");
  await expect(decisions).toContainText("not actionable");
  await expect(decisions).toContainText("eligibility.tier/tier_below_minimum");

  await page.getByTestId("paper-campaign-selector").selectOption("paper-v1");
  await expect(panel).toContainText("paper-v1");
  await expect(panel).toContainText("frozen");
  await expect(page.getByTestId("paper-campaign-decisions")).toHaveCount(0);
});

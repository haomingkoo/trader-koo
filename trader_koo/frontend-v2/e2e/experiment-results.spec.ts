import { expect, test } from "@playwright/test";

test("failed tournament remains visible with sealed holdout untouched", async ({ page }) => {
  await page.routeWebSocket(/\/ws\//, (socket) => socket.close());

  const api = await page.request.get("http://127.0.0.1:8000/api/research/experiments");
  expect(api.ok()).toBeTruthy();
  const contract = await api.json();
  const tournament = contract.experiments.find(
    (item: { experiment_id: string }) => item.experiment_id === "challenger-tournament",
  );
  expect(tournament).toMatchObject({
    evidence_label: "invalid",
    status: "blocked_before_validation",
    selected: false,
    automatic_promotion: false,
    heldout: { accessed: false, access_log: [] },
  });

  await page.goto("/experiments");
  await page.getByRole("button", { name: "Non-TA challenger tournament" }).click();

  const results = page.getByTestId("experiment-results-page");
  await expect(results).toContainText("invalid");
  await expect(results).toContainText("consistent total return basis required");
  await expect(results).toContainText("held-out untouched");
  await expect(results).toContainText("C1");
  await expect(results).toContainText("C2");
  await expect(results).toContainText("C3");
  await expect(results).toContainText("Comparable curves are unavailable");
  await expect(results.getByText("N/A").first()).toBeVisible();
  await expect(results).toContainText("Complete ledger unavailable");
  await expect(page.getByRole("button", { name: /activate/i })).toHaveCount(0);
});

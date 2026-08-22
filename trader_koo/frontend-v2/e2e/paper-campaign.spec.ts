import { expect, test } from "@playwright/test";

test("paper campaign funnel exposes policy, exact gates, and unhealthy streak", async ({ page }) => {
  await page.routeWebSocket(/\/ws\//, (socket) => socket.close());
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    await route.fulfill({ json: { ok: false } });
  });
  await page.route("**/api/paper-trades?*", async (route) => {
    await route.fulfill({ json: { ok: true, count: 0, trades: [] } });
  });
  await page.route("**/api/paper-trades/summary", async (route) => {
    await route.fulfill({
      json: {
        ok: true,
        overall: {
          total_trades: 42,
          open_count: 0,
          win_rate_pct: 52.4,
          avg_pnl_pct: 1.53,
          total_pnl_pct: 39.7,
          avg_r_multiple: 0.29,
          total_return_pct: 3.97,
        },
        by_direction: {}, by_family: {}, by_tier: {}, by_exit_reason: {},
        equity_curve: [], recent_trades: [], feedback: [], benchmarks: {},
        policy: {
          bot_version: "v2.0.0", decision_version: "paper-campaign-v2.0",
          min_tier: "B", min_score: 60, max_open: 20, expiry_days: 10,
          min_reward_r_multiple: 2.0, high_vol_atr_pct: 6,
          qualifying_tiers: ["A", "B"],
          qualifying_actionability: ["higher-probability", "conditional"],
          position_size_pct: { A: 12, B: 8, C: 5 },
          caution_position_scale: 0.65, high_vol_position_scale: 0.75,
          earnings_position_scale: 0.6,
        },
        campaign_health: {
          available: true,
          campaign_id: "paper-v2",
          label: "Paper Campaign V2",
          policy_version: "paper-campaign-v2.0",
          status: "active",
          starting_capital: 1000000,
          consecutive_eligible_zero_admission_reports: 3,
          zero_admission_streak_limit: 3,
          replay_live_parity: "not_measured",
          healthy: false,
          latest_report: {
            report_run_id: "report-run-3", report_date: "2026-08-21",
            generated_ts: "report-run-3", ranked: 40, eligible: 14,
            rejected: 40, admitted: 0, exposure_pct: 0,
            conversion_rate_pct: 0,
            rejections_by_gate: [
              { gate: "reward_risk", reason_code: "minimum_reward_r_not_met", count: 13 },
              { gate: "critic", reason_code: "critic_rejected", count: 1 },
            ],
          },
          campaigns: [
            { campaign_id: "paper-v1", label: "Paper Campaign v1", policy_version: "paper-trade-eval-v1", status: "frozen", starting_capital: 1000000, trade_count: 42 },
            { campaign_id: "paper-v2", label: "Paper Campaign V2", policy_version: "paper-campaign-v2.0", status: "active", starting_capital: 1000000, trade_count: 0 },
          ],
        },
      },
    });
  });

  await page.goto("/paper-trades");

  const panel = page.getByTestId("paper-campaign-health");
  await expect(panel).toContainText("paper-v2");
  await expect(panel).toContainText("paper-campaign-v2.0");
  await expect(panel).toContainText("Ranked");
  await expect(panel).toContainText("40");
  await expect(panel).toContainText("14");
  await expect(panel).toContainText("0.0%");
  await expect(panel).toContainText("3/3");
  await expect(panel).toContainText("reward_risk/minimum_reward_r_not_met (13)");
  await expect(panel).toContainText("Paper Campaign v1: 42 immutable trade(s), frozen");
  await expect(panel.getByText("unhealthy")).toBeVisible();
});

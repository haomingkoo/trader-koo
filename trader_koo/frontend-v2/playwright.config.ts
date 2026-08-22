import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.PLAYWRIGHT_PORT ?? 3000);
const python = process.env.E2E_PYTHON || "python3";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: "line",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: `cd ../.. && TRADER_KOO_DB_PATH=/tmp/trader-koo-paper-e2e.db ${python} -m uvicorn tests.e2e_paper_app:app --host 127.0.0.1 --port 8000`,
      url: "http://127.0.0.1:8000/api/paper-trades/summary",
      reuseExistingServer: false,
    },
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${port}`,
      url: `http://127.0.0.1:${port}`,
      reuseExistingServer: !process.env.CI,
    },
  ],
});

import { defineRailway, preserve, project, service, volume } from "railway/iac";

export default defineRailway(() => {
  const traderKooVolume = volume("trader-koo-volume", { alerts: { usage: { "100": {}, "80": {}, "95": {} } }, allowOnlineResize: true, region: "asia-southeast1-eqsg3a", sizeMB: 5000 });
  const traderKoo = service("trader-koo", {
    // No source: releases go through .github/workflows/dark-deploy.yml (railway up), which refuses a GitHub source.
    build: { builder: "NIXPACKS", buildCommand: "bash build.sh" },
    deploy: {
      startCommand: "bash start.sh",
      healthcheckPath: "/api/health",
      healthcheckTimeout: 3600,
      restartPolicyType: "ON_FAILURE",
    },
    replicas: { "asia-southeast1-eqsg3a": 1 },
    domains: ["trader.kooexperience.com"],
    volumeMounts: {
      "/data": traderKooVolume,
    },
    env: {
      FINNHUB_API_KEY: preserve(),
      FRED_API_KEY: preserve(),
      GOOGLE_API_KEY: preserve(),
      TELEGRAM_BOT_TOKEN: preserve(),
      TELEGRAM_CHAT_ID: preserve(),
      TRADER_KOO_ADMIN_AUTH_BLOCK_SEC: preserve(),
      TRADER_KOO_ADMIN_AUTH_MAX_FAILS: preserve(),
      TRADER_KOO_ADMIN_AUTH_WINDOW_SEC: preserve(),
      TRADER_KOO_ALPHA_VANTAGE_KEY: preserve(),
      TRADER_KOO_API_KEY: preserve(),
      TRADER_KOO_AUTO_EMAIL: preserve(),
      TRADER_KOO_AUTO_RESUME_INTERRUPTED_PIPELINE: preserve(),
      TRADER_KOO_AUTO_RESUME_POST_INGEST: preserve(),
      TRADER_KOO_CORS_ORIGINS: preserve(),
      TRADER_KOO_DB_PATH: preserve(),
      TRADER_KOO_EMAIL_TRANSPORT: preserve(),
      TRADER_KOO_EXPOSE_STATUS_INTERNAL: preserve(),
      TRADER_KOO_GEMINI_MODEL: preserve(),
      TRADER_KOO_GIT_SHA: preserve(),
      TRADER_KOO_INGEST_RETRY_FAILED_BACKOFF_SEC: preserve(),
      TRADER_KOO_INGEST_RETRY_FAILED_PASSES: preserve(),
      TRADER_KOO_INGEST_RUNNING_STALE_MIN: preserve(),
      TRADER_KOO_LLM_ENABLED: preserve(),
      TRADER_KOO_LLM_FAIL_ALERT_TO: preserve(),
      TRADER_KOO_LOG_DIR: preserve(),
      TRADER_KOO_PAPER_TRADE_ENABLED: preserve(),
      TRADER_KOO_REPORT_DIR: preserve(),
      TRADER_KOO_REPORT_EMAIL_TO: preserve(),
      TRADER_KOO_REQUIRE_FULL_DATASET: preserve(),
      TRADER_KOO_RESEND_API_KEY: preserve(),
      TRADER_KOO_RESEND_FROM: preserve(),
      TRADER_KOO_SMTP_FROM: preserve(),
      TRADER_KOO_SMTP_HOST: preserve(),
      TRADER_KOO_SMTP_PASS: preserve(),
      TRADER_KOO_SMTP_PORT: preserve(),
      TRADER_KOO_SMTP_SECURITY: preserve(),
      TRADER_KOO_SMTP_USER: preserve(),
    },
  });

  return project("graceful-perfection", {
    resources: [traderKoo, traderKooVolume],
  });
});

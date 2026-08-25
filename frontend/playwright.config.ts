import { defineConfig, devices } from "@playwright/test";

const managesWebServer = process.env.PLAYWRIGHT_MANAGED_SERVER !== "1";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: managesWebServer
    ? {
        command: "npm run dev -- --host 0.0.0.0 --port 4173",
        url: "http://127.0.0.1:4173",
        reuseExistingServer: !process.env.CI,
      }
    : undefined,
});
